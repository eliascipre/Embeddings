#!/usr/bin/env python3
"""
Cliente optimizado para Qwen3-Embedding-0.6B
Con circuit breaker, rate limiting y procesamiento en lotes
"""

import asyncio
import aiohttp
import logging
import time
import backoff
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class Qwen3Config:
    """Configuración específica para Qwen3-Embedding-0.6B"""
    endpoint: str = "https://wfuosp4mnqkimde9.us-east-1.aws.endpoints.huggingface.cloud"
    model_name: str = "Qwen3-Embedding-0.6B"
    max_context_length: int = 32000  # 32k tokens de contexto
    embedding_dimensions: int = 1024  # Dimensiones por defecto (configurable de 32-1024) - máxima calidad
    max_batch_size: int = 32  # Tamaño máximo de lote optimizado
    rate_limit_per_minute: int = 100  # Límite de requests por minuto
    timeout_seconds: int = 300
    retry_attempts: int = 5
    token: str = ""

class CircuitBreaker:
    """Circuit breaker para manejo de errores"""
    
    def __init__(self, failure_threshold: int = 10, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def can_execute(self) -> bool:
        """Verificar si se puede ejecutar la operación"""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """Registrar éxito"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def record_failure(self):
        """Registrar fallo"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker abierto después de {self.failure_count} fallos")

class Qwen3EmbeddingClient:
    """Cliente optimizado para Qwen3-Embedding-0.6B"""
    
    def __init__(self, config: Qwen3Config):
        self.config = config
        self.session = None
        self.rate_limiter = asyncio.Semaphore(10)  # Máximo 10 requests concurrentes
        self.request_delay = 0.6  # Delay entre requests (60/100 = 0.6s)
        self.circuit_breaker = CircuitBreaker()
        
        # Estadísticas
        self.stats = {
            'requests_made': 0,
            'requests_failed': 0,
            'rate_limit_hits': 0,
            'total_response_time': 0.0,
            'total_tokens_processed': 0,
            'total_embeddings_generated': 0,
            'circuit_breaker_activations': 0
        }
    
    async def __aenter__(self):
        """Context manager para inicializar la sesión"""
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        # Solo agregar token si está disponible (para endpoints privados)
        if self.config.token and self.config.token.strip():
            headers['Authorization'] = f'Bearer {self.config.token}'
        
        self.session = aiohttp.ClientSession(
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager para cerrar la sesión"""
        if self.session:
            await self.session.close()
    
    def _prepare_payload(self, texts: List[str], dimensions: Optional[int] = None) -> Dict[str, Any]:
        """Preparar el payload para la API de Qwen3"""
        # Usar dimensiones especificadas o las por defecto
        embedding_dimensions = dimensions or self.config.embedding_dimensions
        
        # Validar dimensiones (Qwen3 soporta de 32 a 1024)
        if not (32 <= embedding_dimensions <= 1024):
            logger.warning(f"Dimensiones {embedding_dimensions} fuera del rango [32, 1024], usando {self.config.embedding_dimensions}")
            embedding_dimensions = self.config.embedding_dimensions
        
        payload = {
            "inputs": texts,
            "parameters": {
                "dimensions": embedding_dimensions,
                "normalize": True,  # Normalizar embeddings para mejor similitud coseno
                "instruction": "Represent the following text for retrieval:"  # Instrucción para mejor rendimiento
            }
        }
        
        return payload
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=5,
        base=2,
        max_value=30
    )
    async def generate_embeddings_batch(self, texts: List[str], 
                                      dimensions: Optional[int] = None,
                                      batch_id: Optional[str] = None) -> List[List[float]]:
        """Generar embeddings para un lote de textos"""
        if not self.circuit_breaker.can_execute():
            raise Exception("Circuit breaker está abierto")
        
        if not texts:
            return []
        
        # Validar tamaño del lote
        if len(texts) > self.config.max_batch_size:
            logger.warning(f"Lote demasiado grande ({len(texts)}), dividiendo en lotes más pequeños")
            return await self._process_large_batch(texts, dimensions)
        
        async with self.rate_limiter:
            try:
                start_time = time.time()
                
                # Preparar payload
                payload = self._prepare_payload(texts, dimensions)
                
                # Realizar request
                async with self.session.post(
                    self.config.endpoint,
                    json=payload
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        # Extraer embeddings
                        if isinstance(result, list) and len(result) > 0:
                            embeddings = result
                        elif isinstance(result, dict) and 'embeddings' in result:
                            embeddings = result['embeddings']
                        else:
                            raise ValueError(f"Formato de respuesta inesperado: {result}")
                        
                        # Actualizar estadísticas
                        self.stats['requests_made'] += 1
                        self.stats['total_response_time'] += time.time() - start_time
                        self.stats['total_tokens_processed'] += sum(len(text.split()) for text in texts)
                        self.stats['total_embeddings_generated'] += len(embeddings)
                        
                        # Registrar éxito
                        self.circuit_breaker.record_success()
                        
                        # Delay para rate limiting
                        await asyncio.sleep(self.request_delay)
                        
                        return embeddings
                    
                    elif response.status == 429:  # Rate limit
                        self.stats['rate_limit_hits'] += 1
                        wait_time = 60  # Esperar 1 minuto
                        logger.warning(f"Rate limit alcanzado, esperando {wait_time} segundos")
                        await asyncio.sleep(wait_time)
                        raise aiohttp.ClientError("Rate limit exceeded")
                    
                    elif response.status == 503:  # Service unavailable
                        wait_time = 30  # Esperar 30 segundos
                        logger.warning(f"Servicio no disponible, esperando {wait_time} segundos")
                        await asyncio.sleep(wait_time)
                        raise aiohttp.ClientError("Service unavailable")
                    
                    else:
                        error_text = await response.text()
                        raise aiohttp.ClientError(f"HTTP {response.status}: {error_text}")
            
            except Exception as e:
                self.stats['requests_failed'] += 1
                self.circuit_breaker.record_failure()
                
                if self.circuit_breaker.state == "OPEN":
                    self.stats['circuit_breaker_activations'] += 1
                    logger.error(f"Circuit breaker activado: {e}")
                
                raise
    
    async def _process_large_batch(self, texts: List[str], dimensions: Optional[int] = None) -> List[List[float]]:
        """Procesar lote grande dividiéndolo en lotes más pequeños"""
        all_embeddings = []
        
        for i in range(0, len(texts), self.config.max_batch_size):
            batch_texts = texts[i:i + self.config.max_batch_size]
            batch_embeddings = await self.generate_embeddings_batch(batch_texts, dimensions)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    async def generate_single_embedding(self, text: str, dimensions: Optional[int] = None) -> np.ndarray:
        """Generar embedding para un solo texto"""
        embeddings = await self.generate_embeddings_batch([text], dimensions)
        return np.array(embeddings[0])
    
    async def generate_embeddings_async(self, texts: List[str], 
                                      dimensions: Optional[int] = None,
                                      batch_size: Optional[int] = None) -> List[np.ndarray]:
        """Generar embeddings de forma asíncrona para múltiples textos"""
        if not texts:
            return []
        
        batch_size = batch_size or self.config.max_batch_size
        all_embeddings = []
        
        # Procesar en lotes
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = await self.generate_embeddings_batch(batch_texts, dimensions)
            
            # Convertir a numpy arrays
            for embedding in batch_embeddings:
                all_embeddings.append(np.array(embedding))
        
        return all_embeddings
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cliente"""
        stats = self.stats.copy()
        
        if stats['requests_made'] > 0:
            stats['avg_response_time'] = stats['total_response_time'] / stats['requests_made']
            stats['success_rate'] = (stats['requests_made'] - stats['requests_failed']) / stats['requests_made']
        else:
            stats['avg_response_time'] = 0.0
            stats['success_rate'] = 0.0
        
        stats['circuit_breaker_state'] = self.circuit_breaker.state
        stats['failure_count'] = self.circuit_breaker.failure_count
        
        return stats
    
    def reset_stats(self):
        """Reiniciar estadísticas"""
        self.stats = {
            'requests_made': 0,
            'requests_failed': 0,
            'rate_limit_hits': 0,
            'total_response_time': 0.0,
            'total_tokens_processed': 0,
            'total_embeddings_generated': 0,
            'circuit_breaker_activations': 0
        }
    
    async def cleanup(self):
        """Limpiar recursos"""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info("Cliente Qwen3 limpiado")

class Qwen3BatchProcessor:
    """Procesador de lotes optimizado para Qwen3"""
    
    def __init__(self, client: Qwen3EmbeddingClient, batch_size: int = 32):
        self.client = client
        self.batch_size = batch_size
        self.queue = asyncio.Queue()
        self.results = {}
        self.processing = False
    
    async def process_batch(self, texts: List[str], 
                          dimensions: Optional[int] = None) -> List[np.ndarray]:
        """Procesar lote de textos"""
        if not texts:
            return []
        
        # Dividir en lotes del tamaño especificado
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = await self.client.generate_embeddings_batch(
                batch_texts, dimensions
            )
            
            # Convertir a numpy arrays
            for embedding in batch_embeddings:
                all_embeddings.append(np.array(embedding))
        
        return all_embeddings
    
    async def process_stream(self, text_stream, dimensions: Optional[int] = None):
        """Procesar stream de textos de forma asíncrona"""
        async for texts in text_stream:
            embeddings = await self.process_batch(texts, dimensions)
            yield embeddings
