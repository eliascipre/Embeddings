#!/usr/bin/env python3
"""
Cliente simplificado para Qwen3-Embedding-0.6B
Sin circuit breaker para evitar dependencias problemáticas
"""

import asyncio
import aiohttp
import logging
import time
import backoff
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

# Configuración local
from config_supabase import get_embedding_config

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Qwen3Config:
    """Configuración específica para Qwen3-Embedding-0.6B"""
    endpoint: str = "https://wfuosp4mnqkimde9.us-east-1.aws.endpoints.huggingface.cloud"
    model_name: str = "Qwen3-Embedding-0.6B"
    max_context_length: int = 32000  # 32k tokens de contexto
    embedding_dimensions: int = 1024  # Dimensiones por defecto (configurable de 32-1024) - máxima calidad
    max_batch_size: int = 50  # Tamaño máximo de lote
    rate_limit_per_minute: int = 100  # Límite de requests por minuto
    timeout_seconds: int = 30
    retry_attempts: int = 3

class Qwen3EmbeddingClient:
    """Cliente simplificado para Qwen3-Embedding-0.6B"""
    
    def __init__(self, config: Optional[Qwen3Config] = None):
        self.config = config or Qwen3Config()
        self.session = None
        self.rate_limiter = asyncio.Semaphore(10)  # Máximo 10 requests concurrentes
        self.request_delay = 0.6  # Delay entre requests (60/100 = 0.6s)
        
        # Estadísticas
        self.stats = {
            'requests_made': 0,
            'requests_failed': 0,
            'rate_limit_hits': 0,
            'total_response_time': 0.0,
            'total_tokens_processed': 0,
            'total_embeddings_generated': 0
        }
    
    async def __aenter__(self):
        """Context manager para inicializar la sesión"""
        self.session = aiohttp.ClientSession(
            headers={
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
            timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager para cerrar la sesión"""
        if self.session:
            await self.session.close()
    
    def _prepare_payload(self, texts: List[str], dimensions: Optional[int] = None) -> Dict[str, Any]:
        """Prepara el payload para la API de Qwen3"""
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
        max_tries=3,
        base=2,
        max_value=10
    )
    async def generate_embeddings_batch(self, texts: List[str], 
                                      dimensions: Optional[int] = None,
                                      batch_id: Optional[str] = None) -> List[List[float]]:
        """Genera embeddings en lote usando Qwen3-Embedding-0.6B"""
        
        if not texts:
            return []
        
        # Validar tamaño del lote
        if len(texts) > self.config.max_batch_size:
            logger.warning(f"Lote demasiado grande ({len(texts)}), dividiendo en lotes más pequeños")
            return await self._process_large_batch(texts, dimensions, batch_id)
        
        async with self.rate_limiter:
            start_time = time.time()
            
            try:
                payload = self._prepare_payload(texts, dimensions)
                
                async with self.session.post(
                    self.config.endpoint,
                    json=payload
                ) as response:
                    
                    # Manejar rate limiting
                    if response.status == 429:
                        self.stats['rate_limit_hits'] += 1
                        wait_time = 2 ** self.stats['rate_limit_hits']
                        logger.warning(f"Rate limit alcanzado, esperando {wait_time}s")
                        await asyncio.sleep(wait_time)
                        raise aiohttp.ClientError("Rate limit exceeded")
                    
                    # Manejar otros errores HTTP
                    if response.status >= 400:
                        error_text = await response.text()
                        logger.error(f"Error HTTP {response.status}: {error_text}")
                        response.raise_for_status()
                    
                    # Procesar respuesta
                    embeddings = await response.json()
                    
                    # Validar respuesta
                    if not isinstance(embeddings, list):
                        logger.error(f"Respuesta inesperada: {type(embeddings)}")
                        raise ValueError("Respuesta de API inválida")
                    
                    # Actualizar estadísticas
                    self.stats['requests_made'] += 1
                    self.stats['total_response_time'] += time.time() - start_time
                    self.stats['total_tokens_processed'] += sum(len(text.split()) for text in texts)
                    self.stats['total_embeddings_generated'] += len(embeddings)
                    
                    # Log de progreso
                    if batch_id:
                        logger.info(f"Lote {batch_id}: {len(embeddings)} embeddings generados en {time.time() - start_time:.2f}s")
                    
                    # Delay entre requests
                    await asyncio.sleep(self.request_delay)
                    
                    return embeddings
                    
            except Exception as e:
                self.stats['requests_failed'] += 1
                logger.error(f"Error generando embeddings: {e}")
                raise
    
    async def _process_large_batch(self, texts: List[str], 
                                 dimensions: Optional[int] = None,
                                 batch_id: Optional[str] = None) -> List[List[float]]:
        """Procesa lotes grandes dividiéndolos en lotes más pequeños"""
        all_embeddings = []
        batch_size = self.config.max_batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            sub_batch_id = f"{batch_id}_{i//batch_size + 1}" if batch_id else None
            
            try:
                batch_embeddings = await self.generate_embeddings_batch(
                    batch_texts, dimensions, sub_batch_id
                )
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error procesando sub-lote {sub_batch_id}: {e}")
                # Continuar con el siguiente lote
                continue
        
        return all_embeddings
    
    async def generate_single_embedding(self, text: str, 
                                      dimensions: Optional[int] = None) -> List[float]:
        """Genera embedding para un solo texto"""
        embeddings = await self.generate_embeddings_batch([text], dimensions)
        return embeddings[0] if embeddings else []
    
    def get_embedding_dimensions(self) -> int:
        """Retorna las dimensiones de embedding configuradas"""
        return self.config.embedding_dimensions
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del cliente"""
        stats = self.stats.copy()
        if stats['requests_made'] > 0:
            stats['avg_response_time'] = stats['total_response_time'] / stats['requests_made']
            stats['success_rate'] = (stats['requests_made'] - stats['requests_failed']) / stats['requests_made']
        else:
            stats['avg_response_time'] = 0.0
            stats['success_rate'] = 0.0
        
        return stats
    
    def reset_stats(self):
        """Reinicia las estadísticas"""
        self.stats = {
            'requests_made': 0,
            'requests_failed': 0,
            'rate_limit_hits': 0,
            'total_response_time': 0.0,
            'total_tokens_processed': 0,
            'total_embeddings_generated': 0
        }

class Qwen3BatchProcessor:
    """Procesador de lotes optimizado para Qwen3"""
    
    def __init__(self, client: Qwen3EmbeddingClient):
        self.client = client
        self.config = client.config
    
    async def process_document_chunks(self, chunks: List[Dict[str, Any]], 
                                    batch_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Procesa chunks de documentos en lotes optimizados"""
        if not chunks:
            return []
        
        batch_size = batch_size or self.config.max_batch_size
        processed_chunks = []
        
        # Agrupar chunks por tamaño para optimizar el procesamiento
        chunks_by_size = self._group_chunks_by_size(chunks)
        
        for size_group, size_chunks in chunks_by_size.items():
            logger.info(f"Procesando {len(size_chunks)} chunks de tamaño {size_group}")
            
            # Procesar en lotes
            for i in range(0, len(size_chunks), batch_size):
                batch_chunks = size_chunks[i:i + batch_size]
                batch_id = f"size_{size_group}_batch_{i//batch_size + 1}"
                
                try:
                    # Extraer textos
                    texts = [chunk['text'] for chunk in batch_chunks]
                    
                    # Generar embeddings
                    embeddings = await self.client.generate_embeddings_batch(
                        texts, batch_id=batch_id
                    )
                    
                    # Combinar con metadatos
                    for chunk, embedding in zip(batch_chunks, embeddings):
                        chunk['embedding'] = embedding
                        chunk['embedding_dimensions'] = len(embedding)
                        chunk['generated_at'] = time.time()
                        processed_chunks.append(chunk)
                    
                except Exception as e:
                    logger.error(f"Error procesando lote {batch_id}: {e}")
                    # Agregar chunks sin embedding para procesamiento posterior
                    for chunk in batch_chunks:
                        chunk['embedding'] = None
                        chunk['error'] = str(e)
                        processed_chunks.append(chunk)
        
        return processed_chunks
    
    def _group_chunks_by_size(self, chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Agrupa chunks por tamaño para optimizar el procesamiento"""
        size_groups = {
            'small': [],    # < 100 tokens
            'medium': [],   # 100-500 tokens
            'large': []     # > 500 tokens
        }
        
        for chunk in chunks:
            token_count = chunk.get('token_count', 0)
            if token_count < 100:
                size_groups['small'].append(chunk)
            elif token_count <= 500:
                size_groups['medium'].append(chunk)
            else:
                size_groups['large'].append(chunk)
        
        return {k: v for k, v in size_groups.items() if v}

# Función de utilidad para crear cliente
def create_qwen3_client(dimensions: int = 1024) -> Qwen3EmbeddingClient:
    """Crea un cliente Qwen3 con dimensiones específicas"""
    config = Qwen3Config(embedding_dimensions=dimensions)
    return Qwen3EmbeddingClient(config)

# Ejemplo de uso
async def main():
    """Ejemplo de uso del cliente Qwen3"""
    config = Qwen3Config(embedding_dimensions=1024)
    
    async with Qwen3EmbeddingClient(config) as client:
        # Probar con un texto de ejemplo
        texts = [
            "La Administración Nacional de Aduanas de México es responsable del control del comercio exterior.",
            "El comercio exterior está regulado por la Ley Aduanera y sus reglamentos correspondientes.",
            "Los procedimientos aduaneros deben cumplir con las disposiciones legales vigentes.",
            "La importación y exportación de mercancías requiere el cumplimiento de normativas específicas.",
            "El sistema aduanero mexicano se rige por principios de eficiencia y transparencia."
        ]
        
        print("Generando embeddings con Qwen3-Embedding-0.6B...")
        embeddings = await client.generate_embeddings_batch(texts)
        
        print(f"Embeddings generados: {len(embeddings)}")
        print(f"Dimensiones: {len(embeddings[0]) if embeddings else 0}")
        print(f"Estadísticas: {client.get_stats()}")

if __name__ == "__main__":
    asyncio.run(main())
