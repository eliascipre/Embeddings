"""
Cliente optimizado para API de Hugging Face con A100 (2 GPUs, 160GB VRAM)
Maximiza utilización de recursos para procesamiento masivo
"""
import asyncio
import aiohttp
import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
from concurrent.futures import ThreadPoolExecutor
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class HFConfig:
    """Configuración para API de Hugging Face"""
    api_url: str = "https://wfuosp4mnqkimde9.us-east-1.aws.endpoints.huggingface.cloud"
    token: str = os.getenv("HF_TOKEN", "")
    max_concurrent_requests: int = 10  # Reducido para evitar 503
    batch_size: int = 16  # Reducido para menor carga
    timeout: int = 300  # 5 minutos timeout
    retry_attempts: int = 5  # Más intentos
    retry_delay: float = 2.0  # Delay inicial mayor
    batch_delay: float = 0.5  # Delay entre lotes
    circuit_breaker_threshold: int = 10  # Errores consecutivos para activar circuit breaker

class HuggingFaceAPIClient:
    """Cliente optimizado para API de Hugging Face con máxima utilización de recursos"""
    
    def __init__(self, config: HFConfig = None):
        self.config = config or HFConfig()
        self.session = None
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        self.request_count = 0
        self.total_tokens_processed = 0
        self.start_time = None
        self.consecutive_errors = 0  # Para circuit breaker
        self.circuit_breaker_active = False
        self.last_error_time = 0
        
    async def __aenter__(self):
        """Context manager entry"""
        connector = aiohttp.TCPConnector(
            limit=100,  # Máximo de conexiones
            limit_per_host=50,  # Máximo por host
            ttl_dns_cache=300,  # Cache DNS
            use_dns_cache=True,
        )
        
        timeout = aiohttp.ClientTimeout(
            total=self.config.timeout,
            connect=30,
            sock_read=60
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.config.token}",
                "Content-Type": "application/json"
            }
        )
        
        self.start_time = time.time()
        logger.info(f"🚀 Cliente HF iniciado - Max concurrent: {self.config.max_concurrent_requests}")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.session:
            await self.session.close()
        
        if self.start_time:
            elapsed = time.time() - self.start_time
            logger.info(f"📊 Estadísticas HF API:")
            logger.info(f"   Total requests: {self.request_count}")
            logger.info(f"   Total tokens: {self.total_tokens_processed}")
            logger.info(f"   Tiempo total: {elapsed:.2f}s")
            logger.info(f"   Requests/segundo: {self.request_count/elapsed:.2f}")
            logger.info(f"   Tokens/segundo: {self.total_tokens_processed/elapsed:.2f}")
    
    async def generate_embeddings_batch(
        self, 
        texts: List[str], 
        batch_size: int = None
    ) -> List[np.ndarray]:
        """Generar embeddings en lotes optimizados para A100"""
        batch_size = batch_size or self.config.batch_size
        
        if not texts:
            return []
        
        logger.info(f"🔄 Generando embeddings para {len(texts)} textos en lotes de {batch_size}")
        
        # Dividir en lotes
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        # Procesar lotes con delay entre ellos para evitar 503
        all_embeddings = []
        for i, batch in enumerate(batches):
            # Verificar circuit breaker
            if self.circuit_breaker_active:
                wait_time = min(60, 2 ** min(self.consecutive_errors, 6))  # Max 60s
                logger.warning(f"🔒 Circuit breaker activo, esperando {wait_time}s")
                await asyncio.sleep(wait_time)
                self.circuit_breaker_active = False
                self.consecutive_errors = 0
            
            # Procesar lote
            result = await self._process_batch_async(batch, i)
            if result:
                all_embeddings.extend(result)
                self.consecutive_errors = 0  # Reset en éxito
            else:
                self.consecutive_errors += 1
                if self.consecutive_errors >= self.config.circuit_breaker_threshold:
                    self.circuit_breaker_active = True
                    logger.error(f"🚨 Circuit breaker activado después de {self.consecutive_errors} errores")
            
            # Delay entre lotes para evitar saturar el servidor
            if i < len(batches) - 1:  # No delay en el último lote
                await asyncio.sleep(self.config.batch_delay)
        
        logger.info(f"✅ Embeddings generados: {len(all_embeddings)}")
        return all_embeddings
    
    async def _process_batch_async(self, texts: List[str], batch_id: int) -> List[np.ndarray]:
        """Procesar un lote de textos de forma asíncrona"""
        async with self.semaphore:
            return await self._generate_embeddings_single_batch(texts, batch_id)
    
    async def _generate_embeddings_single_batch(
        self, 
        texts: List[str], 
        batch_id: int
    ) -> List[np.ndarray]:
        """Generar embeddings para un lote específico"""
        for attempt in range(self.config.retry_attempts):
            try:
                payload = {
                    "inputs": texts,
                    "parameters": {
                        "normalize": True,  # Normalizar embeddings
                        "return_tensors": "numpy"  # Devolver como numpy
                    }
                }
                
                start_time = time.time()
                async with self.session.post(
                    self.config.api_url,
                    json=payload
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        # Procesar respuesta
                        if isinstance(result, list) and len(result) > 0:
                            # Si es lista de embeddings
                            embeddings = np.array(result)
                            
                            # Verificar dimensiones
                            if len(embeddings.shape) == 1:
                                embeddings = embeddings.reshape(1, -1)
                            
                            # Normalizar si no está normalizado
                            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                            embeddings = embeddings / (norms + 1e-8)
                            
                            # Estadísticas
                            self.request_count += 1
                            self.total_tokens_processed += sum(len(text.split()) for text in texts)
                            
                            elapsed = time.time() - start_time
                            logger.debug(f"✅ Lote {batch_id}: {len(texts)} textos, {elapsed:.2f}s")
                            
                            return [emb for emb in embeddings]
                        else:
                            logger.warning(f"⚠️ Respuesta inesperada del lote {batch_id}: {result}")
                            return []
                    
                    elif response.status == 429:  # Rate limit
                        wait_time = self.config.retry_delay * (2 ** attempt)
                        logger.warning(f"⏳ Rate limit en lote {batch_id}, esperando {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    elif response.status == 503:  # Service Unavailable
                        wait_time = self.config.retry_delay * (2 ** attempt) + 5  # Extra delay para 503
                        logger.warning(f"⚠️ Servicio no disponible en lote {batch_id}, esperando {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Error HTTP {response.status} en lote {batch_id}: {error_text}")
                        if attempt < self.config.retry_attempts - 1:
                            wait_time = self.config.retry_delay * (2 ** attempt)
                            await asyncio.sleep(wait_time)
                            continue
                        return []
            
            except asyncio.TimeoutError:
                logger.warning(f"⏰ Timeout en lote {batch_id}, intento {attempt + 1}")
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay)
                    continue
                return []
            
            except Exception as e:
                logger.error(f"❌ Error en lote {batch_id}, intento {attempt + 1}: {e}")
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay)
                    continue
                return []
        
        return []
    
    async def generate_embeddings_single(self, text: str) -> np.ndarray:
        """Generar embedding para un solo texto"""
        results = await self.generate_embeddings_batch([text], batch_size=1)
        return results[0] if results else np.array([])
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cliente"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            "total_requests": self.request_count,
            "total_tokens": self.total_tokens_processed,
            "elapsed_time": elapsed,
            "requests_per_second": self.request_count / elapsed if elapsed > 0 else 0,
            "tokens_per_second": self.total_tokens_processed / elapsed if elapsed > 0 else 0,
            "max_concurrent": self.config.max_concurrent_requests,
            "batch_size": self.config.batch_size
        }

# Función de conveniencia para uso directo
async def generate_embeddings_hf(
    texts: List[str], 
    api_url: str = None,
    token: str = None,
    batch_size: int = 16
) -> List[np.ndarray]:
    """Función de conveniencia para generar embeddings"""
    config = HFConfig(
        api_url=api_url or "https://wfuosp4mnqkimde9.us-east-1.aws.endpoints.huggingface.cloud",
        token=token or os.getenv("HF_TOKEN", ""),
        batch_size=batch_size
    )
    
    async with HuggingFaceAPIClient(config) as client:
        return await client.generate_embeddings_batch(texts)
