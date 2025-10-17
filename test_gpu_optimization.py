#!/usr/bin/env python3
"""
Script de prueba para verificar la optimización de GPU
"""
import asyncio
import aiohttp
import time
import logging
from typing import List

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_hf_endpoint_performance():
    """Probar el rendimiento del endpoint de Hugging Face"""
    
    endpoint = "https://wfuosp4mnqkimde9.us-east-1.aws.endpoints.huggingface.cloud"
    
    # Textos de prueba
    test_texts = [
        "Este es un documento legal sobre contratos mercantiles en México.",
        "La Constitución Política de los Estados Unidos Mexicanos establece los derechos fundamentales.",
        "El Código Civil Federal regula las relaciones civiles entre particulares.",
        "La Ley Federal del Trabajo protege los derechos de los trabajadores.",
        "El Código de Comercio regula las actividades comerciales en el país."
    ] * 10  # 50 textos en total
    
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    }
    
    async with aiohttp.ClientSession(headers=headers) as session:
        # Probar diferentes tamaños de lote
        batch_sizes = [1, 5, 10, 16, 32]
        
        for batch_size in batch_sizes:
            logger.info(f"🧪 Probando lote de {batch_size} textos...")
            
            # Dividir textos en lotes
            batches = [test_texts[i:i + batch_size] for i in range(0, len(test_texts), batch_size)]
            
            total_time = 0
            total_embeddings = 0
            errors = 0
            
            for i, batch in enumerate(batches):
                try:
                    start_time = time.time()
                    
                    payload = {
                        "inputs": batch,
                        "parameters": {
                            "dimensions": 1024,
                            "normalize": True,
                            "instruction": "Represent the following text for retrieval:"
                        }
                    }
                    
                    async with session.post(endpoint, json=payload) as response:
                        if response.status == 200:
                            result = await response.json()
                            
                            if isinstance(result, list):
                                embeddings = result
                            elif isinstance(result, dict) and 'embeddings' in result:
                                embeddings = result['embeddings']
                            else:
                                raise ValueError(f"Formato inesperado: {result}")
                            
                            batch_time = time.time() - start_time
                            total_time += batch_time
                            total_embeddings += len(embeddings)
                            
                            logger.info(f"   Lote {i+1}: {len(batch)} textos en {batch_time:.2f}s")
                        else:
                            error_text = await response.text()
                            logger.error(f"   Error en lote {i+1}: HTTP {response.status} - {error_text}")
                            errors += 1
                            
                except Exception as e:
                    logger.error(f"   Error en lote {i+1}: {e}")
                    errors += 1
            
            # Calcular estadísticas
            if total_time > 0:
                texts_per_second = total_embeddings / total_time
                avg_time_per_text = total_time / total_embeddings if total_embeddings > 0 else 0
                
                logger.info(f"📊 RESULTADOS PARA LOTE DE {batch_size}:")
                logger.info(f"   - Total embeddings: {total_embeddings}")
                logger.info(f"   - Tiempo total: {total_time:.2f}s")
                logger.info(f"   - Textos/segundo: {texts_per_second:.2f}")
                logger.info(f"   - Tiempo promedio por texto: {avg_time_per_text:.4f}s")
                logger.info(f"   - Errores: {errors}")
                logger.info("   " + "="*50)

async def test_concurrent_requests():
    """Probar requests concurrentes"""
    
    endpoint = "https://wfuosp4mnqkimde9.us-east-1.aws.endpoints.huggingface.cloud"
    
    async def single_request(session, request_id):
        """Realizar una request individual"""
        try:
            start_time = time.time()
            
            payload = {
                "inputs": [f"Texto de prueba {request_id} para verificar el rendimiento del endpoint."],
                "parameters": {
                    "dimensions": 1024,
                    "normalize": True
                }
            }
            
            async with session.post(endpoint, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    processing_time = time.time() - start_time
                    return {
                        'request_id': request_id,
                        'success': True,
                        'time': processing_time,
                        'embeddings_count': len(result) if isinstance(result, list) else 1
                    }
                else:
                    return {
                        'request_id': request_id,
                        'success': False,
                        'error': f"HTTP {response.status}"
                    }
        except Exception as e:
            return {
                'request_id': request_id,
                'success': False,
                'error': str(e)
            }
    
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    }
    
    # Probar diferentes niveles de concurrencia
    concurrency_levels = [1, 5, 10, 20]
    
    for concurrency in concurrency_levels:
        logger.info(f"🚀 Probando {concurrency} requests concurrentes...")
        
        async with aiohttp.ClientSession(headers=headers) as session:
            start_time = time.time()
            
            # Crear semáforo para limitar concurrencia
            semaphore = asyncio.Semaphore(concurrency)
            
            async def limited_request(request_id):
                async with semaphore:
                    return await single_request(session, request_id)
            
            # Ejecutar requests concurrentes
            tasks = [limited_request(i) for i in range(20)]  # 20 requests total
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            total_time = time.time() - start_time
            
            # Analizar resultados
            successful = [r for r in results if isinstance(r, dict) and r.get('success', False)]
            failed = [r for r in results if isinstance(r, dict) and not r.get('success', False)]
            
            if successful:
                avg_time = sum(r['time'] for r in successful) / len(successful)
                total_embeddings = sum(r['embeddings_count'] for r in successful)
                
                logger.info(f"📊 RESULTADOS CON {concurrency} CONCURRENTES:")
                logger.info(f"   - Requests exitosos: {len(successful)}")
                logger.info(f"   - Requests fallidos: {len(failed)}")
                logger.info(f"   - Tiempo total: {total_time:.2f}s")
                logger.info(f"   - Tiempo promedio por request: {avg_time:.4f}s")
                logger.info(f"   - Embeddings generados: {total_embeddings}")
                logger.info(f"   - Requests/segundo: {len(successful)/total_time:.2f}")
                logger.info("   " + "="*50)

async def main():
    """Función principal de prueba"""
    logger.info("🧪 INICIANDO PRUEBAS DE RENDIMIENTO DE GPU")
    logger.info("="*60)
    
    # Prueba 1: Diferentes tamaños de lote
    logger.info("PRUEBA 1: Tamaños de lote")
    await test_hf_endpoint_performance()
    
    logger.info("\n" + "="*60)
    
    # Prueba 2: Requests concurrentes
    logger.info("PRUEBA 2: Requests concurrentes")
    await test_concurrent_requests()
    
    logger.info("\n🎉 PRUEBAS COMPLETADAS")

if __name__ == "__main__":
    asyncio.run(main())
