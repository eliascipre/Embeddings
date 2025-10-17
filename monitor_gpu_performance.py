#!/usr/bin/env python3
"""
Monitor de rendimiento para verificar la utilización de GPU
"""
import asyncio
import aiohttp
import time
import logging
from datetime import datetime
from typing import List, Dict, Any
import json

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GPUPerformanceMonitor:
    """Monitor de rendimiento para GPU A100"""
    
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_chunks_processed': 0,
            'total_time': 0.0,
            'start_time': None,
            'batch_sizes_tested': [],
            'performance_history': []
        }
    
    async def test_batch_performance(self, batch_size: int, num_batches: int = 5) -> Dict[str, Any]:
        """Probar rendimiento con un tamaño de lote específico"""
        
        # Textos de prueba legales
        legal_texts = [
            "La Constitución Política de los Estados Unidos Mexicanos establece los derechos fundamentales de los ciudadanos.",
            "El Código Civil Federal regula las relaciones civiles entre particulares en el territorio nacional.",
            "La Ley Federal del Trabajo protege los derechos de los trabajadores y establece las obligaciones patronales.",
            "El Código de Comercio regula las actividades comerciales y mercantiles en México.",
            "La Ley de Amparo establece los procedimientos para la protección de los derechos constitucionales.",
            "El Código Penal Federal define los delitos y las penas aplicables en el territorio nacional.",
            "La Ley Federal de Procedimiento Administrativo regula los procedimientos de la administración pública.",
            "El Código Nacional de Procedimientos Penales establece las reglas del proceso penal acusatorio.",
            "La Ley General de Salud regula las actividades de salud pública y privada en México.",
            "El Código Fiscal de la Federación establece las obligaciones fiscales de los contribuyentes."
        ]
        
        # Repetir textos para crear lotes más grandes
        test_texts = (legal_texts * ((batch_size // len(legal_texts)) + 1))[:batch_size]
        
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        batch_stats = {
            'batch_size': batch_size,
            'num_batches': num_batches,
            'total_chunks': 0,
            'total_time': 0.0,
            'successful_batches': 0,
            'failed_batches': 0,
            'chunks_per_second': 0.0,
            'avg_time_per_batch': 0.0
        }
        
        async with aiohttp.ClientSession(headers=headers) as session:
            for i in range(num_batches):
                try:
                    start_time = time.time()
                    
                    payload = {
                        "inputs": test_texts,
                        "parameters": {
                            "dimensions": 1024,
                            "normalize": True,
                            "instruction": "Represent the following text for retrieval:"
                        }
                    }
                    
                    async with session.post(self.endpoint, json=payload) as response:
                        if response.status == 200:
                            result = await response.json()
                            
                            if isinstance(result, list):
                                embeddings = result
                            elif isinstance(result, dict) and 'embeddings' in result:
                                embeddings = result['embeddings']
                            else:
                                raise ValueError(f"Formato inesperado: {result}")
                            
                            batch_time = time.time() - start_time
                            batch_stats['total_time'] += batch_time
                            batch_stats['total_chunks'] += len(embeddings)
                            batch_stats['successful_batches'] += 1
                            
                            logger.info(f"✅ Lote {i+1}/{num_batches}: {len(embeddings)} chunks en {batch_time:.2f}s")
                            
                        else:
                            error_text = await response.text()
                            logger.error(f"❌ Error en lote {i+1}: HTTP {response.status} - {error_text}")
                            batch_stats['failed_batches'] += 1
                            
                except Exception as e:
                    logger.error(f"❌ Error en lote {i+1}: {e}")
                    batch_stats['failed_batches'] += 1
        
        # Calcular estadísticas finales
        if batch_stats['total_time'] > 0:
            batch_stats['chunks_per_second'] = batch_stats['total_chunks'] / batch_stats['total_time']
            batch_stats['avg_time_per_batch'] = batch_stats['total_time'] / batch_stats['successful_batches'] if batch_stats['successful_batches'] > 0 else 0
        
        return batch_stats
    
    async def run_comprehensive_test(self):
        """Ejecutar prueba comprensiva de rendimiento"""
        
        logger.info("🚀 INICIANDO PRUEBA COMPRENSIVA DE RENDIMIENTO GPU A100")
        logger.info("="*70)
        
        self.stats['start_time'] = datetime.now()
        
        # Probar diferentes tamaños de lote
        batch_sizes = [1, 5, 10, 16, 32, 64]
        
        for batch_size in batch_sizes:
            logger.info(f"\n🧪 PROBANDO LOTE DE {batch_size} CHUNKS")
            logger.info("-" * 50)
            
            batch_stats = await self.test_batch_performance(batch_size, num_batches=3)
            
            # Guardar estadísticas
            self.stats['batch_sizes_tested'].append(batch_size)
            self.stats['performance_history'].append(batch_stats)
            
            # Mostrar resultados
            logger.info(f"📊 RESULTADOS PARA LOTE DE {batch_size}:")
            logger.info(f"   - Lotes exitosos: {batch_stats['successful_batches']}/{batch_stats['num_batches']}")
            logger.info(f"   - Total chunks: {batch_stats['total_chunks']}")
            logger.info(f"   - Tiempo total: {batch_stats['total_time']:.2f}s")
            logger.info(f"   - Chunks/segundo: {batch_stats['chunks_per_second']:.2f}")
            logger.info(f"   - Tiempo promedio por lote: {batch_stats['avg_time_per_batch']:.2f}s")
            
            # Actualizar estadísticas globales
            self.stats['total_requests'] += batch_stats['num_batches']
            self.stats['successful_requests'] += batch_stats['successful_batches']
            self.stats['failed_requests'] += batch_stats['failed_batches']
            self.stats['total_chunks_processed'] += batch_stats['total_chunks']
            self.stats['total_time'] += batch_stats['total_time']
        
        # Mostrar resumen final
        self._show_final_summary()
    
    def _show_final_summary(self):
        """Mostrar resumen final de rendimiento"""
        
        logger.info("\n" + "="*70)
        logger.info("📈 RESUMEN FINAL DE RENDIMIENTO")
        logger.info("="*70)
        
        if self.stats['total_time'] > 0:
            overall_chunks_per_second = self.stats['total_chunks_processed'] / self.stats['total_time']
            success_rate = self.stats['successful_requests'] / self.stats['total_requests'] if self.stats['total_requests'] > 0 else 0
            
            logger.info(f"🎯 ESTADÍSTICAS GLOBALES:")
            logger.info(f"   - Total requests: {self.stats['total_requests']}")
            logger.info(f"   - Requests exitosos: {self.stats['successful_requests']}")
            logger.info(f"   - Requests fallidos: {self.stats['failed_requests']}")
            logger.info(f"   - Tasa de éxito: {success_rate:.2%}")
            logger.info(f"   - Total chunks procesados: {self.stats['total_chunks_processed']}")
            logger.info(f"   - Tiempo total: {self.stats['total_time']:.2f}s")
            logger.info(f"   - Chunks/segundo promedio: {overall_chunks_per_second:.2f}")
        
        # Encontrar el mejor tamaño de lote
        if self.stats['performance_history']:
            best_batch = max(self.stats['performance_history'], key=lambda x: x['chunks_per_second'])
            logger.info(f"\n🏆 MEJOR CONFIGURACIÓN:")
            logger.info(f"   - Tamaño de lote óptimo: {best_batch['batch_size']}")
            logger.info(f"   - Chunks/segundo: {best_batch['chunks_per_second']:.2f}")
            logger.info(f"   - Tiempo promedio por lote: {best_batch['avg_time_per_batch']:.2f}s")
        
        # Recomendaciones
        logger.info(f"\n💡 RECOMENDACIONES:")
        if best_batch['batch_size'] >= 32:
            logger.info("   ✅ Usar lotes de 32+ chunks para máxima eficiencia")
        else:
            logger.info("   ⚠️ El endpoint puede tener limitaciones, usar lotes más pequeños")
        
        logger.info("   ✅ Implementar procesamiento asíncrono con lotes")
        logger.info("   ✅ Usar semáforos para controlar concurrencia")
        logger.info("   ✅ Monitorear métricas en tiempo real")
    
    def save_results(self, filename: str = "gpu_performance_results.json"):
        """Guardar resultados en archivo JSON"""
        
        results = {
            'test_timestamp': self.stats['start_time'].isoformat() if self.stats['start_time'] else None,
            'endpoint': self.endpoint,
            'summary': {
                'total_requests': self.stats['total_requests'],
                'successful_requests': self.stats['successful_requests'],
                'failed_requests': self.stats['failed_requests'],
                'total_chunks_processed': self.stats['total_chunks_processed'],
                'total_time': self.stats['total_time'],
                'overall_chunks_per_second': self.stats['total_chunks_processed'] / self.stats['total_time'] if self.stats['total_time'] > 0 else 0,
                'success_rate': self.stats['successful_requests'] / self.stats['total_requests'] if self.stats['total_requests'] > 0 else 0
            },
            'batch_tests': self.stats['performance_history']
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Resultados guardados en: {filename}")

async def main():
    """Función principal"""
    
    endpoint = "https://wfuosp4mnqkimde9.us-east-1.aws.endpoints.huggingface.cloud"
    
    monitor = GPUPerformanceMonitor(endpoint)
    
    try:
        await monitor.run_comprehensive_test()
        monitor.save_results()
        
    except KeyboardInterrupt:
        logger.info("⏹️ Prueba interrumpida por el usuario")
    except Exception as e:
        logger.error(f"❌ Error en la prueba: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
