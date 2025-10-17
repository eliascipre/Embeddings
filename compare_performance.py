#!/usr/bin/env python3
"""
Script para comparar el rendimiento entre la implementación original y la optimizada
"""
import asyncio
import time
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Agregar src al path
sys.path.append(str(Path(__file__).parent / "src"))

from ultra_simple_processing import UltraSimpleRAGSystem
from optimized_gpu_processing import OptimizedGPURAGSystem

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_original_system(supabase_url: str, supabase_key: str, test_directory: str) -> Dict[str, Any]:
    """Probar el sistema original"""
    
    logger.info("🔄 Probando sistema original...")
    
    start_time = time.time()
    
    try:
        # Crear sistema original
        rag_system = UltraSimpleRAGSystem(supabase_url, supabase_key)
        rag_system.initialize()
        
        # Procesar documentos
        results = await rag_system.process_documents(test_directory, max_workers=4)
        
        end_time = time.time()
        
        return {
            'system': 'original',
            'total_time': end_time - start_time,
            'processed_files': results['processed_files'],
            'total_chunks': results['total_chunks'],
            'total_embeddings': results['total_embeddings'],
            'success_rate': results['success_rate'],
            'errors': len(results['errors']),
            'files_per_second': results['processed_files'] / (end_time - start_time),
            'chunks_per_second': results['total_chunks'] / (end_time - start_time),
            'embeddings_per_second': results['total_embeddings'] / (end_time - start_time)
        }
        
    except Exception as e:
        logger.error(f"❌ Error en sistema original: {e}")
        return {
            'system': 'original',
            'error': str(e),
            'total_time': time.time() - start_time
        }

async def test_optimized_system(supabase_url: str, supabase_key: str, test_directory: str) -> Dict[str, Any]:
    """Probar el sistema optimizado"""
    
    logger.info("🚀 Probando sistema optimizado...")
    
    start_time = time.time()
    
    try:
        # Crear sistema optimizado
        rag_system = OptimizedGPURAGSystem(
            supabase_url, 
            supabase_key, 
            "https://wfuosp4mnqkimde9.us-east-1.aws.endpoints.huggingface.cloud"
        )
        rag_system.initialize()
        
        # Procesar documentos
        results = await rag_system.process_documents_optimized(test_directory, max_workers=8)
        
        end_time = time.time()
        
        return {
            'system': 'optimized',
            'total_time': end_time - start_time,
            'processed_files': results['processed_files'],
            'total_chunks': results['total_chunks'],
            'total_embeddings': results['total_embeddings'],
            'success_rate': results['success_rate'],
            'errors': len(results['errors']),
            'files_per_second': results['processed_files'] / (end_time - start_time),
            'chunks_per_second': results['total_chunks'] / (end_time - start_time),
            'embeddings_per_second': results['total_embeddings'] / (end_time - start_time)
        }
        
    except Exception as e:
        logger.error(f"❌ Error en sistema optimizado: {e}")
        return {
            'system': 'optimized',
            'error': str(e),
            'total_time': time.time() - start_time
        }

def compare_results(original_results: Dict[str, Any], optimized_results: Dict[str, Any]):
    """Comparar resultados de ambos sistemas"""
    
    logger.info("\n" + "="*80)
    logger.info("📊 COMPARACIÓN DE RENDIMIENTO")
    logger.info("="*80)
    
    # Verificar si hay errores
    if 'error' in original_results:
        logger.error(f"❌ Sistema original falló: {original_results['error']}")
        return
    
    if 'error' in optimized_results:
        logger.error(f"❌ Sistema optimizado falló: {optimized_results['error']}")
        return
    
    # Comparar métricas
    metrics = [
        ('Tiempo total (s)', 'total_time'),
        ('Archivos procesados', 'processed_files'),
        ('Chunks creados', 'total_chunks'),
        ('Embeddings generados', 'total_embeddings'),
        ('Tasa de éxito (%)', 'success_rate'),
        ('Errores', 'errors'),
        ('Archivos/segundo', 'files_per_second'),
        ('Chunks/segundo', 'chunks_per_second'),
        ('Embeddings/segundo', 'embeddings_per_second')
    ]
    
    logger.info(f"{'Métrica':<25} {'Original':<15} {'Optimizado':<15} {'Mejora':<15}")
    logger.info("-" * 80)
    
    for metric_name, metric_key in metrics:
        original_value = original_results.get(metric_key, 0)
        optimized_value = optimized_results.get(metric_key, 0)
        
        if metric_key in ['total_time', 'errors']:
            # Para tiempo y errores, menor es mejor
            if original_value > 0 and optimized_value > 0:
                improvement = ((original_value - optimized_value) / original_value) * 100
                improvement_str = f"{improvement:+.1f}%"
            else:
                improvement_str = "N/A"
        else:
            # Para el resto, mayor es mejor
            if original_value > 0 and optimized_value > 0:
                improvement = ((optimized_value - original_value) / original_value) * 100
                improvement_str = f"{improvement:+.1f}%"
            else:
                improvement_str = "N/A"
        
        # Formatear valores
        if metric_key == 'success_rate':
            original_str = f"{original_value:.1%}"
            optimized_str = f"{optimized_value:.1%}"
        elif metric_key in ['files_per_second', 'chunks_per_second', 'embeddings_per_second']:
            original_str = f"{original_value:.2f}"
            optimized_str = f"{optimized_value:.2f}"
        else:
            original_str = f"{original_value:.0f}"
            optimized_str = f"{optimized_value:.0f}"
        
        logger.info(f"{metric_name:<25} {original_str:<15} {optimized_str:<15} {improvement_str:<15}")
    
    # Análisis de mejoras
    logger.info("\n" + "="*80)
    logger.info("📈 ANÁLISIS DE MEJORAS")
    logger.info("="*80)
    
    time_improvement = ((original_results['total_time'] - optimized_results['total_time']) / original_results['total_time']) * 100
    chunks_improvement = ((optimized_results['chunks_per_second'] - original_results['chunks_per_second']) / original_results['chunks_per_second']) * 100
    embeddings_improvement = ((optimized_results['embeddings_per_second'] - original_results['embeddings_per_second']) / original_results['embeddings_per_second']) * 100
    
    logger.info(f"⏱️ Reducción de tiempo: {time_improvement:.1f}%")
    logger.info(f"🚀 Mejora en chunks/segundo: {chunks_improvement:.1f}%")
    logger.info(f"🧮 Mejora en embeddings/segundo: {embeddings_improvement:.1f}%")
    
    # Recomendaciones
    logger.info("\n💡 RECOMENDACIONES:")
    
    if time_improvement > 50:
        logger.info("   ✅ El sistema optimizado es significativamente más rápido")
    elif time_improvement > 20:
        logger.info("   ✅ El sistema optimizado es notablemente más rápido")
    elif time_improvement > 0:
        logger.info("   ✅ El sistema optimizado es ligeramente más rápido")
    else:
        logger.info("   ⚠️ El sistema optimizado no muestra mejoras significativas")
    
    if chunks_improvement > 100:
        logger.info("   🚀 Mejora dramática en procesamiento de chunks")
    elif chunks_improvement > 50:
        logger.info("   🚀 Mejora significativa en procesamiento de chunks")
    elif chunks_improvement > 0:
        logger.info("   ✅ Mejora moderada en procesamiento de chunks")
    
    if embeddings_improvement > 100:
        logger.info("   🧮 Mejora dramática en generación de embeddings")
    elif embeddings_improvement > 50:
        logger.info("   🧮 Mejora significativa en generación de embeddings")
    elif embeddings_improvement > 0:
        logger.info("   ✅ Mejora moderada en generación de embeddings")

async def main():
    """Función principal de comparación"""
    
    parser = argparse.ArgumentParser(description="Comparar rendimiento entre sistemas")
    parser.add_argument('test_directory', help='Directorio de prueba con documentos')
    parser.add_argument('--supabase-url', required=True, help='URL de Supabase')
    parser.add_argument('--supabase-key', required=True, help='Clave de Supabase')
    parser.add_argument('--skip-original', action='store_true', 
                       help='Saltar prueba del sistema original')
    parser.add_argument('--skip-optimized', action='store_true', 
                       help='Saltar prueba del sistema optimizado')
    
    args = parser.parse_args()
    
    logger.info("🧪 INICIANDO COMPARACIÓN DE RENDIMIENTO")
    logger.info("="*80)
    
    original_results = None
    optimized_results = None
    
    # Probar sistema original
    if not args.skip_original:
        original_results = await test_original_system(
            args.supabase_url, 
            args.supabase_key, 
            args.test_directory
        )
    
    # Probar sistema optimizado
    if not args.skip_optimized:
        optimized_results = await test_optimized_system(
            args.supabase_url, 
            args.supabase_key, 
            args.test_directory
        )
    
    # Comparar resultados
    if original_results and optimized_results:
        compare_results(original_results, optimized_results)
    else:
        logger.warning("⚠️ No se pudieron comparar los resultados")
    
    logger.info("\n🎉 Comparación completada")

if __name__ == "__main__":
    asyncio.run(main())
