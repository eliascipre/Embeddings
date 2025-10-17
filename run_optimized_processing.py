#!/usr/bin/env python3
"""
Script principal para ejecutar el procesamiento optimizado
"""
import asyncio
import argparse
import logging
import sys
from pathlib import Path

# Agregar src al path
sys.path.append(str(Path(__file__).parent / "src"))

from optimized_gpu_processing import OptimizedGPURAGSystem

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Función principal"""
    
    parser = argparse.ArgumentParser(description="Sistema RAG optimizado para GPU A100")
    parser.add_argument('source_directory', help='Directorio con documentos')
    parser.add_argument('--supabase-url', required=True, help='URL de Supabase')
    parser.add_argument('--supabase-key', required=True, help='Clave de Supabase')
    parser.add_argument('--hf-endpoint', 
                       default="https://wfuosp4mnqkimde9.us-east-1.aws.endpoints.huggingface.cloud", 
                       help='Endpoint de Hugging Face')
    parser.add_argument('--hf-token', help='Token de Hugging Face (opcional)')
    parser.add_argument('--max-workers', type=int, default=8, help='Workers paralelos')
    parser.add_argument('--test-performance', action='store_true', 
                       help='Ejecutar prueba de rendimiento antes del procesamiento')
    
    args = parser.parse_args()
    
    try:
        # Ejecutar prueba de rendimiento si se solicita
        if args.test_performance:
            logger.info("🧪 Ejecutando prueba de rendimiento...")
            from monitor_gpu_performance import GPUPerformanceMonitor
            
            monitor = GPUPerformanceMonitor(args.hf_endpoint)
            await monitor.run_comprehensive_test()
            monitor.save_results()
            
            logger.info("✅ Prueba de rendimiento completada")
            logger.info("🚀 Iniciando procesamiento optimizado...")
        
        # Crear sistema optimizado
        rag_system = OptimizedGPURAGSystem(
            args.supabase_url, 
            args.supabase_key, 
            args.hf_endpoint, 
            args.hf_token
        )
        rag_system.initialize()
        
        # Procesar documentos
        results = await rag_system.process_documents_optimized(
            args.source_directory, 
            args.max_workers
        )
        
        # Mostrar resultados finales
        logger.info("🎉 PROCESAMIENTO COMPLETADO")
        logger.info(f"📄 Archivos procesados: {results['processed_files']}/{results['total_files']}")
        logger.info(f"📝 Chunks creados: {results['total_chunks']}")
        logger.info(f"🧮 Embeddings generados: {results['total_embeddings']}")
        logger.info(f"📈 Tasa de éxito: {results['success_rate']:.2%}")
        
        if results['errors']:
            logger.warning(f"⚠️ Errores encontrados: {len(results['errors'])}")
            for error in results['errors'][:5]:  # Mostrar solo los primeros 5
                logger.warning(f"   - {error}")
        
        return results
        
    except KeyboardInterrupt:
        logger.info("⏹️ Proceso interrumpido por el usuario")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
