#!/usr/bin/env python3
"""
Script principal corregido para RTX 5090
- Barra de progreso funcional
- Gestión de memoria GPU optimizada
- Uso máximo de GPU
"""
import asyncio
import logging
import argparse
import sys
from pathlib import Path
from typing import Optional
import time

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import config
from src.fixed_embeddings import FixedEmbeddingsManager

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('legal_rag_fixed.log'),
        logging.StreamHandler()
    ]
)

# Reducir logs verbosos
logging.getLogger("unstructured").setLevel(logging.ERROR)
logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

async def main():
    """Función principal corregida"""
    parser = argparse.ArgumentParser(description="Sistema de Embeddings Legales Corregido")
    parser.add_argument("command", choices=["process", "stats", "search"], help="Comando a ejecutar")
    parser.add_argument("path", nargs="?", help="Ruta del directorio o consulta de búsqueda")
    parser.add_argument("--batch-size", type=int, default=20, help="Tamaño del lote (default: 20)")
    parser.add_argument("--max-workers", type=int, default=16, help="Número máximo de workers paralelos (default: 16)")
    parser.add_argument("--force-rebuild", action="store_true", help="Forzar reconstrucción completa")
    parser.add_argument("--query", type=str, help="Consulta de búsqueda")
    parser.add_argument("--limit", type=int, default=10, help="Límite de resultados de búsqueda")
    
    args = parser.parse_args()
    
    try:
        # Crear gestor corregido
        manager = FixedEmbeddingsManager()
        
        if args.command == "process":
            if not args.path:
                logger.error("❌ Se requiere la ruta del directorio para procesar")
                return
            
            source_path = Path(args.path)
            if not source_path.exists():
                logger.error(f"❌ Directorio no encontrado: {source_path}")
                return
            
            logger.info("🚀 Iniciando procesamiento corregido...")
            logger.info(f"📁 Directorio: {source_path}")
            logger.info(f"📦 Tamaño de lote: {args.batch_size}")
            logger.info(f"👥 Workers paralelos: {args.max_workers}")
            logger.info(f"🔄 Reconstruir: {args.force_rebuild}")
            
            # Mostrar información del sistema
            import psutil
            import torch
            
            logger.info("🖥️ Información del sistema:")
            logger.info(f"   CPU cores: {psutil.cpu_count(logical=True)}")
            logger.info(f"   RAM total: {psutil.virtual_memory().total / 1024**3:.1f} GB")
            logger.info(f"   GPU disponible: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                gpu_props = torch.cuda.get_device_properties(0)
                logger.info(f"   GPU: {gpu_props.name}")
                logger.info(f"   GPU memoria: {gpu_props.total_memory / 1024**3:.1f} GB")
            
            # Procesar documentos
            start_time = time.time()
            stats = await manager.process_documents_from_directory(
                source_directory=str(source_path),
                force_rebuild=args.force_rebuild,
                batch_size=args.batch_size,
                max_workers=args.max_workers
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            logger.info("🎉 Procesamiento completado!")
            logger.info(f"⏱️ Tiempo total: {total_time:.2f} segundos")
            logger.info(f"📊 Estadísticas: {stats}")
            
        elif args.command == "stats":
            logger.info("📊 Obteniendo estadísticas de la base de datos...")
            stats = await manager.vector_db.get_database_stats()
            logger.info(f"📈 Estadísticas: {stats}")
            
        elif args.command == "search":
            if not args.query:
                logger.error("❌ Se requiere una consulta de búsqueda")
                return
            
            logger.info(f"🔍 Buscando: {args.query}")
            results = await manager.search_engine.search_similar_chunks(
                query=args.query,
                limit=args.limit
            )
            
            logger.info(f"📋 Resultados encontrados: {len(results)}")
            for i, result in enumerate(results, 1):
                logger.info(f"   {i}. {result.get('content', '')[:100]}...")
        
        else:
            logger.error(f"❌ Comando no reconocido: {args.command}")
            return
    
    except KeyboardInterrupt:
        logger.info("⏹️ Procesamiento interrumpido por el usuario")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise
    finally:
        # Limpiar recursos
        if 'manager' in locals():
            manager.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
