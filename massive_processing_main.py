#!/usr/bin/env python3
"""
Script principal para procesamiento masivo con Hugging Face A100
Optimizado para máxima utilización de recursos
"""
import asyncio
import logging
import argparse
import sys
from pathlib import Path
import json
import time
from datetime import datetime
import os

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent / "src"))

from src.optimized_rag_system import OptimizedRAGSystem
from src.config import config

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('massive_processing.log'),
        logging.StreamHandler()
    ]
)

# Reducir logs verbosos
logging.getLogger("unstructured").setLevel(logging.ERROR)
logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

async def process_all_documents(
    source_directory: str,
    supabase_url: str,
    supabase_key: str,
    hf_token: str,
    max_workers: int = 8,
    batch_size: int = 16,
    force_rebuild: bool = False
):
    """Procesar todos los documentos con máxima paralelización"""
    try:
        source_path = Path(source_directory)
        if not source_path.exists():
            raise FileNotFoundError(f"Directorio no encontrado: {source_directory}")
        
        logger.info("🚀 INICIANDO PROCESAMIENTO MASIVO CON HUGGING FACE A100")
        logger.info(f"📁 Directorio fuente: {source_directory}")
        logger.info(f"👥 Workers paralelos: {max_workers}")
        logger.info(f"📦 Tamaño de lote: {batch_size}")
        logger.info(f"🔄 Reconstruir: {force_rebuild}")
        
        # Crear sistema RAG optimizado
        async with OptimizedRAGSystem(supabase_url, supabase_key, hf_token) as rag_system:
            
            # Verificar configuración de base de datos
            if force_rebuild:
                logger.info("🗑️ Limpiando base de datos existente...")
                rag_system.supabase.table("legal_chunks").delete().neq("chunk_id", 0).execute()
                rag_system.supabase.table("legal_documents").delete().neq("document_id", "").execute()
                logger.info("✅ Base de datos limpiada")
            
            # Procesar todos los documentos
            start_time = time.time()
            results = await rag_system.process_documents_massive(
                source_directory=source_path,
                max_workers=max_workers,
                batch_size=batch_size
            )
            end_time = time.time()
            
            # Guardar resultados
            results_file = Path("massive_processing_results.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📊 Resultados guardados en: {results_file}")
            
            # Mostrar estadísticas finales
            logger.info("🎉 PROCESAMIENTO MASIVO COMPLETADO")
            logger.info(f"⏱️ Tiempo total: {end_time - start_time:.2f} segundos")
            logger.info(f"📄 Archivos procesados: {results['processed_files']}/{results['total_files']}")
            logger.info(f"📝 Chunks generados: {results['total_chunks']}")
            logger.info(f"🧮 Embeddings generados: {results['total_embeddings']}")
            logger.info(f"📈 Tasa de éxito: {results['success_rate']:.2%}")
            logger.info(f"🚀 Archivos/segundo: {results['files_per_second']:.2f}")
            logger.info(f"🚀 Chunks/segundo: {results['chunks_per_second']:.2f}")
            logger.info(f"🚀 Embeddings/segundo: {results['embeddings_per_second']:.2f}")
            logger.info(f"🏛️ Estados procesados: {', '.join(results['states_processed'])}")
            logger.info(f"📚 Tipos de ley procesados: {', '.join(results['law_types_processed'])}")
            
            return results
            
    except Exception as e:
        logger.error(f"❌ Error en procesamiento masivo: {e}")
        raise

async def search_documents(
    query: str,
    supabase_url: str,
    supabase_key: str,
    hf_token: str,
    search_type: str = "hybrid",
    filters: dict = None,
    top_k: int = 20
):
    """Buscar documentos usando el sistema RAG optimizado"""
    try:
        logger.info(f"🔍 Buscando: '{query}' (tipo: {search_type})")
        
        async with OptimizedRAGSystem(supabase_url, supabase_key, hf_token) as rag_system:
            results = await rag_system.search_documents_hybrid(
                query=query,
                search_type=search_type,
                filters=filters or {},
                top_k=top_k
            )
            
            if "error" in results:
                logger.error(f"❌ Error en búsqueda: {results['error']}")
                return results
            
            # Mostrar resultados
            logger.info(f"📊 Encontrados {results['total_found']} resultados")
            
            for i, result in enumerate(results['results'][:10], 1):  # Mostrar solo los primeros 10
                logger.info(f"\n{i}. Similitud: {result.get('similarity', 0.0):.4f}")
                logger.info(f"   Título: {result['metadata'].get('title', 'Sin título')}")
                logger.info(f"   Estado: {result['metadata'].get('state', 'N/A')}")
                logger.info(f"   Tipo: {result['metadata'].get('law_type', 'N/A')}")
                logger.info(f"   Artículo: {result['metadata'].get('article_number', 'N/A')}")
                logger.info(f"   Contenido: {result['content'][:200]}...")
                logger.info("   " + "-" * 50)
            
            return results
            
    except Exception as e:
        logger.error(f"❌ Error en búsqueda: {e}")
        return {"error": str(e)}

async def show_database_stats(
    supabase_url: str,
    supabase_key: str,
    hf_token: str
):
    """Mostrar estadísticas de la base de datos"""
    try:
        logger.info("📊 Obteniendo estadísticas de la base de datos...")
        
        async with OptimizedRAGSystem(supabase_url, supabase_key, hf_token) as rag_system:
            stats = await rag_system.get_database_stats()
            
            logger.info("📈 ESTADÍSTICAS DE LA BASE DE DATOS:")
            logger.info(f"   📄 Total documentos: {stats.get('total_documents', 0)}")
            logger.info(f"   📝 Total chunks: {stats.get('total_chunks', 0)}")
            logger.info(f"   🏛️ Total estados: {stats.get('total_states', 0)}")
            logger.info(f"   📚 Total tipos de ley: {stats.get('total_law_types', 0)}")
            logger.info(f"   📊 Promedio chunks por documento: {stats.get('avg_chunks_per_document', 0):.2f}")
            logger.info(f"   💾 Tamaño total: {stats.get('total_size_mb', 0)} MB")
            
            return stats
            
    except Exception as e:
        logger.error(f"❌ Error obteniendo estadísticas: {e}")
        return {"error": str(e)}

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="Sistema RAG masivo con Hugging Face A100"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comandos disponibles')
    
    # Comando para procesar documentos
    process_parser = subparsers.add_parser('process', help='Procesar todos los documentos')
    process_parser.add_argument(
        'source_directory',
        help='Directorio fuente con documentos PDF'
    )
    process_parser.add_argument(
        '--supabase-url',
        required=True,
        help='URL de Supabase'
    )
    process_parser.add_argument(
        '--supabase-key',
        required=True,
        help='Clave de Supabase'
    )
    process_parser.add_argument(
        '--hf-token',
        required=True,
        help='Token de Hugging Face'
    )
    process_parser.add_argument(
        '--max-workers',
        type=int,
        default=8,
        help='Número máximo de workers paralelos (default: 8)'
    )
    process_parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Tamaño del lote (default: 50)'
    )
    process_parser.add_argument(
        '--force-rebuild',
        action='store_true',
        help='Reconstruir base de datos desde cero'
    )
    
    # Comando para buscar documentos
    search_parser = subparsers.add_parser('search', help='Buscar documentos')
    search_parser.add_argument('query', help='Consulta de búsqueda')
    search_parser.add_argument(
        '--supabase-url',
        required=True,
        help='URL de Supabase'
    )
    search_parser.add_argument(
        '--supabase-key',
        required=True,
        help='Clave de Supabase'
    )
    search_parser.add_argument(
        '--hf-token',
        required=True,
        help='Token de Hugging Face'
    )
    search_parser.add_argument(
        '--search-type',
        choices=['hybrid', 'vector', 'keyword', 'metadata'],
        default='hybrid',
        help='Tipo de búsqueda (default: hybrid)'
    )
    search_parser.add_argument(
        '--state',
        help='Filtrar por estado'
    )
    search_parser.add_argument(
        '--law-type',
        help='Filtrar por tipo de ley'
    )
    search_parser.add_argument(
        '--top-k',
        type=int,
        default=20,
        help='Número máximo de resultados (default: 20)'
    )
    
    # Comando para mostrar estadísticas
    stats_parser = subparsers.add_parser('stats', help='Mostrar estadísticas de la base de datos')
    stats_parser.add_argument(
        '--supabase-url',
        required=True,
        help='URL de Supabase'
    )
    stats_parser.add_argument(
        '--supabase-key',
        required=True,
        help='Clave de Supabase'
    )
    stats_parser.add_argument(
        '--hf-token',
        required=True,
        help='Token de Hugging Face'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'process':
            asyncio.run(process_all_documents(
                source_directory=args.source_directory,
                supabase_url=args.supabase_url,
                supabase_key=args.supabase_key,
                hf_token=args.hf_token,
                max_workers=args.max_workers,
                batch_size=args.batch_size,
                force_rebuild=args.force_rebuild
            ))
            
        elif args.command == 'search':
            filters = {}
            if args.state:
                filters['state'] = args.state
            if args.law_type:
                filters['law_type'] = args.law_type
                
            asyncio.run(search_documents(
                query=args.query,
                supabase_url=args.supabase_url,
                supabase_key=args.supabase_key,
                hf_token=args.hf_token,
                search_type=args.search_type,
                filters=filters,
                top_k=args.top_k
            ))
            
        elif args.command == 'stats':
            asyncio.run(show_database_stats(
                supabase_url=args.supabase_url,
                supabase_key=args.supabase_key,
                hf_token=args.hf_token
            ))
            
    except KeyboardInterrupt:
        logger.info("⏹️ Proceso interrumpido por el usuario")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
