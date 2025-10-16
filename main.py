#!/usr/bin/env python3
"""
Script principal para procesamiento de documentos legales con embeddings
"""
import asyncio
import logging
import argparse
from pathlib import Path
import json
from datetime import datetime
from typing import Optional

from src.embeddings_manager import LegalEmbeddingsManager
from src.config import config

# Configurar logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.logs_dir / "legal_rag.log"),
        logging.StreamHandler()
    ]
)

# Configurar logging específico para warnings de Unstructured
logging.getLogger("unstructured").setLevel(logging.ERROR)
logging.getLogger("pypdf").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

async def process_documents(
    source_directory: str,
    force_rebuild: bool = False,
    batch_size: int = None
):
    """Procesar documentos desde un directorio"""
    try:
        logger.info("🚀 Iniciando sistema de embeddings legales")
        logger.info(f"📁 Directorio fuente: {source_directory}")
        logger.info(f"🔄 Reconstruir: {force_rebuild}")
        
        # Crear gestor de embeddings
        embeddings_manager = LegalEmbeddingsManager()
        
        # Procesar documentos
        results = await embeddings_manager.process_documents_from_directory(
            source_directory=source_directory,
            force_rebuild=force_rebuild,
            batch_size=batch_size
        )
        
        # Guardar resultados
        results_file = config.data_dir / f"processing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Resultados guardados en: {results_file}")
        
        # Mostrar estadísticas del sistema
        stats = await embeddings_manager.get_system_stats()
        stats_file = config.data_dir / f"system_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📈 Estadísticas del sistema guardadas en: {stats_file}")
        
        # Limpiar recursos
        embeddings_manager.cleanup()
        
        logger.info("✅ Procesamiento completado exitosamente")
        return results
        
    except Exception as e:
        logger.error(f"❌ Error en el procesamiento: {e}")
        raise

async def search_documents(
    query: str,
    top_k: int = 10,
    search_mode: str = "hybrid",
    state: Optional[str] = None,
    document_type: Optional[str] = None
):
    """Buscar documentos"""
    try:
        logger.info(f"🔍 Buscando: '{query}'")
        
        # Crear gestor de embeddings
        embeddings_manager = LegalEmbeddingsManager()
        
        # Buscar documentos
        results = await embeddings_manager.search_documents(
            query=query,
            top_k=top_k,
            search_mode=search_mode,
            state=state,
            document_type=document_type
        )
        
        # Mostrar resultados
        print(f"\n🔍 Resultados para: '{query}'")
        print(f"📊 Encontrados {len(results)} resultados\n")
        
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result.get('score', 0):.3f}")
            print(f"   Contenido: {result.get('content', '')[:200]}...")
            print(f"   Fuente: {result.get('metadata', {}).get('source', 'N/A')}")
            print(f"   Estado: {result.get('metadata', {}).get('state', 'N/A')}")
            print(f"   Tipo: {result.get('metadata', {}).get('document_type', 'N/A')}")
            print()
        
        # Limpiar recursos
        embeddings_manager.cleanup()
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error en la búsqueda: {e}")
        raise

async def show_stats():
    """Mostrar estadísticas del sistema"""
    try:
        logger.info("📊 Obteniendo estadísticas del sistema")
        
        # Crear gestor de embeddings
        embeddings_manager = LegalEmbeddingsManager()
        
        # Obtener estadísticas
        stats = await embeddings_manager.get_system_stats()
        
        # Mostrar estadísticas
        print("\n📊 ESTADÍSTICAS DEL SISTEMA")
        print("=" * 50)
        
        # Base de datos
        if "database" in stats:
            db_stats = stats["database"]
            print(f"\n🗄️ BASE DE DATOS:")
            print(f"   Total documentos: {db_stats.get('total_documents', 0)}")
            print(f"   Total chunks: {db_stats.get('total_chunks', 0)}")
            print(f"   Tipo: {db_stats.get('database_type', 'N/A')}")
            print(f"   Dimensiones: {db_stats.get('embedding_dimensions', 0)}")
        
        # LightRAG
        if "lightrag" in stats:
            lightrag_stats = stats["lightrag"]
            print(f"\n🧠 LIGHTRAG:")
            print(f"   Disponible: {lightrag_stats.get('available', False)}")
            if lightrag_stats.get('available'):
                print(f"   Modelo LLM: {lightrag_stats.get('llm_model', 'N/A')}")
                print(f"   Modos disponibles: {', '.join(lightrag_stats.get('modes_available', []))}")
        
        # Procesador
        if "processor" in stats:
            proc_stats = stats["processor"]
            print(f"\n⚙️ PROCESADOR:")
            print(f"   RAG-Anything: {proc_stats.get('rag_anything_available', False)}")
            print(f"   Unstructured: {proc_stats.get('unstructured_available', False)}")
            print(f"   LangChain: {proc_stats.get('langchain_available', False)}")
            print(f"   Procesador preferido: {proc_stats.get('preferred_processor', 'N/A')}")
        
        # Búsqueda
        if "search" in stats:
            search_stats = stats["search"]
            print(f"\n🔍 BÚSQUEDA:")
            print(f"   LightRAG: {search_stats.get('lightrag_available', False)}")
            print(f"   Vectorial: {search_stats.get('vector_search_available', False)}")
            print(f"   Híbrida: {search_stats.get('hybrid_search_available', False)}")
        
        # Modelo de embeddings
        if "embedding_model" in stats:
            emb_stats = stats["embedding_model"]
            print(f"\n🧮 EMBEDDINGS:")
            print(f"   Modelo: {emb_stats.get('name', 'N/A')}")
            print(f"   Dispositivo: {emb_stats.get('device', 'N/A')}")
            print(f"   Dimensiones: {emb_stats.get('dimensions', 0)}")
        
        print("\n" + "=" * 50)
        
        # Limpiar recursos
        embeddings_manager.cleanup()
        
    except Exception as e:
        logger.error(f"❌ Error obteniendo estadísticas: {e}")
        raise

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="Sistema de embeddings legales con RAG-Anything y Supabase"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comandos disponibles')
    
    # Comando para procesar documentos
    process_parser = subparsers.add_parser('process', help='Procesar documentos PDF')
    process_parser.add_argument(
        'source_directory',
        help='Directorio fuente con documentos PDF'
    )
    process_parser.add_argument(
        '--force-rebuild',
        action='store_true',
        help='Reconstruir base de datos desde cero'
    )
    process_parser.add_argument(
        '--batch-size',
        type=int,
        default=config.batch_size,
        help=f'Tamaño del lote (default: {config.batch_size})'
    )
    
    # Comando para buscar documentos
    search_parser = subparsers.add_parser('search', help='Buscar documentos')
    search_parser.add_argument('query', help='Consulta de búsqueda')
    search_parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Número máximo de resultados (default: 10)'
    )
    search_parser.add_argument(
        '--mode',
        choices=['hybrid', 'lightrag', 'vector'],
        default='hybrid',
        help='Modo de búsqueda (default: hybrid)'
    )
    search_parser.add_argument(
        '--state',
        help='Filtrar por estado'
    )
    search_parser.add_argument(
        '--document-type',
        help='Filtrar por tipo de documento'
    )
    
    # Comando para mostrar estadísticas
    stats_parser = subparsers.add_parser('stats', help='Mostrar estadísticas del sistema')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'process':
            asyncio.run(process_documents(
                source_directory=args.source_directory,
                force_rebuild=args.force_rebuild,
                batch_size=args.batch_size
            ))
        elif args.command == 'search':
            asyncio.run(search_documents(
                query=args.query,
                top_k=args.top_k,
                search_mode=args.mode,
                state=args.state,
                document_type=args.document_type
            ))
        elif args.command == 'stats':
            asyncio.run(show_stats())
            
    except KeyboardInterrupt:
        logger.info("⏹️ Proceso interrumpido por el usuario")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
