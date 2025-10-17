#!/usr/bin/env python3
"""
Sistema RAG Optimizado para Documentos SIEM
Procesamiento masivo paralelo con 16 workers y lotes de 50 archivos
Búsqueda híbrida: vectorial + palabras clave + metadatos
"""

import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.optimized_siem_rag_system import OptimizedSIEMRAGSystem
from src.comercio_exterior_processor import ComercioExteriorProcessor
from src.qwen3_embedding_client import Qwen3EmbeddingClient, Qwen3Config
from config_supabase import get_supabase_url, get_supabase_key, get_processing_config

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/massive_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MassiveProcessingMain:
    """Clase principal para procesamiento masivo de documentos SIEM"""
    
    def __init__(self, supabase_url: str, supabase_key: str, hf_token: str):
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.hf_token = hf_token
        self.processing_config = get_processing_config()
        
        # Configurar directorios
        self.setup_directories()
        
        # Inicializar componentes
        self.rag_system = None
        self.document_processor = None
        self.embedding_client = None
        
    def setup_directories(self):
        """Configurar directorios necesarios"""
        directories = ['logs', 'resultados_procesamiento', 'temp']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    async def initialize_components(self):
        """Inicializar todos los componentes del sistema"""
        try:
            logger.info("Inicializando componentes del sistema RAG...")
            
            # Configurar cliente de embeddings Qwen3 (endpoint público)
            qwen3_config = Qwen3Config(
                endpoint="https://wfuosp4mnqkimde9.us-east-1.aws.endpoints.huggingface.cloud",
                token="",  # Endpoint público, no requiere token
                max_batch_size=32,  # Optimizado para Qwen3
                rate_limit_per_minute=100,
                timeout_seconds=300,
                retry_attempts=5
            )
            self.embedding_client = Qwen3EmbeddingClient(qwen3_config)
            logger.info("Cliente de embeddings Qwen3 inicializado (endpoint público)")
            
            # Inicializar procesador de documentos de comercio exterior
            self.document_processor = ComercioExteriorProcessor()
            
            # Inicializar sistema RAG
            self.rag_system = OptimizedSIEMRAGSystem(
                supabase_url=self.supabase_url,
                supabase_key=self.supabase_key,
                embedding_client=self.embedding_client,
                document_processor=self.document_processor
            )
            
            await self.rag_system.initialize()
            logger.info("Componentes inicializados correctamente")
            
        except Exception as e:
            logger.error(f"Error inicializando componentes: {e}")
            raise
    
    async def process_documents(self, data_path: str, max_workers: int = 16, batch_size: int = 50):
        """Procesar documentos en lotes con procesamiento paralelo masivo"""
        try:
            logger.info(f"Iniciando procesamiento masivo de documentos en: {data_path}")
            logger.info(f"Configuración: {max_workers} workers, lotes de {batch_size} archivos")
            
            # Inicializar componentes primero
            await self.initialize_components()
            
            # Obtener lista de archivos
            files = self.get_document_files(data_path)
            total_files = len(files)
            logger.info(f"Total de archivos encontrados: {total_files}")
            
            if total_files == 0:
                logger.warning("No se encontraron archivos para procesar")
                return
            
            # Procesar en lotes
            processed_files = 0
            failed_files = 0
            start_time = datetime.now()
            
            for i in range(0, total_files, batch_size):
                batch_files = files[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (total_files + batch_size - 1) // batch_size
                
                logger.info(f"Procesando lote {batch_num}/{total_batches} ({len(batch_files)} archivos)")
                
                try:
                    # Procesar lote con workers paralelos
                    batch_results = await self.rag_system.process_documents_batch(
                        batch_files, 
                        max_workers=max_workers
                    )
                    
                    # Actualizar estadísticas
                    processed_files += batch_results['processed']
                    failed_files += batch_results['failed']
                    
                    logger.info(f"Lote {batch_num} completado: {batch_results['processed']} procesados, {batch_results['failed']} fallidos")
                    
                except Exception as e:
                    logger.error(f"Error procesando lote {batch_num}: {e}")
                    failed_files += len(batch_files)
            
            # Estadísticas finales
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info("=" * 60)
            logger.info("PROCESAMIENTO MASIVO COMPLETADO")
            logger.info("=" * 60)
            logger.info(f"Archivos procesados: {processed_files}")
            logger.info(f"Archivos fallidos: {failed_files}")
            logger.info(f"Tiempo total: {duration:.2f} segundos")
            logger.info(f"Velocidad: {processed_files/duration:.2f} archivos/segundo")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Error en procesamiento masivo: {e}")
            raise
    
    def get_document_files(self, data_path: str) -> List[Path]:
        """Obtener lista de archivos de documentos"""
        data_dir = Path(data_path)
        if not data_dir.exists():
            raise ValueError(f"Directorio no encontrado: {data_path}")
        
        supported_extensions = self.processing_config['supported_extensions']
        files = []
        
        for ext in supported_extensions:
            files.extend(data_dir.rglob(f"*{ext}"))
        
        # Filtrar archivos válidos
        valid_files = []
        for file_path in files:
            if file_path.is_file() and file_path.stat().st_size > 0:
                valid_files.append(file_path)
        
        return sorted(valid_files)
    
    async def search_documents(self, query: str, search_type: str = "hybrid", 
                             filters: Optional[Dict[str, Any]] = None, 
                             top_k: int = 20, **kwargs):
        """Buscar documentos usando el sistema RAG"""
        try:
            logger.info(f"Buscando: '{query}' (tipo: {search_type})")
            
            if not self.rag_system:
                await self.initialize_components()
            
            # Realizar búsqueda
            results = await self.rag_system.search_documents(
                query=query,
                search_type=search_type,
                filters=filters or {},
                top_k=top_k
            )
            
            # Mostrar resultados
            self.display_search_results(results, query)
            
            return results
            
        except Exception as e:
            logger.error(f"Error en búsqueda: {e}")
            raise
    
    def display_search_results(self, results: Dict[str, Any], query: str):
        """Mostrar resultados de búsqueda"""
        print(f"\n{'='*80}")
        print(f"RESULTADOS DE BÚSQUEDA: '{query}'")
        print(f"{'='*80}")
        print(f"Total encontrados: {results.get('total_found', 0)}")
        print(f"Tiempo de búsqueda: {results.get('search_time', 0):.3f} segundos")
        print(f"{'='*80}")
        
        for i, result in enumerate(results.get('results', []), 1):
            print(f"\n{i}. {result.get('file_name', 'N/A')}")
            print(f"   Similitud: {result.get('similarity_score', 0):.4f}")
            print(f"   Tipo: {result.get('chunk_type', 'N/A')}")
            print(f"   Contenido: {result.get('chunk_text', '')[:200]}...")
            print(f"   Metadatos: {result.get('metadata', {})}")
            print("-" * 80)
    
    async def get_database_stats(self):
        """Obtener estadísticas de la base de datos"""
        try:
            if not self.rag_system:
                await self.initialize_components()
            
            stats = await self.rag_system.get_database_stats()
            
            print(f"\n{'='*60}")
            print("ESTADÍSTICAS DE LA BASE DE DATOS")
            print(f"{'='*60}")
            print(f"Documentos totales: {stats.get('total_documents', 0)}")
            print(f"Chunks totales: {stats.get('total_chunks', 0)}")
            print(f"Embeddings totales: {stats.get('total_embeddings', 0)}")
            print(f"Tamaño total: {stats.get('total_size_mb', 0):.2f} MB")
            print(f"Última actualización: {stats.get('last_updated', 'N/A')}")
            print(f"{'='*60}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            raise
    
    async def cleanup(self):
        """Limpiar recursos"""
        try:
            if self.rag_system:
                await self.rag_system.cleanup()
            if self.embedding_client:
                await self.embedding_client.cleanup()
            logger.info("Recursos limpiados correctamente")
        except Exception as e:
            logger.error(f"Error en limpieza: {e}")

async def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Sistema RAG Optimizado para Documentos SIEM')
    parser.add_argument('command', choices=['process', 'search', 'stats'], 
                       help='Comando a ejecutar')
    parser.add_argument('data_path', nargs='?', 
                       help='Ruta a los datos (para comando process)')
    parser.add_argument('--supabase-url', required=True, 
                       help='URL de Supabase')
    parser.add_argument('--supabase-key', required=True, 
                       help='API Key de Supabase')
    parser.add_argument('--hf-token', required=False, default='', 
                       help='Token de Hugging Face (opcional para endpoints públicos)')
    parser.add_argument('--max-workers', type=int, default=16, 
                       help='Número máximo de workers paralelos')
    parser.add_argument('--batch-size', type=int, default=50, 
                       help='Tamaño del lote de archivos')
    parser.add_argument('--search-type', choices=['vectorial', 'keyword', 'hybrid', 'metadata'], 
                       default='hybrid', help='Tipo de búsqueda')
    parser.add_argument('--top-k', type=int, default=20, 
                       help='Número de resultados a retornar')
    parser.add_argument('--category', help='Filtro por categoría')
    parser.add_argument('--legal-type', help='Filtro por tipo legal')
    
    args = parser.parse_args()
    
    # Validar argumentos
    if args.command == 'process' and not args.data_path:
        parser.error("Se requiere data_path para el comando process")
    
    if args.command == 'search' and not args.data_path:
        parser.error("Se requiere data_path para el comando search")
    
    # Crear instancia principal
    main_processor = MassiveProcessingMain(
        supabase_url=args.supabase_url,
        supabase_key=args.supabase_key,
        hf_token=args.hf_token
    )
    
    try:
        if args.command == 'process':
            await main_processor.process_documents(
                data_path=args.data_path,
                max_workers=args.max_workers,
                batch_size=args.batch_size
            )
        
        elif args.command == 'search':
            filters = {}
            if args.category:
                filters['category'] = args.category
            if args.legal_type:
                filters['legal_type'] = args.legal_type
            
            await main_processor.search_documents(
                query=args.data_path,  # En search, data_path es la query
                search_type=args.search_type,
                filters=filters,
                top_k=args.top_k
            )
        
        elif args.command == 'stats':
            await main_processor.get_database_stats()
    
    except KeyboardInterrupt:
        logger.info("Procesamiento interrumpido por el usuario")
    except Exception as e:
        logger.error(f"Error en ejecución: {e}")
        sys.exit(1)
    finally:
        await main_processor.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
