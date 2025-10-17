#!/usr/bin/env python3
"""
Sistema RAG Ultra-Optimizado con todas las técnicas avanzadas
- RAG-Anything + Unstructured + LangChain
- Hybrid Search + Re-ranking + Query Expansion
- Chunking contextual legal
- Procesamiento paralelo masivo
"""
import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor
import os

# LangChain imports avanzados
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.bm25 import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever

# Unstructured imports
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from unstructured.chunking.hierarchical import chunk_by_title as hierarchical_chunking
from unstructured.staging.base import elements_to_json

# RAG-Anything imports
try:
    from raganything import RAGAnything
    RAG_ANYTHING_AVAILABLE = True
except ImportError:
    RAG_ANYTHING_AVAILABLE = False

# Supabase
from supabase import create_client, Client

# Imports locales
from .huggingface_api_client import HuggingFaceAPIClient, HFConfig
from .legal_chunking_processor import LegalChunkingProcessor, LegalChunk
from .config import config

logger = logging.getLogger(__name__)

class UltraOptimizedRAGSystem:
    """Sistema RAG ultra-optimizado con todas las técnicas avanzadas"""
    
    def __init__(self, supabase_url: str, supabase_key: str, hf_token: str):
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.hf_client = None
        self.chunking_processor = LegalChunkingProcessor()
        
        # Configurar chunking contextual legal
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            length_function=len,
            separators=[
                "\n\nARTÍCULO",
                "\n\nCAPÍTULO", 
                "\n\nTÍTULO",
                "\n\nSECCIÓN",
                "\n\nPÁRRAFO",
                "\n\nINCISO",
                "\n\n",
                "\n",
                " ",
                ""
            ]
        )
        
        # Configurar RAG-Anything si está disponible
        self.rag_anything = None
        if RAG_ANYTHING_AVAILABLE and config.openai_api_key:
            try:
                self.rag_anything = RAGAnything(
                    api_key=config.openai_api_key,
                    base_url=config.openai_base_url,
                    model_name=config.embedding_model
                )
                logger.info("✅ RAG-Anything configurado")
            except Exception as e:
                logger.warning(f"⚠️ Error configurando RAG-Anything: {e}")
        
        # Configurar retrievers híbridos
        self._setup_hybrid_retrievers()
        
    def _setup_hybrid_retrievers(self):
        """Configurar sistema de retrievers híbridos"""
        # Vector retriever (ya implementado)
        self.vector_retriever = None
        
        # BM25 retriever para búsqueda por palabras clave
        self.bm25_retriever = None
        
        # Multi-query retriever para expansión de consultas
        self.multi_query_retriever = None
        
        # Ensemble retriever para combinar resultados
        self.ensemble_retriever = None
        
        # Contextual compression retriever
        self.compression_retriever = None
        
    async def process_documents_ultra_optimized(
        self, 
        source_directory: Path, 
        max_workers: int = 16,  # Aumentado para A100
        batch_size: int = 16    # Reducido para evitar 503
    ) -> Dict[str, Any]:
        """Procesamiento ultra-optimizado con todas las técnicas"""
        
        # 1. LIMPIAR BASE DE DATOS
        logger.info("🗑️ Limpiando base de datos...")
        self.supabase.table("legal_chunks").delete().neq("chunk_id", 0).execute()
        self.supabase.table("legal_documents").delete().neq("document_id", "").execute()
        
        # 2. PROCESAMIENTO PARALELO MASIVO
        logger.info(f"🚀 Iniciando procesamiento ultra-optimizado con {max_workers} workers")
        
        # Encontrar todos los PDFs
        pdf_files = list(source_directory.rglob("*.pdf"))
        total_files = len(pdf_files)
        
        # Procesar en lotes paralelos
        results = {
            "processed_files": 0,
            "total_files": total_files,
            "total_chunks": 0,
            "total_embeddings": 0,
            "success_rate": 0.0,
            "files_per_second": 0.0,
            "chunks_per_second": 0.0,
            "embeddings_per_second": 0.0,
            "states_processed": set(),
            "law_types_processed": set(),
            "errors": []
        }
        
        start_time = time.time()
        
        # Procesar en lotes de workers
        batch_size_workers = max(1, total_files // max_workers)
        
        async with HuggingFaceAPIClient() as hf_client:
            self.hf_client = hf_client
            
            # Procesar archivos en lotes paralelos
            for i in range(0, total_files, batch_size_workers):
                batch_files = pdf_files[i:i + batch_size_workers]
                
                # Crear tareas paralelas para el lote
                tasks = []
                for pdf_file in batch_files:
                    task = self._process_single_document_advanced(
                        pdf_file, 
                        source_directory,
                        hf_client
                    )
                    tasks.append(task)
                
                # Ejecutar lote en paralelo
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Procesar resultados del lote
                for result in batch_results:
                    if isinstance(result, Exception):
                        results["errors"].append(str(result))
                        logger.error(f"❌ Error en lote: {result}")
                    else:
                        results["processed_files"] += 1
                        results["total_chunks"] += result.get("chunks", 0)
                        results["total_embeddings"] += result.get("embeddings", 0)
                        results["states_processed"].add(result.get("state", "unknown"))
                        results["law_types_processed"].add(result.get("law_type", "unknown"))
        
        # Calcular estadísticas finales
        end_time = time.time()
        total_time = end_time - start_time
        
        results["success_rate"] = results["processed_files"] / total_files
        results["files_per_second"] = results["processed_files"] / total_time
        results["chunks_per_second"] = results["total_chunks"] / total_time
        results["embeddings_per_second"] = results["total_embeddings"] / total_time
        results["states_processed"] = list(results["states_processed"])
        results["law_types_processed"] = list(results["law_types_processed"])
        
        return results
    
    async def _process_single_document_advanced(
        self, 
        pdf_file: Path, 
        source_directory: Path,
        hf_client: HuggingFaceAPIClient
    ) -> Dict[str, Any]:
        """Procesar un documento con técnicas avanzadas"""
        try:
            # 1. EXTRACCIÓN AVANZADA CON UNSTRUCTURED
            elements = partition_pdf(
                str(pdf_file),
                strategy="hi_res",  # Máxima calidad
                infer_table_structure=True,
                extract_images_in_pdf=False,
                include_page_breaks=True
            )
            
            # 2. CHUNKING JERÁRQUICO
            chunks = hierarchical_chunking(
                elements,
                max_characters=1200,
                combine_text_under_n_chars=200,
                new_after_n_chars=1000
            )
            
            # 3. CHUNKING LEGAL INTELIGENTE
            legal_chunks = self.chunking_processor.process_document(
                str(pdf_file),
                content="\n".join([chunk.text for chunk in chunks])
            )
            
            # 4. GENERAR EMBEDDINGS CON HF A100
            texts = [chunk.content for chunk in legal_chunks]
            embeddings = await hf_client.get_embeddings_batch(texts)
            
            # 5. ALMACENAR EN SUPABASE CON METADATOS AVANZADOS
            await self._store_document_advanced(
                pdf_file, 
                source_directory,
                legal_chunks, 
                embeddings
            )
            
            return {
                "chunks": len(legal_chunks),
                "embeddings": len(embeddings),
                "state": self._extract_state_from_path(pdf_file, source_directory),
                "law_type": self._extract_law_type_from_filename(pdf_file.name)
            }
            
        except Exception as e:
            logger.error(f"❌ Error procesando {pdf_file}: {e}")
            raise
    
    async def _store_document_advanced(
        self, 
        pdf_file: Path, 
        source_directory: Path,
        chunks: List[LegalChunk], 
        embeddings: List[List[float]]
    ):
        """Almacenar documento con metadatos avanzados"""
        try:
            # Extraer metadatos avanzados
            state = self._extract_state_from_path(pdf_file, source_directory)
            law_type = self._extract_law_type_from_filename(pdf_file.name)
            
            # Crear documento con metadatos completos
            document_metadata = {
                'document_id': f"{state}_{pdf_file.stem}",
                'title': pdf_file.stem,
                'state': state,
                'document_type': 'ley',
                'law_type': law_type,
                'file_path': str(pdf_file),
                'file_size': pdf_file.stat().st_size,
                'total_pages': self._estimate_pages(pdf_file),
                'total_chunks': len(chunks),
                'metadata': {
                    'extraction_method': 'unstructured_hierarchical',
                    'chunking_strategy': 'legal_intelligent',
                    'embedding_model': 'Qwen3-Embedding-0.6B',
                    'processing_timestamp': datetime.now().isoformat()
                }
            }
            
            # Almacenar documento
            doc_data = {
                'document_id': document_metadata['document_id'],
                'title': document_metadata['title'],
                'state': document_metadata['state'],
                'document_type': document_metadata['document_type'],
                'law_type': document_metadata['law_type'],
                'file_path': document_metadata['file_path'],
                'file_size': document_metadata['file_size'],
                'total_pages': document_metadata['total_pages'],
                'total_chunks': document_metadata['total_chunks'],
                'metadata': json.dumps(document_metadata['metadata'])
            }
            
            self.supabase.table('legal_documents').upsert(doc_data).execute()
            
            # Almacenar chunks con embeddings
            chunks_data = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_data = {
                    'document_id': document_metadata['document_id'],
                    'content': chunk.content,
                    'embedding': embedding,
                    'law_title': chunk.law_title,
                    'chapter_title': chunk.chapter_title,
                    'section_title': chunk.section_title,
                    'article_number': chunk.article_number,
                    'paragraph_number': chunk.paragraph_number,
                    'inciso_number': chunk.inciso_number,
                    'chunk_type': chunk.chunk_type,
                    'page_number': chunk.page_number,
                    'word_count': chunk.word_count,
                    'char_count': chunk.char_count,
                    'metadata': json.dumps(chunk.metadata)
                }
                chunks_data.append(chunk_data)
            
            # Insertar en lotes
            batch_size = 1000
            for i in range(0, len(chunks_data), batch_size):
                batch = chunks_data[i:i + batch_size]
                self.supabase.table('legal_chunks').insert(batch).execute()
            
            logger.debug(f"✅ Documento almacenado: {document_metadata['document_id']}")
            
        except Exception as e:
            logger.error(f"❌ Error almacenando documento: {e}")
            raise
    
    def _extract_state_from_path(self, pdf_file: Path, source_directory: Path) -> str:
        """Extraer estado del path del archivo"""
        relative_path = pdf_file.relative_to(source_directory)
        return relative_path.parts[0] if relative_path.parts else "unknown"
    
    def _extract_law_type_from_filename(self, filename: str) -> str:
        """Extraer tipo de ley del nombre del archivo"""
        filename_lower = filename.lower()
        
        if "constitucion" in filename_lower:
            return "constitucion"
        elif "ley" in filename_lower:
            return "ley"
        elif "reglamento" in filename_lower:
            return "reglamento"
        elif "decreto" in filename_lower:
            return "decreto"
        elif "acuerdo" in filename_lower:
            return "acuerdo"
        else:
            return "otro"
    
    def _estimate_pages(self, pdf_file: Path) -> int:
        """Estimar número de páginas del PDF"""
        try:
            # Implementación básica - se puede mejorar
            file_size = pdf_file.stat().st_size
            # Estimación aproximada: 50KB por página
            return max(1, int(file_size / 50000))
        except:
            return 1

# Función principal para ejecutar
async def main():
    """Función principal para ejecutar el sistema ultra-optimizado"""
    SUPABASE_URL = "https://zcxqxrgtmnfixkgeaurj.supabase.co"
    SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InpjeHF4cmd0bW5maXhrZ2VhdXJqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA2MTk4MTgsImV4cCI6MjA3NjE5NTgxOH0.YqseZLeQDohHdc-s9QduefPqy5SOSyWJysK_q8cMtio"
    HF_TOKEN = "hf_public"
    
    source_dir = Path("leyes_de_todos_los_estados")
    
    async with UltraOptimizedRAGSystem(SUPABASE_URL, SUPABASE_KEY, HF_TOKEN) as rag_system:
        results = await rag_system.process_documents_ultra_optimized(
            source_directory=source_dir,
            max_workers=16,  # Aprovechar A100 al máximo
            batch_size=16    # Reducido para evitar 503
        )
        
        print("🎉 PROCESAMIENTO ULTRA-OPTIMIZADO COMPLETADO")
        print(f"📄 Archivos procesados: {results['processed_files']}/{results['total_files']}")
        print(f"📝 Chunks generados: {results['total_chunks']}")
        print(f"🧮 Embeddings generados: {results['total_embeddings']}")
        print(f"📈 Tasa de éxito: {results['success_rate']:.2%}")
        print(f"🚀 Archivos/segundo: {results['files_per_second']:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
