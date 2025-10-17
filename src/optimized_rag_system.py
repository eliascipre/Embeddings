"""
Sistema RAG optimizado con Hugging Face A100, LangChain y búsqueda híbrida
Maximiza utilización de recursos para procesamiento masivo
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

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline

# Supabase
from supabase import create_client, Client

# Imports locales
from .huggingface_api_client import HuggingFaceAPIClient, HFConfig
from .legal_chunking_processor import LegalChunkingProcessor, LegalChunk
from .config import config

logger = logging.getLogger(__name__)

class OptimizedRAGSystem:
    """Sistema RAG optimizado para procesamiento masivo con A100"""
    
    def __init__(self, supabase_url: str, supabase_key: str, hf_token: str):
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.hf_client = None
        self.chunking_processor = LegalChunkingProcessor()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            length_function=len,
            separators=[
                "\n\nARTÍCULO",
                "\n\nCAPÍTULO", 
                "\n\nTÍTULO",
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )
        
        # Configuración para A100
        self.hf_config = HFConfig(
            token=hf_token,
            max_concurrent_requests=50,  # Máximo para A100
            batch_size=32,  # Límite máximo del endpoint HF
            timeout=300
        )
        
        # Estadísticas
        self.stats = {
            'documents_processed': 0,
            'chunks_generated': 0,
            'embeddings_generated': 0,
            'queries_processed': 0,
            'total_processing_time': 0,
            'start_time': None
        }
    
    async def __aenter__(self):
        """Context manager entry"""
        self.hf_client = HuggingFaceAPIClient(self.hf_config)
        await self.hf_client.__aenter__()
        self.stats['start_time'] = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.hf_client:
            await self.hf_client.__aexit__(exc_type, exc_val, exc_tb)
        
        if self.stats['start_time']:
            self.stats['total_processing_time'] = time.time() - self.stats['start_time']
            self._log_stats()
    
    async def process_documents_massive(
        self, 
        source_directory: Path,
        max_workers: int = 16,  # Optimizado para A100
        batch_size: int = 50   # PDFs por lote
    ) -> Dict[str, Any]:
        """Procesar todos los documentos de forma masiva con máxima paralelización"""
        try:
            logger.info(f"🚀 Iniciando procesamiento masivo desde: {source_directory}")
            logger.info(f"👥 Workers paralelos: {max_workers}")
            logger.info(f"📦 Tamaño de lote: {batch_size}")
            
            # Encontrar todos los PDFs
            pdf_files = list(source_directory.rglob("*.pdf"))
            logger.info(f"📄 Encontrados {len(pdf_files)} archivos PDF")
            
            if not pdf_files:
                raise ValueError("No se encontraron archivos PDF")
            
            # Estadísticas de procesamiento
            processing_stats = {
                'total_files': len(pdf_files),
                'processed_files': 0,
                'failed_files': 0,
                'total_chunks': 0,
                'total_embeddings': 0,
                'processing_time': 0,
                'start_time': datetime.now().isoformat(),
                'states_processed': set(),
                'law_types_processed': set()
            }
            
            start_time = time.time()
            
            # Procesar en lotes paralelos
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Dividir archivos en lotes
                batches = [pdf_files[i:i + batch_size] for i in range(0, len(pdf_files), batch_size)]
                
                # Procesar lotes en paralelo
                batch_tasks = []
                for i, batch in enumerate(batches):
                    task = asyncio.create_task(
                        self._process_batch_parallel(batch, i, executor)
                    )
                    batch_tasks.append(task)
                
                # Esperar a que terminen todos los lotes
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Consolidar resultados
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"❌ Error en lote: {result}")
                        processing_stats['failed_files'] += batch_size
                        continue
                    
                    processing_stats['processed_files'] += result['processed_files']
                    processing_stats['failed_files'] += result['failed_files']
                    processing_stats['total_chunks'] += result['total_chunks']
                    processing_stats['total_embeddings'] += result['total_embeddings']
                    processing_stats['states_processed'].update(result['states_processed'])
                    processing_stats['law_types_processed'].update(result['law_types_processed'])
            
            # Finalizar estadísticas
            end_time = time.time()
            processing_stats['processing_time'] = end_time - start_time
            processing_stats['end_time'] = datetime.now().isoformat()
            processing_stats['success_rate'] = processing_stats['processed_files'] / processing_stats['total_files']
            processing_stats['files_per_second'] = processing_stats['processed_files'] / processing_stats['processing_time']
            processing_stats['chunks_per_second'] = processing_stats['total_chunks'] / processing_stats['processing_time']
            processing_stats['embeddings_per_second'] = processing_stats['total_embeddings'] / processing_stats['processing_time']
            
            # Convertir sets a listas para JSON
            processing_stats['states_processed'] = list(processing_stats['states_processed'])
            processing_stats['law_types_processed'] = list(processing_stats['law_types_processed'])
            
            logger.info(f"✅ Procesamiento masivo completado:")
            logger.info(f"   📄 Archivos procesados: {processing_stats['processed_files']}/{processing_stats['total_files']}")
            logger.info(f"   📝 Chunks generados: {processing_stats['total_chunks']}")
            logger.info(f"   🧮 Embeddings generados: {processing_stats['total_embeddings']}")
            logger.info(f"   ⏱️ Tiempo total: {processing_stats['processing_time']:.2f} segundos")
            logger.info(f"   📈 Tasa de éxito: {processing_stats['success_rate']:.2%}")
            logger.info(f"   🚀 Archivos/segundo: {processing_stats['files_per_second']:.2f}")
            logger.info(f"   🚀 Chunks/segundo: {processing_stats['chunks_per_second']:.2f}")
            logger.info(f"   🚀 Embeddings/segundo: {processing_stats['embeddings_per_second']:.2f}")
            
            return processing_stats
            
        except Exception as e:
            logger.error(f"❌ Error en procesamiento masivo: {e}")
            raise
    
    async def _process_batch_parallel(
        self, 
        pdf_files: List[Path], 
        batch_id: int,
        executor: ThreadPoolExecutor
    ) -> Dict[str, Any]:
        """Procesar un lote de PDFs en paralelo"""
        batch_stats = {
            'processed_files': 0,
            'failed_files': 0,
            'total_chunks': 0,
            'total_embeddings': 0,
            'states_processed': set(),
            'law_types_processed': set()
        }
        
        # Procesar archivos del lote
        for pdf_file in pdf_files:
            try:
                result = await self._process_single_document(pdf_file)
                
                batch_stats['processed_files'] += 1
                batch_stats['total_chunks'] += result['total_chunks']
                batch_stats['total_embeddings'] += result['total_embeddings']
                batch_stats['states_processed'].add(result['state'])
                batch_stats['law_types_processed'].add(result['law_type'])
                
                logger.debug(f"✅ Lote {batch_id}: {pdf_file.name} procesado")
                
            except Exception as e:
                logger.error(f"❌ Error procesando {pdf_file.name}: {e}")
                batch_stats['failed_files'] += 1
                continue
        
        return batch_stats
    
    async def _process_single_document(self, pdf_file: Path) -> Dict[str, Any]:
        """Procesar un documento individual"""
        try:
            # Detectar estado
            state = self._detect_state_from_path(pdf_file)
            
            # Procesar documento con chunking legal
            processing_result = self.chunking_processor.process_document(pdf_file, state)
            
            document_metadata = processing_result['document_metadata']
            chunks = processing_result['chunks']
            
            # Generar embeddings para todos los chunks
            chunk_contents = [chunk.content for chunk in chunks]
            embeddings = await self.hf_client.generate_embeddings_batch(chunk_contents)
            
            # Preparar datos para Supabase
            await self._store_document_in_supabase(document_metadata, chunks, embeddings)
            
            # Actualizar estadísticas
            self.stats['documents_processed'] += 1
            self.stats['chunks_generated'] += len(chunks)
            self.stats['embeddings_generated'] += len(embeddings)
            
            return {
                'state': state,
                'law_type': document_metadata['law_type'],
                'total_chunks': len(chunks),
                'total_embeddings': len(embeddings)
            }
            
        except Exception as e:
            logger.error(f"❌ Error procesando documento {pdf_file}: {e}")
            raise
    
    def _detect_state_from_path(self, file_path: Path) -> str:
        """Detectar estado basado en la ruta del archivo"""
        path_parts = file_path.parts
        
        for part in path_parts:
            part_lower = part.lower().replace(' ', '_').replace('-', '_')
            if part_lower in self.chunking_processor.mexican_states:
                return part_lower
        
        return 'unknown'
    
    async def _store_document_in_supabase(
        self, 
        document_metadata: Dict[str, Any], 
        chunks: List[LegalChunk], 
        embeddings: List[np.ndarray]
    ):
        """Almacenar documento y chunks en Supabase"""
        try:
            # Insertar documento principal
            doc_data = {
                'document_id': document_metadata['document_id'],
                'title': document_metadata['title'],
                'state': document_metadata['state'],
                'document_type': document_metadata['document_type'],
                'law_type': document_metadata['law_type'],
                'effective_date': document_metadata.get('effective_date'),
                'amendment_date': document_metadata.get('amendment_date'),
                'status': document_metadata.get('status', 'vigente'),
                'file_path': document_metadata['file_path'],
                'file_size': document_metadata['file_size'],
                'total_pages': document_metadata.get('total_pages', 0),
                'total_chunks': len(chunks),
                'metadata': json.dumps(document_metadata.get('metadata', {}))
            }
            
            self.supabase.table('legal_documents').upsert(doc_data).execute()
            
            # Preparar chunks para inserción
            chunks_data = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_data = {
                    'document_id': document_metadata['document_id'],
                    'content': chunk.content,
                    'embedding': embedding.tolist(),
                    'article_number': chunk.article_number,
                    'paragraph_number': chunk.paragraph_number,
                    'inciso_number': chunk.inciso_number,
                    'chapter_title': chunk.chapter_title,
                    'section_title': chunk.section_title,
                    'law_title': chunk.law_title,
                    'chunk_type': chunk.chunk_type,
                    'page_number': chunk.page_number,
                    'word_count': chunk.word_count,
                    'char_count': chunk.char_count,
                    'metadata': json.dumps(chunk.metadata or {})
                }
                chunks_data.append(chunk_data)
            
            # Insertar chunks en lotes
            batch_size = 1000  # Tamaño de lote para Supabase
            for i in range(0, len(chunks_data), batch_size):
                batch = chunks_data[i:i + batch_size]
                self.supabase.table('legal_chunks').insert(batch).execute()
            
            logger.debug(f"✅ Documento almacenado: {document_metadata['document_id']}")
            
        except Exception as e:
            logger.error(f"❌ Error almacenando en Supabase: {e}")
            raise
    
    async def search_documents_hybrid(
        self,
        query: str,
        search_type: str = "hybrid",
        filters: Dict[str, Any] = None,
        top_k: int = 20,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> Dict[str, Any]:
        """Búsqueda híbrida optimizada con múltiples estrategias"""
        try:
            logger.info(f"🔍 Búsqueda {search_type}: '{query}'")
            
            if search_type == "hybrid":
                return await self._hybrid_search(query, filters, top_k, vector_weight, keyword_weight)
            elif search_type == "vector":
                return await self._vector_search(query, filters, top_k)
            elif search_type == "keyword":
                return await self._keyword_search(query, filters, top_k)
            elif search_type == "metadata":
                return await self._metadata_search(query, filters, top_k)
            else:
                raise ValueError(f"Tipo de búsqueda no soportado: {search_type}")
                
        except Exception as e:
            logger.error(f"❌ Error en búsqueda: {e}")
            return {"error": str(e), "results": []}
    
    async def _hybrid_search(
        self, 
        query: str, 
        filters: Dict[str, Any], 
        top_k: int,
        vector_weight: float,
        keyword_weight: float
    ) -> Dict[str, Any]:
        """Búsqueda híbrida combinando vectorial y palabras clave"""
        try:
            # Generar embedding de la consulta
            query_embedding = await self.hf_client.generate_embeddings_single(query)
            
            # Búsqueda híbrida usando función RPC optimizada
            result = self.supabase.rpc(
                "search_legal_chunks_hybrid",
                {
                    "query_embedding": query_embedding.tolist(),
                    "search_text": query,
                    "vector_weight": vector_weight,
                    "keyword_weight": keyword_weight,
                    "match_count": top_k,
                    "state_filter": filters.get("state") if filters else None,
                    "law_type_filter": filters.get("law_type") if filters else None,
                    "document_type_filter": filters.get("document_type") if filters else None
                }
            ).execute()
            
            if result.data:
                # Procesar resultados
                results = []
                for chunk in result.data:
                    result_item = {
                        "chunk_id": chunk.get("chunk_id"),
                        "document_id": chunk.get("document_id"),
                        "content": chunk.get("content"),
                        "similarity": chunk.get("combined_score", 0.0),
                        "vector_similarity": chunk.get("vector_similarity", 0.0),
                        "keyword_rank": chunk.get("keyword_rank", 0.0),
                        "metadata": {
                            "title": chunk.get("title"),
                            "state": chunk.get("state"),
                            "law_type": chunk.get("law_type"),
                            "document_type": chunk.get("document_type"),
                            "article_number": chunk.get("article_number"),
                            "chapter_title": chunk.get("chapter_title"),
                            "chunk_type": chunk.get("chunk_type"),
                            "page_number": chunk.get("page_number"),
                            "metadata": chunk.get("metadata", {})
                        }
                    }
                    results.append(result_item)
                
                # Actualizar estadísticas
                self.stats['queries_processed'] += 1
                
                return {
                    "results": results,
                    "total_found": len(results),
                    "search_type": "hybrid",
                    "query": query,
                    "filters": filters or {},
                    "vector_weight": vector_weight,
                    "keyword_weight": keyword_weight
                }
            else:
                return {"results": [], "total_found": 0, "search_type": "hybrid"}
                
        except Exception as e:
            logger.error(f"❌ Error en búsqueda híbrida: {e}")
            return {"error": str(e), "results": []}
    
    async def _vector_search(
        self, 
        query: str, 
        filters: Dict[str, Any], 
        top_k: int
    ) -> Dict[str, Any]:
        """Búsqueda vectorial pura"""
        try:
            query_embedding = await self.hf_client.generate_embeddings_single(query)
            
            result = self.supabase.rpc(
                "search_legal_chunks_optimized",
                {
                    "query_embedding": query_embedding.tolist(),
                    "match_threshold": 0.6,
                    "match_count": top_k,
                    "state_filter": filters.get("state") if filters else None,
                    "law_type_filter": filters.get("law_type") if filters else None,
                    "document_type_filter": filters.get("document_type") if filters else None
                }
            ).execute()
            
            if result.data:
                results = []
                for chunk in result.data:
                    result_item = {
                        "chunk_id": chunk.get("chunk_id"),
                        "document_id": chunk.get("document_id"),
                        "content": chunk.get("content"),
                        "similarity": chunk.get("similarity", 0.0),
                        "metadata": {
                            "title": chunk.get("title"),
                            "state": chunk.get("state"),
                            "law_type": chunk.get("law_type"),
                            "document_type": chunk.get("document_type"),
                            "article_number": chunk.get("article_number"),
                            "chapter_title": chunk.get("chapter_title"),
                            "chunk_type": chunk.get("chunk_type"),
                            "page_number": chunk.get("page_number"),
                            "metadata": chunk.get("metadata", {})
                        }
                    }
                    results.append(result_item)
                
                return {
                    "results": results,
                    "total_found": len(results),
                    "search_type": "vector"
                }
            else:
                return {"results": [], "total_found": 0, "search_type": "vector"}
                
        except Exception as e:
            logger.error(f"❌ Error en búsqueda vectorial: {e}")
            return {"error": str(e), "results": []}
    
    async def _keyword_search(
        self, 
        query: str, 
        filters: Dict[str, Any], 
        top_k: int
    ) -> Dict[str, Any]:
        """Búsqueda por palabras clave"""
        try:
            result = self.supabase.rpc(
                "search_legal_chunks_keyword",
                {
                    "search_text": query,
                    "match_count": top_k,
                    "state_filter": filters.get("state") if filters else None,
                    "law_type_filter": filters.get("law_type") if filters else None,
                    "document_type_filter": filters.get("document_type") if filters else None
                }
            ).execute()
            
            if result.data:
                results = []
                for chunk in result.data:
                    result_item = {
                        "chunk_id": chunk.get("chunk_id"),
                        "document_id": chunk.get("document_id"),
                        "content": chunk.get("content"),
                        "rank": chunk.get("rank", 0.0),
                        "metadata": {
                            "title": chunk.get("title"),
                            "state": chunk.get("state"),
                            "law_type": chunk.get("law_type"),
                            "document_type": chunk.get("document_type"),
                            "article_number": chunk.get("article_number"),
                            "chapter_title": chunk.get("chapter_title"),
                            "chunk_type": chunk.get("chunk_type"),
                            "page_number": chunk.get("page_number"),
                            "metadata": chunk.get("metadata", {})
                        }
                    }
                    results.append(result_item)
                
                return {
                    "results": results,
                    "total_found": len(results),
                    "search_type": "keyword"
                }
            else:
                return {"results": [], "total_found": 0, "search_type": "keyword"}
                
        except Exception as e:
            logger.error(f"❌ Error en búsqueda por palabras clave: {e}")
            return {"error": str(e), "results": []}
    
    async def _metadata_search(
        self, 
        query: str, 
        filters: Dict[str, Any], 
        top_k: int
    ) -> Dict[str, Any]:
        """Búsqueda por metadatos específicos"""
        try:
            # Construir consulta de metadatos
            query_builder = self.supabase.table("legal_chunks").select(
                "*, legal_documents!inner(title, state, law_type, document_type)"
            )
            
            # Aplicar filtros
            if filters:
                if filters.get("state"):
                    query_builder = query_builder.eq("legal_documents.state", filters["state"])
                if filters.get("law_type"):
                    query_builder = query_builder.eq("legal_documents.law_type", filters["law_type"])
                if filters.get("document_type"):
                    query_builder = query_builder.eq("legal_documents.document_type", filters["document_type"])
                if filters.get("article_number"):
                    query_builder = query_builder.ilike("article_number", f"%{filters['article_number']}%")
                if filters.get("chunk_type"):
                    query_builder = query_builder.eq("chunk_type", filters["chunk_type"])
            
            # Ejecutar consulta
            result = await query_builder.limit(top_k).execute()
            
            if result.data:
                results = []
                for chunk in result.data:
                    result_item = {
                        "chunk_id": chunk.get("chunk_id"),
                        "document_id": chunk.get("document_id"),
                        "content": chunk.get("content"),
                        "metadata": {
                            "title": chunk.get("legal_documents", {}).get("title"),
                            "state": chunk.get("legal_documents", {}).get("state"),
                            "law_type": chunk.get("legal_documents", {}).get("law_type"),
                            "document_type": chunk.get("legal_documents", {}).get("document_type"),
                            "article_number": chunk.get("article_number"),
                            "chapter_title": chunk.get("chapter_title"),
                            "chunk_type": chunk.get("chunk_type"),
                            "page_number": chunk.get("page_number"),
                            "metadata": chunk.get("metadata", {})
                        }
                    }
                    results.append(result_item)
                
                return {
                    "results": results,
                    "total_found": len(results),
                    "search_type": "metadata"
                }
            else:
                return {"results": [], "total_found": 0, "search_type": "metadata"}
                
        except Exception as e:
            logger.error(f"❌ Error en búsqueda por metadatos: {e}")
            return {"error": str(e), "results": []}
    
    def _log_stats(self):
        """Registrar estadísticas del sistema"""
        logger.info("📊 ESTADÍSTICAS DEL SISTEMA RAG:")
        logger.info(f"   📄 Documentos procesados: {self.stats['documents_processed']}")
        logger.info(f"   📝 Chunks generados: {self.stats['chunks_generated']}")
        logger.info(f"   🧮 Embeddings generados: {self.stats['embeddings_generated']}")
        logger.info(f"   🔍 Consultas procesadas: {self.stats['queries_processed']}")
        logger.info(f"   ⏱️ Tiempo total: {self.stats['total_processing_time']:.2f}s")
        
        if self.hf_client:
            hf_stats = self.hf_client.get_stats()
            logger.info(f"   🚀 Requests HF: {hf_stats['total_requests']}")
            logger.info(f"   🚀 Tokens HF: {hf_stats['total_tokens']}")
            logger.info(f"   🚀 Requests/segundo: {hf_stats['requests_per_second']:.2f}")
            logger.info(f"   🚀 Tokens/segundo: {hf_stats['tokens_per_second']:.2f}")
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de la base de datos"""
        try:
            result = self.supabase.rpc("get_database_stats").execute()
            return result.data[0] if result.data else {}
        except Exception as e:
            logger.error(f"❌ Error obteniendo estadísticas: {e}")
            return {"error": str(e)}
