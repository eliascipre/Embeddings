#!/usr/bin/env python3
"""
Sistema RAG Optimizado para Documentos SIEM
Procesamiento masivo paralelo con 16 workers y lotes de 50 archivos
Búsqueda híbrida: vectorial + palabras clave + metadatos
"""

import asyncio
import logging
import hashlib
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import numpy as np
from supabase import create_client, Client

try:
    from .comercio_exterior_processor import ComercioExteriorProcessor
    from .smart_legal_chunker import SmartLegalChunker, LegalChunk
    from .qwen3_embedding_client import Qwen3EmbeddingClient
except ImportError:
    from comercio_exterior_processor import ComercioExteriorProcessor
    from smart_legal_chunker import SmartLegalChunker, LegalChunk
    from qwen3_embedding_client import Qwen3EmbeddingClient
from config_supabase import get_database_tables, get_processing_config

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Resultado del procesamiento de un documento"""
    file_path: str
    success: bool
    chunks_created: int
    embeddings_generated: int
    processing_time: float
    error_message: Optional[str] = None

@dataclass
class SearchResult:
    """Resultado de búsqueda"""
    chunk_id: int
    document_id: int
    file_name: str
    chunk_text: str
    similarity_score: float
    chunk_type: str
    metadata: Dict[str, Any]

class OptimizedSIEMRAGSystem:
    """Sistema RAG optimizado para documentos SIEM"""
    
    def __init__(self, supabase_url: str, supabase_key: str, 
                 embedding_client: Qwen3EmbeddingClient,
                 document_processor: ComercioExteriorProcessor):
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.embedding_client = embedding_client
        self.document_processor = document_processor
        self.tables = get_database_tables()
        self.processing_config = get_processing_config()
        
        # Cliente Supabase
        self.supabase: Optional[Client] = None
        
        # Estadísticas
        self.stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'embeddings_generated': 0,
            'processing_time': 0.0,
            'errors': 0
        }
    
    async def initialize(self):
        """Inicializar el sistema"""
        try:
            logger.info("Inicializando sistema RAG optimizado...")
            
            # Conectar a Supabase con configuración correcta
            self.supabase = create_client(self.supabase_url, self.supabase_key)
            logger.info("Conexión con Supabase establecida")
            
            # Inicializar cliente de embeddings si está disponible
            if self.embedding_client:
                await self.embedding_client.__aenter__()
                logger.info("Cliente de embeddings inicializado")
            
            # Verificar tablas
            await self._verify_database_schema()
            
            logger.info("Sistema RAG inicializado correctamente")
            
        except Exception as e:
            logger.error(f"Error inicializando sistema RAG: {e}")
            raise
    
    def _get_headers(self):
        """Obtener headers para autenticación con Supabase"""
        return {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json"
        }
    
    async def _verify_database_schema(self):
        """Verificar que el esquema de la base de datos esté correcto"""
        try:
            # Usar el mismo método que funcionó en el test
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.supabase_url}/rest/v1/siem_documents?select=id&limit=1", 
                                     headers=self._get_headers()) as response:
                    if response.status == 200:
                        logger.info("Tabla siem_documents verificada correctamente")
                    else:
                        raise Exception(f"Error verificando siem_documents: {response.status}")
            
        except Exception as e:
            logger.error(f"Error verificando esquema de base de datos: {e}")
            raise
    
    async def process_documents_batch(self, file_paths: List[Path], 
                                    max_workers: int = 16) -> Dict[str, int]:
        """Procesar un lote de documentos con workers paralelos"""
        logger.info(f"Procesando lote de {len(file_paths)} archivos con {max_workers} workers")
        
        results = {
            'processed': 0,
            'failed': 0,
            'chunks_created': 0,
            'embeddings_generated': 0
        }
        
        # Procesar archivos en paralelo
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Crear tareas
            future_to_path = {
                executor.submit(self._process_single_document, file_path): file_path
                for file_path in file_paths
            }
            
            # Procesar resultados conforme se completan
            for future in as_completed(future_to_path):
                file_path = future_to_path[future]
                try:
                    result = future.result()
                    
                    if result.success:
                        results['processed'] += 1
                        results['chunks_created'] += result.chunks_created
                        results['embeddings_generated'] += result.embeddings_generated
                        logger.info(f"✓ {file_path.name}: {result.chunks_created} chunks, {result.embeddings_generated} embeddings")
                    else:
                        results['failed'] += 1
                        logger.error(f"✗ {file_path.name}: {result.error_message}")
                    
                except Exception as e:
                    results['failed'] += 1
                    logger.error(f"✗ {file_path.name}: Error inesperado - {e}")
        
        logger.info(f"Lote completado: {results['processed']} procesados, {results['failed']} fallidos")
        return results
    
    def _process_single_document(self, file_path: Path) -> ProcessingResult:
        """Procesar un solo documento (ejecutado en worker)"""
        start_time = time.time()
        
        try:
            # Verificar si el documento ya fue procesado
            file_hash = self._calculate_file_hash(file_path)
            
            # Verificar duplicados
            existing_doc = self.supabase.table('siem_documents').select('id').eq('file_hash', file_hash).execute()
            if existing_doc.data:
                logger.info(f"Documento ya procesado: {file_path.name}")
                return ProcessingResult(
                    file_path=str(file_path),
                    success=True,
                    chunks_created=0,
                    embeddings_generated=0,
                    processing_time=0.0
                )
            
            # Extraer texto del documento
            text_content = self.document_processor.extract_text(file_path)
            if not text_content.strip():
                raise ValueError("No se pudo extraer texto del documento")
            
            # Crear chunks de comercio exterior usando chunking inteligente
            chunks = self.document_processor.create_comercio_chunks(text_content, file_path)
            if not chunks:
                raise ValueError("No se pudieron crear chunks del documento")
            
            # Filtrar chunks de calidad (eliminar chunks extraños)
            quality_chunks = self._filter_quality_chunks(chunks)
            if not quality_chunks:
                raise ValueError("No se encontraron chunks de calidad en el documento")
            
            logger.info(f"Chunks de calidad: {len(quality_chunks)}/{len(chunks)} para {file_path.name}")
            
            # Guardar documento en base de datos
            document_data = {
                'file_name': file_path.name,
                'file_path': str(file_path),
                'file_hash': file_hash,
                'file_size': file_path.stat().st_size,
                'file_extension': file_path.suffix,
                'processing_status': 'processing',
                'processing_started_at': datetime.now().isoformat()
            }
            
            doc_result = self.supabase.table('siem_documents').insert(document_data).execute()
            document_id = doc_result.data[0]['id']
            
            # Procesar chunks y embeddings
            chunks_created = 0
            embeddings_generated = 0
            
            for chunk in quality_chunks:
                # Guardar chunk con metadatos avanzados
                chunk_data = {
                    'document_id': document_id,
                    'chunk_text': chunk['text'],
                    'chunk_text_clean': chunk.get('clean_text', chunk['text']),
                    'chunk_type': chunk['type'],
                    'chunk_index': chunk['index'],
                    'word_count': chunk.get('word_count', len(chunk['text'].split())),
                    'chunk_hash': hashlib.md5(chunk['text'].encode()).hexdigest(),
                    # Metadatos del chunking inteligente
                    'chunk_id': chunk.get('chunk_id'),
                    'hierarchy_level': chunk.get('hierarchy_level', 1),
                    'parent_chunk_id': chunk.get('parent_chunk_id'),
                    'article_number': chunk.get('article_number'),
                    'paragraph_number': chunk.get('paragraph_number'),
                    'inciso_number': chunk.get('inciso_number'),
                    'chapter_title': chunk.get('chapter_title'),
                    'section_title': chunk.get('section_title'),
                    'law_title': chunk.get('law_title'),
                    'page_number': chunk.get('page_number'),
                    'char_count': chunk.get('char_count', len(chunk['text'])),
                    'is_compressed': chunk.get('is_compressed', False),
                    'compression_ratio': chunk.get('compression_ratio', 1.0),
                    'metadata': json.dumps(chunk.get('metadata', {}))
                }
                
                chunk_result = self.supabase.table('siem_chunks').insert(chunk_data).execute()
                chunk_id = chunk_result.data[0]['id']
                chunks_created += 1
                
                # Generar embedding (solo si hay cliente disponible)
                if self.embedding_client:
                    try:
                        embedding = asyncio.run(self.embedding_client.generate_single_embedding(chunk['text']))
                        
                        # Guardar embedding
                        embedding_data = {
                            'chunk_id': chunk_id,
                            'embedding_vector': embedding.tolist(),
                            'model_name': 'Qwen3-Embedding-0.6B',
                            'dimensions': len(embedding),
                            'generated_at': datetime.now().isoformat()
                        }
                        
                        self.supabase.table('siem_embeddings').insert(embedding_data).execute()
                        embeddings_generated += 1
                        
                    except Exception as e:
                        logger.warning(f"Error generando embedding para chunk {chunk_id}: {e}")
                else:
                    logger.info(f"Chunk {chunk_id} guardado sin embedding (cliente no disponible)")
            
            # Actualizar estado del documento
            processing_time = time.time() - start_time
            self.supabase.table('siem_documents').update({
                'processing_status': 'completed',
                'processing_completed_at': datetime.now().isoformat()
            }).eq('id', document_id).execute()
            
            return ProcessingResult(
                file_path=str(file_path),
                success=True,
                chunks_created=chunks_created,
                embeddings_generated=embeddings_generated,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error procesando {file_path.name}: {e}")
            return ProcessingResult(
                file_path=str(file_path),
                success=False,
                chunks_created=0,
                embeddings_generated=0,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calcular hash MD5 del archivo"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    async def search_documents(self, query: str, search_type: str = "hybrid",
                             filters: Dict[str, Any] = None, top_k: int = 20) -> Dict[str, Any]:
        """Buscar documentos usando búsqueda híbrida"""
        start_time = time.time()
        
        try:
            if search_type == "vectorial":
                results = await self._search_vectorial(query, filters, top_k)
            elif search_type == "keyword":
                results = await self._search_keyword(query, filters, top_k)
            elif search_type == "metadata":
                results = await self._search_metadata(query, filters, top_k)
            else:  # hybrid
                results = await self._search_hybrid(query, filters, top_k)
            
            search_time = time.time() - start_time
            results['search_time'] = search_time
            
            return results
            
        except Exception as e:
            logger.error(f"Error en búsqueda: {e}")
            raise
    
    async def _search_hybrid(self, query: str, filters: Dict[str, Any], top_k: int) -> Dict[str, Any]:
        """Búsqueda híbrida combinando vectorial y palabras clave"""
        try:
            # Generar embedding de la consulta
            query_embedding = await self.embedding_client.generate_single_embedding(query)
            
            # Búsqueda vectorial
            vectorial_results = await self._search_vectorial_embeddings(query_embedding, filters, top_k)
            
            # Búsqueda por palabras clave
            keyword_results = await self._search_keyword_text(query, filters, top_k)
            
            # Combinar resultados
            combined_results = self._combine_search_results(vectorial_results, keyword_results, top_k)
            
            return {
                'results': combined_results,
                'total_found': len(combined_results),
                'search_type': 'hybrid'
            }
            
        except Exception as e:
            logger.error(f"Error en búsqueda híbrida: {e}")
            raise
    
    async def _search_vectorial_embeddings(self, query_embedding: np.ndarray, 
                                         filters: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
        """Búsqueda vectorial usando embeddings"""
        try:
            # Construir consulta SQL para búsqueda vectorial
            query_sql = """
            SELECT 
                c.id as chunk_id,
                c.document_id,
                d.file_name,
                c.chunk_text,
                c.chunk_type,
                c.legal_hierarchy,
                1 - (e.embedding_vector <=> %s) as similarity_score
            FROM siem_chunks c
            JOIN siem_documents d ON c.document_id = d.id
            JOIN siem_embeddings e ON c.id = e.chunk_id
            WHERE 1=1
            """
            
            params = [query_embedding.tolist()]
            
            # Aplicar filtros
            if filters.get('category'):
                query_sql += " AND d.file_name ILIKE %s"
                params.append(f"%{filters['category']}%")
            
            if filters.get('legal_type'):
                query_sql += " AND c.chunk_type = %s"
                params.append(filters['legal_type'])
            
            query_sql += " ORDER BY similarity_score DESC LIMIT %s"
            params.append(top_k)
            
            # Ejecutar consulta
            result = self.supabase.rpc('search_embeddings_cosine', {
                'query_embedding': query_embedding.tolist(),
                'match_threshold': 0.7,
                'match_count': top_k
            }).execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            logger.error(f"Error en búsqueda vectorial: {e}")
            return []
    
    async def _search_keyword_text(self, query: str, filters: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
        """Búsqueda por palabras clave"""
        try:
            # Construir consulta de búsqueda por texto
            search_query = self.supabase.table('siem_chunks').select(
                'id, document_id, chunk_text, chunk_type, legal_hierarchy, siem_documents(file_name)'
            ).text_search('chunk_text_clean', query)
            
            # Aplicar filtros
            if filters.get('category'):
                search_query = search_query.ilike('siem_documents.file_name', f"%{filters['category']}%")
            
            if filters.get('legal_type'):
                search_query = search_query.eq('chunk_type', filters['legal_type'])
            
            result = search_query.limit(top_k).execute()
            
            # Formatear resultados
            formatted_results = []
            for item in result.data:
                formatted_results.append({
                    'chunk_id': item['id'],
                    'document_id': item['document_id'],
                    'file_name': item['siem_documents']['file_name'],
                    'chunk_text': item['chunk_text'],
                    'chunk_type': item['chunk_type'],
                    'similarity_score': 0.8,  # Score fijo para búsqueda por texto
                    'metadata': item.get('legal_hierarchy', {})
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error en búsqueda por palabras clave: {e}")
            return []
    
    async def _search_metadata(self, query: str, filters: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
        """Búsqueda por metadatos"""
        try:
            # Buscar en metadatos de documentos
            search_query = self.supabase.table('siem_documents').select(
                'id, file_name, file_path, created_at'
            )
            
            # Aplicar filtros de metadatos
            if filters.get('category'):
                search_query = search_query.ilike('file_name', f"%{filters['category']}%")
            
            if filters.get('date_from'):
                search_query = search_query.gte('created_at', filters['date_from'])
            
            if filters.get('date_to'):
                search_query = search_query.lte('created_at', filters['date_to'])
            
            result = search_query.limit(top_k).execute()
            
            # Formatear resultados
            formatted_results = []
            for item in result.data:
                formatted_results.append({
                    'document_id': item['id'],
                    'file_name': item['file_name'],
                    'file_path': item['file_path'],
                    'similarity_score': 1.0,  # Score fijo para búsqueda por metadatos
                    'metadata': {
                        'created_at': item['created_at'],
                        'search_type': 'metadata'
                    }
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error en búsqueda por metadatos: {e}")
            return []
    
    def _combine_search_results(self, vectorial_results: List[Dict[str, Any]], 
                              keyword_results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Combinar resultados de búsqueda vectorial y por palabras clave"""
        # Crear diccionario de resultados únicos
        combined = {}
        
        # Agregar resultados vectoriales (peso 0.7)
        for result in vectorial_results:
            chunk_id = result['chunk_id']
            combined[chunk_id] = {
                **result,
                'combined_score': result['similarity_score'] * 0.7
            }
        
        # Agregar resultados por palabras clave (peso 0.3)
        for result in keyword_results:
            chunk_id = result['chunk_id']
            if chunk_id in combined:
                # Combinar scores
                combined[chunk_id]['combined_score'] += result['similarity_score'] * 0.3
            else:
                combined[chunk_id] = {
                    **result,
                    'combined_score': result['similarity_score'] * 0.3
                }
        
        # Ordenar por score combinado y retornar top_k
        sorted_results = sorted(combined.values(), key=lambda x: x['combined_score'], reverse=True)
        return sorted_results[:top_k]
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de la base de datos"""
        try:
            # Contar documentos
            docs_result = self.supabase.table('siem_documents').select('id', count='exact').execute()
            total_documents = docs_result.count
            
            # Contar chunks
            chunks_result = self.supabase.table('siem_chunks').select('id', count='exact').execute()
            total_chunks = chunks_result.count
            
            # Contar embeddings
            embeddings_result = self.supabase.table('siem_embeddings').select('id', count='exact').execute()
            total_embeddings = embeddings_result.count
            
            # Calcular tamaño total
            size_result = self.supabase.table('siem_documents').select('file_size').execute()
            total_size_mb = sum(item['file_size'] for item in size_result.data) / (1024 * 1024)
            
            return {
                'total_documents': total_documents,
                'total_chunks': total_chunks,
                'total_embeddings': total_embeddings,
                'total_size_mb': total_size_mb,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {}
    
    def _filter_quality_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filtrar chunks de calidad, eliminando chunks extraños o inútiles"""
        import re
        
        quality_chunks = []
        
        for chunk in chunks:
            # Verificar que el chunk tenga contenido útil
            if not self._is_quality_chunk(chunk):
                continue
            
            # Verificar tamaño mínimo y máximo
            word_count = chunk.get('word_count', len(chunk['text'].split()))
            char_count = chunk.get('char_count', len(chunk['text']))
            
            if word_count < 10:  # Muy corto
                continue
            if char_count > 10000:  # Muy largo
                continue
            
            # Verificar que no sea contenido inútil
            text = chunk['text'].lower()
            useless_indicators = [
                'página', 'sin texto', 'dof - diario oficial',
                'fe de erratas', 'nota: el presente documento',
                'página 1 de', '--- página'
            ]
            
            if any(indicator in text for indicator in useless_indicators):
                continue
            
            # Verificar que tenga contenido legal relevante
            legal_indicators = [
                'artículo', 'capítulo', 'título', 'sección', 'párrafo',
                'fracción', 'transitorio', 'definiciones', 'disposiciones',
                'ley', 'código', 'reglamento', 'decreto', 'acuerdo',
                'convenio', 'tratado', 'comercio', 'exterior', 'aduana'
            ]
            
            if not any(indicator in text for indicator in legal_indicators):
                # Si no tiene indicadores legales, verificar que sea suficientemente largo
                if word_count < 50:
                    continue
            
            quality_chunks.append(chunk)
        
        return quality_chunks
    
    def _is_quality_chunk(self, chunk: Dict[str, Any]) -> bool:
        """Verificar si un chunk es de calidad"""
        import re
        
        text = chunk.get('text', '')
        
        # Verificar que no esté vacío
        if not text or not text.strip():
            return False
        
        # Verificar que no sea solo números o caracteres especiales
        if re.match(r'^[\d\s\-\.]+$', text.strip()):
            return False
        
        # Verificar que no sea solo números romanos
        if re.match(r'^[IVX\s]+$', text.strip()):
            return False
        
        # Verificar que tenga al menos algunas letras
        if not re.search(r'[a-zA-ZáéíóúñÁÉÍÓÚÑ]', text):
            return False
        
        return True

    async def cleanup(self):
        """Limpiar recursos"""
        try:
            if self.embedding_client:
                await self.embedding_client.cleanup()
            logger.info("Recursos del sistema RAG limpiados")
        except Exception as e:
            logger.error(f"Error en limpieza: {e}")
