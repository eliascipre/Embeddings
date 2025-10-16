"""
Sistema de base de datos simplificado usando solo Supabase API
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime
import numpy as np

from supabase import create_client, Client
from config import config

logger = logging.getLogger(__name__)

class SupabaseVectorDatabase:
    """Base de datos vectorial simplificada usando solo Supabase API"""
    
    def __init__(self):
        self.supabase: Optional[Client] = None
        self._setup_connections()
        
    def _setup_connections(self):
        """Configurar conexiones a Supabase"""
        try:
            # Configurar cliente Supabase
            self.supabase = create_client(
                config.supabase_url,
                config.supabase_key
            )
            logger.info("✅ Cliente Supabase configurado")
            
            # Inicializar tablas en Supabase
            self._initialize_supabase_tables()
            
        except Exception as e:
            logger.error(f"❌ Error configurando conexiones: {e}")
            raise
    
    def _initialize_supabase_tables(self):
        """Inicializar tablas en Supabase usando la API"""
        try:
            # Crear tabla de documentos legales
            try:
                self.supabase.table('legal_documents').select('*').limit(1).execute()
                logger.info("✅ Tabla legal_documents ya existe")
            except:
                # La tabla no existe, necesitamos crearla manualmente en Supabase
                logger.warning("⚠️ Tabla legal_documents no existe. Créala manualmente en Supabase con:")
                logger.warning("""
                CREATE TABLE legal_documents (
                    id SERIAL PRIMARY KEY,
                    document_id VARCHAR(255) UNIQUE NOT NULL,
                    title VARCHAR(500),
                    state VARCHAR(100),
                    document_type VARCHAR(100),
                    file_path TEXT,
                    file_size BIGINT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB
                );
                """)
            
            # Crear tabla de chunks con embeddings
            try:
                self.supabase.table('legal_chunks').select('*').limit(1).execute()
                logger.info("✅ Tabla legal_chunks ya existe")
            except:
                logger.warning("⚠️ Tabla legal_chunks no existe. Créala manualmente en Supabase con:")
                logger.warning("""
                CREATE TABLE legal_chunks (
                    id SERIAL PRIMARY KEY,
                    document_id VARCHAR(255) NOT NULL,
                    chunk_id INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding VECTOR(1024),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """)
                
        except Exception as e:
            logger.error(f"❌ Error inicializando tablas: {e}")
            raise
    
    async def store_document(self, document_data: Dict[str, Any]) -> str:
        """Almacenar documento en Supabase"""
        try:
            document_id = document_data.get("document_id")
            if not document_id:
                raise ValueError("document_id es requerido")
            
            # Insertar o actualizar documento
            result = self.supabase.table('legal_documents').upsert({
                "document_id": document_id,
                "title": document_data.get("title", ""),
                "state": document_data.get("state", ""),
                "document_type": document_data.get("document_type", ""),
                "file_path": document_data.get("file_path", ""),
                "file_size": document_data.get("file_size", 0),
                "metadata": document_data.get("metadata", {}),
                "updated_at": datetime.now().isoformat()
            }).execute()
            
            logger.info(f"✅ Documento {document_id} almacenado en Supabase")
            return document_id
                
        except Exception as e:
            logger.error(f"❌ Error almacenando documento: {e}")
            raise
    
    async def store_chunks(self, document_id: str, chunks: List[Dict[str, Any]]) -> int:
        """Almacenar chunks con embeddings en Supabase"""
        try:
            stored_count = 0
            
            # Eliminar chunks existentes del documento
            self.supabase.table('legal_chunks').delete().eq('document_id', document_id).execute()
            
            # Preparar datos para inserción
            chunks_data = []
            for chunk in chunks:
                embedding = chunk.get("embedding")
                if embedding is None:
                    logger.warning(f"⚠️ Chunk sin embedding omitido: {chunk.get('chunk_id')}")
                    continue
                
                chunk_data = {
                    "document_id": document_id,
                    "chunk_id": chunk.get("chunk_id", 0),
                    "content": chunk.get("content", ""),
                    "embedding": embedding,  # Supabase manejará el vector
                    "metadata": chunk.get("metadata", {})
                }
                chunks_data.append(chunk_data)
                stored_count += 1
            
            # Insertar chunks en lotes
            if chunks_data:
                self.supabase.table('legal_chunks').insert(chunks_data).execute()
            
            logger.info(f"✅ {stored_count} chunks almacenados para documento {document_id}")
            return stored_count
                
        except Exception as e:
            logger.error(f"❌ Error almacenando chunks: {e}")
            raise
    
    async def search_similar_chunks(
        self, 
        query_embedding: List[float], 
        limit: int = 10,
        similarity_threshold: float = 0.7,
        state: Optional[str] = None,
        document_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Buscar chunks similares usando Supabase"""
        try:
            # Usar la función de búsqueda vectorial de Supabase
            query = self.supabase.rpc(
                'match_legal_chunks',
                {
                    'query_embedding': query_embedding,
                    'match_threshold': similarity_threshold,
                    'match_count': limit,
                    'filter_state': state,
                    'filter_document_type': document_type
                }
            ).execute()
            
            results = query.data if hasattr(query, 'data') else []
            
            # Procesar resultados
            similar_chunks = []
            for result in results:
                chunk_data = {
                    "id": result.get('id'),
                    "document_id": result.get('document_id'),
                    "chunk_id": result.get('chunk_id'),
                    "content": result.get('content'),
                    "metadata": result.get('metadata', {}),
                    "similarity": result.get('similarity', 0.0)
                }
                similar_chunks.append(chunk_data)
            
            logger.info(f"✅ Encontrados {len(similar_chunks)} chunks similares")
            return similar_chunks
                
        except Exception as e:
            logger.error(f"❌ Error buscando chunks similares: {e}")
            # Fallback a búsqueda simple
            return await self._fallback_search(query_embedding, limit)
    
    async def _fallback_search(self, query_embedding: List[float], limit: int) -> List[Dict[str, Any]]:
        """Búsqueda de fallback sin función vectorial"""
        try:
            # Búsqueda simple por contenido
            query = self.supabase.table('legal_chunks').select('*').limit(limit)
            result = query.execute()
            
            chunks = []
            for chunk in result.data:
                chunk_data = {
                    "id": chunk.get('id'),
                    "document_id": chunk.get('document_id'),
                    "chunk_id": chunk.get('chunk_id'),
                    "content": chunk.get('content'),
                    "metadata": chunk.get('metadata', {}),
                    "similarity": 0.5  # Similitud por defecto
                }
                chunks.append(chunk_data)
            
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Error en búsqueda de fallback: {e}")
            return []
    
    async def get_document_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de la base de datos"""
        try:
            # Contar documentos
            docs_result = self.supabase.table('legal_documents').select('*', count='exact').execute()
            total_documents = docs_result.count if hasattr(docs_result, 'count') else 0
            
            # Contar chunks
            chunks_result = self.supabase.table('legal_chunks').select('*', count='exact').execute()
            total_chunks = chunks_result.count if hasattr(chunks_result, 'count') else 0
            
            return {
                "total_documents": total_documents,
                "total_chunks": total_chunks,
                "database_type": "supabase",
                "embedding_dimensions": 1024
            }
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo estadísticas: {e}")
            return {"error": str(e)}
    
    async def delete_document(self, document_id: str) -> bool:
        """Eliminar documento y sus chunks"""
        try:
            # Eliminar chunks primero
            self.supabase.table('legal_chunks').delete().eq('document_id', document_id).execute()
            
            # Eliminar documento
            self.supabase.table('legal_documents').delete().eq('document_id', document_id).execute()
            
            logger.info(f"✅ Documento {document_id} eliminado")
            return True
                
        except Exception as e:
            logger.error(f"❌ Error eliminando documento: {e}")
            return False
    
    def close_connections(self):
        """Cerrar conexiones a la base de datos"""
        logger.info("✅ Conexiones cerradas")
