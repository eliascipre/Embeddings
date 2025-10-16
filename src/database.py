"""
Sistema de base de datos con pgvector para Supabase
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import sql
import json
from datetime import datetime

from supabase import create_client, Client
from .config import config

logger = logging.getLogger(__name__)

class LegalVectorDatabase:
    """Base de datos vectorial para documentos legales con pgvector"""
    
    def __init__(self):
        self.supabase: Optional[Client] = None
        self.pg_connection = None
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
            
            # Configurar conexión PostgreSQL a través de Supabase
            # Construir URL de conexión desde Supabase
            supabase_db_url = f"postgresql://postgres.{config.supabase_url.split('//')[1].split('.')[0]}:{config.supabase_key}@db.{config.supabase_url.split('//')[1].split('.')[0]}.supabase.co:5432/postgres"
            
            self.pg_connection = psycopg2.connect(
                supabase_db_url,
                cursor_factory=RealDictCursor
            )
            logger.info("✅ Conexión PostgreSQL a Supabase configurada")
            
            # Inicializar extensión pgvector
            self._initialize_pgvector()
            
        except Exception as e:
            logger.error(f"❌ Error configurando conexiones: {e}")
            raise
    
    def _initialize_pgvector(self):
        """Inicializar extensión pgvector en la base de datos"""
        try:
            with self.pg_connection.cursor() as cursor:
                # Crear extensión pgvector si no existe
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # Crear tabla de documentos legales
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS legal_documents (
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
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS legal_chunks (
                        id SERIAL PRIMARY KEY,
                        document_id VARCHAR(255) NOT NULL,
                        chunk_id INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        embedding VECTOR(4096),
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (document_id) REFERENCES legal_documents(document_id)
                    );
                """)
                
                # Crear índices para optimizar búsquedas
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_legal_chunks_embedding 
                    ON legal_chunks USING ivfflat (embedding vector_cosine_ops) 
                    WITH (lists = 100);
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_legal_chunks_document_id 
                    ON legal_chunks(document_id);
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_legal_documents_state 
                    ON legal_documents(state);
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_legal_documents_type 
                    ON legal_documents(document_type);
                """)
                
                self.pg_connection.commit()
                logger.info("✅ Base de datos pgvector inicializada correctamente")
                
        except Exception as e:
            logger.error(f"❌ Error inicializando pgvector: {e}")
            self.pg_connection.rollback()
            raise
    
    async def store_document(self, document_data: Dict[str, Any]) -> str:
        """Almacenar documento en la base de datos"""
        try:
            document_id = document_data.get("document_id")
            if not document_id:
                raise ValueError("document_id es requerido")
            
            with self.pg_connection.cursor() as cursor:
                # Insertar o actualizar documento
                cursor.execute("""
                    INSERT INTO legal_documents 
                    (document_id, title, state, document_type, file_path, file_size, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (document_id) 
                    DO UPDATE SET
                        title = EXCLUDED.title,
                        state = EXCLUDED.state,
                        document_type = EXCLUDED.document_type,
                        file_path = EXCLUDED.file_path,
                        file_size = EXCLUDED.file_size,
                        metadata = EXCLUDED.metadata,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING id;
                """, (
                    document_id,
                    document_data.get("title", ""),
                    document_data.get("state", ""),
                    document_data.get("document_type", ""),
                    document_data.get("file_path", ""),
                    document_data.get("file_size", 0),
                    json.dumps(document_data.get("metadata", {}))
                ))
                
                result = cursor.fetchone()
                doc_db_id = result['id'] if result else None
                
                self.pg_connection.commit()
                logger.info(f"✅ Documento {document_id} almacenado con ID {doc_db_id}")
                return document_id
                
        except Exception as e:
            logger.error(f"❌ Error almacenando documento: {e}")
            self.pg_connection.rollback()
            raise
    
    async def store_chunks(self, document_id: str, chunks: List[Dict[str, Any]]) -> int:
        """Almacenar chunks con embeddings en la base de datos"""
        try:
            stored_count = 0
            
            with self.pg_connection.cursor() as cursor:
                # Eliminar chunks existentes del documento
                cursor.execute(
                    "DELETE FROM legal_chunks WHERE document_id = %s",
                    (document_id,)
                )
                
                # Insertar nuevos chunks
                for chunk in chunks:
                    embedding = chunk.get("embedding")
                    if embedding is None:
                        logger.warning(f"⚠️ Chunk sin embedding omitido: {chunk.get('chunk_id')}")
                        continue
                    
                    # Convertir embedding a formato pgvector
                    embedding_str = f"[{','.join(map(str, embedding))}]"
                    
                    cursor.execute("""
                        INSERT INTO legal_chunks 
                        (document_id, chunk_id, content, embedding, metadata)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (
                        document_id,
                        chunk.get("chunk_id", 0),
                        chunk.get("content", ""),
                        embedding_str,
                        json.dumps(chunk.get("metadata", {}))
                    ))
                    stored_count += 1
                
                self.pg_connection.commit()
                logger.info(f"✅ {stored_count} chunks almacenados para documento {document_id}")
                return stored_count
                
        except Exception as e:
            logger.error(f"❌ Error almacenando chunks: {e}")
            self.pg_connection.rollback()
            raise
    
    async def search_similar_chunks(
        self, 
        query_embedding: List[float], 
        limit: int = 10,
        similarity_threshold: float = 0.7,
        state: Optional[str] = None,
        document_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Buscar chunks similares usando pgvector"""
        try:
            with self.pg_connection.cursor() as cursor:
                # Construir consulta SQL
                query_parts = [
                    """
                    SELECT 
                        lc.id,
                        lc.document_id,
                        lc.chunk_id,
                        lc.content,
                        lc.metadata,
                        ld.title,
                        ld.state,
                        ld.document_type,
                        1 - (lc.embedding <=> %s) as similarity
                    FROM legal_chunks lc
                    JOIN legal_documents ld ON lc.document_id = ld.document_id
                    WHERE 1 - (lc.embedding <=> %s) > %s
                    """
                ]
                
                params = [query_embedding, query_embedding, similarity_threshold]
                
                # Agregar filtros opcionales
                if state:
                    query_parts.append("AND ld.state = %s")
                    params.append(state)
                
                if document_type:
                    query_parts.append("AND ld.document_type = %s")
                    params.append(document_type)
                
                query_parts.append("ORDER BY similarity DESC LIMIT %s")
                params.append(limit)
                
                query = " ".join(query_parts)
                cursor.execute(query, params)
                
                results = cursor.fetchall()
                
                # Convertir resultados a diccionarios
                similar_chunks = []
                for row in results:
                    chunk_data = {
                        "id": row['id'],
                        "document_id": row['document_id'],
                        "chunk_id": row['chunk_id'],
                        "content": row['content'],
                        "metadata": json.loads(row['metadata']) if row['metadata'] else {},
                        "title": row['title'],
                        "state": row['state'],
                        "document_type": row['document_type'],
                        "similarity": float(row['similarity'])
                    }
                    similar_chunks.append(chunk_data)
                
                logger.info(f"✅ Encontrados {len(similar_chunks)} chunks similares")
                return similar_chunks
                
        except Exception as e:
            logger.error(f"❌ Error buscando chunks similares: {e}")
            raise
    
    async def get_document_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de la base de datos"""
        try:
            with self.pg_connection.cursor() as cursor:
                # Contar documentos
                cursor.execute("SELECT COUNT(*) as total_documents FROM legal_documents")
                total_documents = cursor.fetchone()['total_documents']
                
                # Contar chunks
                cursor.execute("SELECT COUNT(*) as total_chunks FROM legal_chunks")
                total_chunks = cursor.fetchone()['total_chunks']
                
                # Contar por estado
                cursor.execute("""
                    SELECT state, COUNT(*) as count 
                    FROM legal_documents 
                    GROUP BY state 
                    ORDER BY count DESC
                """)
                states_stats = cursor.fetchall()
                
                # Contar por tipo de documento
                cursor.execute("""
                    SELECT document_type, COUNT(*) as count 
                    FROM legal_documents 
                    GROUP BY document_type 
                    ORDER BY count DESC
                """)
                types_stats = cursor.fetchall()
                
                return {
                    "total_documents": total_documents,
                    "total_chunks": total_chunks,
                    "states_distribution": [dict(row) for row in states_stats],
                    "document_types_distribution": [dict(row) for row in types_stats],
                    "database_type": "pgvector",
                    "embedding_dimensions": 4096
                }
                
        except Exception as e:
            logger.error(f"❌ Error obteniendo estadísticas: {e}")
            return {"error": str(e)}
    
    async def delete_document(self, document_id: str) -> bool:
        """Eliminar documento y sus chunks"""
        try:
            with self.pg_connection.cursor() as cursor:
                # Eliminar chunks primero (por la foreign key)
                cursor.execute(
                    "DELETE FROM legal_chunks WHERE document_id = %s",
                    (document_id,)
                )
                
                # Eliminar documento
                cursor.execute(
                    "DELETE FROM legal_documents WHERE document_id = %s",
                    (document_id,)
                )
                
                self.pg_connection.commit()
                logger.info(f"✅ Documento {document_id} eliminado")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error eliminando documento: {e}")
            self.pg_connection.rollback()
            return False
    
    def close_connections(self):
        """Cerrar conexiones a la base de datos"""
        if self.pg_connection:
            self.pg_connection.close()
            logger.info("✅ Conexiones cerradas")
