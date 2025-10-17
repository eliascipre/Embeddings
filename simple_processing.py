#!/usr/bin/env python3
"""
Sistema de procesamiento simplificado sin asyncio
Basado en el proyecto original que funcionaba
"""
import os
import sys
import logging
import hashlib
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from supabase import create_client, Client

# Agregar el directorio src al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.comercio_exterior_processor import ComercioExteriorProcessor
from src.smart_legal_chunker import SmartLegalChunker, LegalChunk

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simple_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimpleEmbeddingClient:
    """Cliente simple de embeddings sin asyncio"""
    
    def __init__(self, endpoint: str = "https://wfuosp4mnqkimde9.us-east-1.aws.endpoints.huggingface.cloud"):
        self.endpoint = endpoint
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generar embedding de forma síncrona"""
        try:
            # Limpiar texto
            text = self._clean_text(text)
            if not text or len(text.strip()) < 10:
                return None
            
            # Truncar texto si es muy largo (para evitar embeddings muy grandes)
            if len(text) > 2000:
                text = text[:2000]
            
            payload = {
                "inputs": text,
                "parameters": {}  # Usar dimensiones completas (1024)
            }
            
            response = self.session.post(self.endpoint, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0]
            elif isinstance(result, dict) and 'embeddings' in result:
                return result['embeddings'][0]
            else:
                logger.warning(f"Formato de respuesta inesperado: {type(result)}")
                return None
                
        except Exception as e:
            logger.error(f"Error generando embedding: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Limpiar texto para embeddings"""
        if not text:
            return ""
        
        # Limpiar caracteres Unicode problemáticos de forma más agresiva
        text = text.replace('\u0000', '')
        text = text.replace('\x00', '')
        text = text.replace('\\u0000', '')
        text = text.replace('\\x00', '')
        
        # Limpiar caracteres de control y secuencias de escape
        import re
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        text = re.sub(r'\\[0-9A-Fa-f]{4}', '', text)
        text = re.sub(r'\\u[0-9A-Fa-f]{4}', '', text)
        text = re.sub(r'\\x[0-9A-Fa-f]{2}', '', text)
        
        # Limpiar caracteres nulos restantes
        text = ''.join(char for char in text if ord(char) != 0)
        
        # Normalizar espacios
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text

class SimpleRAGSystem:
    """Sistema RAG simplificado sin asyncio"""
    
    def __init__(self, supabase_url: str, supabase_key: str):
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.supabase: Optional[Client] = None
        self.embedding_client = SimpleEmbeddingClient()
        self.document_processor = ComercioExteriorProcessor()
        
        # Estadísticas
        self.stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'embeddings_generated': 0,
            'errors': 0
        }
    
    def initialize(self):
        """Inicializar el sistema"""
        try:
            logger.info("Inicializando sistema RAG simplificado...")
            
            # Conectar a Supabase
            self.supabase = create_client(self.supabase_url, self.supabase_key)
            logger.info("Conexión con Supabase establecida")
            
            # Verificar tablas
            self._verify_database_schema()
            logger.info("Sistema RAG inicializado correctamente")
            
        except Exception as e:
            logger.error(f"Error inicializando sistema RAG: {e}")
            raise
    
    def _verify_database_schema(self):
        """Verificar esquema de base de datos"""
        try:
            # Verificar tabla de documentos
            result = self.supabase.table('siem_documents').select('id').limit(1).execute()
            logger.info("Tabla siem_documents verificada correctamente")
            
        except Exception as e:
            logger.error(f"Error verificando esquema de base de datos: {e}")
            raise
    
    def process_document(self, file_path: Path) -> Dict[str, Any]:
        """Procesar un documento individual"""
        try:
            logger.info(f"Procesando: {file_path.name}")
            
            # Verificar si ya existe
            file_hash = hashlib.md5(str(file_path).encode()).hexdigest()
            existing = self.supabase.table('siem_documents').select('id').eq('file_hash', file_hash).execute()
            
            if existing.data:
                logger.info(f"Documento ya procesado: {file_path.name}")
                return {
                    'file_path': str(file_path),
                    'success': True,
                    'chunks_created': 0,
                    'embeddings_generated': 0,
                    'processing_time': 0.0,
                    'message': 'Ya procesado'
                }
            
            # Extraer texto
            text = self.document_processor.extract_text(file_path)
            if not text:
                raise ValueError("No se pudo extraer texto del documento")
            
            # Crear chunks
            chunks = self.document_processor.create_comercio_chunks(text, file_path)
            if not chunks:
                raise ValueError("No se pudieron crear chunks del documento")
            
            # Guardar documento
            document_data = {
                'file_name': file_path.name,
                'file_path': str(file_path),
                'file_extension': file_path.suffix.lower(),
                'file_hash': file_hash,
                'file_size': file_path.stat().st_size,
                'processing_status': 'processing',
                'created_at': datetime.now().isoformat()
            }
            
            doc_result = self.supabase.table('siem_documents').insert(document_data).execute()
            document_id = doc_result.data[0]['id']
            
            # Guardar chunks y generar embeddings
            chunks_created = 0
            embeddings_generated = 0
            
            for chunk in chunks:
                # Limpiar texto del chunk
                clean_text = self.embedding_client._clean_text(chunk['text'])
                
                # Guardar chunk
                chunk_data = {
                    'document_id': document_id,
                    'chunk_text': clean_text,
                    'chunk_text_clean': clean_text,
                    'chunk_type': chunk['type'],
                    'chunk_index': chunk['index'],
                    'word_count': chunk.get('word_count', len(clean_text.split())),
                    'chunk_hash': hashlib.md5(clean_text.encode()).hexdigest(),
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
                    'char_count': chunk.get('char_count', len(clean_text)),
                    'is_compressed': chunk.get('is_compressed', False),
                    'compression_ratio': chunk.get('compression_ratio', 1.0),
                    'metadata': json.dumps(chunk.get('metadata', {}))
                }
                
                chunk_result = self.supabase.table('siem_chunks').insert(chunk_data).execute()
                chunk_id = chunk_result.data[0]['id']
                chunks_created += 1
                
                # Generar embedding
                try:
                    embedding = self.embedding_client.generate_embedding(clean_text)
                    if embedding:
                        embedding_data = {
                            'chunk_id': chunk_id,
                            'embedding_vector': embedding,
                            'model_name': 'Qwen3-Embedding-0.6B',
                            'dimensions': len(embedding),
                            'generated_at': datetime.now().isoformat()
                        }
                        
                        self.supabase.table('siem_embeddings').insert(embedding_data).execute()
                        embeddings_generated += 1
                        logger.info(f"✅ Embedding generado para chunk {chunk_id} ({len(embedding)} dimensiones)")
                        
                except Exception as e:
                    error_msg = str(e)
                    if "index row size" in error_msg:
                        logger.warning(f"⚠️ Embedding muy grande para chunk {chunk_id}, guardando sin embedding")
                        # Guardar chunk sin embedding
                        pass
                    else:
                        logger.warning(f"Error generando embedding para chunk {chunk_id}: {e}")
            
            # Actualizar estado del documento
            self.supabase.table('siem_documents').update({
                'processing_status': 'completed',
                'updated_at': datetime.now().isoformat()
            }).eq('id', document_id).execute()
            
            # Actualizar estadísticas
            self.stats['documents_processed'] += 1
            self.stats['chunks_created'] += chunks_created
            self.stats['embeddings_generated'] += embeddings_generated
            
            logger.info(f"✓ {file_path.name}: {chunks_created} chunks, {embeddings_generated} embeddings")
            
            return {
                'file_path': str(file_path),
                'success': True,
                'chunks_created': chunks_created,
                'embeddings_generated': embeddings_generated,
                'processing_time': 0.0
            }
            
        except Exception as e:
            logger.error(f"✗ {file_path.name}: {e}")
            self.stats['errors'] += 1
            return {
                'file_path': str(file_path),
                'success': False,
                'chunks_created': 0,
                'embeddings_generated': 0,
                'processing_time': 0.0,
                'error_message': str(e)
            }
    
    def process_documents_parallel(self, source_directory: str, max_workers: int = 4, batch_size: int = 10):
        """Procesar documentos en paralelo"""
        try:
            source_path = Path(source_directory)
            if not source_path.exists():
                raise FileNotFoundError(f"Directorio no encontrado: {source_directory}")
            
            # Encontrar archivos
            files = []
            for ext in ['*.pdf', '*.docx', '*.txt']:
                files.extend(source_path.glob(ext))
            
            logger.info(f"Total de archivos encontrados: {len(files)}")
            
            # Procesar en lotes
            results = []
            for i in range(0, len(files), batch_size):
                batch_files = files[i:i + batch_size]
                logger.info(f"Procesando lote {i//batch_size + 1}/{(len(files) + batch_size - 1)//batch_size} ({len(batch_files)} archivos)")
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_file = {
                        executor.submit(self.process_document, file_path): file_path 
                        for file_path in batch_files
                    }
                    
                    for future in as_completed(future_to_file):
                        file_path = future_to_file[future]
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            logger.error(f"Error procesando {file_path}: {e}")
                            results.append({
                                'file_path': str(file_path),
                                'success': False,
                                'error_message': str(e)
                            })
            
            # Mostrar estadísticas finales
            successful = sum(1 for r in results if r['success'])
            total_chunks = sum(r.get('chunks_created', 0) for r in results)
            total_embeddings = sum(r.get('embeddings_generated', 0) for r in results)
            
            logger.info("🎉 PROCESAMIENTO COMPLETADO")
            logger.info(f"📄 Archivos procesados: {successful}/{len(files)}")
            logger.info(f"📝 Chunks creados: {total_chunks}")
            logger.info(f"🧮 Embeddings generados: {total_embeddings}")
            logger.info(f"📈 Tasa de éxito: {successful/len(files):.2%}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error en procesamiento paralelo: {e}")
            raise

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Sistema RAG simplificado")
    parser.add_argument('source_directory', help='Directorio fuente')
    parser.add_argument('--supabase-url', required=True, help='URL de Supabase')
    parser.add_argument('--supabase-key', required=True, help='Clave de Supabase')
    parser.add_argument('--max-workers', type=int, default=4, help='Workers paralelos')
    parser.add_argument('--batch-size', type=int, default=10, help='Tamaño de lote')
    
    args = parser.parse_args()
    
    try:
        # Crear sistema
        rag_system = SimpleRAGSystem(args.supabase_url, args.supabase_key)
        rag_system.initialize()
        
        # Procesar documentos
        results = rag_system.process_documents_parallel(
            source_directory=args.source_directory,
            max_workers=args.max_workers,
            batch_size=args.batch_size
        )
        
        # Guardar resultados
        with open('simple_processing_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("Resultados guardados en simple_processing_results.json")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
