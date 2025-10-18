#!/usr/bin/env python3
"""
Sistema RAG optimizado con guardado inmediato de chunks
Basado en el código de referencia proporcionado
"""
import asyncio
import aiohttp
import logging
import sys
import json
import hashlib
import time
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import numpy as np

# Agregar src al path
sys.path.append(str(Path(__file__).parent / "src"))

from supabase import create_client, Client
from simple_document_processor import SimpleDocumentProcessor

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('immediate_save_processing.log')
    ]
)

logger = logging.getLogger(__name__)

class OptimizedGPUEmbeddingClient:
    """Cliente optimizado para embeddings con Hugging Face"""
    
    def __init__(self, hf_endpoint: str, hf_token: Optional[str] = None):
        self.hf_endpoint = hf_endpoint
        self.hf_token = hf_token
        self.batch_size = 32  # Límite máximo de la API de Hugging Face
        self.max_concurrent_requests = 25  # Reducido para evitar timeouts
        self.session = None
        self.semaphore = asyncio.Semaphore(25)  # Control de concurrencia optimizado
        self.stats = {
            'total_requests': 0,
            'total_chunks_processed': 0,
            'total_time': 0.0,
            'errors': 0
        }
    
    async def __aenter__(self):
        """Context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120, connect=30),  # Timeout aumentado para documentos grandes
            connector=aiohttp.TCPConnector(limit=50, limit_per_host=10),  # Menos conexiones para estabilidad
            headers={
                'Authorization': f'Bearer {self.hf_token}' if self.hf_token else '',
                'Content-Type': 'application/json'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.session:
            await self.session.close()
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generar embeddings para un lote de textos con control de concurrencia y reintentos"""
        async with self.semaphore:  # Control de concurrencia
            max_retries = 3
            retry_delay = 2  # segundos
            
            for attempt in range(max_retries):
                try:
                    start_time = time.time()
                    
                    payload = {
                        "inputs": texts,
                        "parameters": {}
                    }
                    
                    async with self.session.post(self.hf_endpoint, json=payload) as response:
                        if response.status == 200:
                            result = await response.json()
                            embeddings = result if isinstance(result, list) else result.get('embeddings', [])
                            
                            self.stats['total_requests'] += 1
                            self.stats['total_chunks_processed'] += len(texts)
                            self.stats['total_time'] += time.time() - start_time
                            
                            logger.info(f"✅ Lote procesado: {len(texts)} chunks en {time.time() - start_time:.2f}s")
                            return embeddings
                        else:
                            error_text = await response.text()
                            logger.warning(f"⚠️ Error en API (intento {attempt + 1}/{max_retries}): {response.status} - {error_text}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(retry_delay * (attempt + 1))  # Backoff exponencial
                                continue
                            else:
                                self.stats['errors'] += 1
                                return []
                            
                except asyncio.TimeoutError:
                    logger.warning(f"⚠️ Timeout en intento {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        logger.error(f"❌ Timeout después de {max_retries} intentos")
                        self.stats['errors'] += 1
                        return []
                        
                except Exception as e:
                    logger.warning(f"⚠️ Error en intento {attempt + 1}/{max_retries}: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        logger.error(f"❌ Error después de {max_retries} intentos: {e}")
                        self.stats['errors'] += 1
                        return []
            
            return []

class OptimizedImmediateSaveRAGSystem:
    """Sistema RAG con guardado inmediato de chunks"""
    
    def __init__(self, supabase_url: str, supabase_key: str, hf_endpoint: str, hf_token: Optional[str] = None):
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.hf_endpoint = hf_endpoint
        self.hf_token = hf_token
        self.supabase: Client = None
        self.document_processor = SimpleDocumentProcessor(max_chunk_size=1500, chunk_overlap=150)  # Chunks más pequeños para mejor rendimiento
        self.embedding_client = OptimizedGPUEmbeddingClient(hf_endpoint, hf_token)
        
        # Inicializar automáticamente
        self.initialize()
    
    def initialize(self):
        """Inicializar el sistema"""
        try:
            logger.info("🚀 Inicializando sistema RAG con guardado inmediato...")
            
            # Conectar a Supabase
            self.supabase = create_client(self.supabase_url, self.supabase_key)
            logger.info("✅ Conexión con Supabase establecida")
            
            logger.info("🎉 Sistema inicializado correctamente")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando sistema: {e}")
            raise
    
    async def process_documents_immediate_save(self, source_directory: str, max_workers: int = 20, max_files: int = None) -> Dict[str, Any]:
        """Procesar documentos con guardado inmediato de chunks"""
        try:
            source_path = Path(source_directory)
            if not source_path.exists():
                raise FileNotFoundError(f"Directorio no encontrado: {source_directory}")
            
            # Encontrar todos los archivos
            files = []
            for ext in ['.pdf', '.txt', '.md']:
                files.extend(list(source_path.glob(f"**/*{ext}")))
            
            # Limitar número de archivos si se especifica
            if max_files and max_files > 0:
                files = files[:max_files]
                logger.info(f"📁 Limitando a {len(files)} archivos para procesamiento")
            
            logger.info(f"📁 Encontrados {len(files)} archivos para procesar")
            
            # Inicializar resultados
            results = {
                'total_files': len(files),
                'processed_files': 0,
                'total_chunks': 0,
                'total_embeddings': 0,
                'errors': [],
                'start_time': datetime.now().isoformat()
            }
            
            # Procesar documentos en paralelo con guardado inmediato
            logger.info("🔄 Procesando documentos en paralelo con guardado inmediato...")
            
            async with self.embedding_client as client:
                # Procesar en lotes de documentos para paralelismo (aumentado para documentos grandes)
                batch_size = 12  # Procesar 12 documentos en paralelo (optimizado para documentos grandes)
                for batch_start in range(0, len(files), batch_size):
                    batch_end = min(batch_start + batch_size, len(files))
                    batch_files = files[batch_start:batch_end]
                    
                    # Crear tareas para procesar documentos en paralelo
                    tasks = []
                    for file_path in batch_files:
                        task = self._process_single_document(file_path, client, results)
                        tasks.append(task)
                    
                    # Ejecutar tareas en paralelo con timeout para evitar bloqueos
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*tasks, return_exceptions=True),
                            timeout=300  # 5 minutos de timeout por lote
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"⚠️ Timeout en lote {batch_start}-{batch_end}, continuando...")
                    
                    logger.info(f"📊 Progreso: {min(batch_end, len(files))}/{len(files)} archivos procesados")
            
            # Finalizar resultados
            results['end_time'] = datetime.now().isoformat()
            results['success_rate'] = results['processed_files'] / results['total_files'] if results['total_files'] > 0 else 0
            
            logger.info("🎉 PROCESAMIENTO CON GUARDADO INMEDIATO COMPLETADO")
            logger.info(f"📄 Archivos procesados: {results['processed_files']}/{results['total_files']}")
            logger.info(f"📝 Chunks creados: {results['total_chunks']}")
            logger.info(f"🧮 Embeddings generados: {results['total_embeddings']}")
            logger.info(f"📈 Tasa de éxito: {results['success_rate']:.2%}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error en procesamiento: {e}")
            raise
    
    async def _process_single_document(self, file_path: Path, client, results: Dict[str, Any]):
        """Procesar un solo documento de forma asíncrona"""
        try:
            logger.info(f"📄 Procesando: {file_path.name}")
            
            # Calcular hash del archivo (necesario para todos los casos)
            file_hash = self._calculate_file_hash(file_path)
            
            # Verificar si ya existe (optimizado: solo verificar por nombre de archivo primero)
            existing_doc = self.supabase.table('siem_documents').select('id,file_hash').eq('file_name', file_path.name).execute()
            
            if existing_doc.data:
                # Verificar si el hash coincide
                for doc in existing_doc.data:
                    if doc['file_hash'] == file_hash:
                        logger.info(f"⏭️ Saltando {file_path.name} (ya procesado)")
                        return
            
            # Procesar documento
            chunks = self.document_processor.process_document(file_path)
            if not chunks:
                logger.warning(f"⚠️ No se crearon chunks para {file_path.name}")
                return
            
            # Agregar metadata de documento a cada chunk
            for i, chunk in enumerate(chunks):
                chunk['file_name'] = file_path.name
                chunk['file_path'] = str(file_path)
                chunk['file_hash'] = file_hash
                chunk['file_extension'] = file_path.suffix.lower()
                chunk['file_size'] = file_path.stat().st_size if file_path.exists() else 0
                # Generar hash único para el chunk
                chunk['chunk_hash'] = hashlib.md5(f"{file_hash}_{i}_{chunk['text'][:100]}".encode()).hexdigest()
            
            # Guardar documento primero
            document_id = self._save_document(file_path, file_hash, len(chunks))
            logger.info(f"✅ Documento guardado: {file_path.name} (ID: {document_id})")
            
            # Procesar chunks con estrategia adaptativa según el tamaño del documento
            chunks_saved = 0
            embeddings_saved = 0
            
            # Estrategia adaptativa: documentos grandes se procesan secuencialmente para evitar timeouts
            if len(chunks) > 500:  # Documentos muy grandes (>500 chunks)
                logger.info(f"📊 Documento grande detectado ({len(chunks)} chunks), procesando secuencialmente...")
                chunks_saved, embeddings_saved = await self._process_large_document_sequentially(
                    chunks, document_id, client
                )
            else:
                # Documentos normales: procesamiento paralelo
                logger.info(f"📊 Documento normal ({len(chunks)} chunks), procesando en paralelo...")
                chunks_saved, embeddings_saved = await self._process_document_parallel(
                    chunks, document_id, client
                )
            
            # Actualizar estado del documento
            self._update_document_status(document_id, 'completed')
            
            # Actualizar resultados de forma thread-safe
            results['processed_files'] += 1
            results['total_chunks'] += chunks_saved
            results['total_embeddings'] += embeddings_saved
            
            logger.info(f"✅ {file_path.name}: {chunks_saved} chunks y {embeddings_saved} embeddings guardados")
            
        except Exception as e:
            error_msg = f"Error procesando {file_path.name}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            results['errors'].append(error_msg)
    
    async def _process_chunk_batch(self, batch_chunks: List[Dict], document_id: int, client) -> tuple:
        """Procesar un lote de chunks de forma asíncrona"""
        try:
            # Generar embeddings para el lote
            texts = [chunk['text'] for chunk in batch_chunks]
            embeddings = await client.generate_embeddings_batch(texts)
            
            if not embeddings:
                return 0, 0
            
            # Preparar datos para guardado
            chunks_to_save = []
            embeddings_to_save = []
            
            for chunk, embedding in zip(batch_chunks, embeddings):
                chunk['document_id'] = document_id
                chunks_to_save.append(chunk)
                embeddings_to_save.append({
                    'embedding': embedding,
                    'chunk_text': chunk['text']
                })
            
            # Guardar en lote
            chunks_saved, embeddings_saved = await self._save_chunks_and_embeddings_batch(
                chunks_to_save, embeddings_to_save, document_id
            )
            
            return chunks_saved, embeddings_saved
            
        except Exception as e:
            logger.error(f"❌ Error procesando lote de chunks: {e}")
            return 0, 0
    
    async def _process_large_document_sequentially(self, chunks: List[Dict], document_id: int, client) -> tuple:
        """Procesar documento grande de forma secuencial para evitar timeouts"""
        chunks_saved = 0
        embeddings_saved = 0
        batch_size = 16  # Lotes más pequeños para documentos grandes
        
        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            
            logger.info(f"📝 Procesando lote {batch_start//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size} ({len(batch_chunks)} chunks)")
            
            # Procesar lote individual
            batch_saved, batch_embeddings = await self._process_chunk_batch(batch_chunks, document_id, client)
            chunks_saved += batch_saved
            embeddings_saved += batch_embeddings
            
            # Pausa entre lotes para evitar sobrecarga
            if batch_start + batch_size < len(chunks):
                await asyncio.sleep(1)  # 1 segundo de pausa entre lotes
        
        return chunks_saved, embeddings_saved
    
    async def _process_document_parallel(self, chunks: List[Dict], document_id: int, client) -> tuple:
        """Procesar documento normal en paralelo"""
        chunks_saved = 0
        embeddings_saved = 0
        batch_size = 32
        tasks = []
        
        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            
            # Crear tarea asíncrona para procesar el lote
            task = self._process_chunk_batch(batch_chunks, document_id, client)
            tasks.append(task)
        
        # Ejecutar todas las tareas en paralelo
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Consolidar resultados
        for result in batch_results:
            if isinstance(result, tuple) and len(result) == 2:
                chunks_saved += result[0]
                embeddings_saved += result[1]
            elif isinstance(result, Exception):
                logger.error(f"❌ Error en lote: {result}")
        
        return chunks_saved, embeddings_saved
    
    async def _save_chunks_and_embeddings_batch(self, chunks: List[Dict], embeddings_data: List[Dict], document_id: int) -> tuple:
        """Guardar chunks y embeddings en lotes para mejor rendimiento"""
        try:
            # Preparar datos de chunks para inserción masiva
            chunks_data = []
            for chunk in chunks:
                chunk_data = {
                    'document_id': document_id,
                    'chunk_text': chunk['text'],
                    'chunk_text_clean': chunk.get('clean_text', chunk['text']),
                    'chunk_type': chunk.get('chunk_type', 'paragraph'),
                    'chunk_index': chunk.get('chunk_index', 0),
                    'word_count': chunk.get('word_count', 0),
                    'char_count': chunk.get('char_count', 0),
                    'chunk_hash': chunk.get('chunk_hash', ''),
                    'created_at': datetime.now().isoformat()
                }
                chunks_data.append(chunk_data)
            
            # Insertar chunks en lote
            chunks_result = self.supabase.table('siem_chunks').insert(chunks_data).execute()
            chunk_ids = [chunk['id'] for chunk in chunks_result.data]
            
            # Preparar datos de embeddings para inserción masiva
            embeddings_data_list = []
            for i, (chunk_id, embedding_info) in enumerate(zip(chunk_ids, embeddings_data)):
                embedding_data = {
                    'chunk_id': chunk_id,
                    'embedding_vector': embedding_info['embedding'],
                    'dimensions': len(embedding_info['embedding']),
                    'model_name': 'qwen3-gpu-optimized',
                    'created_at': datetime.now().isoformat()
                }
                embeddings_data_list.append(embedding_data)
            
            # Insertar embeddings en lote
            self.supabase.table('siem_embeddings').insert(embeddings_data_list).execute()
            
            return len(chunk_ids), len(embeddings_data_list)
            
        except Exception as e:
            logger.error(f"❌ Error guardando lote: {e}")
            return 0, 0
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calcular hash del archivo"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _save_document(self, file_path: Path, file_hash: str, chunk_count: int) -> int:
        """Guardar documento en la base de datos"""
        document_data = {
            'file_name': file_path.name,
            'file_path': str(file_path),
            'file_hash': file_hash,
            'file_size': file_path.stat().st_size if file_path.exists() else 0,
            'file_extension': file_path.suffix.lower(),
            'processing_status': 'processing',
            'created_at': datetime.now().isoformat()
        }
        
        result = self.supabase.table('siem_documents').insert(document_data).execute()
        return result.data[0]['id']
    
    def _save_chunk(self, chunk: Dict[str, Any], document_id: int) -> int:
        """Guardar chunk en la base de datos"""
        chunk_data = {
            'document_id': document_id,
            'chunk_text': chunk['text'],
            'chunk_text_clean': chunk.get('clean_text', chunk['text']),
            'chunk_type': chunk.get('chunk_type', 'paragraph'),
            'chunk_index': chunk.get('chunk_index', 0),
            'word_count': chunk.get('word_count', 0),
            'char_count': chunk.get('char_count', 0),
            'chunk_hash': chunk.get('chunk_hash', ''),
            'created_at': datetime.now().isoformat()
        }
        
        result = self.supabase.table('siem_chunks').insert(chunk_data).execute()
        return result.data[0]['id']
    
    def _save_embedding(self, embedding: List[float], chunk_id: int, document_id: int):
        """Guardar embedding en la base de datos"""
        embedding_data = {
            'chunk_id': chunk_id,
            'embedding_vector': embedding,
            'dimensions': len(embedding),
            'model_name': 'qwen3-gpu-optimized',
            'created_at': datetime.now().isoformat()
        }
        
        self.supabase.table('siem_embeddings').insert(embedding_data).execute()
    
    def _update_document_status(self, document_id: int, status: str):
        """Actualizar estado del documento"""
        self.supabase.table('siem_documents').update({
            'processing_status': status,
            'updated_at': datetime.now().isoformat()
        }).eq('id', document_id).execute()

async def main():
    """Función principal"""
    try:
        # Inicializar sistema
        logger.info("🚀 Inicializando sistema RAG con guardado inmediato...")
        system = OptimizedImmediateSaveRAGSystem(
            supabase_url="https://rygrdlradxyykzuudgtu.supabase.co",
            supabase_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJ5Z3JkbHJhZHh5eWt6dXVkZ3R1Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MDcxODQwNiwiZXhwIjoyMDc2Mjk0NDA2fQ.jbnFbBjlq_NsLaNamGJ98a4uXay4lAZX_BVWQCobpCw",
            hf_endpoint="https://wfuosp4mnqkimde9.us-east-1.aws.endpoints.huggingface.cloud"
        )
        
        # Procesar todos los archivos con guardado inmediato optimizado
        results = await system.process_documents_immediate_save("SIEM", max_workers=32)
        
        logger.info(f"✅ Procesamiento completado: {results}")
        
    except Exception as e:
        logger.error(f"❌ Error en procesamiento: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(main())
