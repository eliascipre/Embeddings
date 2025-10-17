#!/usr/bin/env python3
"""
Sistema RAG ultra optimizado para GPU A100
Procesamiento masivo con lotes para máxima utilización de GPU
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

from simple_document_processor import SimpleDocumentProcessor
from supabase import create_client, Client

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OptimizedGPUEmbeddingClient:
    """Cliente optimizado para GPU A100 con procesamiento por lotes"""
    
    def __init__(self, endpoint: str, token: Optional[str] = None):
        self.endpoint = endpoint
        self.token = token
        self.session = None
        self.batch_size = 32  # Tamaño óptimo para A100
        self.max_concurrent_requests = 10  # Máximo de requests concurrentes
        self.semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        # Estadísticas
        self.stats = {
            'total_requests': 0,
            'total_chunks_processed': 0,
            'total_embeddings_generated': 0,
            'total_time': 0.0,
            'errors': 0
        }
    
    async def __aenter__(self):
        """Context manager para inicializar la sesión"""
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        
        self.session = aiohttp.ClientSession(
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=300)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager para cerrar la sesión"""
        if self.session:
            await self.session.close()
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generar embeddings para un lote de textos optimizado para GPU"""
        if not texts:
            return []
        
        async with self.semaphore:
            try:
                start_time = time.time()
                
                # Preparar payload optimizado para GPU
                payload = {
                    "inputs": texts,
                    "parameters": {
                        "dimensions": 1024,  # Máxima calidad
                        "normalize": True,
                        "instruction": "Represent the following text for retrieval:"
                    }
                }
                
                # Realizar request
                async with self.session.post(self.endpoint, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Extraer embeddings
                        if isinstance(result, list):
                            embeddings = result
                        elif isinstance(result, dict) and 'embeddings' in result:
                            embeddings = result['embeddings']
                        else:
                            raise ValueError(f"Formato de respuesta inesperado: {result}")
                        
                        # Actualizar estadísticas
                        processing_time = time.time() - start_time
                        self.stats['total_requests'] += 1
                        self.stats['total_chunks_processed'] += len(texts)
                        self.stats['total_embeddings_generated'] += len(embeddings)
                        self.stats['total_time'] += processing_time
                        
                        logger.info(f"✅ Lote procesado: {len(texts)} chunks en {processing_time:.2f}s")
                        return embeddings
                    
                    else:
                        error_text = await response.text()
                        raise Exception(f"HTTP {response.status}: {error_text}")
            
            except Exception as e:
                self.stats['errors'] += 1
                logger.error(f"❌ Error en lote: {e}")
                raise
    
    async def process_chunks_in_batches(self, chunks: List[Dict[str, Any]]) -> List[List[float]]:
        """Procesar chunks en lotes optimizados para GPU"""
        all_embeddings = []
        
        # Dividir chunks en lotes del tamaño óptimo
        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i:i + self.batch_size]
            batch_texts = [chunk['text'] for chunk in batch_chunks]
            
            try:
                batch_embeddings = await self.generate_embeddings_batch(batch_texts)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"❌ Error procesando lote {i//self.batch_size + 1}: {e}")
                # Continuar con el siguiente lote
                continue
        
        return all_embeddings
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cliente"""
        stats = self.stats.copy()
        if stats['total_requests'] > 0:
            stats['avg_time_per_request'] = stats['total_time'] / stats['total_requests']
            stats['chunks_per_second'] = stats['total_chunks_processed'] / stats['total_time'] if stats['total_time'] > 0 else 0
            stats['success_rate'] = (stats['total_requests'] - stats['errors']) / stats['total_requests']
        return stats

class OptimizedGPURAGSystem:
    """Sistema RAG ultra optimizado para GPU A100"""
    
    def __init__(self, supabase_url: str, supabase_key: str, hf_endpoint: str, hf_token: Optional[str] = None):
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.supabase: Client = None
        self.document_processor = SimpleDocumentProcessor(max_chunk_size=2000, chunk_overlap=200)  # Chunks más grandes
        self.embedding_client = OptimizedGPUEmbeddingClient(hf_endpoint, hf_token)
        
        # Inicializar automáticamente
        self.initialize()
        
    def initialize(self):
        """Inicializar el sistema"""
        try:
            logger.info("🚀 Inicializando sistema RAG optimizado para GPU A100...")
            
            # Conectar a Supabase
            self.supabase = create_client(self.supabase_url, self.supabase_key)
            logger.info("✅ Conexión con Supabase establecida")
            
            logger.info("🎉 Sistema inicializado correctamente")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando sistema: {e}")
            raise
    
    async def process_documents_optimized(self, source_directory: str, max_workers: int = 8, max_files: int = None) -> Dict[str, Any]:
        """Procesar todos los documentos con acumulación de chunks para máxima optimización GPU"""
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
                logger.info(f"📁 Limitando a {len(files)} archivos para debug")
            
            logger.info(f"📁 Encontrados {len(files)} archivos para procesar")
            logger.info(f"🚀 Configuración optimizada para GPU A100:")
            logger.info(f"   - Workers paralelos: {max_workers}")
            logger.info(f"   - Tamaño de lote: {self.embedding_client.batch_size}")
            logger.info(f"   - Requests concurrentes: {self.embedding_client.max_concurrent_requests}")
            logger.info(f"   - Chunk size: 2000 caracteres")
            logger.info(f"   - Estrategia: Acumulación de chunks para lotes grandes")
            
            # Procesar archivos
            results = {
                'total_files': len(files),
                'processed_files': 0,
                'total_chunks': 0,
                'total_embeddings': 0,
                'errors': [],
                'start_time': datetime.now().isoformat()
            }
            
            # FASE 1: Procesar todos los documentos y acumular chunks (SIN GUARDAR EN BD)
            logger.info("🔄 FASE 1: Procesando documentos y acumulando chunks...")
            all_chunks = []
            document_metadata = {}  # Mapear metadatos de documentos
            
            for i, file_path in enumerate(files, 1):
                try:
                    logger.info(f"📄 Procesando ({i}/{len(files)}): {file_path.name}")
                    
                    # Verificar si ya existe
                    file_hash = self._calculate_file_hash(file_path)
                    existing_doc = self.supabase.table('siem_documents').select('id').eq('file_hash', file_hash).execute()
                    
                    if existing_doc.data:
                        logger.info(f"⏭️ Saltando {file_path.name} (ya procesado)")
                        continue
                    
                    # Procesar documento
                    logger.info(f"🔍 Procesando documento: {file_path.name}")
                    chunks = self.document_processor.process_document(file_path)
                    if not chunks:
                        logger.warning(f"⚠️ No se crearon chunks para {file_path.name}")
                        continue
                    
                    # Agregar metadata de documento a cada chunk (SIN GUARDAR EN BD AÚN)
                    for chunk in chunks:
                        chunk['file_name'] = file_path.name
                        chunk['file_path'] = str(file_path)
                        chunk['file_hash'] = file_hash
                        chunk['file_extension'] = file_path.suffix.lower()
                        chunk['file_size'] = file_path.stat().st_size if file_path.exists() else 0
                    
                    # Acumular chunks
                    all_chunks.extend(chunks)
                    document_metadata[file_path.name] = {
                        'file_path': file_path,
                        'file_hash': file_hash,
                        'chunks': chunks,
                        'chunk_count': len(chunks)
                    }
                    
                    results['processed_files'] += 1
                    results['total_chunks'] += len(chunks)
                    
                    logger.info(f"✅ {file_path.name}: {len(chunks)} chunks acumulados (Total acumulado: {len(all_chunks)})")
                    
                except Exception as e:
                    error_msg = f"Error procesando {file_path.name}: {str(e)}"
                    logger.error(f"❌ {error_msg}")
                    logger.error(f"❌ Traceback: {traceback.format_exc()}")
                    results['errors'].append(error_msg)
                    continue
            
            logger.info(f"📊 FASE 1 COMPLETADA: {len(all_chunks)} chunks acumulados de {results['processed_files']} archivos")
            
            # FASE 2: Procesar chunks en lotes optimizados para GPU
            if all_chunks:
                logger.info(f"🚀 Iniciando FASE 2 con {len(all_chunks)} chunks...")
                logger.info("🚀 FASE 2: Generando embeddings en lotes optimizados...")
                
                async with self.embedding_client as client:
                    # Procesar todos los chunks en lotes grandes
                    embeddings = await client.process_chunks_in_batches(all_chunks)
                    
                    # FASE 3: Guardar documentos, chunks y embeddings en la base de datos
                    logger.info("💾 FASE 3: Guardando documentos, chunks y embeddings...")
                    
                    chunks_saved = 0
                    embeddings_saved = 0
                    documents_saved = 0
                    
                    # Primero guardar todos los documentos
                    document_id_map = {}  # Mapear file_name a document_id
                    for file_name, doc_info in document_metadata.items():
                        try:
                            document_id = self._save_document(
                                doc_info['file_path'], 
                                doc_info['file_hash'], 
                                doc_info['chunk_count']
                            )
                            document_id_map[file_name] = document_id
                            documents_saved += 1
                            logger.info(f"✅ Documento guardado: {file_name} (ID: {document_id})")
                        except Exception as e:
                            logger.error(f"❌ Error guardando documento {file_name}: {e}")
                            continue
                    
                    # Luego guardar chunks y embeddings
                    for i, chunk in enumerate(all_chunks):
                        try:
                            # Obtener document_id del chunk
                            file_name = chunk['file_name']
                            if file_name not in document_id_map:
                                logger.error(f"❌ Document ID no encontrado para {file_name}")
                                continue
                            
                            document_id = document_id_map[file_name]
                            
                            # Guardar chunk
                            chunk_id = self._save_chunk(chunk, document_id)
                            chunks_saved += 1
                            
                            # Guardar embedding si existe
                            if i < len(embeddings):
                                self._save_embedding(embeddings[i], chunk_id, document_id)
                                embeddings_saved += 1
                            
                        except Exception as e:
                            logger.error(f"❌ Error guardando chunk {i}: {e}")
                            continue
                    
                    # Finalmente actualizar documentos con estadísticas
                    for file_name, doc_info in document_metadata.items():
                        if file_name in document_id_map:
                            try:
                                self.supabase.table('siem_documents').update({
                                    'total_chunks': doc_info['chunk_count'],
                                    'processing_status': 'completed'
                                }).eq('id', document_id_map[file_name]).execute()
                            except Exception as e:
                                logger.error(f"❌ Error actualizando documento {file_name}: {e}")
                    
                    results['total_embeddings'] = embeddings_saved
                    
                    # Mostrar estadísticas del cliente
                    client_stats = client.get_stats()
                    logger.info("📊 ESTADÍSTICAS DEL CLIENTE DE EMBEDDINGS:")
                    logger.info(f"   - Requests realizados: {client_stats['total_requests']}")
                    logger.info(f"   - Chunks procesados: {client_stats['total_chunks_processed']}")
                    logger.info(f"   - Embeddings generados: {client_stats['total_embeddings_generated']}")
                    logger.info(f"   - Tiempo total: {client_stats['total_time']:.2f}s")
                    logger.info(f"   - Chunks/segundo: {client_stats['chunks_per_second']:.2f}")
                    logger.info(f"   - Tasa de éxito: {client_stats['success_rate']:.2%}")
                    
                    logger.info(f"✅ FASE 3 COMPLETADA: {documents_saved} documentos, {chunks_saved} chunks, {embeddings_saved} embeddings guardados")
            
            results['end_time'] = datetime.now().isoformat()
            results['success_rate'] = results['processed_files'] / results['total_files'] if results['total_files'] > 0 else 0
            
            logger.info("🎉 PROCESAMIENTO OPTIMIZADO COMPLETADO")
            logger.info(f"📄 Archivos procesados: {results['processed_files']}/{results['total_files']}")
            logger.info(f"📝 Chunks creados: {results['total_chunks']}")
            logger.info(f"🧮 Embeddings generados: {results['total_embeddings']}")
            logger.info(f"📈 Tasa de éxito: {results['success_rate']:.2%}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error en procesamiento: {e}")
            raise
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calcular hash del archivo"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return hashlib.md5(str(file_path).encode()).hexdigest()
    
    def _save_document(self, file_path: Path, file_hash: str, total_chunks: int) -> int:
        """Guardar documento en la base de datos"""
        document_data = {
            'file_name': file_path.name,
            'file_path': str(file_path),
            'file_hash': file_hash,
            'file_extension': file_path.suffix.lower(),
            'file_size': file_path.stat().st_size if file_path.exists() else 0,
            'total_chunks': total_chunks,
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
            'chunk_text_clean': chunk['clean_text'],
            'chunk_type': chunk['type'],
            'chunk_index': chunk['index'],
            'word_count': chunk['word_count'],
            'chunk_hash': hashlib.md5(chunk['text'].encode('utf-8')).hexdigest(),
            'hierarchy_level': chunk['hierarchy_level'],
            'parent_chunk_id': chunk['parent_chunk_id'],
            'article_number': chunk['article_number'],
            'paragraph_number': chunk['paragraph_number'],
            'inciso_number': chunk['inciso_number'],
            'chapter_title': chunk['chapter_title'],
            'section_title': chunk['section_title'],
            'law_title': chunk['law_title'],
            'page_number': chunk['page_number'],
            'char_count': chunk['char_count'],
            'is_compressed': chunk['is_compressed'],
            'compression_ratio': chunk['compression_ratio'],
            'metadata': json.dumps(chunk['metadata'])
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

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Sistema RAG optimizado para GPU A100")
    parser.add_argument('source_directory', help='Directorio con documentos')
    parser.add_argument('--supabase-url', required=True, help='URL de Supabase')
    parser.add_argument('--supabase-key', required=True, help='Clave de Supabase')
    parser.add_argument('--hf-endpoint', default="https://wfuosp4mnqkimde9.us-east-1.aws.endpoints.huggingface.cloud", help='Endpoint de Hugging Face')
    parser.add_argument('--hf-token', help='Token de Hugging Face (opcional)')
    parser.add_argument('--max-workers', type=int, default=8, help='Workers paralelos')
    
    args = parser.parse_args()
    
    try:
        # Crear sistema optimizado
        rag_system = OptimizedGPURAGSystem(
            args.supabase_url, 
            args.supabase_key, 
            args.hf_endpoint, 
            args.hf_token
        )
        rag_system.initialize()
        
        # Procesar documentos
        results = asyncio.run(rag_system.process_documents_optimized(args.source_directory, args.max_workers))
        
        # Guardar resultados
        with open('optimized_gpu_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info("📊 Resultados guardados en optimized_gpu_results.json")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
