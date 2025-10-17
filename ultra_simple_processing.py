#!/usr/bin/env python3
"""
Procesamiento ultra simplificado para RAG
Basado en el proyecto original exitoso
"""
import logging
import sys
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime

# Agregar src al path
sys.path.append(str(Path(__file__).parent / "src"))

from simple_document_processor import SimpleDocumentProcessor
from src.qwen3_embedding_client import Qwen3EmbeddingClient, Qwen3Config
from supabase import create_client, Client

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UltraSimpleRAGSystem:
    """Sistema RAG ultra simplificado"""
    
    def __init__(self, supabase_url: str, supabase_key: str):
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.supabase: Client = None
        self.document_processor = SimpleDocumentProcessor(max_chunk_size=500, chunk_overlap=100)  # Chunks más pequeños
        self.embedding_client = None
        
    def initialize(self):
        """Inicializar el sistema"""
        try:
            logger.info("🚀 Inicializando sistema RAG ultra simplificado...")
            
            # Conectar a Supabase
            self.supabase = create_client(self.supabase_url, self.supabase_key)
            logger.info("✅ Conexión con Supabase establecida")
            
            # Inicializar cliente de embeddings
            config = Qwen3Config(
                endpoint="https://wfuosp4mnqkimde9.us-east-1.aws.endpoints.huggingface.cloud",
                token=None  # Endpoint público
            )
            self.embedding_client = Qwen3EmbeddingClient(config)
            logger.info("✅ Cliente de embeddings inicializado")
            
            logger.info("🎉 Sistema inicializado correctamente")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando sistema: {e}")
            raise
    
    async def process_documents(self, source_directory: str, max_workers: int = 4) -> Dict[str, Any]:
        """Procesar todos los documentos de manera simple"""
        try:
            source_path = Path(source_directory)
            if not source_path.exists():
                raise FileNotFoundError(f"Directorio no encontrado: {source_directory}")
            
            # Encontrar todos los archivos
            files = []
            for ext in ['.pdf', '.txt', '.md']:
                files.extend(list(source_path.glob(f"**/*{ext}")))
            
            logger.info(f"📁 Encontrados {len(files)} archivos para procesar")
            
            # Procesar archivos
            results = {
                'total_files': len(files),
                'processed_files': 0,
                'total_chunks': 0,
                'total_embeddings': 0,
                'errors': [],
                'start_time': datetime.now().isoformat()
            }
            
            # Usar el cliente de embeddings como context manager
            async with self.embedding_client as client:
                for file_path in files:
                    try:
                        logger.info(f"📄 Procesando: {file_path.name}")
                        
                        # Verificar si ya existe
                        file_hash = self._calculate_file_hash(file_path)
                        existing_doc = self.supabase.table('siem_documents').select('id').eq('file_hash', file_hash).execute()
                        
                        if existing_doc.data:
                            logger.info(f"⏭️ Saltando {file_path.name} (ya procesado)")
                            continue
                        
                        # Procesar documento
                        chunks = self.document_processor.process_document(file_path)
                        if not chunks:
                            logger.warning(f"⚠️ No se crearon chunks para {file_path.name}")
                            continue
                        
                        # Guardar documento
                        document_id = self._save_document(file_path, file_hash, len(chunks))
                        
                        # Guardar chunks y embeddings
                        chunks_saved = 0
                        embeddings_saved = 0
                        
                        # Procesar chunks en lotes para optimizar GPU
                        try:
                            logger.info(f"🔄 Generando embeddings para {len(chunks)} chunks en lotes...")
                            
                            # Dividir chunks en lotes de 32 (óptimo para A100)
                            batch_size = 32
                            all_embeddings = []
                            
                            for i in range(0, len(chunks), batch_size):
                                batch_chunks = chunks[i:i + batch_size]
                                batch_texts = [chunk['text'] for chunk in batch_chunks]
                                
                                try:
                                    # Generar embeddings para el lote
                                    batch_embeddings = await client.generate_embeddings_batch(batch_texts)
                                    all_embeddings.extend(batch_embeddings)
                                    logger.info(f"✅ Lote {i//batch_size + 1}: {len(batch_texts)} chunks procesados")
                                except Exception as e:
                                    logger.warning(f"⚠️ Error en lote {i//batch_size + 1}: {e}")
                                    # Continuar con el siguiente lote
                                    continue
                            
                            # Guardar chunks y embeddings
                            for i, chunk in enumerate(chunks):
                                try:
                                    # Guardar chunk
                                    chunk_id = self._save_chunk(chunk, document_id)
                                    chunks_saved += 1
                                    
                                    # Guardar embedding si existe
                                    if i < len(all_embeddings):
                                        self._save_embedding(all_embeddings[i], chunk_id, document_id)
                                        embeddings_saved += 1
                                    
                                except Exception as e:
                                    logger.error(f"❌ Error guardando chunk {i}: {e}")
                                    continue
                                    
                        except Exception as e:
                            logger.error(f"❌ Error procesando embeddings: {e}")
                            # Continuar sin embeddings
                            for chunk in chunks:
                                try:
                                    chunk_id = self._save_chunk(chunk, document_id)
                                    chunks_saved += 1
                                except Exception as e:
                                    logger.error(f"❌ Error guardando chunk: {e}")
                                    continue
                        
                        # Actualizar documento con estadísticas
                        self.supabase.table('siem_documents').update({
                            'total_chunks': chunks_saved,
                            'processing_status': 'completed'
                        }).eq('id', document_id).execute()
                        
                        results['processed_files'] += 1
                        results['total_chunks'] += chunks_saved
                        results['total_embeddings'] += embeddings_saved
                        
                        logger.info(f"✅ {file_path.name}: {chunks_saved} chunks, {embeddings_saved} embeddings")
                        
                    except Exception as e:
                        error_msg = f"Error procesando {file_path.name}: {e}"
                        logger.error(f"❌ {error_msg}")
                        results['errors'].append(error_msg)
                        continue
                
                results['end_time'] = datetime.now().isoformat()
                results['success_rate'] = results['processed_files'] / results['total_files'] if results['total_files'] > 0 else 0
                
                logger.info("🎉 PROCESAMIENTO COMPLETADO")
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
            'document_id': document_id,
            'embedding_vector': embedding,
            'dimensions': len(embedding),
            'model_name': 'qwen3',
            'created_at': datetime.now().isoformat()
        }
        
        self.supabase.table('siem_embeddings').insert(embedding_data).execute()

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Sistema RAG ultra simplificado")
    parser.add_argument('source_directory', help='Directorio con documentos')
    parser.add_argument('--supabase-url', required=True, help='URL de Supabase')
    parser.add_argument('--supabase-key', required=True, help='Clave de Supabase')
    parser.add_argument('--max-workers', type=int, default=4, help='Workers paralelos')
    
    args = parser.parse_args()
    
    try:
        # Crear sistema
        rag_system = UltraSimpleRAGSystem(args.supabase_url, args.supabase_key)
        rag_system.initialize()
        
        # Procesar documentos
        import asyncio
        results = asyncio.run(rag_system.process_documents(args.source_directory, args.max_workers))
        
        # Guardar resultados
        with open('ultra_simple_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info("📊 Resultados guardados en ultra_simple_results.json")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
