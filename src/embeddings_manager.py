"""
Gestor principal de embeddings para documentos legales
"""
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from .config import config
from .document_processor import LegalDocumentProcessor
from .supabase_database import SupabaseVectorDatabase
from .lightrag_search import LightRAGSearchEngine

logger = logging.getLogger(__name__)

class LegalEmbeddingsManager:
    """Gestor principal para creación y gestión de embeddings legales"""
    
    def __init__(self):
        self.document_processor = LegalDocumentProcessor()
        self.vector_db = SupabaseVectorDatabase()
        self.search_engine = LightRAGSearchEngine(self.vector_db)
        self.embeddings_model = None
        self._setup_embeddings()
        self._setup_file_logger()
        
    def _setup_embeddings(self):
        """Configurar modelo de embeddings"""
        try:
            logger.info(f"🔄 Cargando modelo de embeddings: {config.embedding_model}")
            
            self.embeddings_model = SentenceTransformer(
                config.embedding_model,
                device=config.embedding_device,
                trust_remote_code=True
            )
            
            # Verificar dimensiones
            test_embedding = self.embeddings_model.encode(
                ["test"], 
                normalize_embeddings=True
            )
            
            logger.info(f"✅ Modelo cargado: {config.embedding_model}")
            logger.info(f"📊 Dimensiones: {test_embedding.shape[1]}")
            logger.info(f"🎯 Dispositivo: {config.embedding_device}")
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo de embeddings: {e}")
            raise
    
    def _setup_file_logger(self):
        """Configurar logger para archivos individuales"""
        self.file_logger = logging.getLogger('embeddings_processing')
        file_handler = logging.FileHandler(
            config.logs_dir / f"embeddings_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.file_logger.addHandler(file_handler)
        self.file_logger.setLevel(logging.INFO)
    
    async def process_documents_from_directory(
        self, 
        source_directory: str,
        force_rebuild: bool = False,
        batch_size: int = None
    ) -> Dict[str, Any]:
        """Procesar todos los documentos PDF de un directorio"""
        source_path = Path(source_directory)
        if not source_path.exists():
            raise FileNotFoundError(f"Directorio no encontrado: {source_directory}")
        
        batch_size = batch_size or config.batch_size
        
        try:
            logger.info(f"🚀 Iniciando procesamiento de documentos desde: {source_directory}")
            logger.info(f"📦 Tamaño de lote: {batch_size}")
            logger.info(f"🔄 Reconstruir: {force_rebuild}")
            
            # Encontrar todos los PDFs
            pdf_files = list(source_path.rglob("*.pdf"))
            logger.info(f"📄 Encontrados {len(pdf_files)} archivos PDF")
            
            if not pdf_files:
                logger.warning("⚠️ No se encontraron archivos PDF")
                return {"error": "No se encontraron archivos PDF"}
            
            # Estadísticas de procesamiento
            stats = {
                "total_files": len(pdf_files),
                "processed_files": 0,
                "total_chunks": 0,
                "failed_files": 0,
                "processing_time": 0,
                "start_time": datetime.now().isoformat()
            }
            
            start_time = datetime.now()
            
            # Procesar archivos en lotes
            with tqdm(total=len(pdf_files), desc="Procesando PDFs", unit="archivo") as pbar:
                for i in range(0, len(pdf_files), batch_size):
                    batch_files = pdf_files[i:i + batch_size]
                    batch_results = await self._process_batch(batch_files, source_path)
                    
                    # Actualizar estadísticas
                    stats["processed_files"] += batch_results["processed_files"]
                    stats["total_chunks"] += batch_results["total_chunks"]
                    stats["failed_files"] += batch_results["failed_files"]
                    
                    pbar.update(len(batch_files))
                    pbar.set_postfix({
                        'Procesados': stats["processed_files"],
                        'Chunks': stats["total_chunks"],
                        'Fallidos': stats["failed_files"]
                    })
            
            # Finalizar estadísticas
            end_time = datetime.now()
            stats["processing_time"] = (end_time - start_time).total_seconds()
            stats["end_time"] = end_time.isoformat()
            stats["success_rate"] = stats["processed_files"] / stats["total_files"] if stats["total_files"] > 0 else 0
            
            logger.info(f"✅ Procesamiento completado:")
            logger.info(f"   📄 Archivos procesados: {stats['processed_files']}/{stats['total_files']}")
            logger.info(f"   📝 Chunks creados: {stats['total_chunks']}")
            logger.info(f"   ⏱️ Tiempo total: {stats['processing_time']:.2f} segundos")
            logger.info(f"   📈 Tasa de éxito: {stats['success_rate']:.2%}")
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error procesando directorio: {e}")
            raise
    
    async def _process_batch(
        self, 
        pdf_files: List[Path], 
        source_path: Path
    ) -> Dict[str, Any]:
        """Procesar un lote de archivos PDF"""
        batch_stats = {
            "processed_files": 0,
            "total_chunks": 0,
            "failed_files": 0
        }
        
        for pdf_file in pdf_files:
            try:
                # Determinar estado basado en la estructura de directorios
                relative_path = pdf_file.relative_to(source_path)
                state = relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown"
                
                # Procesar documento
                chunks = await self.document_processor.process_document(pdf_file)
                
                if not chunks:
                    logger.warning(f"⚠️ No se generaron chunks para {pdf_file.name}")
                    batch_stats["failed_files"] += 1
                    continue
                
                # Generar embeddings para los chunks
                embeddings = await self._generate_embeddings_batch(chunks)
                
                # Preparar datos del documento
                document_data = {
                    "document_id": f"{state}_{pdf_file.stem}",
                    "title": pdf_file.stem,
                    "state": state,
                    "document_type": self._detect_document_type(pdf_file.name),
                    "file_path": str(pdf_file),
                    "file_size": pdf_file.stat().st_size,
                    "metadata": {
                        "source_directory": str(source_path),
                        "processed_at": datetime.now().isoformat(),
                        "total_chunks": len(chunks)
                    }
                }
                
                # Almacenar en base de datos
                await self.vector_db.store_document(document_data)
                
                # Preparar chunks con embeddings
                chunks_with_embeddings = []
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    chunk_data = {
                        "chunk_id": i,
                        "content": chunk["content"],
                        "embedding": embedding.tolist(),
                        "metadata": {
                            **chunk["metadata"],
                            "document_id": document_data["document_id"],
                            "state": state
                        }
                    }
                    chunks_with_embeddings.append(chunk_data)
                
                # Almacenar chunks
                await self.vector_db.store_chunks(
                    document_data["document_id"], 
                    chunks_with_embeddings
                )
                
                # Agregar a LightRAG si está disponible
                await self.search_engine.add_documents_to_lightrag(chunks_with_embeddings)
                
                batch_stats["processed_files"] += 1
                batch_stats["total_chunks"] += len(chunks)
                
                # Log detallado solo en archivo
                self.file_logger.info(f"✅ COMPLETADO - {pdf_file.name}: {len(chunks)} chunks, {len(embeddings)} embeddings")
                
            except Exception as e:
                logger.error(f"❌ Error procesando {pdf_file.name}: {e}")
                batch_stats["failed_files"] += 1
                continue
        
        return batch_stats
    
    async def _generate_embeddings_batch(self, chunks: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Generar embeddings para un lote de chunks"""
        try:
            # Extraer contenido de los chunks
            contents = [chunk["content"] for chunk in chunks]
            
            # Generar embeddings con GPU si está disponible
            with torch.no_grad():
                embeddings = self.embeddings_model.encode(
                    contents,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_tensor=True,
                    device=config.embedding_device
                )
                
                # Convertir a numpy
                if isinstance(embeddings, torch.Tensor):
                    embeddings = embeddings.cpu().numpy()
            
            # Limpiar memoria GPU
            if config.embedding_device == "cuda":
                torch.cuda.empty_cache()
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Error generando embeddings: {e}")
            raise
    
    def _detect_document_type(self, filename: str) -> str:
        """Detectar tipo de documento basado en el nombre del archivo"""
        name_lower = filename.lower()
        
        if any(keyword in name_lower for keyword in ['ley', 'codigo', 'código']):
            return 'ley'
        elif any(keyword in name_lower for keyword in ['decreto']):
            return 'decreto'
        elif any(keyword in name_lower for keyword in ['acuerdo']):
            return 'acuerdo'
        elif any(keyword in name_lower for keyword in ['reglamento']):
            return 'reglamento'
        elif any(keyword in name_lower for keyword in ['constitución', 'constitucion']):
            return 'constitucion'
        else:
            return 'documento'
    
    async def search_documents(
        self, 
        query: str, 
        top_k: int = 10,
        search_mode: str = "hybrid",
        state: Optional[str] = None,
        document_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Buscar documentos usando el motor de búsqueda configurado"""
        try:
            logger.info(f"🔍 Buscando: '{query}' (modo: {search_mode})")
            
            if search_mode == "hybrid":
                results = await self.search_engine.hybrid_search(query, top_k)
            elif search_mode == "lightrag":
                results = await self.search_engine.search_with_lightrag(query, top_k=top_k)
            else:  # vector search
                # Generar embedding de la consulta
                query_embedding = self.embeddings_model.encode(
                    [query], 
                    normalize_embeddings=True,
                    convert_to_tensor=False
                )[0].tolist()
                
                # Buscar en base de datos
                results = await self.vector_db.search_similar_chunks(
                    query_embedding=query_embedding,
                    limit=top_k,
                    state=state,
                    document_type=document_type
                )
                
                # Convertir a formato estándar
                results = [{
                    "content": r["content"],
                    "metadata": r["metadata"],
                    "score": r["similarity"],
                    "source": "vector_search"
                } for r in results]
            
            logger.info(f"✅ Encontrados {len(results)} resultados")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error buscando documentos: {e}")
            return []
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas completas del sistema"""
        try:
            # Estadísticas de la base de datos
            db_stats = await self.vector_db.get_document_stats()
            
            # Estadísticas de LightRAG
            lightrag_stats = await self.search_engine.get_lightrag_stats()
            
            # Estadísticas del procesador
            processor_info = self.document_processor.get_processor_info()
            
            # Estadísticas de búsqueda
            search_capabilities = self.search_engine.get_search_capabilities()
            
            return {
                "database": db_stats,
                "lightrag": lightrag_stats,
                "processor": processor_info,
                "search": search_capabilities,
                "embedding_model": {
                    "name": config.embedding_model,
                    "device": config.embedding_device,
                    "dimensions": config.embedding_dimensions
                },
                "config": {
                    "chunk_size": config.chunk_size,
                    "chunk_overlap": config.chunk_overlap,
                    "batch_size": config.batch_size,
                    "similarity_threshold": config.similarity_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo estadísticas: {e}")
            return {"error": str(e)}
    
    def cleanup(self):
        """Limpiar recursos"""
        if self.vector_db:
            self.vector_db.close_connections()
        logger.info("✅ Recursos limpiados")
