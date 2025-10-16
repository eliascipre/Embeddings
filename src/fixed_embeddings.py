"""
Sistema de embeddings corregido para RTX 5090
- Barra de progreso funcional
- Gestión de memoria GPU optimizada
- Uso máximo de GPU
"""
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import numpy as np
import torch
import time
import psutil
import GPUtil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import gc
from torch.utils.tensorboard import SummaryWriter

from sentence_transformers import SentenceTransformer
from config import config
from document_processor import LegalDocumentProcessor
from supabase_database import SupabaseVectorDatabase
from lightrag_search import LightRAGSearchEngine

logger = logging.getLogger(__name__)

class FixedEmbeddingsManager:
    """Sistema de embeddings corregido para máximo rendimiento"""
    
    def __init__(self):
        self.document_processor = LegalDocumentProcessor()
        self.vector_db = SupabaseVectorDatabase()
        self.search_engine = LightRAGSearchEngine(self.vector_db)
        self.embeddings_model = None
        self._setup_embeddings()
        self._setup_file_logger()
        self._setup_tensorboard()
        
        # Estadísticas
        self.stats = {
            "pdfs_processed": 0,
            "chunks_generated": 0,
            "embeddings_generated": 0,
            "gpu_utilization": 0,
            "cpu_utilization": 0
        }
    
    def _setup_embeddings(self):
        """Configurar modelo de embeddings optimizado"""
        try:
            logger.info(f"🔄 Cargando modelo de embeddings: {config.embedding_model}")
            
            # Configuración optimizada para RTX 5090
            self.embeddings_model = SentenceTransformer(
                config.embedding_model,
                device=config.embedding_device,
                trust_remote_code=True
            )
            
            # Optimizaciones para RTX 5090
            if config.embedding_device == "cuda":
                # Habilitar optimizaciones
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                # Configurar memoria GPU
                torch.cuda.empty_cache()
                torch.cuda.set_per_process_memory_fraction(0.90)  # Usar 90% de la GPU
                
                # Verificar que la GPU esté disponible
                if torch.cuda.is_available():
                    gpu_props = torch.cuda.get_device_properties(0)
                    logger.info(f"🎮 GPU: {gpu_props.name}")
                    logger.info(f"💾 Memoria GPU: {gpu_props.total_memory / 1024**3:.1f}GB")
            
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
        self.file_logger = logging.getLogger('fixed_embeddings')
        file_handler = logging.FileHandler(
            config.logs_dir / f"fixed_embeddings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.file_logger.addHandler(file_handler)
        self.file_logger.setLevel(logging.INFO)
    
    def _setup_tensorboard(self):
        """Configurar TensorBoard para monitoreo avanzado"""
        try:
            # Crear directorio para logs de TensorBoard
            tb_dir = config.logs_dir / "tensorboard"
            tb_dir.mkdir(exist_ok=True)
            
            # Inicializar TensorBoard
            self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
            logger.info(f"📊 TensorBoard configurado en: {tb_dir}")
            
        except Exception as e:
            logger.warning(f"⚠️ No se pudo configurar TensorBoard: {e}")
            self.tb_writer = None
    
    def _get_system_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del sistema en tiempo real"""
        stats = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "gpu_memory_used": 0,
            "gpu_memory_total": 0,
            "gpu_utilization": 0
        }
        
        if torch.cuda.is_available():
            stats["gpu_memory_used"] = torch.cuda.memory_allocated() / 1024**3
            stats["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
            stats["gpu_utilization"] = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
        
        return stats
    
    async def process_documents_from_directory(
        self, 
        source_directory: str,
        force_rebuild: bool = False,
        batch_size: int = None,
        max_workers: int = None
    ) -> Dict[str, Any]:
        """Procesar documentos con gestión de memoria optimizada"""
        source_path = Path(source_directory)
        if not source_path.exists():
            raise FileNotFoundError(f"Directorio no encontrado: {source_directory}")
        
        # Configuración optimizada
        batch_size = batch_size or 20  # Lotes pequeños para evitar OOM
        max_workers = max_workers or min(16, psutil.cpu_count(logical=True))
        
        try:
            logger.info(f"🚀 Iniciando procesamiento optimizado desde: {source_directory}")
            logger.info(f"📦 Tamaño de lote: {batch_size}")
            logger.info(f"👥 Workers paralelos: {max_workers}")
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
                "start_time": datetime.now().isoformat(),
                "max_workers": max_workers,
                "batch_size": batch_size
            }
            
            start_time = time.time()
            
            # Procesar archivos con barra de progreso funcional
            with tqdm(total=len(pdf_files), desc="🚀 Procesando PDFs", unit="archivo", 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                
                # Procesar en lotes pequeños
                for i in range(0, len(pdf_files), batch_size):
                    batch_files = pdf_files[i:i + batch_size]
                    
                    # Procesar lote
                    batch_results = await self._process_batch_fixed(
                        batch_files, source_path, max_workers
                    )
                    
                    # Actualizar estadísticas
                    stats["processed_files"] += batch_results["processed_files"]
                    stats["total_chunks"] += batch_results["total_chunks"]
                    stats["failed_files"] += batch_results["failed_files"]
                    
                    # Obtener estadísticas del sistema
                    system_stats = self._get_system_stats()
                    
                    # Calcular ETA
                    elapsed_time = time.time() - start_time
                    if stats["processed_files"] > 0:
                        eta_seconds = (elapsed_time / stats["processed_files"]) * (stats["total_files"] - stats["processed_files"])
                        eta_str = f"{int(eta_seconds//3600):02d}:{int((eta_seconds%3600)//60):02d}:{int(eta_seconds%60):02d}"
                    else:
                        eta_str = "Calculando..."
                    
                    pbar.update(len(batch_files))
                    pbar.set_postfix({
                        'ETA': eta_str,
                        'GPU': f"{system_stats['gpu_memory_used']:.1f}GB",
                        'CPU': f"{system_stats['cpu_percent']:.1f}%",
                        'Chunks': stats["total_chunks"]
                    })
                    
                    # Log a TensorBoard
                    if self.tb_writer:
                        step = stats["processed_files"]
                        self.tb_writer.add_scalar('Progress/Files_Processed', step, step)
                        self.tb_writer.add_scalar('Progress/Chunks_Generated', stats["total_chunks"], step)
                        self.tb_writer.add_scalar('System/CPU_Usage', system_stats['cpu_percent'], step)
                        self.tb_writer.add_scalar('System/GPU_Memory_Used', system_stats['gpu_memory_used'], step)
                        self.tb_writer.add_scalar('System/GPU_Utilization', system_stats['gpu_utilization'], step)
                        self.tb_writer.add_scalar('Performance/Files_per_Second', 
                                                stats["processed_files"] / elapsed_time if elapsed_time > 0 else 0, step)
                        self.tb_writer.add_scalar('Performance/Chunks_per_Second', 
                                                stats["total_chunks"] / elapsed_time if elapsed_time > 0 else 0, step)
                        self.tb_writer.flush()
                    
                    # Limpiar memoria GPU después de cada lote
                    if config.embedding_device == "cuda":
                        torch.cuda.empty_cache()
                        gc.collect()
            
            # Finalizar estadísticas
            end_time = time.time()
            stats["processing_time"] = end_time - start_time
            stats["end_time"] = datetime.now().isoformat()
            stats["success_rate"] = stats["processed_files"] / stats["total_files"] if stats["total_files"] > 0 else 0
            stats["files_per_second"] = stats["processed_files"] / stats["processing_time"] if stats["processing_time"] > 0 else 0
            stats["chunks_per_second"] = stats["total_chunks"] / stats["processing_time"] if stats["processing_time"] > 0 else 0
            
            logger.info(f"✅ Procesamiento completado:")
            logger.info(f"   📄 Archivos procesados: {stats['processed_files']}/{stats['total_files']}")
            logger.info(f"   📝 Chunks creados: {stats['total_chunks']}")
            logger.info(f"   ⏱️ Tiempo total: {stats['processing_time']:.2f} segundos")
            logger.info(f"   📈 Tasa de éxito: {stats['success_rate']:.2%}")
            logger.info(f"   🚀 Archivos/segundo: {stats['files_per_second']:.2f}")
            logger.info(f"   🚀 Chunks/segundo: {stats['chunks_per_second']:.2f}")
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error procesando directorio: {e}")
            raise
    
    async def _process_batch_fixed(
        self, 
        pdf_files: List[Path], 
        source_path: Path,
        max_workers: int
    ) -> Dict[str, Any]:
        """Procesar lote con gestión de memoria optimizada"""
        batch_stats = {
            "processed_files": 0,
            "total_chunks": 0,
            "failed_files": 0
        }
        
        # Procesar archivos secuencialmente para evitar OOM
        for pdf_file in pdf_files:
            try:
                result = await self._process_single_file_fixed(pdf_file, source_path)
                batch_stats["processed_files"] += result.get("processed_files", 0)
                batch_stats["total_chunks"] += result.get("total_chunks", 0)
                batch_stats["failed_files"] += result.get("failed_files", 0)
                
                # Limpiar memoria después de cada archivo
                if config.embedding_device == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()
                    
            except Exception as e:
                logger.error(f"❌ Error procesando {pdf_file.name}: {e}")
                batch_stats["failed_files"] += 1
        
        return batch_stats
    
    async def _process_single_file_fixed(
        self, 
        pdf_file: Path, 
        source_path: Path
    ) -> Dict[str, Any]:
        """Procesar un archivo individual con gestión de memoria optimizada"""
        try:
            # Determinar estado basado en la estructura de directorios
            relative_path = pdf_file.relative_to(source_path)
            state = relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown"
            
            # Procesar documento (CPU)
            chunks = await self.document_processor.process_document(pdf_file)
            
            if not chunks:
                self.file_logger.warning(f"⚠️ No se generaron chunks para {pdf_file.name}")
                return {"processed_files": 0, "total_chunks": 0, "failed_files": 1}
            
            # Generar embeddings en lotes pequeños
            embeddings = await self._generate_embeddings_fixed(chunks)
            
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
            
            # Log detallado
            self.file_logger.info(f"✅ COMPLETADO - {pdf_file.name}: {len(chunks)} chunks, {len(embeddings)} embeddings")
            
            return {
                "processed_files": 1,
                "total_chunks": len(chunks),
                "failed_files": 0
            }
            
        except Exception as e:
            self.file_logger.error(f"❌ ERROR - {pdf_file.name}: {str(e)}")
            return {"processed_files": 0, "total_chunks": 0, "failed_files": 1}
    
    async def _generate_embeddings_fixed(self, chunks: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Generar embeddings con gestión de memoria optimizada"""
        try:
            # Extraer contenido de los chunks
            contents = [chunk["content"] for chunk in chunks]
            
            # Generar embeddings en lotes pequeños para evitar OOM
            all_embeddings = []
            batch_size = 8  # Lotes muy pequeños para evitar OOM
            
            for i in range(0, len(contents), batch_size):
                batch_contents = contents[i:i + batch_size]
                
                with torch.no_grad():
                    batch_embeddings = self.embeddings_model.encode(
                        batch_contents,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        convert_to_tensor=True,
                        device=config.embedding_device,
                        batch_size=len(batch_contents)
                    )
                    
                    # Convertir a numpy
                    if isinstance(batch_embeddings, torch.Tensor):
                        batch_embeddings = batch_embeddings.cpu().numpy()
                    
                    all_embeddings.extend(batch_embeddings)
                
                # Limpiar memoria después de cada lote pequeño
                if config.embedding_device == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()
            
            return all_embeddings
            
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
    
    def cleanup(self):
        """Limpiar recursos"""
        if self.vector_db:
            self.vector_db.close_connections()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.tb_writer:
            self.tb_writer.close()
        gc.collect()
        logger.info("✅ Recursos limpiados")
