#!/usr/bin/env python3
"""
Pipeline optimizado para procesamiento de documentos SIEM
Optimizado para GPU A100, HuggingFace API y RAG híbrido
"""

import os
import json
import logging
import hashlib
import asyncio
import aiohttp
import time
import io
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import re
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import queue
import backoff
import pymupdf4llm
from unstructured.partition.pdf import partition_pdf
import pdfplumber
from unstructured.chunking.title import chunk_by_title
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from supabase import create_client, Client
# from tensorboard import SummaryWriter  # Temporalmente deshabilitado
import psutil
import GPUtil

# Cliente Qwen3 personalizado (versión simplificada)
from qwen3_embedding_client_simple import Qwen3EmbeddingClient, Qwen3BatchProcessor, Qwen3Config

# Configuración local
from config_supabase import (
    get_supabase_url, get_supabase_key, get_database_tables,
    get_embedding_config, get_chunking_config, get_processing_config
)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/optimized_siem_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingStats:
    """Estadísticas de procesamiento"""
    documents_processed: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    api_calls_made: int = 0
    errors: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_processing_time: float = 0.0
    gpu_memory_used: float = 0.0
    cpu_usage: float = 0.0

class CircuitBreakerConfig:
    """Configuración del circuit breaker"""
    FAILURE_THRESHOLD = 5
    RECOVERY_TIMEOUT = 60
    EXPECTED_EXCEPTION = Exception

class HuggingFaceAPIClient:
    """Cliente simplificado para HuggingFace API usando Qwen3"""
    
    def __init__(self, base_url: str = "https://wfuosp4mnqkimde9.us-east-1.aws.endpoints.huggingface.cloud"):
        self.base_url = base_url
        self.session = None
        self.rate_limiter = asyncio.Semaphore(5)  # Reducido a 5 requests concurrentes
        self.request_delay = 0.6  # Delay entre lotes
        self.stats = {
            'requests_made': 0,
            'requests_failed': 0,
            'rate_limit_hits': 0,
            'total_response_time': 0.0
        }
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3,
        base=2
    )
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Genera embeddings en lote usando Qwen3-Embedding-0.6B"""
        
        async with self.rate_limiter:
            start_time = time.time()
            
            try:
                payload = {
                    "inputs": texts,
                    "parameters": {
                        "dimensions": 1024,
                        "normalize": True,
                        "instruction": "Represent the following text for retrieval:"
                    }
                }
                
                async with self.session.post(
                    self.base_url,
                    json=payload
                ) as response:
                    
                    if response.status == 429:  # Rate limit
                        self.stats['rate_limit_hits'] += 1
                        await asyncio.sleep(2)
                        raise aiohttp.ClientError("Rate limit exceeded")
                    
                    response.raise_for_status()
                    embeddings = await response.json()
                    
                    self.stats['requests_made'] += 1
                    self.stats['total_response_time'] += time.time() - start_time
                    
                    # Delay entre requests
                    await asyncio.sleep(self.request_delay)
                    
                    return embeddings
                    
            except Exception as e:
                self.stats['requests_failed'] += 1
                logger.error(f"Error en API de Qwen3: {e}")
                raise

class GPUOptimizedProcessor:
    """Procesador optimizado para GPU A100"""
    
    def __init__(self):
        self.device = self._setup_gpu()
        self.model = None
        self.tokenizer = None
        self.memory_monitor = GPUMemoryMonitor()
    
    def _setup_gpu(self) -> str:
        """Configura la GPU A100"""
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"GPUs disponibles: {gpu_count}")
            
            # Usar la primera GPU A100 disponible
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                if "A100" in gpu_name:
                    torch.cuda.set_device(i)
                    logger.info(f"Usando GPU A100: {gpu_name}")
                    return f"cuda:{i}"
            
            # Si no hay A100, usar la primera GPU disponible
            torch.cuda.set_device(0)
            logger.info(f"Usando GPU: {torch.cuda.get_device_name(0)}")
            return "cuda:0"
        else:
            logger.warning("CUDA no disponible, usando CPU")
            return "cpu"
    
    def load_model(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """Carga el modelo optimizado para GPU"""
        try:
            logger.info(f"Cargando modelo: {model_name}")
            self.model = SentenceTransformer(model_name, device=self.device)
            
            # Optimizaciones para A100
            if self.device != "cpu":
                self.model.half()  # Usar FP16 para mejor rendimiento
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            logger.info("Modelo cargado exitosamente")
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            raise

class GPUMemoryMonitor:
    """Monitor de memoria GPU"""
    
    def __init__(self):
        self.lock = Lock()
    
    def get_memory_info(self) -> Dict[str, float]:
        """Obtiene información de memoria GPU"""
        with self.lock:
            try:
                if torch.cuda.is_available():
                    gpu = GPUtil.getGPUs()[0]
                    return {
                        'total_memory': gpu.memoryTotal,
                        'used_memory': gpu.memoryUsed,
                        'free_memory': gpu.memoryFree,
                        'utilization': gpu.load * 100
                    }
                else:
                    return {'total_memory': 0, 'used_memory': 0, 'free_memory': 0, 'utilization': 0}
            except Exception as e:
                logger.warning(f"Error obteniendo info de GPU: {e}")
                return {'total_memory': 0, 'used_memory': 0, 'free_memory': 0, 'utilization': 0}

class LegalDocumentChunker:
    """Chunker inteligente para documentos legales"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.legal_patterns = config['legal_patterns']
    
    def extract_with_pymupdf(self, file_path: Path) -> str:
        """Extrae texto usando PyMuPDF con manejo robusto de errores"""
        extraction_methods = [
            ("pymupdf4llm", self._extract_with_pymupdf4llm),
            ("pymupdf_direct", self._extract_with_pymupdf_direct),
            ("pdfplumber", self._extract_with_pdfplumber),
            ("unstructured", self._extract_with_unstructured),
            ("ocr", self._extract_with_ocr)
        ]
        
        for method_name, method_func in extraction_methods:
            try:
                logger.info(f"Intentando extracción con {method_name} para {file_path.name}")
                text = method_func(file_path)
                
                if text and text.strip() and len(text.strip()) > 50:  # Mínimo 50 caracteres
                    logger.info(f"Extracción exitosa con {method_name} para {file_path.name} ({len(text)} caracteres)")
                    return text
                else:
                    logger.warning(f"{method_name} extrajo texto insuficiente de {file_path.name}")
                    
            except Exception as e:
                logger.warning(f"Error con {method_name} en {file_path.name}: {e}")
                continue
        
        logger.error(f"No se pudo extraer texto de {file_path.name} con ningún método")
        return ""
    
    def _extract_with_pymupdf4llm(self, file_path: Path) -> str:
        """Extrae texto usando pymupdf4llm con manejo de errores específicos"""
        try:
            # Verificar que el archivo existe y es válido
            if not file_path.exists():
                raise Exception("Archivo no encontrado")
            
            # Intentar extracción con pymupdf4llm
            text = pymupdf4llm.to_markdown(str(file_path))
            
            if not text or not text.strip():
                raise Exception("pymupdf4llm extrajo texto vacío")
            
            # Verificar que el texto tiene contenido útil
            if len(text.strip()) < 50:
                raise Exception("Texto extraído muy corto")
            
            return text
            
        except Exception as e:
            # Si pymupdf4llm falla, lanzar excepción para que se use el siguiente método
            raise Exception(f"pymupdf4llm falló: {str(e)}")
    
    def _extract_with_pymupdf_direct(self, file_path: Path) -> str:
        """Extrae texto usando PyMuPDF directo con manejo robusto de páginas"""
        import fitz
        doc = None
        try:
            doc = fitz.open(str(file_path))
            text = ""
            pages_with_text = 0
            total_pages = doc.page_count
            
            logger.info(f"Procesando {total_pages} páginas con PyMuPDF directo")
            
            for page_num in range(total_pages):
                try:
                    page = doc[page_num]
                    
                    # Intentar extracción de texto
                    page_text = page.get_text()
                    
                    if page_text and page_text.strip():
                        # Limpiar texto extraído
                        clean_text = page_text.strip()
                        if len(clean_text) > 10:  # Solo páginas con contenido significativo
                            text += f"\n--- PÁGINA {page_num + 1} ---\n{clean_text}\n"
                            pages_with_text += 1
                        else:
                            logger.debug(f"Página {page_num + 1} tiene poco contenido, saltando")
                    else:
                        logger.debug(f"Página {page_num + 1} no contiene texto (posible imagen)")
                        
                except Exception as page_error:
                    logger.warning(f"Error procesando página {page_num + 1} de {file_path.name}: {page_error}")
                    continue
            
            if pages_with_text == 0:
                raise Exception("No se extrajo texto de ninguna página")
            
            logger.info(f"Extraídas {pages_with_text} páginas con texto de {total_pages} totales")
            return text
            
        except Exception as e:
            raise Exception(f"Error con PyMuPDF directo: {str(e)}")
        finally:
            if doc:
                doc.close()
    
    def _extract_with_pdfplumber(self, file_path: Path) -> str:
        """Extrae texto usando pdfplumber"""
        try:
            text = ""
            with pdfplumber.open(str(file_path)) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text += f"\n--- PÁGINA {page_num + 1} ---\n{page_text}\n"
                    except Exception as page_error:
                        logger.warning(f"Error procesando página {page_num + 1} con pdfplumber: {page_error}")
                        continue
            
            if not text.strip():
                raise Exception("pdfplumber no extrajo texto")
            
            return text
            
        except Exception as e:
            raise Exception(f"Error con pdfplumber: {str(e)}")
    
    def _extract_with_unstructured(self, file_path: Path) -> str:
        """Extrae texto usando Unstructured"""
        try:
            elements = partition_pdf(str(file_path))
            text = ""
            for element in elements:
                if hasattr(element, 'text') and element.text:
                    text += element.text + "\n"
            
            if not text.strip():
                raise Exception("Unstructured no extrajo texto")
            
            return text
            
        except Exception as e:
            raise Exception(f"Error con Unstructured: {str(e)}")
    
    def _extract_with_ocr(self, file_path: Path) -> str:
        """Extrae texto usando OCR (último recurso para documentos escaneados)"""
        try:
            import pytesseract
            from PIL import Image
            import fitz
            
            doc = fitz.open(str(file_path))
            text = ""
            
            for page_num in range(doc.page_count):
                try:
                    page = doc[page_num]
                    # Convertir página a imagen
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("png")
                    
                    # Usar OCR en la imagen
                    image = Image.open(io.BytesIO(img_data))
                    page_text = pytesseract.image_to_string(image, lang='spa')  # Español
                    
                    if page_text.strip():
                        text += page_text + "\n"
                        logger.info(f"OCR extrajo texto de página {page_num} de {file_path}")
                    else:
                        logger.warning(f"OCR no encontró texto en página {page_num} de {file_path}")
                        
                except Exception as page_error:
                    logger.warning(f"Error en OCR página {page_num} de {file_path}: {page_error}")
                    continue
            
            doc.close()
            return text
            
        except ImportError:
            logger.error("pytesseract no está instalado. Instala con: pip install pytesseract")
            return ""
        except Exception as e:
            logger.error(f"Error en OCR para {file_path}: {e}")
            return ""
    
    def extract_with_unstructured(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extrae texto usando Unstructured (mejor estructura)"""
        try:
            elements = partition_pdf(str(file_path))
            return [{"text": str(el), "type": el.__class__.__name__} for el in elements]
        except Exception as e:
            logger.error(f"Error con Unstructured en {file_path}: {e}")
            return []
    
    def create_legal_chunks(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Crea chunks preservando estructura legal"""
        chunks = []
        
        # Dividir por secciones legales
        section_pattern = '|'.join(self.legal_patterns.values())
        sections = re.split(f'({section_pattern})', text)
        
        current_section = ""
        current_tokens = 0
        chunk_idx = 0
        
        for i, part in enumerate(sections):
            if re.match(section_pattern, part):
                # Procesar sección anterior
                if current_section.strip():
                    section_chunks = self._split_section_into_chunks(
                        current_section, metadata, chunk_idx
                    )
                    chunks.extend(section_chunks)
                    chunk_idx += len(section_chunks)
                
                current_section = part + " "
                current_tokens = len(part.split())
            else:
                current_section += part
                current_tokens += len(part.split())
        
        # Procesar última sección
        if current_section.strip():
            section_chunks = self._split_section_into_chunks(
                current_section, metadata, chunk_idx
            )
            chunks.extend(section_chunks)
        
        return chunks
    
    def _split_section_into_chunks(self, section: str, metadata: Dict[str, Any], start_idx: int) -> List[Dict[str, Any]]:
        """Divide una sección en chunks más pequeños"""
        section_tokens = len(section.split())
        
        if section_tokens <= self.config['max_chunk_size']:
            return [{
                'text': section.strip(),
                'chunk_type': 'legal_section',
                'chunk_index': start_idx,
                'token_count': section_tokens,
                'metadata': metadata.copy(),
                'legal_hierarchy': self._extract_legal_hierarchy(section)
            }]
        
        # Dividir por párrafos
        paragraphs = re.split(self.config['paragraph_splitter'], section)
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_idx = start_idx
        
        for para in paragraphs:
            para_tokens = len(para.split())
            
            if current_tokens + para_tokens <= self.config['max_chunk_size']:
                current_chunk += para + "\n\n"
                current_tokens += para_tokens
            else:
                if current_chunk.strip() and current_tokens >= self.config['min_chunk_size']:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'chunk_type': 'paragraph_group',
                        'chunk_index': chunk_idx,
                        'token_count': current_tokens,
                        'metadata': metadata.copy(),
                        'legal_hierarchy': self._extract_legal_hierarchy(current_chunk)
                    })
                    chunk_idx += 1
                
                current_chunk = para + "\n\n"
                current_tokens = para_tokens
        
        if current_chunk.strip() and current_tokens >= self.config['min_chunk_size']:
            chunks.append({
                'text': current_chunk.strip(),
                'chunk_type': 'paragraph_group',
                'chunk_index': chunk_idx,
                'token_count': current_tokens,
                'metadata': metadata.copy(),
                'legal_hierarchy': self._extract_legal_hierarchy(current_chunk)
            })
        
        return chunks
    
    def _extract_legal_hierarchy(self, text: str) -> Dict[str, Any]:
        """Extrae jerarquía legal del texto"""
        hierarchy = {
            'has_articulo': bool(re.search(self.legal_patterns['articulo'], text)),
            'has_capitulo': bool(re.search(self.legal_patterns['capitulo'], text)),
            'has_titulo': bool(re.search(self.legal_patterns['titulo'], text)),
            'has_seccion': bool(re.search(self.legal_patterns['seccion'], text)),
            'has_inciso': bool(re.search(self.legal_patterns['inciso'], text)),
            'has_fraccion': bool(re.search(self.legal_patterns['fraccion'], text)),
            'has_parrafo': bool(re.search(self.legal_patterns['parrafo'], text)),
            'legal_elements_count': 0
        }
        
        hierarchy['legal_elements_count'] = sum(hierarchy.values())
        return hierarchy

class OptimizedSIEMPipeline:
    """Pipeline principal optimizado"""
    
    def __init__(self, siem_path: str):
        self.siem_path = Path(siem_path)
        
        # Configuraciones
        self.embedding_config = get_embedding_config()
        self.chunking_config = get_chunking_config()
        self.processing_config = get_processing_config()
        self.tables = get_database_tables()
        
        # Componentes
        self.supabase = self._init_supabase()
        self.gpu_processor = GPUOptimizedProcessor()
        self.chunker = LegalDocumentChunker(self.chunking_config)
        # self.tensorboard_writer = SummaryWriter('logs/tensorboard')  # Temporalmente deshabilitado
        
        # Cliente Qwen3
        qwen3_config = Qwen3Config(
            embedding_dimensions=self.embedding_config['dimension']
        )
        self.qwen3_client = Qwen3EmbeddingClient(qwen3_config)
        self.batch_processor = Qwen3BatchProcessor(self.qwen3_client)
        
        # Estadísticas
        self.stats = ProcessingStats()
        self.batch_queue = queue.Queue(maxsize=50)
        self.processing_lock = Lock()
    
    def _init_supabase(self) -> Client:
        """Inicializa Supabase"""
        try:
            url = get_supabase_url()
            key = get_supabase_key()
            supabase = create_client(url, key)
            logger.info("Conexión con Supabase establecida")
            return supabase
        except Exception as e:
            logger.error(f"Error conectando con Supabase: {e}")
            raise
    
    async def process_document_batch(self, file_paths: List[Path]) -> List[Dict[str, Any]]:
        """Procesa un lote de documentos"""
        batch_id = f"batch_{int(time.time())}"
        logger.info(f"Procesando lote {batch_id} con {len(file_paths)} documentos")
        
        processed_documents = []
        
        # Procesar documentos en paralelo
        with ThreadPoolExecutor(max_workers=16) as executor:
            future_to_path = {
                executor.submit(self._process_single_document, file_path, batch_id): file_path
                for file_path in file_paths
            }
            
            for future in as_completed(future_to_path):
                file_path = future_to_path[future]
                try:
                    result = future.result()
                    if result:
                        processed_documents.append(result)
                except Exception as e:
                    logger.error(f"Error procesando {file_path}: {e}")
                    self.stats.errors += 1
        
        return processed_documents
    
    def _process_single_document(self, file_path: Path, batch_id: str) -> Optional[Dict[str, Any]]:
        """Procesa un documento individual con manejo robusto de errores"""
        try:
            logger.info(f"Procesando documento: {file_path.name}")
            
            # Extraer texto con múltiples métodos de fallback
            text = self._extract_text_robust(file_path)
            
            # Verificar que el texto extraído es válido
            if not self._is_valid_text(text):
                logger.warning(f"Documento {file_path.name} no contiene texto válido, saltando...")
                return None
            
            # Crear metadatos
            metadata = {
                'file_name': file_path.name,
                'file_path': str(file_path.relative_to(self.siem_path)),
                'file_size': file_path.stat().st_size,
                'file_extension': file_path.suffix.lower(),
                'file_hash': hashlib.md5(text.encode()).hexdigest(),
                'processed_at': datetime.now().isoformat(),
                'text_length': len(text),
                'token_count': len(text.split())
            }
            
            # Crear chunks
            chunks = self.chunker.create_legal_chunks(text, metadata)
            
            # Generar embeddings usando HuggingFace API
            embeddings_data = asyncio.run(self._generate_embeddings_async(chunks, batch_id))
            
            document_data = {
                'file_path': str(file_path),
                'metadata': metadata,
                'chunks': embeddings_data,
                'chunk_count': len(embeddings_data),
                'batch_id': batch_id
            }
            
            # Guardar en Supabase
            if self._save_document_to_supabase(document_data):
                with self.processing_lock:
                    self.stats.documents_processed += 1
                    self.stats.chunks_created += len(chunks)
                    self.stats.embeddings_generated += len(embeddings_data)
                
                return document_data
            
        except Exception as e:
            logger.error(f"Error procesando {file_path}: {e}")
            with self.processing_lock:
                self.stats.errors += 1
            return None
    
    async def _generate_embeddings_async(self, chunks: List[Dict[str, Any]], batch_id: str) -> List[Dict[str, Any]]:
        """Genera embeddings usando Qwen3-Embedding-0.6B"""
        try:
            # Extraer textos y limitar tamaño
            texts = [chunk['text'][:1000] for chunk in chunks]  # Limitar a 1000 caracteres por texto
            
            # Procesar en lotes más pequeños para evitar "Payload Too Large"
            batch_size = 3  # Reducido de 50 a 3
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                async with HuggingFaceAPIClient() as client:
                    batch_embeddings = await client.generate_embeddings_batch(batch_texts)
                    all_embeddings.extend(batch_embeddings)
            
            # Combinar con metadatos
            for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
                chunk['embedding'] = embedding
                chunk['embedding_id'] = f"{chunk['metadata']['file_name']}_{i}_{int(time.time())}"
                chunk['batch_id'] = batch_id
                chunk['model_name'] = self.embedding_config['model_name']
                chunk['generation_method'] = 'qwen3_api'
                chunk['api_endpoint'] = self.embedding_config['huggingface_endpoint']
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error generando embeddings con Qwen3: {e}")
            # Retornar chunks sin embedding para procesamiento posterior
            for chunk in chunks:
                chunk['embedding'] = None
                chunk['error'] = str(e)
            return chunks
    
    def _extract_text_robust(self, file_path: Path) -> str:
        """Extrae texto usando múltiples métodos con fallback"""
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.pdf':
            return self.chunker.extract_with_pymupdf(file_path)
        elif file_ext in ['.docx', '.doc']:
            return self._extract_word_document(file_path)
        elif file_ext == '.txt':
            return self._extract_text_file(file_path)
        else:
            logger.warning(f"Formato no soportado: {file_ext}")
            return ""
    
    def _extract_word_document(self, file_path: Path) -> str:
        """Extrae texto de documentos Word"""
        try:
            from docx import Document
            doc = Document(str(file_path))
            text = ""
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.warning(f"Error extrayendo Word {file_path.name}: {e}")
            return ""
    
    def _extract_text_file(self, file_path: Path) -> str:
        """Extrae texto de archivos de texto plano"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"Error leyendo archivo de texto {file_path.name}: {e}")
                return ""
        except Exception as e:
            logger.warning(f"Error extrayendo texto {file_path.name}: {e}")
            return ""
    
    def _is_valid_text(self, text: str) -> bool:
        """Valida si el texto extraído es útil"""
        if not text or not text.strip():
            return False
        
        # Verificar longitud mínima
        if len(text.strip()) < 50:  # Mínimo 50 caracteres
            return False
        
        # Verificar que no es solo espacios, números o caracteres especiales
        clean_text = ''.join(c for c in text if c.isalnum() or c.isspace())
        if len(clean_text.strip()) < 30:  # Mínimo 30 caracteres alfanuméricos
            return False
        
        # Verificar que contiene palabras (no solo números)
        words = clean_text.split()
        if len(words) < 5:  # Mínimo 5 palabras
            return False
        
        return True
    
    def _save_document_to_supabase(self, document_data: Dict[str, Any]) -> bool:
        """Guarda documento en Supabase"""
        try:
            # Insertar documento
            doc_record = {
                'file_name': document_data['metadata']['file_name'],
                'file_path': document_data['metadata']['file_path'],
                'file_hash': document_data['metadata']['file_hash'],
                'file_size': document_data['metadata']['file_size'],
                'file_extension': document_data['metadata']['file_extension'],
                'text_length': document_data['metadata']['text_length'],
                'token_count': document_data['metadata']['token_count'],
                'chunk_count': document_data['chunk_count'],
                'processing_status': 'completed',
                'processing_completed_at': datetime.now().isoformat()
            }
            
            result = self.supabase.table(self.tables['documents']).insert(doc_record).execute()
            document_id = result.data[0]['id'] if result.data else None
            
            if not document_id:
                return False
            
            # Insertar chunks y embeddings en lotes
            self._save_chunks_and_embeddings_batch(document_id, document_data['chunks'])
            
            return True
            
        except Exception as e:
            error_msg = str(e)
            if "duplicate key value violates unique constraint" in error_msg:
                logger.warning(f"Documento ya procesado (duplicado): {document_data.get('filename', 'unknown')}")
                return True  # No es un error crítico, el documento ya existe
            elif "Server disconnected" in error_msg:
                logger.error(f"Error de conexión con Supabase: {e}")
                return False
            else:
                logger.error(f"Error guardando en Supabase: {e}")
                return False
    
    def _save_chunks_and_embeddings_batch(self, document_id: int, chunks: List[Dict[str, Any]]):
        """Guarda chunks y embeddings en lotes de 1000"""
        batch_size = 1000
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            
            # Preparar datos para inserción
            chunk_records = []
            embedding_records = []
            
            for chunk in batch_chunks:
                # Generar hash único para el chunk
                chunk_hash = hashlib.md5(chunk['text'].encode()).hexdigest()
                
                # Verificar si el chunk ya existe (verificación opcional para optimización)
                # Nota: El UPSERT manejará los duplicados de forma atómica
                try:
                    existing_chunk = self.supabase.table(self.tables['chunks']).select('id').eq('chunk_hash', chunk_hash).execute()
                    if existing_chunk.data:
                        logger.info(f"Chunk ya existe (hash: {chunk_hash[:8]}...), será actualizado por UPSERT")
                except Exception as e:
                    logger.warning(f"Error verificando chunk duplicado: {e}")
                    # Continuar de todas formas, el UPSERT manejará el duplicado
                
                chunk_record = {
                    'document_id': document_id,
                    'chunk_text': chunk['text'],
                    'chunk_text_clean': chunk['text'].lower().strip(),
                    'chunk_type': chunk['chunk_type'],
                    'legal_hierarchy': chunk.get('legal_hierarchy', {}),
                    'chunk_index': chunk['chunk_index'],
                    'token_count': chunk['token_count'],
                    'character_count': len(chunk['text']),
                    'word_count': len(chunk['text'].split()),
                    'chunk_hash': chunk_hash,
                    'embedding_id': chunk['embedding_id']
                }
                chunk_records.append(chunk_record)
            
            # Insertar chunks usando UPSERT para manejar duplicados
            try:
                chunk_result = self.supabase.table(self.tables['chunks']).upsert(
                    chunk_records, 
                    on_conflict='chunk_hash'
                ).execute()
                chunk_ids = [item['id'] for item in chunk_result.data]
            except Exception as e:
                logger.error(f"Error en upsert de chunks: {e}")
                # Fallback: insertar uno por uno con manejo de duplicados
                chunk_ids = []
                for chunk_record in chunk_records:
                    try:
                        result = self.supabase.table(self.tables['chunks']).insert([chunk_record]).execute()
                        if result.data:
                            chunk_ids.append(result.data[0]['id'])
                    except Exception as insert_error:
                        if "duplicate key value violates unique constraint" in str(insert_error):
                            # Obtener el ID del chunk existente
                            existing = self.supabase.table(self.tables['chunks']).select('id').eq('chunk_hash', chunk_record['chunk_hash']).execute()
                            if existing.data:
                                chunk_ids.append(existing.data[0]['id'])
                        else:
                            logger.error(f"Error insertando chunk individual: {insert_error}")
            
            # Preparar embeddings
            for i, (chunk, chunk_id) in enumerate(zip(batch_chunks, chunk_ids)):
                embedding_record = {
                    'chunk_id': chunk_id,
                    'embedding_vector': chunk['embedding'],
                    'embedding_id': chunk['embedding_id'],
                    'model_name': self.embedding_config['model_name'],
                    'dimension': self.embedding_config['dimension'],
                    'generation_method': 'huggingface_api',
                    'batch_id': chunk.get('batch_id', '')
                }
                embedding_records.append(embedding_record)
            
            # Insertar embeddings
            self.supabase.table(self.tables['embeddings']).insert(embedding_records).execute()
    
    def log_metrics(self):
        """Registra métricas en TensorBoard (temporalmente deshabilitado)"""
        # TensorBoard temporalmente deshabilitado debido a problemas de compatibilidad con Python 3.12
        pass
        # if self.stats.start_time:
        #     elapsed_time = (datetime.now() - self.stats.start_time).total_seconds()
        #     
        #     self.tensorboard_writer.add_scalar('Documents/Processed', self.stats.documents_processed, elapsed_time)
        #     self.tensorboard_writer.add_scalar('Chunks/Created', self.stats.chunks_created, elapsed_time)
        #     self.tensorboard_writer.add_scalar('Embeddings/Generated', self.stats.embeddings_generated, elapsed_time)
        #     self.tensorboard_writer.add_scalar('Processing/Errors', self.stats.errors, elapsed_time)
        #     self.tensorboard_writer.add_scalar('Processing/Documents_per_second', 
        #                                      self.stats.documents_processed / elapsed_time, elapsed_time)
        #     
        #     # Métricas de GPU
        #     gpu_info = self.gpu_processor.memory_monitor.get_memory_info()
        #     self.tensorboard_writer.add_scalar('GPU/Memory_Used_GB', gpu_info['used_memory'] / 1024, elapsed_time)
        #     self.tensorboard_writer.add_scalar('GPU/Utilization_Percent', gpu_info['utilization'], elapsed_time)
        #     
        #     # Métricas de CPU
        #     cpu_percent = psutil.cpu_percent()
        #     self.tensorboard_writer.add_scalar('CPU/Usage_Percent', cpu_percent, elapsed_time)
    
    async def run_pipeline(self, max_files: Optional[int] = None):
        """Ejecuta el pipeline completo"""
        logger.info("Iniciando pipeline optimizado de SIEM...")
        self.stats.start_time = datetime.now()
        
        # Obtener archivos a procesar
        files_to_process = []
        for ext in self.processing_config['supported_extensions']:
            files_to_process.extend(self.siem_path.rglob(f'*{ext}'))
        
        if max_files:
            files_to_process = files_to_process[:max_files]
        
        logger.info(f"Procesando {len(files_to_process)} archivos en lotes de {self.processing_config['max_files_per_batch']}")
        
        # Procesar en lotes
        batch_size = self.processing_config['max_files_per_batch']
        for i in range(0, len(files_to_process), batch_size):
            batch_files = files_to_process[i:i + batch_size]
            logger.info(f"Procesando lote {i//batch_size + 1}/{(len(files_to_process) + batch_size - 1)//batch_size}")
            
            await self.process_document_batch(batch_files)
            
            # Log de métricas
            self.log_metrics()
            
            # Log de progreso
            logger.info(f"Progreso: {self.stats.documents_processed} documentos procesados, "
                       f"{self.stats.chunks_created} chunks creados, "
                       f"{self.stats.embeddings_generated} embeddings generados")
        
        self.stats.end_time = datetime.now()
        self.stats.total_processing_time = (self.stats.end_time - self.stats.start_time).total_seconds()
        
        logger.info("Pipeline completado exitosamente")
        self._print_final_stats()
    
    def _print_final_stats(self):
        """Imprime estadísticas finales"""
        print("\n" + "="*80)
        print("PIPELINE SIEM OPTIMIZADO - ESTADÍSTICAS FINALES")
        print("="*80)
        print(f"Documentos procesados: {self.stats.documents_processed}")
        print(f"Chunks creados: {self.stats.chunks_created}")
        print(f"Embeddings generados: {self.stats.embeddings_generated}")
        print(f"Errores: {self.stats.errors}")
        print(f"Tiempo total: {self.stats.total_processing_time:.2f} segundos")
        print(f"Documentos por segundo: {self.stats.documents_processed / self.stats.total_processing_time:.2f}")
        print(f"Chunks por segundo: {self.stats.chunks_created / self.stats.total_processing_time:.2f}")
        print("="*80)

def main():
    """Función principal"""
    siem_path = "/home/elias/Documentos/Embeddings/SIEM"
    
    if not os.path.exists(siem_path):
        logger.error(f"La ruta {siem_path} no existe")
        return
    
    # Crear directorios necesarios
    Path('logs').mkdir(exist_ok=True)
    # Path('logs/tensorboard').mkdir(exist_ok=True)  # Temporalmente deshabilitado
    
    # Ejecutar pipeline
    pipeline = OptimizedSIEMPipeline(siem_path)
    asyncio.run(pipeline.run_pipeline(max_files=10))  # Probar con 10 archivos

if __name__ == "__main__":
    main()
