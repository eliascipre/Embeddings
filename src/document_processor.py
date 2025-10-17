"""
Procesador de documentos legales con RAG-Anything y Unstructured como fallback
"""
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime

# RAG-Anything imports
try:
    from raganything import RAGAnything
    from raganything.utils import DocumentProcessor
    RAG_ANYTHING_AVAILABLE = True
except ImportError:
    RAG_ANYTHING_AVAILABLE = False
    logging.warning("RAG-Anything no disponible, usando Unstructured como fallback")

# Unstructured imports
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from unstructured.staging.base import elements_to_json

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from config import config

logger = logging.getLogger(__name__)

class LegalDocumentProcessor:
    """Procesador avanzado de documentos legales con RAG-Anything y Unstructured"""
    
    def __init__(self):
        self.rag_anything = None
        self.unstructured_available = True
        self._setup_processors()
        self._setup_file_logger()
        
    def _setup_processors(self):
        """Configurar procesadores de documentos"""
        try:
            if RAG_ANYTHING_AVAILABLE:
                # Configurar RAG-Anything con la API correcta
                self.rag_anything = RAGAnything(
                    api_key=config.openai_api_key,  # Necesario para LLM
                    base_url=config.openai_base_url,  # Opcional
                    model_name=config.embedding_model,
                    device=config.embedding_device,
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                    output_dir=str(config.cache_directory / "rag_anything_output")
                )
                logger.info("✅ RAG-Anything configurado correctamente")
            else:
                logger.warning("⚠️ RAG-Anything no disponible, usando Unstructured")
                
        except Exception as e:
            logger.error(f"❌ Error configurando RAG-Anything: {e}")
            self.rag_anything = None
            
        # Configurar LangChain como fallback
        self.langchain_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=[
                "\n\nARTÍCULO",     # Separar por artículos
                "\n\nCAPÍTULO",     # Separar por capítulos
                "\n\nTÍTULO",       # Separar por títulos
                "\n\n",             # Párrafos
                "\n",               # Líneas
                ". ",               # Oraciones
                " ",                # Palabras
                ""                  # Caracteres
            ]
        )
    
    def _setup_file_logger(self):
        """Configurar logger para archivos individuales"""
        self.file_logger = logging.getLogger('file_processing')
        file_handler = logging.FileHandler(
            config.logs_dir / f"pdf_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.file_logger.addHandler(file_handler)
        self.file_logger.setLevel(logging.INFO)
        
    async def process_pdf_with_rag_anything(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Procesar PDF con RAG-Anything"""
        if not self.rag_anything:
            raise ValueError("RAG-Anything no está disponible")
            
        try:
            logger.info(f"🔄 Procesando {pdf_path.name} con RAG-Anything...")
            
            # Procesar documento con RAG-Anything usando la API correcta
            result = await self.rag_anything.process_document_complete(
                file_path=str(pdf_path),
                output_dir=str(config.cache_directory / "rag_anything_output"),
                parse_method="auto",  # auto, ocr, txt
                parser="mineru",      # mineru o docling
                display_stats=True,
                doc_id=f"legal_doc_{pdf_path.stem}"
            )
            
            # Extraer contenido procesado
            processed_chunks = []
            if hasattr(result, 'content') and result.content:
                # Si el resultado tiene contenido directo
                content = result.content
                chunks = self._split_content_into_chunks(content)
                
                for i, chunk in enumerate(chunks):
                    chunk_data = {
                        "content": chunk,
                        "metadata": {
                            "source": str(pdf_path),
                            "type": "legal_document",
                            "processed_at": datetime.now().isoformat(),
                            "processor": "rag_anything",
                            "chunk_id": i,
                            "chunk_size": len(chunk),
                            "parser": "mineru",
                            "parse_method": "auto"
                        }
                    }
                    processed_chunks.append(chunk_data)
            else:
                # Fallback si no hay contenido directo
                logger.warning(f"⚠️ No se pudo extraer contenido de {pdf_path.name}, usando fallback")
                return await self.process_pdf_with_unstructured(pdf_path)
                
            logger.info(f"✅ RAG-Anything procesó {len(processed_chunks)} chunks de {pdf_path.name}")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"❌ Error procesando con RAG-Anything {pdf_path}: {e}")
            # Fallback a Unstructured
            return await self.process_pdf_with_unstructured(pdf_path)
    
    def _split_content_into_chunks(self, content: str) -> List[str]:
        """Dividir contenido en chunks usando el splitter semántico"""
        return self.semantic_splitter.split_text(content)
    
    async def process_pdf_with_unstructured(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Procesar PDF con Unstructured como fallback"""
        try:
            # Log detallado solo en archivo, no en consola
            self.file_logger.info(f"🔄 INICIO - Procesando {pdf_path.name} con Unstructured")
            
            # Procesar PDF con Unstructured
            elements = partition_pdf(
                filename=str(pdf_path),
                strategy="hi_res",  # Estrategia de alta resolución
                infer_table_structure=True,
                extract_images_in_pdf=False,
                include_page_breaks=True,
                languages=["spa", "es"]  # Configurar para español
            )
            
            # Chunking inteligente por títulos
            chunks = chunk_by_title(
                elements,
                max_characters=config.chunk_size,
                new_after_n_chars=config.chunk_size - config.chunk_overlap,
                combine_text_under_n_chars=100
            )
            
            # Convertir a formato estándar
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    "content": str(chunk),
                    "metadata": {
                        "source": str(pdf_path),
                        "type": "legal_document",
                        "processed_at": datetime.now().isoformat(),
                        "processor": "unstructured",
                        "chunk_id": i,
                        "chunk_size": len(str(chunk)),
                        "element_type": getattr(chunk, 'category', 'unknown')
                    }
                }
                processed_chunks.append(chunk_data)
                
            # Log detallado solo en archivo
            self.file_logger.info(f"✅ FIN - {pdf_path.name}: {len(processed_chunks)} chunks generados con Unstructured")
            return processed_chunks
            
        except Exception as e:
            self.file_logger.error(f"❌ ERROR - {pdf_path.name}: {str(e)}")
            # Fallback a LangChain
            return await self.process_pdf_with_langchain(pdf_path)
    
    async def process_pdf_with_langchain(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Procesar PDF con LangChain como último recurso"""
        try:
            logger.info(f"🔄 Procesando {pdf_path.name} con LangChain...")
            
            # Cargar PDF con LangChain
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            
            # Procesar cada página
            processed_chunks = []
            for page_num, page in enumerate(pages):
                if not page.page_content.strip():
                    continue
                    
                # Chunking semántico
                chunks = self.langchain_splitter.split_text(page.page_content)
                
                for chunk_num, chunk in enumerate(chunks):
                    if len(chunk.strip()) < 50:  # Filtrar chunks muy pequeños
                        continue
                        
                    chunk_data = {
                        "content": chunk.strip(),
                        "metadata": {
                            "source": str(pdf_path),
                            "type": "legal_document",
                            "processed_at": datetime.now().isoformat(),
                            "processor": "langchain",
                            "page_number": page_num + 1,
                            "chunk_id": chunk_num,
                            "chunk_size": len(chunk.strip())
                        }
                    }
                    processed_chunks.append(chunk_data)
                    
            logger.info(f"✅ LangChain procesó {len(processed_chunks)} chunks de {pdf_path.name}")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"❌ Error procesando con LangChain {pdf_path}: {e}")
            raise
    
    async def process_document(self, pdf_path: Path, prefer_rag_anything: bool = True) -> List[Dict[str, Any]]:
        """Procesar documento con el mejor procesador disponible"""
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {pdf_path}")
            
        if not pdf_path.suffix.lower() == '.pdf':
            raise ValueError(f"El archivo debe ser un PDF: {pdf_path}")
        
        # Intentar con RAG-Anything primero si está disponible y preferido
        if prefer_rag_anything and self.rag_anything:
            try:
                return await self.process_pdf_with_rag_anything(pdf_path)
            except Exception as e:
                logger.warning(f"⚠️ RAG-Anything falló, usando Unstructured: {e}")
                return await self.process_pdf_with_unstructured(pdf_path)
        
        # Usar Unstructured como primera opción
        try:
            return await self.process_pdf_with_unstructured(pdf_path)
        except Exception as e:
            logger.warning(f"⚠️ Unstructured falló, usando LangChain: {e}")
            return await self.process_pdf_with_langchain(pdf_path)
    
    def get_processor_info(self) -> Dict[str, Any]:
        """Obtener información sobre los procesadores disponibles"""
        return {
            "rag_anything_available": RAG_ANYTHING_AVAILABLE and self.rag_anything is not None,
            "unstructured_available": self.unstructured_available,
            "langchain_available": True,
            "preferred_processor": "rag_anything" if (RAG_ANYTHING_AVAILABLE and self.rag_anything) else "unstructured",
            "config": {
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "embedding_model": config.embedding_model,
                "device": config.embedding_device
            }
        }
