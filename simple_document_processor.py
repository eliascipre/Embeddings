#!/usr/bin/env python3
"""
Procesador simplificado de documentos para RAG
Basado en el proyecto original exitoso
"""
import logging
import re
from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)

class SimpleDocumentProcessor:
    """Procesador simple y robusto para cualquier tipo de documento"""
    
    def __init__(self, max_chunk_size: int = 500, chunk_overlap: int = 100):  # Chunks más pequeños
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        
    def process_document(self, file_path: Path) -> List[Dict[str, Any]]:
        """Procesar cualquier documento y crear chunks simples"""
        try:
            logger.info(f"📄 Procesando: {file_path.name}")
            
            # Extraer texto
            text = self._extract_text(file_path)
            if not text or len(text.strip()) < 10:
                logger.warning(f"⚠️ Sin contenido útil en {file_path.name}")
                return []
            
            # Limpiar texto
            clean_text = self._clean_text(text)
            
            # Crear chunks simples
            chunks = self._create_simple_chunks(clean_text, file_path)
            
            logger.info(f"✅ {file_path.name}: {len(chunks)} chunks creados")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Error procesando {file_path}: {e}")
            return []
    
    def _extract_text(self, file_path: Path) -> str:
        """Extraer texto de cualquier archivo"""
        try:
            if file_path.suffix.lower() == '.pdf':
                return self._extract_pdf_text(file_path)
            elif file_path.suffix.lower() in ['.txt', '.md']:
                return self._extract_text_file(file_path)
            else:
                logger.warning(f"⚠️ Tipo de archivo no soportado: {file_path.suffix}")
                return ""
        except Exception as e:
            logger.error(f"❌ Error extrayendo texto de {file_path}: {e}")
            return ""
    
    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extraer texto de PDF con PyMuPDF optimizado para documentos grandes"""
        try:
            doc = fitz.open(str(file_path))
            text = ""
            
            # Procesar páginas en lotes para documentos grandes
            total_pages = len(doc)
            batch_size = 10  # Procesar 10 páginas a la vez
            
            for batch_start in range(0, total_pages, batch_size):
                batch_end = min(batch_start + batch_size, total_pages)
                
                for page_num in range(batch_start, batch_end):
                    try:
                        page = doc.load_page(page_num)
                        page_text = page.get_text()
                        if page_text.strip():
                            text += f"\n--- PÁGINA {page_num + 1} ---\n{page_text}\n"
                    except Exception as page_error:
                        logger.warning(f"⚠️ Error en página {page_num + 1}: {page_error}")
                        continue
                
                # Log de progreso para documentos grandes
                if total_pages > 50:
                    logger.info(f"📄 Procesadas páginas {batch_start + 1}-{batch_end} de {total_pages}")
            
            doc.close()
            return text
            
        except Exception as e:
            logger.error(f"❌ Error extrayendo PDF {file_path}: {e}")
            return ""
    
    def _extract_text_file(self, file_path: Path) -> str:
        """Extraer texto de archivo de texto"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            logger.error(f"❌ Error leyendo archivo de texto {file_path}: {e}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """Limpiar texto de manera robusta"""
        if not text:
            return ""
        
        # Limpiar caracteres Unicode problemáticos
        text = text.replace('\u0000', '')  # Eliminar caracteres nulos
        text = text.replace('\x00', '')    # Eliminar bytes nulos
        
        # Eliminar caracteres de control problemáticos
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Eliminar secuencias de escape Unicode problemáticas
        text = re.sub(r'\\u[0-9A-Fa-f]{4}', '', text)
        text = re.sub(r'\\x[0-9A-Fa-f]{2}', '', text)
        
        # Normalizar espacios y saltos de línea
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        # Limpiar espacios al inicio y final
        text = text.strip()
        
        return text
    
    def _create_simple_chunks(self, text: str, file_path: Path) -> List[Dict[str, Any]]:
        """Crear chunks simples dividiendo por caracteres con overlap optimizado"""
        chunks = []
        
        # Si el texto es menor al tamaño máximo, crear un solo chunk
        if len(text) <= self.max_chunk_size:
            if text.strip():
                chunks.append(self._create_chunk_dict(
                    text.strip(), 
                    0, 
                    file_path
                ))
            return chunks
        
        # Dividir por caracteres con overlap optimizado
        start = 0
        chunk_index = 0
        text_length = len(text)
        
        # Log de progreso para textos muy grandes
        if text_length > 100000:  # Más de 100KB de texto
            logger.info(f"📝 Creando chunks para texto de {text_length:,} caracteres")
        
        while start < text_length:
            # Calcular el final del chunk
            end = start + self.max_chunk_size
            
            # Si no es el último chunk, buscar un buen punto de corte
            if end < text_length:
                # Buscar el último punto, coma o salto de línea dentro del chunk
                chunk_text = text[start:end]
                
                # Buscar puntos de corte naturales (optimizado)
                cut_points = [
                    chunk_text.rfind('. '),
                    chunk_text.rfind('\n'),
                    chunk_text.rfind(', '),
                    chunk_text.rfind('; '),
                    chunk_text.rfind(': '),
                    chunk_text.rfind(' '),  # Cualquier espacio como último recurso
                ]
                
                # Usar el mejor punto de corte encontrado
                best_cut = max([p for p in cut_points if p > 0])
                if best_cut > 0:
                    end = start + best_cut + 1  # +1 para incluir el carácter de corte
            
            # Extraer el chunk
            chunk_text = text[start:end].strip()
            
            if chunk_text and len(chunk_text) >= 30:  # Mínimo 30 caracteres (reducido)
                chunks.append(self._create_chunk_dict(
                    chunk_text, 
                    chunk_index, 
                    file_path
                ))
                chunk_index += 1
                
                # Log de progreso cada 100 chunks
                if chunk_index % 100 == 0:
                    logger.info(f"📝 Creados {chunk_index} chunks...")
            
            # Mover el inicio considerando el overlap
            start = end - self.chunk_overlap
            if start >= text_length:
                break
        
        logger.info(f"✅ Total de chunks creados: {len(chunks)}")
        return chunks
    
    def _create_chunk_dict(self, content: str, index: int, file_path: Path) -> Dict[str, Any]:
        """Crear diccionario de chunk con metadatos básicos"""
        # Limpiar contenido para búsqueda
        clean_content = self._clean_text_for_search(content)
        
        return {
            'text': content,
            'type': 'paragraph',  # Usar tipo válido para RAG
            'index': index,
            'file_path': str(file_path),
            'file_name': file_path.name,
            'clean_text': clean_content,
            'hierarchy': self._extract_simple_hierarchy(content),
            'chunk_id': f"{file_path.stem}_{index}",
            'hierarchy_level': 1,
            'parent_chunk_id': None,
            'article_number': None,
            'paragraph_number': str(index + 1),
            'inciso_number': None,
            'chapter_title': None,
            'section_title': None,
            'law_title': None,
            'page_number': self._extract_page_number(content),
            'word_count': len(content.split()),
            'char_count': len(content),
            'is_compressed': False,
            'compression_ratio': 1.0,
            'metadata': {
                'file_name': file_path.name,  # Usar file_name en lugar de filename
                'file_type': file_path.suffix.lower(),
                'processed_at': datetime.now().isoformat(),
                'chunking_method': 'simple_paragraph'
            }
        }
    
    def _clean_text_for_search(self, text: str) -> str:
        """Limpiar texto específicamente para búsqueda"""
        if not text:
            return ""
        
        # Convertir a minúsculas para búsqueda
        text = text.lower()
        
        # Remover caracteres especiales pero mantener espacios
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Normalizar espacios
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _extract_simple_hierarchy(self, content: str) -> str:
        """Extraer jerarquía simple del contenido"""
        content_upper = content.upper()
        
        if 'ARTÍCULO' in content_upper:
            return 'articulo'
        elif 'CAPÍTULO' in content_upper:
            return 'capitulo'
        elif 'TÍTULO' in content_upper:
            return 'titulo'
        elif 'SECCIÓN' in content_upper:
            return 'seccion'
        else:
            return 'paragrafo'
    
    def _extract_page_number(self, content: str) -> int:
        """Extraer número de página del contenido"""
        page_match = re.search(r'--- PÁGINA (\d+) ---', content)
        if page_match:
            return int(page_match.group(1))
        return 1
