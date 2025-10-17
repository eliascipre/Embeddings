"""
Procesador de chunking legal inteligente para documentos legales mexicanos
Preserva jerarquía legal y optimiza para RAG
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import fitz  # PyMuPDF
import json

logger = logging.getLogger(__name__)

@dataclass
class LegalChunk:
    """Estructura de un chunk legal con metadatos completos"""
    content: str
    chunk_type: str  # articulo, paragrafo, inciso, titulo, capitulo, etc.
    article_number: Optional[str] = None
    paragraph_number: Optional[str] = None
    inciso_number: Optional[str] = None
    chapter_title: Optional[str] = None
    section_title: Optional[str] = None
    law_title: Optional[str] = None
    page_number: Optional[int] = None
    word_count: int = 0
    char_count: int = 0
    metadata: Dict[str, Any] = None

class LegalChunkingProcessor:
    """Procesador inteligente de chunking para documentos legales"""
    
    def __init__(self):
        self.legal_patterns = {
            # Patrones para artículos
            'articulo': re.compile(r'ARTÍCULO\s+(\d+[A-Za-z]*)', re.IGNORECASE),
            'articulo_romano': re.compile(r'ARTÍCULO\s+([IVX]+)', re.IGNORECASE),
            
            # Patrones para capítulos
            'capitulo': re.compile(r'CAPÍTULO\s+([IVX]+|\d+)', re.IGNORECASE),
            'capitulo_romano': re.compile(r'CAPÍTULO\s+([IVX]+)', re.IGNORECASE),
            
            # Patrones para títulos
            'titulo': re.compile(r'TÍTULO\s+([IVX]+|\d+)', re.IGNORECASE),
            'titulo_romano': re.compile(r'TÍTULO\s+([IVX]+)', re.IGNORECASE),
            
            # Patrones para secciones
            'seccion': re.compile(r'SECCIÓN\s+([IVX]+|\d+)', re.IGNORECASE),
            'seccion_romano': re.compile(r'SECCIÓN\s+([IVX]+)', re.IGNORECASE),
            
            # Patrones para párrafos
            'paragrafo': re.compile(r'PÁRRAFO\s+([IVX]+|\d+)', re.IGNORECASE),
            'paragrafo_romano': re.compile(r'PÁRRAFO\s+([IVX]+)', re.IGNORECASE),
            
            # Patrones para incisos
            'inciso': re.compile(r'(\d+[A-Za-z]*)\.', re.IGNORECASE),
            'inciso_letra': re.compile(r'([A-Za-z])\)', re.IGNORECASE),
            'inciso_romano': re.compile(r'([IVX]+)\.', re.IGNORECASE),
            
            # Patrones para fracciones
            'fraccion': re.compile(r'FRACCIÓN\s+([IVX]+|\d+)', re.IGNORECASE),
            
            # Patrones para transitorios
            'transitorio': re.compile(r'TRANSITORIO\s+([IVX]+|\d+)', re.IGNORECASE),
            
            # Patrones para definiciones
            'definicion': re.compile(r'DEFINICIONES?', re.IGNORECASE),
            
            # Patrones para disposiciones
            'disposicion': re.compile(r'DISPOSICIONES?\s+(FINALES?|TRANSITORIAS?)', re.IGNORECASE),
        }
        
        # Palabras clave para detectar tipos de ley
        self.law_type_keywords = {
            'constitucion': ['constitución', 'constitucional', 'carta magna'],
            'codigo': ['código', 'codigo'],
            'ley': ['ley', 'legislación'],
            'reglamento': ['reglamento', 'reglamentario'],
            'decreto': ['decreto', 'decretado'],
            'acuerdo': ['acuerdo', 'acordado'],
            'circular': ['circular'],
            'oficio': ['oficio'],
            'resolucion': ['resolución', 'resolucion'],
            'norma': ['norma', 'normativa'],
            'manual': ['manual'],
            'guia': ['guía', 'guia'],
            'procedimiento': ['procedimiento'],
            'protocolo': ['protocolo']
        }
        
        # Estados de México
        self.mexican_states = [
            'aguascalientes', 'baja_california', 'baja_california_sur', 'campeche',
            'chiapas', 'chihuahua', 'ciudad_de_mexico', 'coahuila', 'colima',
            'durango', 'guanajuato', 'guerrero', 'hidalgo', 'jalisco', 'mexico',
            'michoacan', 'morelos', 'nayarit', 'nuevo_leon', 'oaxaca', 'puebla',
            'queretaro', 'quintana_roo', 'san_luis_potosi', 'sinaloa', 'sonora',
            'tabasco', 'tamaulipas', 'tlaxcala', 'veracruz', 'yucatan', 'zacatecas'
        ]
    
    def process_document(self, file_path: Path, state: str = None) -> Dict[str, Any]:
        """Procesar un documento legal completo"""
        try:
            logger.info(f"📄 Procesando documento legal: {file_path.name}")
            
            # Extraer texto del PDF
            content = self._extract_text_from_pdf(file_path)
            if not content:
                raise ValueError(f"No se pudo extraer texto del archivo: {file_path}")
            
            # Detectar estado si no se proporciona
            if not state:
                state = self._detect_state_from_path(file_path)
            
            # Detectar tipo de ley
            law_type = self._detect_law_type(file_path.name, content)
            
            # Procesar contenido con chunking legal
            chunks = self._chunk_legal_content(content, file_path.name)
            
            # Crear metadatos del documento
            document_metadata = self._create_document_metadata(
                file_path, state, law_type, content, chunks
            )
            
            logger.info(f"✅ Documento procesado: {len(chunks)} chunks generados")
            
            return {
                'document_metadata': document_metadata,
                'chunks': chunks,
                'total_chunks': len(chunks),
                'processing_stats': {
                    'total_pages': document_metadata.get('total_pages', 0),
                    'total_words': sum(chunk.word_count for chunk in chunks),
                    'total_chars': sum(chunk.char_count for chunk in chunks),
                    'articles_count': len([c for c in chunks if c.chunk_type == 'articulo']),
                    'chapters_count': len([c for c in chunks if c.chunk_type == 'capitulo']),
                    'paragraphs_count': len([c for c in chunks if c.chunk_type == 'paragrafo'])
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error procesando documento {file_path}: {e}")
            raise
    
    def _extract_text_from_pdf(self, file_path: Path) -> str:
        """Extraer texto de un PDF con información de páginas"""
        try:
            doc = fitz.open(str(file_path))
            content = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                if text.strip():
                    content += f"\n--- PÁGINA {page_num + 1} ---\n{text}\n"
            
            doc.close()
            return content
            
        except Exception as e:
            logger.error(f"❌ Error extrayendo texto de {file_path}: {e}")
            return ""
    
    def _detect_state_from_path(self, file_path: Path) -> str:
        """Detectar estado basado en la ruta del archivo"""
        path_parts = file_path.parts
        
        for part in path_parts:
            part_lower = part.lower().replace(' ', '_').replace('-', '_')
            if part_lower in self.mexican_states:
                return part_lower
        
        return 'unknown'
    
    def _detect_law_type(self, filename: str, content: str) -> str:
        """Detectar tipo de ley basado en nombre y contenido"""
        filename_lower = filename.lower()
        content_lower = content.lower()
        
        # Buscar en el nombre del archivo primero
        for law_type, keywords in self.law_type_keywords.items():
            for keyword in keywords:
                if keyword in filename_lower:
                    return law_type
        
        # Buscar en el contenido
        for law_type, keywords in self.law_type_keywords.items():
            for keyword in keywords:
                if keyword in content_lower:
                    return law_type
        
        return 'ley'  # Default
    
    def _chunk_legal_content(self, content: str, filename: str) -> List[LegalChunk]:
        """Chunking inteligente preservando jerarquía legal"""
        chunks = []
        
        # Dividir por páginas primero
        pages = content.split('--- PÁGINA')
        current_page = 1
        
        for page_content in pages:
            if not page_content.strip():
                continue
            
            # Extraer número de página
            page_match = re.search(r'(\d+)\s+---', page_content)
            if page_match:
                current_page = int(page_match.group(1))
            
            # Procesar contenido de la página
            page_chunks = self._process_page_content(page_content, current_page, filename)
            chunks.extend(page_chunks)
        
        # Post-procesamiento para mejorar chunks
        chunks = self._post_process_chunks(chunks)
        
        return chunks
    
    def _process_page_content(self, content: str, page_num: int, filename: str) -> List[LegalChunk]:
        """Procesar contenido de una página específica"""
        chunks = []
        
        # Limpiar contenido
        content = content.replace('--- PÁGINA', '').strip()
        if not content:
            return chunks
        
        # Dividir por artículos primero
        articles = re.split(r'ARTÍCULO\s+\d+[A-Za-z]*', content)
        
        if len(articles) > 1:
            # Procesar cada artículo
            for i, article_content in enumerate(articles[1:], 1):
                article_chunks = self._process_article(article_content, i, page_num, filename)
                chunks.extend(article_chunks)
        else:
            # Si no hay artículos, procesar por otros patrones
            other_chunks = self._process_non_article_content(content, page_num, filename)
            chunks.extend(other_chunks)
        
        return chunks
    
    def _process_article(self, content: str, article_num: int, page_num: int, filename: str) -> List[LegalChunk]:
        """Procesar un artículo específico"""
        chunks = []
        
        # Crear chunk del artículo completo
        article_chunk = LegalChunk(
            content=content.strip(),
            chunk_type='articulo',
            article_number=str(article_num),
            page_number=page_num,
            word_count=len(content.split()),
            char_count=len(content),
            metadata={
                'filename': filename,
                'article_number': str(article_num),
                'page_number': page_num
            }
        )
        chunks.append(article_chunk)
        
        # Dividir artículo en párrafos
        paragraphs = re.split(r'PÁRRAFO\s+[IVX\d]+', content)
        
        for i, paragraph in enumerate(paragraphs[1:], 1):
            if len(paragraph.strip()) > 50:  # Filtrar párrafos muy cortos
                para_chunk = LegalChunk(
                    content=paragraph.strip(),
                    chunk_type='paragrafo',
                    article_number=str(article_num),
                    paragraph_number=str(i),
                    page_number=page_num,
                    word_count=len(paragraph.split()),
                    char_count=len(paragraph),
                    metadata={
                        'filename': filename,
                        'article_number': str(article_num),
                        'paragraph_number': str(i),
                        'page_number': page_num
                    }
                )
                chunks.append(para_chunk)
        
        # Dividir párrafos en incisos
        for para_chunk in chunks:
            if para_chunk.chunk_type == 'paragrafo':
                incisos = self._extract_incisos(para_chunk.content)
                for i, inciso in enumerate(incisos, 1):
                    if len(inciso.strip()) > 20:
                        inciso_chunk = LegalChunk(
                            content=inciso.strip(),
                            chunk_type='inciso',
                            article_number=para_chunk.article_number,
                            paragraph_number=para_chunk.paragraph_number,
                            inciso_number=str(i),
                            page_number=page_num,
                            word_count=len(inciso.split()),
                            char_count=len(inciso),
                            metadata={
                                'filename': filename,
                                'article_number': para_chunk.article_number,
                                'paragraph_number': para_chunk.paragraph_number,
                                'inciso_number': str(i),
                                'page_number': page_num
                            }
                        )
                        chunks.append(inciso_chunk)
        
        return chunks
    
    def _process_non_article_content(self, content: str, page_num: int, filename: str) -> List[LegalChunk]:
        """Procesar contenido que no tiene estructura de artículos"""
        chunks = []
        
        # Buscar capítulos
        chapters = re.split(r'CAPÍTULO\s+[IVX\d]+', content)
        if len(chapters) > 1:
            for i, chapter_content in enumerate(chapters[1:], 1):
                if len(chapter_content.strip()) > 100:
                    chapter_chunk = LegalChunk(
                        content=chapter_content.strip(),
                        chunk_type='capitulo',
                        chapter_title=f"Capítulo {i}",
                        page_number=page_num,
                        word_count=len(chapter_content.split()),
                        char_count=len(chapter_content),
                        metadata={
                            'filename': filename,
                            'chapter_number': str(i),
                            'page_number': page_num
                        }
                    )
                    chunks.append(chapter_chunk)
        
        # Buscar títulos
        titles = re.split(r'TÍTULO\s+[IVX\d]+', content)
        if len(titles) > 1:
            for i, title_content in enumerate(titles[1:], 1):
                if len(title_content.strip()) > 100:
                    title_chunk = LegalChunk(
                        content=title_content.strip(),
                        chunk_type='titulo',
                        law_title=f"Título {i}",
                        page_number=page_num,
                        word_count=len(title_content.split()),
                        char_count=len(title_content),
                        metadata={
                            'filename': filename,
                            'title_number': str(i),
                            'page_number': page_num
                        }
                    )
                    chunks.append(title_chunk)
        
        # Si no hay estructura clara, dividir por párrafos
        if not chunks:
            paragraphs = content.split('\n\n')
            for i, paragraph in enumerate(paragraphs):
                if len(paragraph.strip()) > 100:
                    para_chunk = LegalChunk(
                        content=paragraph.strip(),
                        chunk_type='paragrafo',
                        page_number=page_num,
                        word_count=len(paragraph.split()),
                        char_count=len(paragraph),
                        metadata={
                            'filename': filename,
                            'paragraph_number': str(i + 1),
                            'page_number': page_num
                        }
                    )
                    chunks.append(para_chunk)
        
        return chunks
    
    def _extract_incisos(self, content: str) -> List[str]:
        """Extraer incisos de un párrafo"""
        incisos = []
        
        # Patrones para incisos
        patterns = [
            r'(\d+[A-Za-z]*\.\s+[^0-9]+?)(?=\d+[A-Za-z]*\.|$)',
            r'([A-Za-z]\)\s+[^A-Za-z]+?)(?=[A-Za-z]\)|$)',
            r'([IVX]+\.\s+[^IVX]+?)(?=[IVX]+\.|$)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            incisos.extend(matches)
        
        return incisos
    
    def _post_process_chunks(self, chunks: List[LegalChunk]) -> List[LegalChunk]:
        """Post-procesamiento para mejorar calidad de chunks"""
        processed_chunks = []
        
        for chunk in chunks:
            # Filtrar chunks muy cortos
            if chunk.word_count < 10:
                continue
            
            # Limpiar contenido
            chunk.content = self._clean_content(chunk.content)
            
            # Actualizar conteos
            chunk.word_count = len(chunk.content.split())
            chunk.char_count = len(chunk.content)
            
            # Agregar metadatos adicionales
            if not chunk.metadata:
                chunk.metadata = {}
            
            chunk.metadata.update({
                'processed_at': datetime.now().isoformat(),
                'chunk_type': chunk.chunk_type,
                'word_count': chunk.word_count,
                'char_count': chunk.char_count
            })
            
            processed_chunks.append(chunk)
        
        return processed_chunks
    
    def _clean_content(self, content: str) -> str:
        """Limpiar contenido del chunk"""
        # Remover caracteres especiales excesivos
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'\n+', '\n', content)
        
        # Remover números de página
        content = re.sub(r'--- PÁGINA \d+ ---', '', content)
        
        # Limpiar espacios
        content = content.strip()
        
        return content
    
    def _create_document_metadata(
        self, 
        file_path: Path, 
        state: str, 
        law_type: str, 
        content: str, 
        chunks: List[LegalChunk]
    ) -> Dict[str, Any]:
        """Crear metadatos completos del documento"""
        # Contar páginas
        page_count = len(set(chunk.page_number for chunk in chunks if chunk.page_number))
        
        # Extraer información adicional
        title = file_path.stem
        file_size = file_path.stat().st_size if file_path.exists() else 0
        
        # Detectar fechas
        effective_date = self._extract_effective_date(content)
        amendment_date = self._extract_amendment_date(content)
        
        # Detectar estado de la ley
        status = self._detect_law_status(content)
        
        return {
            'document_id': f"{state}_{file_path.stem}",
            'title': title,
            'state': state,
            'document_type': 'pdf',
            'law_type': law_type,
            'effective_date': effective_date,
            'amendment_date': amendment_date,
            'status': status,
            'file_path': str(file_path),
            'file_size': file_size,
            'total_pages': page_count,
            'total_chunks': len(chunks),
            'metadata': {
                'filename': file_path.name,
                'processed_at': datetime.now().isoformat(),
                'articles_count': len([c for c in chunks if c.chunk_type == 'articulo']),
                'chapters_count': len([c for c in chunks if c.chunk_type == 'capitulo']),
                'paragraphs_count': len([c for c in chunks if c.chunk_type == 'paragrafo']),
                'incisos_count': len([c for c in chunks if c.chunk_type == 'inciso']),
                'total_words': sum(chunk.word_count for chunk in chunks),
                'total_chars': sum(chunk.char_count for chunk in chunks)
            }
        }
    
    def _extract_effective_date(self, content: str) -> Optional[str]:
        """Extraer fecha de vigencia de la ley"""
        patterns = [
            r'vigente\s+desde\s+el?\s+(\d{1,2}\s+de\s+\w+\s+de\s+\d{4})',
            r'entra\s+en\s+vigor\s+el?\s+(\d{1,2}\s+de\s+\w+\s+de\s+\d{4})',
            r'vigencia\s+(\d{1,2}\s+de\s+\w+\s+de\s+\d{4})',
            r'(\d{1,2}\s+de\s+\w+\s+de\s+\d{4})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                return self._convert_spanish_date_to_iso(date_str)
        
        return None
    
    def _extract_amendment_date(self, content: str) -> Optional[str]:
        """Extraer fecha de última reforma"""
        patterns = [
            r'última\s+reforma\s+(\d{1,2}\s+de\s+\w+\s+de\s+\d{4})',
            r'reformado\s+el?\s+(\d{1,2}\s+de\s+\w+\s+de\s+\d{4})',
            r'última\s+modificación\s+(\d{1,2}\s+de\s+\w+\s+de\s+\d{4})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                return self._convert_spanish_date_to_iso(date_str)
        
        return None
    
    def _convert_spanish_date_to_iso(self, date_str: str) -> Optional[str]:
        """Convertir fecha española a formato ISO (YYYY-MM-DD)"""
        try:
            # Limpiar la fecha
            date_str = re.sub(r'\s+', ' ', date_str.strip())
            
            # Mapeo de meses en español
            months = {
                'enero': '01', 'febrero': '02', 'marzo': '03', 'abril': '04',
                'mayo': '05', 'junio': '06', 'julio': '07', 'agosto': '08',
                'septiembre': '09', 'octubre': '10', 'noviembre': '11', 'diciembre': '12'
            }
            
            # Patrones de fecha
            patterns = [
                r'(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})',
                r'(\d{1,2})\s+DE\s+(\w+)\s+DE\s+(\d{4})',
                r'(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, date_str, re.IGNORECASE)
                if match:
                    day = match.group(1).zfill(2)
                    month_name = match.group(2).lower()
                    year = match.group(3)
                    
                    if month_name in months:
                        month = months[month_name]
                        return f"{year}-{month}-{day}"
            
            return None
            
        except Exception as e:
            logger.warning(f"Error convirtiendo fecha '{date_str}': {e}")
            return None
    
    def _detect_law_status(self, content: str) -> str:
        """Detectar estado de la ley"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['derogada', 'abrogada', 'repealed']):
            return 'derogada'
        elif any(word in content_lower for word in ['reformada', 'modificada', 'amended']):
            return 'reformada'
        else:
            return 'vigente'
