#!/usr/bin/env python3
"""
Procesador de Documentos de Comercio Exterior para SIEM
Chunking inteligente para documentos de aduanas, tratados, reglamentos, etc.
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import fitz  # PyMuPDF
import docx
from docx2txt import process as docx2txt_process
import pdfplumber
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
try:
    from .smart_legal_chunker import SmartLegalChunker, LegalChunk
except ImportError:
    from smart_legal_chunker import SmartLegalChunker, LegalChunk

logger = logging.getLogger(__name__)

class ComercioExteriorProcessor:
    """Procesador especializado para documentos de comercio exterior"""
    
    def __init__(self):
        # Inicializar chunker inteligente
        self.smart_chunker = SmartLegalChunker(max_chunk_size=2000, compression_threshold=1500)
        
        # Patrones específicos para documentos de comercio exterior
        self.comercio_patterns = {
            # Documentos oficiales
            'dof': re.compile(r'DOF\s*(\d{2}\d{2}\d{2})', re.IGNORECASE),
            'resolucion': re.compile(r'RESOLUCIÓN\s+([A-Za-z\s]+)', re.IGNORECASE),
            'reglamento': re.compile(r'REGLAMENTO\s+([A-Za-z\s]+)', re.IGNORECASE),
            'ley': re.compile(r'LEY\s+([A-Za-z\s]+)', re.IGNORECASE),
            'codigo': re.compile(r'CÓDIGO\s+([A-Za-z\s]+)', re.IGNORECASE),
            'decreto': re.compile(r'DECRETO\s+([A-Za-z\s]+)', re.IGNORECASE),
            'acuerdo': re.compile(r'ACUERDO\s+([A-Za-z\s]+)', re.IGNORECASE),
            'convenio': re.compile(r'CONVENIO\s+([A-Za-z\s]+)', re.IGNORECASE),
            'tratado': re.compile(r'TRATADO\s+([A-Za-z\s]+)', re.IGNORECASE),
            
            # Estructura de documentos
            'capitulo': re.compile(r'CAPÍTULO\s+([IVX]+|\d+)', re.IGNORECASE),
            'titulo': re.compile(r'TÍTULO\s+([IVX]+|\d+)', re.IGNORECASE),
            'seccion': re.compile(r'SECCIÓN\s+([IVX]+|\d+)', re.IGNORECASE),
            'articulo': re.compile(r'ARTÍCULO\s+(\d+[A-Za-z]*)', re.IGNORECASE),
            'paragrafo': re.compile(r'PÁRRAFO\s+([IVX]+|\d+)', re.IGNORECASE),
            'inciso': re.compile(r'(\d+[A-Za-z]*\.\s+[^0-9]+?)(?=\d+[A-Za-z]*\.|$)', re.IGNORECASE),
            'fraccion': re.compile(r'FRACCIÓN\s+([IVX]+|\d+)', re.IGNORECASE),
            
            # Anexos y anexos
            'anexo': re.compile(r'ANEXO\s+([IVX]+|\d+)', re.IGNORECASE),
            'apendice': re.compile(r'APÉNDICE\s+([IVX]+|\d+)', re.IGNORECASE),
            
            # Comercio exterior específico
            'clasificacion_arancelaria': re.compile(r'CLASIFICACIÓN\s+ARANCELARIA', re.IGNORECASE),
            'valor_aduana': re.compile(r'VALOR\s+EN\s+ADUANA', re.IGNORECASE),
            'origen_mercancia': re.compile(r'ORIGEN\s+DE\s+LA\s+MERCANCÍA', re.IGNORECASE),
            'destino_mercancia': re.compile(r'DESTINO\s+DE\s+LA\s+MERCANCÍA', re.IGNORECASE),
            'tipo_operacion': re.compile(r'TIPO\s+DE\s+OPERACIÓN', re.IGNORECASE),
            'regimen_aduana': re.compile(r'RÉGIMEN\s+ADUANERO', re.IGNORECASE),
            
            # Documentos de transporte
            'bill_of_lading': re.compile(r'BILL\s+OF\s+LADING', re.IGNORECASE),
            'air_waybill': re.compile(r'AIR\s+WAYBILL', re.IGNORECASE),
            'conocimiento_embarque': re.compile(r'CONOCIMIENTO\s+DE\s+EMBARQUE', re.IGNORECASE),
            'certificado_origen': re.compile(r'CERTIFICADO\s+DE\s+ORIGEN', re.IGNORECASE),
            'factura_comercial': re.compile(r'FACTURA\s+COMERCIAL', re.IGNORECASE),
            
            # Tratados internacionales
            'tlc': re.compile(r'TLC\s+([A-Za-z\s]+)', re.IGNORECASE),
            'tmec': re.compile(r'T-MEC', re.IGNORECASE),
            'ace': re.compile(r'ACE\s*(\d+)', re.IGNORECASE),
            'aladi': re.compile(r'ALADI', re.IGNORECASE),
            'mercosur': re.compile(r'MERCOSUR', re.IGNORECASE),
            'alianza_pacifico': re.compile(r'ALIANZA\s+DEL\s+PACÍFICO', re.IGNORECASE),
            
            # Países y regiones
            'pais': re.compile(r'(México|Mexico|Estados Unidos|Canadá|Canada|Brasil|Argentina|Chile|Colombia|Perú|Peru|Ecuador|Venezuela|Uruguay|Paraguay|Bolivia|Costa Rica|Panamá|Panama|Guatemala|Honduras|El Salvador|Nicaragua|Cuba|República Dominicana|Dominican Republic)', re.IGNORECASE),
            
            # Fechas
            'fecha': re.compile(r'(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})', re.IGNORECASE),
            'año': re.compile(r'(19|20)\d{2}', re.IGNORECASE),
        }
        
        # Separadores para chunking
        self.section_separators = [
            r'\n\s*CAPÍTULO\s+[IVX]+',
            r'\n\s*TÍTULO\s+[IVX]+',
            r'\n\s*SECCIÓN\s+[IVX]+',
            r'\n\s*ARTÍCULO\s+\d+',
            r'\n\s*ANEXO\s+[IVX]+',
            r'\n\s*APÉNDICE\s+[IVX]+',
            r'\n\s*RESOLUCIÓN\s+',
            r'\n\s*REGLAMENTO\s+',
            r'\n\s*LEY\s+',
            r'\n\s*CÓDIGO\s+',
            r'\n\s*DECRETO\s+',
            r'\n\s*ACUERDO\s+',
            r'\n\s*CONVENIO\s+',
            r'\n\s*TRATADO\s+',
        ]
        
        # Patrones para filtrar contenido inútil
        self.useless_patterns = [
            r'PÁGINA \d+ \(SIN TEXTO\)',
            r'PÁGINA \d+ ---',
            r'--- PÁGINA \d+ ---',
            r'Página \d+ de \d+',
            r'Nota: El presente documento se da a conocer',
            r'DOF - Diario Oficial de la Federación',
            r'FE de erratas',
            r'^\s*\d+\s*$',  # Solo números
            r'^\s*[IVX]+\s*$',  # Solo números romanos
            r'^\s*$',  # Líneas vacías
        ]
        
        # Patrones para contenido útil
        self.useful_patterns = [
            r'ARTÍCULO\s+\d+',
            r'CAPÍTULO\s+[IVX\d]+',
            r'TÍTULO\s+[IVX\d]+',
            r'SECCIÓN\s+[IVX\d]+',
            r'PÁRRAFO\s+[IVX\d]+',
            r'FRACCIÓN\s+[IVX\d]+',
            r'TRANSITORIO\s+[IVX\d]+',
            r'DEFINICIONES?',
            r'DISPOSICIONES?\s+(FINALES?|TRANSITORIAS?)',
        ]
    
    def extract_text(self, file_path: Path) -> str:
        """Extraer texto de un documento"""
        try:
            file_extension = file_path.suffix.lower()
            
            if file_extension == '.pdf':
                return self._extract_text_from_pdf(file_path)
            elif file_extension in ['.docx', '.doc']:
                return self._extract_text_from_docx(file_path)
            elif file_extension == '.txt':
                return self._extract_text_from_txt(file_path)
            else:
                logger.warning(f"Formato no soportado: {file_extension}")
                return ""
                
        except Exception as e:
            logger.error(f"Error extrayendo texto de {file_path}: {e}")
            return ""
    
    def _extract_text_from_pdf(self, file_path: Path) -> str:
        """Extraer texto de PDF usando múltiples métodos robustos"""
        text = ""
        
        # Método 1: PyMuPDF (fitz) - más robusto para PDFs corruptos
        try:
            doc = fitz.open(str(file_path))
            content = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                
                # Filtrar páginas vacías o con poco contenido útil
                if self._is_useful_content(page_text):
                    content += f"\n--- PÁGINA {page_num + 1} ---\n{page_text}\n"
                else:
                    logger.debug(f"Página {page_num + 1} filtrada (contenido inútil): {file_path.name}")
            
            doc.close()
            if content.strip():
                # Filtrar contenido inútil del texto completo
                filtered_content = self._filter_useless_content(content)
                if filtered_content.strip():
                    logger.info(f"PDF extraído exitosamente con PyMuPDF: {file_path.name}")
                    return filtered_content
        except Exception as e:
            logger.warning(f"PyMuPDF falló para {file_path.name}: {e}")
        
        # Método 2: Unstructured (más preciso para PDFs bien formados)
        try:
            elements = partition_pdf(str(file_path))
            text_parts = []
            
            for element in elements:
                if hasattr(element, 'text') and element.text:
                    text_parts.append(element.text)
            
            text = '\n'.join(text_parts)
            if text.strip():
                logger.info(f"PDF extraído exitosamente con unstructured: {file_path.name}")
                return text
        except Exception as e:
            logger.warning(f"Unstructured falló para {file_path.name}: {e}")
        
        # Método 3: pdfplumber - alternativo
        try:
            import pdfplumber
            with pdfplumber.open(str(file_path)) as pdf:
                text_parts = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_parts.append(page_text)
                
                text = '\n'.join(text_parts)
                if text.strip():
                    logger.info(f"PDF extraído exitosamente con pdfplumber: {file_path.name}")
                    return text
        except Exception as e:
            logger.warning(f"pdfplumber falló para {file_path.name}: {e}")
        
        # Método 4: PyMuPDF con manejo de errores por página
        try:
            doc = fitz.open(str(file_path))
            content = ""
            
            for page_num in range(len(doc)):
                try:
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    if text.strip():
                        content += f"\n--- PÁGINA {page_num + 1} ---\n{text}\n"
                    else:
                        # Página sin texto, intentar extraer como imagen
                        content += f"\n--- PÁGINA {page_num + 1} (SIN TEXTO) ---\n"
                except Exception as page_error:
                    logger.warning(f"Error en página {page_num + 1} de {file_path.name}: {page_error}")
                    content += f"\n--- PÁGINA {page_num + 1} (ERROR) ---\n"
            
            doc.close()
            if content.strip():
                logger.info(f"PDF extraído parcialmente con PyMuPDF: {file_path.name}")
                return content
        except Exception as e:
            logger.warning(f"PyMuPDF con manejo de errores falló para {file_path.name}: {e}")
        
        # Si todos los métodos fallan, devolver contenido mínimo
        logger.error(f"TODOS los métodos fallaron para {file_path.name}")
        return f"[DOCUMENTO NO PROCESABLE: {file_path.name}]"
    
    def _extract_text_from_docx(self, file_path: Path) -> str:
        """Extraer texto de DOCX"""
        try:
            # Intentar con python-docx primero
            try:
                doc = docx.Document(str(file_path))
                content = ""
                for paragraph in doc.paragraphs:
                    if paragraph.text.strip():
                        content += paragraph.text + "\n"
                return content
            except:
                # Fallback a docx2txt
                return docx2txt_process(str(file_path))
                
        except Exception as e:
            logger.error(f"Error extrayendo texto DOCX de {file_path}: {e}")
            return ""
    
    def _extract_text_from_txt(self, file_path: Path) -> str:
        """Extraer texto de TXT"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error extrayendo texto TXT de {file_path}: {e}")
            return ""
    
    def create_comercio_chunks(self, text: str, file_path: Path) -> List[Dict[str, Any]]:
        """Crear chunks para documentos de comercio exterior usando chunking inteligente"""
        try:
            if not text.strip():
                return []
            
            # Verificar si el contenido tiene estructura legal
            if not self._has_legal_structure(text):
                logger.warning(f"Documento sin estructura legal detectada: {file_path.name}")
                return []
            
            # Limpiar texto
            cleaned_text = self._clean_text(text)
            
            # Detectar tipo de documento
            doc_type = self._detect_document_type(file_path, cleaned_text)
            
            # Crear metadatos del documento
            document_metadata = {
                'filename': file_path.name,
                'file_path': str(file_path),
                'document_type': doc_type,
                'file_size': file_path.stat().st_size if file_path.exists() else 0,
                'processed_at': datetime.now().isoformat()
            }
            
            # Usar chunker inteligente
            legal_chunks = self.smart_chunker.chunk_document(
                cleaned_text, 
                file_path, 
                document_metadata
            )
            
            # Convertir a formato compatible con el sistema actual
            chunks = []
            for legal_chunk in legal_chunks:
                chunk_dict = {
                    'text': legal_chunk.content,
                    'type': legal_chunk.chunk_type,
                    'index': len(chunks),
                    'file_path': str(file_path),
                    'file_name': file_path.name,
                    'clean_text': self._clean_text_for_search(legal_chunk.content),
                    'hierarchy': self._extract_comercio_hierarchy(legal_chunk.content),
                    'chunk_id': legal_chunk.chunk_id,
                    'hierarchy_level': legal_chunk.hierarchy_level,
                    'parent_chunk_id': legal_chunk.parent_chunk_id,
                    'article_number': legal_chunk.article_number,
                    'paragraph_number': legal_chunk.paragraph_number,
                    'inciso_number': legal_chunk.inciso_number,
                    'chapter_title': legal_chunk.chapter_title,
                    'section_title': legal_chunk.section_title,
                    'law_title': legal_chunk.law_title,
                    'page_number': legal_chunk.page_number,
                    'word_count': legal_chunk.word_count,
                    'char_count': legal_chunk.char_count,
                    'is_compressed': legal_chunk.is_compressed,
                    'compression_ratio': legal_chunk.compression_ratio,
                    'metadata': legal_chunk.metadata
                }
                chunks.append(chunk_dict)
            
            # Obtener estadísticas del chunking
            stats = self.smart_chunker.get_chunk_statistics(legal_chunks)
            logger.info(f"Creados {len(chunks)} chunks inteligentes para {file_path.name}")
            logger.info(f"Estadísticas: {stats['total_chunks']} chunks, {stats['compressed_chunks']} comprimidos, "
                       f"promedio {stats['avg_chunk_size']:.0f} chars")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error creando chunks para {file_path}: {e}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """Limpiar texto manteniendo estructura de comercio exterior"""
        if not text:
            return ""
        
        # Limpiar caracteres Unicode problemáticos
        text = text.replace('\u0000', '')  # Eliminar caracteres nulos
        text = text.replace('\x00', '')    # Eliminar bytes nulos
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)  # Eliminar caracteres de control
        
        # Limpiar secuencias de escape Unicode problemáticas
        text = re.sub(r'\\u0000', '', text)  # Eliminar secuencias \u0000
        text = re.sub(r'\\x00', '', text)    # Eliminar secuencias \x00
        text = re.sub(r'\\[0-9A-Fa-f]{4}', '', text)  # Eliminar secuencias \uXXXX
        
        # Normalizar espacios
        text = re.sub(r'\s+', ' ', text)
        
        # Preservar estructura de documentos oficiales
        text = re.sub(r'\n\s*ARTÍCULO\s+', '\n\nARTÍCULO ', text)
        text = re.sub(r'\n\s*CAPÍTULO\s+', '\n\nCAPÍTULO ', text)
        text = re.sub(r'\n\s*TÍTULO\s+', '\n\nTÍTULO ', text)
        text = re.sub(r'\n\s*SECCIÓN\s+', '\n\nSECCIÓN ', text)
        text = re.sub(r'\n\s*ANEXO\s+', '\n\nANEXO ', text)
        text = re.sub(r'\n\s*APÉNDICE\s+', '\n\nAPÉNDICE ', text)
        
        return text.strip()
    
    def _clean_text_for_search(self, text: str) -> str:
        """Limpiar texto para búsqueda (más agresivo)"""
        # Remover caracteres especiales
        text = re.sub(r'[^\w\s]', ' ', text)
        # Normalizar espacios
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip()
    
    def _detect_document_type(self, file_path: Path, text: str) -> str:
        """Detectar tipo de documento de comercio exterior"""
        file_name = file_path.name.lower()
        text_lower = text.lower()
        
        # Patrones de detección
        if 'dof' in file_name or 'dof' in text_lower:
            return 'dof'
        elif 'reglas_comercio_exterior' in file_name or 'reglas comercio exterior' in text_lower:
            return 'reglamento'
        elif 'ley_aduanera' in file_name or 'ley aduanera' in text_lower:
            return 'ley'
        elif 'codigo_fiscal' in file_name or 'código fiscal' in text_lower:
            return 'codigo'
        elif 'resolucion' in file_name or 'resolución' in text_lower:
            return 'resolucion'
        elif 'anexo' in file_name or 'anexo' in text_lower:
            return 'anexo'
        elif 'tratado' in file_name or 'tratado' in text_lower:
            return 'tratado'
        elif 'convenio' in file_name or 'convenio' in text_lower:
            return 'convenio'
        elif 'acuerdo' in file_name or 'acuerdo' in text_lower:
            return 'acuerdo'
        elif 'tlc' in file_name or 'tlc' in text_lower:
            return 'tlc'
        elif 'tmec' in file_name or 't-mec' in text_lower:
            return 'tmec'
        elif 'ace' in file_name or 'ace' in text_lower:
            return 'ace'
        elif 'bill_of_lading' in file_name or 'bill of lading' in text_lower:
            return 'bill_of_lading'
        elif 'air_waybill' in file_name or 'air waybill' in text_lower:
            return 'air_waybill'
        elif 'certificado' in file_name or 'certificado' in text_lower:
            return 'certificado'
        else:
            return 'general'
    
    def _create_structured_chunks(self, text: str, doc_type: str) -> List[Dict[str, Any]]:
        """Crear chunks preservando estructura de documentos oficiales"""
        chunks = []
        
        # Dividir por secciones
        sections = self._split_by_sections(text)
        
        for section in sections:
            if not section.strip():
                continue
            
            # Detectar tipo de sección
            section_type = self._detect_section_type(section)
            
            # Validar tipo de chunk según constraint de base de datos
            if section_type not in ['articulo', 'paragrafo', 'anexo', 'titulo', 'capitulo', 'seccion', 
                                  'resolucion', 'reglamento', 'ley', 'codigo', 'decreto', 'acuerdo', 
                                  'convenio', 'tratado', 'dof', 'paragraph', 'section']:
                section_type = 'section'  # Tipo por defecto
            
            # Crear chunk principal de la sección
            chunks.append({
                'text': section.strip(),
                'type': section_type,
                'hierarchy': self._extract_comercio_hierarchy(section)
            })
            
            # Si la sección es muy larga, dividir en párrafos
            if len(section.split()) > 500:
                paragraphs = self._split_into_paragraphs(section)
                for i, paragraph in enumerate(paragraphs):
                    if len(paragraph.split()) >= 10:
                        # Validar tipo de chunk para párrafos
                        paragraph_type = 'paragraph' if section_type == 'section' else 'paragrafo'
                        chunks.append({
                            'text': paragraph.strip(),
                            'type': paragraph_type,
                            'hierarchy': self._extract_comercio_hierarchy(paragraph)
                        })
        
        return chunks
    
    def _create_general_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Crear chunks generales para documentos no estructurados"""
        chunks = []
        
        # Dividir por párrafos
        paragraphs = self._split_into_paragraphs(text)
        
        for paragraph in paragraphs:
            if len(paragraph.split()) >= 10:
                chunks.append({
                    'text': paragraph.strip(),
                    'type': 'paragraph',
                    'hierarchy': {}
                })
        
        return chunks
    
    def _split_by_sections(self, text: str) -> List[str]:
        """Dividir texto por secciones de comercio exterior"""
        # Crear patrón combinado
        pattern = '|'.join(f'({sep})' for sep in self.section_separators)
        
        # Dividir manteniendo los separadores
        sections = re.split(f'({pattern})', text)
        
        # Reconstruir secciones
        result = []
        current_section = ""
        
        for part in sections:
            if part is None:
                continue
            if part and re.match(pattern, part):
                if current_section.strip():
                    result.append(current_section.strip())
                current_section = part
            else:
                current_section += part
        
        if current_section.strip():
            result.append(current_section.strip())
        
        return result
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Dividir texto en párrafos"""
        # Dividir por doble salto de línea
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Filtrar párrafos vacíos
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _detect_section_type(self, text: str) -> str:
        """Detectar tipo de sección de comercio exterior"""
        text_upper = text.upper()
        
        # Tipos permitidos en la base de datos
        allowed_types = ['articulo', 'paragrafo', 'anexo', 'titulo', 'capitulo', 'seccion', 
                        'resolucion', 'reglamento', 'ley', 'codigo', 'decreto', 'acuerdo', 
                        'convenio', 'tratado', 'dof', 'paragraph', 'section']
        
        for pattern_name, pattern in self.comercio_patterns.items():
            if pattern.search(text_upper) and pattern_name in allowed_types:
                return pattern_name
        
        # Detectar por palabras clave
        if 'ARTÍCULO' in text_upper:
            return 'articulo'
        elif 'CAPÍTULO' in text_upper:
            return 'capitulo'
        elif 'TÍTULO' in text_upper:
            return 'titulo'
        elif 'SECCIÓN' in text_upper:
            return 'seccion'
        elif 'ANEXO' in text_upper:
            return 'anexo'
        elif 'APÉNDICE' in text_upper:
            return 'anexo'  # Mapear apendice a anexo
        elif 'RESOLUCIÓN' in text_upper:
            return 'resolucion'
        elif 'REGLAMENTO' in text_upper:
            return 'reglamento'
        elif 'LEY' in text_upper:
            return 'ley'
        elif 'CÓDIGO' in text_upper:
            return 'codigo'
        elif 'DECRETO' in text_upper:
            return 'decreto'
        elif 'ACUERDO' in text_upper:
            return 'acuerdo'
        elif 'CONVENIO' in text_upper:
            return 'convenio'
        elif 'TRATADO' in text_upper:
            return 'tratado'
        else:
            return 'section'
    
    def _extract_comercio_hierarchy(self, text: str) -> Dict[str, Any]:
        """Extraer jerarquía de comercio exterior del texto"""
        hierarchy = {}
        
        # Buscar artículos
        articulo_match = self.comercio_patterns['articulo'].search(text)
        if articulo_match:
            hierarchy['articulo'] = articulo_match.group(1)
        
        # Buscar capítulos
        capitulo_match = self.comercio_patterns['capitulo'].search(text)
        if capitulo_match:
            hierarchy['capitulo'] = capitulo_match.group(1)
        
        # Buscar títulos
        titulo_match = self.comercio_patterns['titulo'].search(text)
        if titulo_match:
            hierarchy['titulo'] = titulo_match.group(1)
        
        # Buscar secciones
        seccion_match = self.comercio_patterns['seccion'].search(text)
        if seccion_match:
            hierarchy['seccion'] = seccion_match.group(1)
        
        # Buscar anexos
        anexo_match = self.comercio_patterns['anexo'].search(text)
        if anexo_match:
            hierarchy['anexo'] = anexo_match.group(1)
        
        # Buscar países
        pais_match = self.comercio_patterns['pais'].search(text)
        if pais_match:
            hierarchy['pais'] = pais_match.group(1)
        
        # Buscar fechas
        fecha_match = self.comercio_patterns['fecha'].search(text)
        if fecha_match:
            hierarchy['fecha'] = fecha_match.group(1)
        
        return hierarchy
    
    def _is_useful_content(self, text: str) -> bool:
        """Verificar si el contenido es útil (no metadatos o páginas vacías)"""
        if not text or not text.strip():
            return False
        
        text_clean = text.strip()
        
        # Verificar si es contenido inútil
        for pattern in self.useless_patterns:
            if re.search(pattern, text_clean, re.IGNORECASE):
                return False
        
        # Verificar si tiene contenido útil
        for pattern in self.useful_patterns:
            if re.search(pattern, text_clean, re.IGNORECASE):
                return True
        
        # Si tiene más de 50 palabras, probablemente es útil
        word_count = len(text_clean.split())
        if word_count > 50:
            return True
        
        # Si tiene menos de 10 palabras, probablemente no es útil
        if word_count < 10:
            return False
        
        # Verificar si contiene información legal relevante
        legal_keywords = [
            'artículo', 'capítulo', 'título', 'sección', 'párrafo', 'fracción',
            'transitorio', 'definiciones', 'disposiciones', 'ley', 'código',
            'reglamento', 'decreto', 'acuerdo', 'convenio', 'tratado',
            'comercio', 'exterior', 'aduana', 'arancel', 'mercancía'
        ]
        
        text_lower = text_clean.lower()
        for keyword in legal_keywords:
            if keyword in text_lower:
                return True
        
        return False
    
    def _filter_useless_content(self, content: str) -> str:
        """Filtrar contenido inútil del texto extraído"""
        lines = content.split('\n')
        filtered_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Verificar si la línea es útil
            is_useful = True
            for pattern in self.useless_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    is_useful = False
                    break
            
            if is_useful:
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def _has_legal_structure(self, text: str) -> bool:
        """Verificar si el texto tiene estructura legal (muy permisivo para RAG)"""
        # Para RAG, procesar TODOS los documentos con contenido
        if not text or len(text.strip()) < 5:
            return False
        
        # Patrones básicos para cualquier contenido útil
        basic_patterns = [
            r'[a-záéíóúñ]',  # Cualquier texto con letras
            r'\d+',  # Cualquier número
            r'[A-Z]',  # Cualquier letra mayúscula
            r'[a-z]',  # Cualquier letra minúscula
        ]
        
        # Si tiene contenido básico, procesarlo
        for pattern in basic_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # También verificar patrones legales específicos
        for pattern in self.useful_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
