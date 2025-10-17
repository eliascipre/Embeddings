#!/usr/bin/env python3
"""
Sistema de Chunking Legal Inteligente
- Preserva jerarquía legal (artículos, capítulos, párrafos, incisos)
- Validaciones robustas para tipos de chunk
- Optimización de tamaño para evitar errores de índice
- Compresión para documentos largos
"""

import re
import logging
import gzip
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class LegalChunk:
    """Estructura de un chunk legal con metadatos completos"""
    content: str
    chunk_type: str
    chunk_id: str
    hierarchy_level: int  # 1=artículo, 2=párrafo, 3=inciso, etc.
    parent_chunk_id: Optional[str] = None
    article_number: Optional[str] = None
    paragraph_number: Optional[str] = None
    inciso_number: Optional[str] = None
    chapter_title: Optional[str] = None
    section_title: Optional[str] = None
    law_title: Optional[str] = None
    page_number: Optional[int] = None
    word_count: int = 0
    char_count: int = 0
    is_compressed: bool = False
    compression_ratio: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        self.word_count = len(self.content.split()) if self.content else 0
        self.char_count = len(self.content) if self.content else 0

class SmartLegalChunker:
    """Sistema de chunking legal inteligente con todas las optimizaciones"""
    
    def __init__(self, max_chunk_size: int = 2000, compression_threshold: int = 1500):
        self.max_chunk_size = max_chunk_size
        self.compression_threshold = compression_threshold
        
        # Tipos de chunk permitidos en la base de datos
        self.allowed_chunk_types = {
            'articulo', 'paragrafo', 'anexo', 'titulo', 'capitulo', 'seccion', 
            'resolucion', 'reglamento', 'ley', 'codigo', 'decreto', 'acuerdo', 
            'convenio', 'tratado', 'dof', 'paragraph', 'section', 'inciso',
            'fraccion', 'transitorio', 'definicion', 'disposicion'
        }
        
        # Patrones legales mejorados
        self.legal_patterns = {
            # Artículos
            'articulo': re.compile(r'ARTÍCULO\s+(\d+[A-Za-z]*)', re.IGNORECASE),
            'articulo_romano': re.compile(r'ARTÍCULO\s+([IVX]+)', re.IGNORECASE),
            
            # Capítulos
            'capitulo': re.compile(r'CAPÍTULO\s+([IVX]+|\d+)', re.IGNORECASE),
            'capitulo_romano': re.compile(r'CAPÍTULO\s+([IVX]+)', re.IGNORECASE),
            
            # Títulos
            'titulo': re.compile(r'TÍTULO\s+([IVX]+|\d+)', re.IGNORECASE),
            'titulo_romano': re.compile(r'TÍTULO\s+([IVX]+)', re.IGNORECASE),
            
            # Secciones
            'seccion': re.compile(r'SECCIÓN\s+([IVX]+|\d+)', re.IGNORECASE),
            'seccion_romano': re.compile(r'SECCIÓN\s+([IVX]+)', re.IGNORECASE),
            
            # Párrafos
            'paragrafo': re.compile(r'PÁRRAFO\s+([IVX]+|\d+)', re.IGNORECASE),
            'paragrafo_romano': re.compile(r'PÁRRAFO\s+([IVX]+)', re.IGNORECASE),
            
            # Incisos
            'inciso': re.compile(r'(\d+[A-Za-z]*\.\s+[^0-9]+?)(?=\d+[A-Za-z]*\.|$)', re.IGNORECASE),
            'inciso_letra': re.compile(r'([A-Za-z])\)\s+([^A-Za-z]+?)(?=[A-Za-z]\)|$)', re.IGNORECASE),
            'inciso_romano': re.compile(r'([IVX]+)\.\s+([^IVX]+?)(?=[IVX]+\.|$)', re.IGNORECASE),
            
            # Fracciones
            'fraccion': re.compile(r'FRACCIÓN\s+([IVX]+|\d+)', re.IGNORECASE),
            
            # Transitorios
            'transitorio': re.compile(r'TRANSITORIO\s+([IVX]+|\d+)', re.IGNORECASE),
            
            # Definiciones
            'definicion': re.compile(r'DEFINICIONES?', re.IGNORECASE),
            
            # Disposiciones
            'disposicion': re.compile(r'DISPOSICIONES?\s+(FINALES?|TRANSITORIAS?)', re.IGNORECASE),
            
            # Comercio exterior específico
            'clasificacion_arancelaria': re.compile(r'CLASIFICACIÓN\s+ARANCELARIA', re.IGNORECASE),
            'valor_aduana': re.compile(r'VALOR\s+EN\s+ADUANA', re.IGNORECASE),
            'origen_mercancia': re.compile(r'ORIGEN\s+DE\s+LA\s+MERCANCÍA', re.IGNORECASE),
            'destino_mercancia': re.compile(r'DESTINO\s+DE\s+LA\s+MERCANCÍA', re.IGNORECASE),
            'tipo_operacion': re.compile(r'TIPO\s+DE\s+OPERACIÓN', re.IGNORECASE),
            'regimen_aduana': re.compile(r'RÉGIMEN\s+ADUANERO', re.IGNORECASE),
        }
        
        # Separadores jerárquicos
        self.hierarchical_separators = [
            r'\n\s*ARTÍCULO\s+\d+',
            r'\n\s*CAPÍTULO\s+[IVX\d]+',
            r'\n\s*TÍTULO\s+[IVX\d]+',
            r'\n\s*SECCIÓN\s+[IVX\d]+',
            r'\n\s*PÁRRAFO\s+[IVX\d]+',
            r'\n\s*ANEXO\s+[IVX\d]+',
            r'\n\s*APÉNDICE\s+[IVX\d]+',
            r'\n\s*FRACCIÓN\s+[IVX\d]+',
            r'\n\s*TRANSITORIO\s+[IVX\d]+',
        ]
    
    def chunk_document(self, content: str, file_path: Path, document_metadata: Dict[str, Any] = None) -> List[LegalChunk]:
        """Chunking principal con todas las optimizaciones"""
        try:
            logger.info(f"🔍 Iniciando chunking inteligente para {file_path.name}")
            
            # Limpiar contenido
            cleaned_content = self._clean_content(content)
            
            # Detectar estructura del documento
            document_structure = self._analyze_document_structure(cleaned_content)
            
            # Crear chunks jerárquicos
            chunks = self._create_hierarchical_chunks(
                cleaned_content, 
                file_path, 
                document_structure,
                document_metadata
            )
            
            # Validar chunks
            chunks = self._validate_chunks(chunks)
            
            # Optimizar tamaño de chunks
            chunks = self._optimize_chunk_sizes(chunks)
            
            # Comprimir chunks largos
            chunks = self._compress_large_chunks(chunks)
            
            # Asignar IDs únicos
            chunks = self._assign_chunk_ids(chunks, file_path)
            
            logger.info(f"✅ Chunking completado: {len(chunks)} chunks generados")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Error en chunking inteligente: {e}")
            raise
    
    def _clean_content(self, content: str) -> str:
        """Limpiar contenido manteniendo estructura legal"""
        # Normalizar espacios
        content = re.sub(r'\s+', ' ', content)
        
        # Preservar estructura de documentos oficiales
        content = re.sub(r'\n\s*ARTÍCULO\s+', '\n\nARTÍCULO ', content)
        content = re.sub(r'\n\s*CAPÍTULO\s+', '\n\nCAPÍTULO ', content)
        content = re.sub(r'\n\s*TÍTULO\s+', '\n\nTÍTULO ', content)
        content = re.sub(r'\n\s*SECCIÓN\s+', '\n\nSECCIÓN ', content)
        content = re.sub(r'\n\s*PÁRRAFO\s+', '\n\nPÁRRAFO ', content)
        content = re.sub(r'\n\s*ANEXO\s+', '\n\nANEXO ', content)
        content = re.sub(r'\n\s*APÉNDICE\s+', '\n\nAPÉNDICE ', content)
        
        return content.strip()
    
    def _analyze_document_structure(self, content: str) -> Dict[str, Any]:
        """Analizar estructura del documento para optimizar chunking"""
        structure = {
            'has_articles': bool(self.legal_patterns['articulo'].search(content)),
            'has_chapters': bool(self.legal_patterns['capitulo'].search(content)),
            'has_titles': bool(self.legal_patterns['titulo'].search(content)),
            'has_sections': bool(self.legal_patterns['seccion'].search(content)),
            'has_paragraphs': bool(self.legal_patterns['paragrafo'].search(content)),
            'has_incisos': bool(self.legal_patterns['inciso'].search(content)),
            'document_type': self._detect_document_type(content),
            'complexity_score': self._calculate_complexity_score(content)
        }
        
        return structure
    
    def _detect_document_type(self, content: str) -> str:
        """Detectar tipo de documento legal"""
        content_upper = content.upper()
        
        if 'CONSTITUCIÓN' in content_upper or 'CARTA MAGNA' in content_upper:
            return 'constitucion'
        elif 'CÓDIGO' in content_upper or 'CODIGO' in content_upper:
            return 'codigo'
        elif 'LEY' in content_upper:
            return 'ley'
        elif 'REGLAMENTO' in content_upper:
            return 'reglamento'
        elif 'DECRETO' in content_upper:
            return 'decreto'
        elif 'ACUERDO' in content_upper:
            return 'acuerdo'
        elif 'CONVENIO' in content_upper:
            return 'convenio'
        elif 'TRATADO' in content_upper:
            return 'tratado'
        elif 'RESOLUCIÓN' in content_upper or 'RESOLUCION' in content_upper:
            return 'resolucion'
        else:
            return 'documento'
    
    def _calculate_complexity_score(self, content: str) -> float:
        """Calcular score de complejidad del documento"""
        score = 0.0
        
        # Puntos por elementos estructurales
        for pattern_name, pattern in self.legal_patterns.items():
            matches = pattern.findall(content)
            score += len(matches) * 0.1
        
        # Puntos por longitud
        word_count = len(content.split())
        score += min(word_count / 1000, 10)  # Máximo 10 puntos por longitud
        
        return min(score, 20)  # Score máximo de 20
    
    def _create_hierarchical_chunks(
        self, 
        content: str, 
        file_path: Path, 
        structure: Dict[str, Any],
        document_metadata: Dict[str, Any] = None
    ) -> List[LegalChunk]:
        """Crear chunks preservando jerarquía legal"""
        chunks = []
        
        if structure['has_articles']:
            chunks.extend(self._chunk_by_articles(content, file_path, document_metadata, structure))
        elif structure['has_chapters']:
            chunks.extend(self._chunk_by_chapters(content, file_path, document_metadata, structure))
        elif structure['has_titles']:
            chunks.extend(self._chunk_by_titles(content, file_path, document_metadata, structure))
        else:
            chunks.extend(self._chunk_by_paragraphs(content, file_path, document_metadata, structure))
        
        return chunks
    
    def _chunk_by_articles(self, content: str, file_path: Path, document_metadata: Dict[str, Any] = None, structure: Dict[str, Any] = None) -> List[LegalChunk]:
        """Chunking por artículos con jerarquía completa"""
        chunks = []
        
        # Dividir por artículos
        articles = re.split(r'ARTÍCULO\s+\d+[A-Za-z]*', content)
        
        for i, article_content in enumerate(articles[1:], 1):
            if not article_content.strip():
                continue
            
            # Agregar "ARTÍCULO X" al inicio del contenido
            full_article_content = f"ARTÍCULO {i}.{article_content.strip()}"
            
            # Crear chunk del artículo completo
            article_chunk = LegalChunk(
                content=full_article_content,
                chunk_type='articulo',
                chunk_id=f"art_{i}",
                hierarchy_level=1,
                article_number=str(i),
                page_number=self._extract_page_number(article_content),
                metadata={
                    'filename': file_path.name,
                    'article_number': str(i),
                    'document_type': structure.get('document_type', 'documento') if structure else 'documento',
                    'complexity_score': structure.get('complexity_score', 0) if structure else 0
                }
            )
            chunks.append(article_chunk)
            
            # Dividir artículo en párrafos
            paragraphs = re.split(r'PÁRRAFO\s+[IVX\d]+', article_content)
            
            for j, paragraph in enumerate(paragraphs[1:], 1):
                if len(paragraph.strip()) > 50:
                    para_chunk = LegalChunk(
                        content=paragraph.strip(),
                        chunk_type='paragrafo',
                        chunk_id=f"art_{i}_para_{j}",
                        hierarchy_level=2,
                        parent_chunk_id=f"art_{i}",
                        article_number=str(i),
                        paragraph_number=str(j),
                        page_number=self._extract_page_number(paragraph),
                        metadata={
                            'filename': file_path.name,
                            'article_number': str(i),
                            'paragraph_number': str(j),
                            'parent_chunk_id': f"art_{i}"
                        }
                    )
                    chunks.append(para_chunk)
                    
                    # Dividir párrafos en incisos
                    incisos = self._extract_incisos(paragraph)
                    for k, inciso in enumerate(incisos, 1):
                        if len(inciso.strip()) > 20:
                            inciso_chunk = LegalChunk(
                                content=inciso.strip(),
                                chunk_type='inciso',
                                chunk_id=f"art_{i}_para_{j}_inc_{k}",
                                hierarchy_level=3,
                                parent_chunk_id=f"art_{i}_para_{j}",
                                article_number=str(i),
                                paragraph_number=str(j),
                                inciso_number=str(k),
                                page_number=self._extract_page_number(inciso),
                                metadata={
                                    'filename': file_path.name,
                                    'article_number': str(i),
                                    'paragraph_number': str(j),
                                    'inciso_number': str(k),
                                    'parent_chunk_id': f"art_{i}_para_{j}"
                                }
                            )
                            chunks.append(inciso_chunk)
        
        return chunks
    
    def _chunk_by_chapters(self, content: str, file_path: Path, document_metadata: Dict[str, Any] = None, structure: Dict[str, Any] = None) -> List[LegalChunk]:
        """Chunking por capítulos"""
        chunks = []
        
        chapters = re.split(r'CAPÍTULO\s+[IVX\d]+', content)
        
        for i, chapter_content in enumerate(chapters[1:], 1):
            if not chapter_content.strip():
                continue
            
            # Agregar "CAPÍTULO X" al inicio del contenido
            full_chapter_content = f"CAPÍTULO {i}.{chapter_content.strip()}"
            
            chapter_chunk = LegalChunk(
                content=full_chapter_content,
                chunk_type='capitulo',
                chunk_id=f"cap_{i}",
                hierarchy_level=1,
                chapter_title=f"Capítulo {i}",
                page_number=self._extract_page_number(chapter_content),
                metadata={
                    'filename': file_path.name,
                    'chapter_number': str(i),
                    'document_type': structure.get('document_type', 'documento') if structure else 'documento'
                }
            )
            chunks.append(chapter_chunk)
        
        return chunks
    
    def _chunk_by_titles(self, content: str, file_path: Path, document_metadata: Dict[str, Any] = None, structure: Dict[str, Any] = None) -> List[LegalChunk]:
        """Chunking por títulos"""
        chunks = []
        
        titles = re.split(r'TÍTULO\s+[IVX\d]+', content)
        
        for i, title_content in enumerate(titles[1:], 1):
            if not title_content.strip():
                continue
            
            # Agregar "TÍTULO X" al inicio del contenido
            full_title_content = f"TÍTULO {i}.{title_content.strip()}"
            
            title_chunk = LegalChunk(
                content=full_title_content,
                chunk_type='titulo',
                chunk_id=f"tit_{i}",
                hierarchy_level=1,
                law_title=f"Título {i}",
                page_number=self._extract_page_number(title_content),
                metadata={
                    'filename': file_path.name,
                    'title_number': str(i),
                    'document_type': structure.get('document_type', 'documento') if structure else 'documento'
                }
            )
            chunks.append(title_chunk)
        
        return chunks
    
    def _chunk_by_paragraphs(self, content: str, file_path: Path, document_metadata: Dict[str, Any] = None, structure: Dict[str, Any] = None) -> List[LegalChunk]:
        """Chunking por párrafos para documentos sin estructura clara"""
        chunks = []
        
        paragraphs = content.split('\n\n')
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph.strip()) > 10:  # Reducir mínimo para RAG
                para_chunk = LegalChunk(
                    content=paragraph.strip(),
                    chunk_type='paragraph',
                    chunk_id=f"para_{i+1}",
                    hierarchy_level=1,
                    page_number=self._extract_page_number(paragraph),
                    metadata={
                        'filename': file_path.name,
                        'paragraph_number': str(i+1),
                        'document_type': structure.get('document_type', 'documento')
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
    
    def _extract_page_number(self, content: str) -> Optional[int]:
        """Extraer número de página del contenido"""
        page_match = re.search(r'--- PÁGINA (\d+) ---', content)
        if page_match:
            return int(page_match.group(1))
        return None
    
    def _validate_chunks(self, chunks: List[LegalChunk]) -> List[LegalChunk]:
        """Validar chunks y corregir tipos no permitidos"""
        validated_chunks = []
        
        for chunk in chunks:
            # Validar tipo de chunk
            if chunk.chunk_type not in self.allowed_chunk_types:
                # Mapear a tipo permitido
                chunk.chunk_type = self._map_to_allowed_type(chunk.chunk_type)
            
            # Validar contenido mínimo
            if chunk.word_count < 5:
                logger.warning(f"⚠️ Chunk muy corto ignorado: {chunk.chunk_id}")
                continue
            
            # Validar contenido máximo
            if chunk.char_count > self.max_chunk_size * 2:
                logger.warning(f"⚠️ Chunk muy largo, será dividido: {chunk.chunk_id}")
                # Dividir chunk largo
                sub_chunks = self._split_large_chunk(chunk)
                validated_chunks.extend(sub_chunks)
            else:
                validated_chunks.append(chunk)
        
        return validated_chunks
    
    def _map_to_allowed_type(self, chunk_type: str) -> str:
        """Mapear tipo de chunk a tipo permitido"""
        mapping = {
            'apendice': 'anexo',
            'inciso': 'paragrafo',  # Mapear incisos a párrafos
            'fraccion': 'seccion',
            'transitorio': 'paragrafo',
            'definicion': 'seccion',
            'disposicion': 'seccion'
        }
        
        return mapping.get(chunk_type, 'section')
    
    def _split_large_chunk(self, chunk: LegalChunk) -> List[LegalChunk]:
        """Dividir chunk muy grande en chunks más pequeños"""
        sub_chunks = []
        content = chunk.content
        max_size = self.max_chunk_size
        
        # Dividir por oraciones
        sentences = re.split(r'[.!?]+', content)
        
        current_chunk = ""
        chunk_count = 1
        
        for sentence in sentences:
            if len(current_chunk + sentence) > max_size and current_chunk:
                # Crear sub-chunk
                sub_chunk = LegalChunk(
                    content=current_chunk.strip(),
                    chunk_type=chunk.chunk_type,
                    chunk_id=f"{chunk.chunk_id}_part_{chunk_count}",
                    hierarchy_level=chunk.hierarchy_level,
                    parent_chunk_id=chunk.chunk_id,
                    article_number=chunk.article_number,
                    paragraph_number=chunk.paragraph_number,
                    inciso_number=chunk.inciso_number,
                    chapter_title=chunk.chapter_title,
                    section_title=chunk.section_title,
                    law_title=chunk.law_title,
                    page_number=chunk.page_number,
                    metadata=chunk.metadata.copy()
                )
                sub_chunks.append(sub_chunk)
                
                current_chunk = sentence
                chunk_count += 1
            else:
                current_chunk += sentence + ". "
        
        # Agregar último chunk
        if current_chunk.strip():
            sub_chunk = LegalChunk(
                content=current_chunk.strip(),
                chunk_type=chunk.chunk_type,
                chunk_id=f"{chunk.chunk_id}_part_{chunk_count}",
                hierarchy_level=chunk.hierarchy_level,
                parent_chunk_id=chunk.chunk_id,
                article_number=chunk.article_number,
                paragraph_number=chunk.paragraph_number,
                inciso_number=chunk.inciso_number,
                chapter_title=chunk.chapter_title,
                section_title=chunk.section_title,
                law_title=chunk.law_title,
                page_number=chunk.page_number,
                metadata=chunk.metadata.copy()
            )
            sub_chunks.append(sub_chunk)
        
        return sub_chunks
    
    def _optimize_chunk_sizes(self, chunks: List[LegalChunk]) -> List[LegalChunk]:
        """Optimizar tamaños de chunks para evitar errores de índice"""
        optimized_chunks = []
        
        for chunk in chunks:
            # Si el chunk es muy pequeño, combinarlo con el siguiente
            if chunk.char_count < 200 and len(optimized_chunks) > 0:
                last_chunk = optimized_chunks[-1]
                if (last_chunk.char_count + chunk.char_count) < self.max_chunk_size:
                    # Combinar chunks
                    last_chunk.content += "\n\n" + chunk.content
                    last_chunk.word_count = len(last_chunk.content.split())
                    last_chunk.char_count = len(last_chunk.content)
                    continue
            
            # Si el chunk es muy grande, dividirlo
            if chunk.char_count > self.max_chunk_size:
                sub_chunks = self._split_large_chunk(chunk)
                optimized_chunks.extend(sub_chunks)
            else:
                optimized_chunks.append(chunk)
        
        return optimized_chunks
    
    def _compress_large_chunks(self, chunks: List[LegalChunk]) -> List[LegalChunk]:
        """Comprimir chunks largos para optimizar almacenamiento"""
        compressed_chunks = []
        
        for chunk in chunks:
            if chunk.char_count > self.compression_threshold:
                # Comprimir contenido
                compressed_content = gzip.compress(chunk.content.encode('utf-8'))
                compression_ratio = len(compressed_content) / len(chunk.content.encode('utf-8'))
                
                # Crear chunk comprimido
                compressed_chunk = LegalChunk(
                    content=compressed_content.decode('latin-1'),  # Para almacenar bytes como string
                    chunk_type=chunk.chunk_type,
                    chunk_id=chunk.chunk_id,
                    hierarchy_level=chunk.hierarchy_level,
                    parent_chunk_id=chunk.parent_chunk_id,
                    article_number=chunk.article_number,
                    paragraph_number=chunk.paragraph_number,
                    inciso_number=chunk.inciso_number,
                    chapter_title=chunk.chapter_title,
                    section_title=chunk.section_title,
                    law_title=chunk.law_title,
                    page_number=chunk.page_number,
                    word_count=chunk.word_count,
                    char_count=chunk.char_count,
                    is_compressed=True,
                    compression_ratio=compression_ratio,
                    metadata=chunk.metadata.copy()
                )
                compressed_chunks.append(compressed_chunk)
            else:
                compressed_chunks.append(chunk)
        
        return compressed_chunks
    
    def _assign_chunk_ids(self, chunks: List[LegalChunk], file_path: Path) -> List[LegalChunk]:
        """Asignar IDs únicos a los chunks"""
        file_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
        
        for i, chunk in enumerate(chunks):
            if not chunk.chunk_id:
                chunk.chunk_id = f"{file_hash}_{i+1}"
            else:
                chunk.chunk_id = f"{file_hash}_{chunk.chunk_id}"
        
        return chunks
    
    def decompress_chunk(self, chunk: LegalChunk) -> str:
        """Descomprimir chunk si está comprimido"""
        if chunk.is_compressed:
            try:
                compressed_bytes = chunk.content.encode('latin-1')
                decompressed_content = gzip.decompress(compressed_bytes).decode('utf-8')
                return decompressed_content
            except Exception as e:
                logger.error(f"❌ Error descomprimiendo chunk {chunk.chunk_id}: {e}")
                return chunk.content
        else:
            return chunk.content
    
    def get_chunk_statistics(self, chunks: List[LegalChunk]) -> Dict[str, Any]:
        """Obtener estadísticas de los chunks"""
        total_chunks = len(chunks)
        compressed_chunks = len([c for c in chunks if c.is_compressed])
        
        chunk_types = {}
        for chunk in chunks:
            chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1
        
        total_words = sum(chunk.word_count for chunk in chunks)
        total_chars = sum(chunk.char_count for chunk in chunks)
        
        avg_chunk_size = total_chars / total_chunks if total_chunks > 0 else 0
        
        return {
            'total_chunks': total_chunks,
            'compressed_chunks': compressed_chunks,
            'compression_ratio': compressed_chunks / total_chunks if total_chunks > 0 else 0,
            'chunk_types': chunk_types,
            'total_words': total_words,
            'total_chars': total_chars,
            'avg_chunk_size': avg_chunk_size,
            'hierarchy_levels': {
                'level_1': len([c for c in chunks if c.hierarchy_level == 1]),
                'level_2': len([c for c in chunks if c.hierarchy_level == 2]),
                'level_3': len([c for c in chunks if c.hierarchy_level == 3]),
            }
        }
