# Sistema de Embeddings para Documentos Legales Mexicanos

## 📋 Descripción General

Este proyecto implementa un sistema RAG (Retrieval-Augmented Generation) ultra-optimizado para procesar y generar embeddings de documentos legales mexicanos. El sistema está diseñado para manejar miles de documentos PDF de leyes, constituciones, códigos y reglamentos de todos los estados de México, utilizando técnicas avanzadas de chunking legal y generación de embeddings con Hugging Face.

## 🏗️ Arquitectura del Sistema

### Componentes Principales

1. **Sistema RAG Optimizado** (`src/optimized_rag_system.py`)
   - Procesamiento masivo paralelo con hasta 16 workers
   - Búsqueda híbrida (vectorial + palabras clave)
   - Integración con Supabase para almacenamiento
   - Optimizado para GPU A100 (2 GPUs, 160GB VRAM)

2. **Procesador de Chunking Legal** (`src/legal_chunking_processor.py`)
   - Chunking inteligente preservando jerarquía legal
   - Detección automática de artículos, capítulos, títulos, párrafos e incisos
   - Extracción de metadatos legales (fechas, estado, tipo de ley)
   - Procesamiento específico para documentos legales mexicanos

3. **Cliente API Hugging Face** (`src/huggingface_api_client.py`)
   - Cliente optimizado para API de Hugging Face
   - Circuit breaker para manejo de errores
   - Procesamiento en lotes con rate limiting
   - Retry automático con backoff exponencial

4. **Sistema de Configuración** (`src/config.py`)
   - Configuración centralizada del sistema
   - Variables de entorno y parámetros optimizados

## 🚀 Cómo se Hicieron los Embeddings

### 1. Extracción de Documentos

El sistema procesa documentos PDF de leyes mexicanas organizados por estado:

```
leyes_de_todos_los_estados/
├── aguascalientes/
├── baja_california/
├── ciudad_de_mexico/
├── jalisco/
└── ... (32 estados)
```

**Estadísticas del Dataset:**
- **Total de archivos:** 5,074 PDFs
- **Total de páginas:** 202,206 páginas
- **Tamaño total:** 2.76 GB
- **Estados procesados:** 32 estados de México
- **Tipos de documentos:** Constituciones, leyes, códigos, reglamentos, decretos

### 2. Procesamiento de Texto

#### Extracción con PyMuPDF
```python
def _extract_text_from_pdf(self, file_path: Path) -> str:
    doc = fitz.open(str(file_path))
    content = ""
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        if text.strip():
            content += f"\n--- PÁGINA {page_num + 1} ---\n{text}\n"
    
    doc.close()
    return content
```

#### Chunking Legal Inteligente
El sistema utiliza patrones específicos para documentos legales:

```python
self.legal_patterns = {
    'articulo': re.compile(r'ARTÍCULO\s+(\d+[A-Za-z]*)', re.IGNORECASE),
    'capitulo': re.compile(r'CAPÍTULO\s+([IVX]+|\d+)', re.IGNORECASE),
    'titulo': re.compile(r'TÍTULO\s+([IVX]+|\d+)', re.IGNORECASE),
    'seccion': re.compile(r'SECCIÓN\s+([IVX]+|\d+)', re.IGNORECASE),
    'paragrafo': re.compile(r'PÁRRAFO\s+([IVX]+|\d+)', re.IGNORECASE),
    'inciso': re.compile(r'(\d+[A-Za-z]*\.\s+[^0-9]+?)(?=\d+[A-Za-z]*\.|$)'),
    # ... más patrones
}
```

### 3. Estructura de Chunks

Cada chunk legal contiene metadatos completos:

```python
@dataclass
class LegalChunk:
    content: str
    chunk_type: str  # articulo, paragrafo, inciso, titulo, capitulo
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
```

### 4. Generación de Embeddings

#### Modelo Utilizado
- **Modelo:** Qwen3-Embedding-0.6B
- **Dimensión:** 768
- **API:** Hugging Face Inference Endpoints
- **Normalización:** L2 normalization aplicada

#### Procesamiento en Lotes
```python
async def generate_embeddings_batch(self, texts: List[str], batch_size: int = 16) -> List[np.ndarray]:
    # Dividir en lotes de 16 textos
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    
    # Procesar con circuit breaker y rate limiting
    for batch in batches:
        embeddings = await self._generate_embeddings_single_batch(batch)
        # Normalizar embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
```

### 5. Almacenamiento en Supabase

#### Estructura de Base de Datos

**Tabla `legal_documents`:**
```sql
CREATE TABLE legal_documents (
    document_id TEXT PRIMARY KEY,
    title TEXT,
    state TEXT,
    document_type TEXT,
    law_type TEXT,
    effective_date DATE,
    amendment_date DATE,
    status TEXT,
    file_path TEXT,
    file_size BIGINT,
    total_pages INTEGER,
    total_chunks INTEGER,
    metadata JSONB
);
```

**Tabla `legal_chunks`:**
```sql
CREATE TABLE legal_chunks (
    chunk_id SERIAL PRIMARY KEY,
    document_id TEXT REFERENCES legal_documents(document_id),
    content TEXT,
    embedding VECTOR(768),
    article_number TEXT,
    paragraph_number TEXT,
    inciso_number TEXT,
    chapter_title TEXT,
    section_title TEXT,
    law_title TEXT,
    chunk_type TEXT,
    page_number INTEGER,
    word_count INTEGER,
    char_count INTEGER,
    metadata JSONB
);
```

#### Índices Optimizados
```sql
-- Índice vectorial para búsqueda por similitud
CREATE INDEX ON legal_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Índices para filtros
CREATE INDEX idx_legal_chunks_document_id ON legal_chunks (document_id);
CREATE INDEX idx_legal_chunks_chunk_type ON legal_chunks (chunk_type);
CREATE INDEX idx_legal_documents_state ON legal_documents (state);
CREATE INDEX idx_legal_documents_law_type ON legal_documents (law_type);
```

## 📊 Resultados del Procesamiento

### Estadísticas Finales
- **Archivos procesados:** 5,053 de 5,074 (99.6% éxito)
- **Chunks generados:** 275,997
- **Embeddings generados:** 275,977
- **Tiempo total:** 2.98 horas
- **Velocidad:** 0.47 archivos/segundo, 25.7 chunks/segundo

### Distribución por Estado
Los 32 estados de México fueron procesados exitosamente:
- Aguascalientes: 136 archivos
- Baja California: 160 archivos
- Ciudad de México: 170 archivos
- Jalisco: 291 archivos (mayor cantidad)
- Yucatán: 351 archivos (mayor cantidad)
- ... y todos los demás estados

### Tipos de Ley Procesados
- **Constituciones:** Documentos constitucionales
- **Leyes:** Leyes estatales y federales
- **Códigos:** Códigos civiles, penales, etc.
- **Reglamentos:** Reglamentos administrativos
- **Decretos:** Decretos ejecutivos

## 🔍 Sistema de Búsqueda

### Búsqueda Híbrida
El sistema implementa búsqueda híbrida que combina:

1. **Búsqueda Vectorial:** Similitud semántica usando embeddings
2. **Búsqueda por Palabras Clave:** BM25 para términos específicos
3. **Búsqueda por Metadatos:** Filtros por estado, tipo de ley, artículo, etc.

```python
async def search_documents_hybrid(
    self,
    query: str,
    search_type: str = "hybrid",
    filters: Dict[str, Any] = None,
    top_k: int = 20,
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3
) -> Dict[str, Any]:
```

### Funciones RPC de Supabase
```sql
-- Búsqueda híbrida optimizada
CREATE OR REPLACE FUNCTION search_legal_chunks_hybrid(
    query_embedding vector(768),
    search_text text,
    vector_weight float DEFAULT 0.7,
    keyword_weight float DEFAULT 0.3,
    match_count int DEFAULT 20,
    state_filter text DEFAULT NULL,
    law_type_filter text DEFAULT NULL
) RETURNS TABLE(...)
```

## 🛠️ Instalación y Uso

### Requisitos
```bash
pip install -r requirements.txt
```

### Variables de Entorno
```bash
export HF_TOKEN="tu_token_huggingface"
export SUPABASE_URL="tu_url_supabase"
export SUPABASE_KEY="tu_clave_supabase"
```

### Ejecutar Procesamiento
```python
# Procesamiento masivo
python massive_processing_main.py process leyes_de_todos_los_estados \
    --supabase-url $SUPABASE_URL \
    --supabase-key $SUPABASE_KEY \
    --hf-token $HF_TOKEN \
    --max-workers 16 \
    --batch-size 50

# Búsqueda
python massive_processing_main.py search "derechos de las mujeres" \
    --supabase-url $SUPABASE_URL \
    --supabase-key $SUPABASE_KEY \
    --hf-token $HF_TOKEN \
    --search-type hybrid \
    --state jalisco \
    --top-k 10
```

## 🔧 Optimizaciones Implementadas

### 1. Procesamiento Paralelo
- **ThreadPoolExecutor** con 16 workers
- Procesamiento en lotes de 50 PDFs
- Circuit breaker para manejo de errores

### 2. Rate Limiting
- Máximo 10 requests concurrentes a Hugging Face
- Delay de 0.5s entre lotes
- Retry con backoff exponencial

### 3. Chunking Inteligente
- Preservación de jerarquía legal
- Separadores específicos para documentos legales
- Filtrado de chunks muy cortos (< 10 palabras)

### 4. Almacenamiento Eficiente
- Inserción en lotes de 1000 chunks
- Índices optimizados para búsqueda
- Compresión de metadatos en JSONB

## 📈 Monitoreo y Logging

### Logs de Procesamiento
- Logs detallados de cada etapa
- Estadísticas de rendimiento
- Métricas de Hugging Face API

### TensorBoard
```python
# Iniciar monitoreo
python start_tensorboard.py
# Abrir http://localhost:6006
```

## 🎯 Casos de Uso

1. **Búsqueda Semántica:** Encontrar leyes relacionadas con conceptos específicos
2. **Análisis Legal:** Comparar leyes entre estados
3. **Investigación Jurídica:** Localizar artículos específicos
4. **Sistema de Recomendaciones:** Sugerir leyes relacionadas

## 🔮 Próximas Mejoras

1. **Fine-tuning del modelo** con datos legales mexicanos
2. **Implementación de RAG-Anything** para mayor flexibilidad
3. **Búsqueda multimodal** con imágenes de documentos
4. **API REST** para integración con otros sistemas
5. **Dashboard web** para visualización de resultados

## 📝 Notas Técnicas

- El sistema está optimizado para GPU A100 con 160GB VRAM
- Utiliza pgvector para almacenamiento eficiente de embeddings
- Implementa circuit breaker para robustez ante fallos de API
- Chunking preserva la estructura legal para mejor contexto en RAG

## 🤝 Contribuciones

Este proyecto está diseñado para ser extensible y modular. Las contribuciones son bienvenidas, especialmente en:
- Mejoras al chunking legal
- Optimizaciones de rendimiento
- Nuevos tipos de búsqueda
- Integración con más fuentes de datos legales

---

**Desarrollado para procesar y analizar el corpus legal mexicano de manera eficiente y escalable.**
