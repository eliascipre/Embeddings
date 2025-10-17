# Sistema de Embeddings para Documentos Legales Mexicanos

## 📋 Descripción General

Este proyecto implementa un sistema RAG (Retrieval-Augmented Generation) ultra-optimizado para procesar y generar embeddings de documentos legales mexicanos. El sistema está diseñado para manejar miles de documentos PDF de leyes, constituciones, códigos y reglamentos de todos los estados de México, utilizando técnicas avanzadas de chunking legal y generación de embeddings con Hugging Face.

**Comando de ejecución utilizado:**
```bash
python massive_processing_main.py process leyes_de_todos_los_estados \
    --supabase-url $SUPABASE_URL \
    --supabase-key $SUPABASE_KEY \
    --hf-token $HF_TOKEN \
    --max-workers 16 \
    --batch-size 50
```

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

### 2. Configuración del Sistema

**Variables de Entorno Requeridas:**
```bash
export HF_TOKEN="tu_token_huggingface"
export SUPABASE_URL="https://zcxqxrgtmnfixkgeaurj.supabase.co"
export SUPABASE_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

**Configuración de Procesamiento:**
- **Workers paralelos:** 16 (optimizado para A100)
- **Tamaño de lote:** 50 PDFs por lote
- **Batch size HF:** 32 textos por request
- **Max requests concurrentes:** 50
- **Timeout:** 300 segundos
- **Retry attempts:** 5 con backoff exponencial

### 3. Procesamiento de Texto

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
El sistema utiliza patrones específicos para documentos legales mexicanos:

```python
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
```

#### Estrategia de Chunking Jerárquico
El sistema implementa un chunking jerárquico que preserva la estructura legal:

1. **División por páginas** - Cada página se procesa independientemente
2. **División por artículos** - Los artículos se mantienen como unidades completas
3. **División por párrafos** - Los párrafos se extraen dentro de cada artículo
4. **División por incisos** - Los incisos se extraen dentro de cada párrafo
5. **Filtrado de calidad** - Se eliminan chunks muy cortos (< 10 palabras)

### 4. Estructura de Chunks

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

#### Tipos de Chunks Generados
- **Artículos** - Unidades completas de artículos legales
- **Párrafos** - Párrafos dentro de artículos
- **Incisos** - Incisos numerados o con letras
- **Capítulos** - Capítulos de leyes
- **Títulos** - Títulos principales
- **Secciones** - Secciones dentro de capítulos
- **Transitorios** - Disposiciones transitorias
- **Definiciones** - Secciones de definiciones

#### Metadatos Extraídos
- **Información del documento:** título, estado, tipo de ley
- **Estructura legal:** número de artículo, párrafo, inciso
- **Ubicación:** número de página, capítulo, sección
- **Estadísticas:** conteo de palabras y caracteres
- **Fechas:** fecha de vigencia, última reforma
- **Estado:** vigente, reformada, derogada

### 5. Generación de Embeddings

#### Modelo Utilizado
- **Modelo:** Qwen3-Embedding-0.6B
- **Dimensión:** 768
- **API:** Hugging Face Inference Endpoints
- **URL del endpoint:** `https://wfuosp4mnqkimde9.us-east-1.aws.endpoints.huggingface.cloud`
- **Normalización:** L2 normalization aplicada

#### Configuración del Cliente HF
```python
@dataclass
class HFConfig:
    api_url: str = "https://wfuosp4mnqkimde9.us-east-1.aws.endpoints.huggingface.cloud"
    token: str = os.getenv("HF_TOKEN", "")
    max_concurrent_requests: int = 10  # Reducido para evitar 503
    batch_size: int = 16  # Reducido para menor carga
    timeout: int = 300  # 5 minutos timeout
    retry_attempts: int = 5  # Más intentos
    retry_delay: float = 2.0  # Delay inicial mayor
    batch_delay: float = 0.5  # Delay entre lotes
    circuit_breaker_threshold: int = 10  # Errores consecutivos para activar circuit breaker
```

#### Procesamiento en Lotes con Circuit Breaker
```python
async def generate_embeddings_batch(self, texts: List[str], batch_size: int = 16) -> List[np.ndarray]:
    # Dividir en lotes de 16 textos
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    
    # Procesar con circuit breaker y rate limiting
    for i, batch in enumerate(batches):
        # Verificar circuit breaker
        if self.circuit_breaker_active:
            wait_time = min(60, 2 ** min(self.consecutive_errors, 6))  # Max 60s
            await asyncio.sleep(wait_time)
            self.circuit_breaker_active = False
        
        # Procesar lote
        embeddings = await self._generate_embeddings_single_batch(batch)
        
        # Normalizar embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        # Delay entre lotes para evitar saturar el servidor
        if i < len(batches) - 1:
            await asyncio.sleep(self.config.batch_delay)
```

#### Manejo de Errores y Rate Limiting
- **Circuit Breaker:** Se activa después de 10 errores consecutivos
- **Retry con Backoff:** 5 intentos con delay exponencial
- **Rate Limiting:** Delay de 0.5s entre lotes
- **Timeout:** 300 segundos por request
- **Manejo de 503:** Delay extra para errores de servicio no disponible

### 6. Almacenamiento en Supabase

#### Configuración de Supabase
- **URL:** `https://zcxqxrgtmnfixkgeaurj.supabase.co`
- **Clave:** Configurada en variables de entorno
- **Extensión:** pgvector para almacenamiento de embeddings
- **Batch size:** 1000 chunks por inserción

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

#### Funciones RPC de Supabase
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

-- Búsqueda vectorial pura
CREATE OR REPLACE FUNCTION search_legal_chunks_optimized(
    query_embedding vector(768),
    match_threshold float DEFAULT 0.6,
    match_count int DEFAULT 20,
    state_filter text DEFAULT NULL,
    law_type_filter text DEFAULT NULL
) RETURNS TABLE(...)

-- Búsqueda por palabras clave
CREATE OR REPLACE FUNCTION search_legal_chunks_keyword(
    search_text text,
    match_count int DEFAULT 20,
    state_filter text DEFAULT NULL,
    law_type_filter text DEFAULT NULL
) RETURNS TABLE(...)
```

## 📊 Resultados del Procesamiento

### Estadísticas Finales
- **Archivos procesados:** 5,053 de 5,074 (99.6% éxito)
- **Chunks generados:** 275,997
- **Embeddings generados:** 275,977
- **Tiempo total:** 2.98 horas
- **Velocidad:** 0.47 archivos/segundo, 25.7 chunks/segundo

### Configuración de Procesamiento Utilizada
```bash
# Comando ejecutado
python massive_processing_main.py process leyes_de_todos_los_estados \
    --supabase-url $SUPABASE_URL \
    --supabase-key $SUPABASE_KEY \
    --hf-token $HF_TOKEN \
    --max-workers 16 \
    --batch-size 50

# Parámetros de configuración
- Workers paralelos: 16
- Tamaño de lote: 50 PDFs
- Batch size HF: 32 textos
- Max requests concurrentes: 50
- Timeout: 300 segundos
- Retry attempts: 5
- Circuit breaker threshold: 10 errores
```

### Optimizaciones Implementadas
1. **Procesamiento Paralelo Masivo**
   - ThreadPoolExecutor con 16 workers
   - Procesamiento en lotes de 50 PDFs
   - Circuit breaker para manejo de errores

2. **Rate Limiting Inteligente**
   - Máximo 10 requests concurrentes a Hugging Face
   - Delay de 0.5s entre lotes
   - Retry con backoff exponencial

3. **Chunking Legal Especializado**
   - Preservación de jerarquía legal
   - Separadores específicos para documentos legales
   - Filtrado de chunks muy cortos (< 10 palabras)

4. **Almacenamiento Eficiente**
   - Inserción en lotes de 1000 chunks
   - Índices optimizados para búsqueda
   - Compresión de metadatos en JSONB

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

### Requisitos del Sistema
- **Python:** 3.8+
- **GPU:** Recomendado A100 o similar (RTX 5090 mínimo)
- **RAM:** 32GB+ recomendado
- **Almacenamiento:** 10GB+ para datos

### Instalación
```bash
# Clonar repositorio
git clone <repository-url>
cd Embeddings

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### Variables de Entorno
```bash
# Configurar variables de entorno
export HF_TOKEN="tu_token_huggingface"
export SUPABASE_URL="https://zcxqxrgtmnfixkgeaurj.supabase.co"
export SUPABASE_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

# O crear archivo .env
echo "HF_TOKEN=tu_token_aqui" > .env
echo "SUPABASE_URL=https://zcxqxrgtmnfixkgeaurj.supabase.co" >> .env
echo "SUPABASE_KEY=tu_clave_aqui" >> .env
```

### Ejecutar Procesamiento
```bash
# Procesamiento masivo (comando utilizado)
python massive_processing_main.py process leyes_de_todos_los_estados \
    --supabase-url $SUPABASE_URL \
    --supabase-key $SUPABASE_KEY \
    --hf-token $HF_TOKEN \
    --max-workers 16 \
    --batch-size 50

# Búsqueda híbrida
python massive_processing_main.py search "derechos de las mujeres" \
    --supabase-url $SUPABASE_URL \
    --supabase-key $SUPABASE_KEY \
    --hf-token $HF_TOKEN \
    --search-type hybrid \
    --state jalisco \
    --top-k 10

# Búsqueda por metadatos
python massive_processing_main.py search "artículo 1" \
    --supabase-url $SUPABASE_URL \
    --supabase-key $SUPABASE_KEY \
    --hf-token $HF_TOKEN \
    --search-type metadata \
    --state jalisco \
    --top-k 5

# Ver estadísticas de la base de datos
python massive_processing_main.py stats \
    --supabase-url $SUPABASE_URL \
    --supabase-key $SUPABASE_KEY \
    --hf-token $HF_TOKEN
```

### Uso Programático
```python
import asyncio
from src.optimized_rag_system import OptimizedRAGSystem

async def main():
    async with OptimizedRAGSystem(
        supabase_url="tu_url",
        supabase_key="tu_clave", 
        hf_token="tu_token"
    ) as rag_system:
        # Búsqueda híbrida
        results = await rag_system.search_documents_hybrid(
            query="derechos de las mujeres",
            search_type="hybrid",
            filters={"state": "jalisco"},
            top_k=10
        )
        print(f"Encontrados {results['total_found']} resultados")

asyncio.run(main())
```

## 🔧 Optimizaciones Implementadas

### 1. Procesamiento Paralelo Masivo
- **ThreadPoolExecutor** con 16 workers
- Procesamiento en lotes de 50 PDFs
- Circuit breaker para manejo de errores
- Procesamiento asíncrono con aiohttp

### 2. Rate Limiting Inteligente
- Máximo 10 requests concurrentes a Hugging Face
- Delay de 0.5s entre lotes
- Retry con backoff exponencial
- Circuit breaker después de 10 errores consecutivos

### 3. Chunking Legal Especializado
- Preservación de jerarquía legal
- Separadores específicos para documentos legales mexicanos
- Filtrado de chunks muy cortos (< 10 palabras)
- Detección automática de estructura legal

### 4. Almacenamiento Eficiente
- Inserción en lotes de 1000 chunks
- Índices optimizados para búsqueda
- Compresión de metadatos en JSONB
- Funciones RPC optimizadas en Supabase

### 5. Manejo de Errores Robusto
- Circuit breaker para APIs externas
- Retry automático con backoff exponencial
- Logging detallado de errores
- Recuperación automática de fallos

### 6. Optimización de Recursos
- Uso eficiente de memoria con procesamiento en lotes
- Conexiones HTTP reutilizables
- Cache de embeddings cuando sea posible
- Monitoreo de uso de GPU y CPU

## 📈 Monitoreo y Logging

### Logs de Procesamiento
- Logs detallados de cada etapa en `massive_processing.log`
- Estadísticas de rendimiento en tiempo real
- Métricas de Hugging Face API
- Monitoreo de errores y recuperación

### Archivos de Log Generados
```
logs/
├── massive_processing.log          # Log principal del procesamiento
├── exploracion_siem.log           # Log de exploración de datos
├── limpieza_agresiva.log          # Log de limpieza de datos
├── limpieza_database.log          # Log de limpieza de base de datos
├── optimized_siem_pipeline.log    # Log del pipeline optimizado
├── setup_and_run.log             # Log de configuración y ejecución
└── tensorboard/                   # Logs de TensorBoard
    ├── events.out.tfevents.*
    └── ...
```

### TensorBoard
```python
# Iniciar monitoreo (si está disponible)
python start_tensorboard.py
# Abrir http://localhost:6006
```

### Métricas Monitoreadas
- **Procesamiento:** archivos/segundo, chunks/segundo, embeddings/segundo
- **API Hugging Face:** requests/segundo, tokens/segundo, errores
- **Base de datos:** inserción/segundo, tamaño de lotes
- **Sistema:** uso de CPU, memoria, GPU

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
6. **Mejoras en chunking** para documentos más complejos
7. **Cache inteligente** de embeddings frecuentes
8. **Análisis de sentimientos** en documentos legales

## 📝 Notas Técnicas

- El sistema está optimizado para GPU A100 con 160GB VRAM (RTX 5090 mínimo)
- Utiliza pgvector para almacenamiento eficiente de embeddings
- Implementa circuit breaker para robustez ante fallos de API
- Chunking preserva la estructura legal para mejor contexto en RAG
- Procesamiento asíncrono para máxima eficiencia
- Rate limiting inteligente para evitar errores 503

## 🚀 Transferencia a Otros Proyectos

Este sistema puede ser adaptado para otros proyectos similares que utilicen:
- **Hugging Face endpoints** para generación de embeddings
- **Supabase** como base de datos vectorial
- **Documentos PDF** con estructura jerárquica
- **Procesamiento masivo** de documentos

### Pasos para Adaptación
1. **Configurar variables de entorno** con las credenciales del nuevo proyecto
2. **Adaptar patrones de chunking** según el tipo de documentos
3. **Modificar metadatos** según las necesidades del dominio
4. **Ajustar parámetros** de procesamiento según los recursos disponibles
5. **Personalizar funciones RPC** de Supabase según el esquema de datos

## 🤝 Contribuciones

Este proyecto está diseñado para ser extensible y modular. Las contribuciones son bienvenidas, especialmente en:
- Mejoras al chunking legal
- Optimizaciones de rendimiento
- Nuevos tipos de búsqueda
- Integración con más fuentes de datos legales
- Adaptación para otros dominios (médico, técnico, etc.)

---

**Desarrollado para procesar y analizar el corpus legal mexicano de manera eficiente y escalable.**
