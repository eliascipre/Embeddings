-- =====================================================
-- ESQUEMA SQL OPTIMIZADO PARA VECTORDB_COMERCIO_EXTERIOR
-- Optimizado para GPU A100, HuggingFace API y RAG híbrido
-- =====================================================

-- Habilitar extensión pgvector para embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- =====================================================
-- ELIMINAR TABLAS EXISTENTES (SI EXISTEN)
-- =====================================================

-- Eliminar tablas en orden inverso de dependencias
DROP TABLE IF EXISTS siem_processing_logs CASCADE;
DROP TABLE IF EXISTS siem_metadata CASCADE;
DROP TABLE IF EXISTS siem_embeddings CASCADE;
DROP TABLE IF EXISTS siem_chunks CASCADE;
DROP TABLE IF EXISTS siem_documents CASCADE;

-- Eliminar vistas
DROP VIEW IF EXISTS processing_stats CASCADE;
DROP VIEW IF EXISTS documents_with_metadata CASCADE;

-- Eliminar funciones RPC
DROP FUNCTION IF EXISTS search_embeddings_cosine(VECTOR(1024), FLOAT, INT) CASCADE;
DROP FUNCTION IF EXISTS search_hybrid(VECTOR(1024), TEXT, FLOAT, INT) CASCADE;
DROP FUNCTION IF EXISTS search_by_metadata(VARCHAR(100), VARCHAR(100), TEXT[], TIMESTAMP WITH TIME ZONE, TIMESTAMP WITH TIME ZONE) CASCADE;
DROP FUNCTION IF EXISTS update_updated_at_column() CASCADE;

-- =====================================================
-- TABLA PRINCIPAL DE DOCUMENTOS
-- =====================================================
CREATE TABLE IF NOT EXISTS siem_documents (
    id SERIAL PRIMARY KEY,
    file_name VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    file_hash VARCHAR(64) UNIQUE NOT NULL, -- MD5 hash para evitar duplicados
    file_size BIGINT NOT NULL,
    file_extension VARCHAR(10) NOT NULL,
    text_length INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    
    -- Metadatos de procesamiento
    processing_status VARCHAR(20) DEFAULT 'pending', -- pending, processing, completed, failed
    processing_started_at TIMESTAMP WITH TIME ZONE,
    processing_completed_at TIMESTAMP WITH TIME ZONE,
    processing_duration_seconds INTEGER,
    
    -- Metadatos del archivo
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Índices para búsqueda rápida
    CONSTRAINT chk_processing_status CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed'))
);

-- =====================================================
-- TABLA DE CHUNKS CON ESTRUCTURA LEGAL PRESERVADA
-- =====================================================
CREATE TABLE IF NOT EXISTS siem_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES siem_documents(id) ON DELETE CASCADE,
    
    -- Contenido del chunk
    chunk_text TEXT NOT NULL,
    chunk_text_clean TEXT, -- Texto limpio para búsqueda
    
    -- Estructura legal preservada
    chunk_type VARCHAR(50) NOT NULL, -- section, paragraph, article, etc.
    legal_hierarchy JSONB, -- Estructura jerárquica del documento legal
    chunk_index INTEGER NOT NULL,
    section_index INTEGER,
    
    -- Metadatos de chunking
    token_count INTEGER NOT NULL,
    character_count INTEGER NOT NULL,
    word_count INTEGER NOT NULL,
    
    -- Identificadores únicos
    chunk_hash VARCHAR(64) UNIQUE NOT NULL, -- Hash del contenido
    embedding_id VARCHAR(255) UNIQUE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_chunk_type CHECK (chunk_type IN ('section', 'paragraph', 'article', 'chapter', 'title', 'clause', 'subsection', 'legal_section', 'legal_paragraph', 'legal_article'))
);

-- =====================================================
-- TABLA DE EMBEDDINGS OPTIMIZADA PARA GPU A100
-- =====================================================
CREATE TABLE IF NOT EXISTS siem_embeddings (
    id SERIAL PRIMARY KEY,
    chunk_id INTEGER REFERENCES siem_chunks(id) ON DELETE CASCADE,
    
    -- Vector de embedding (1024 dimensiones para Qwen3-Embedding-0.6B - máxima calidad)
    embedding_vector VECTOR(1024) NOT NULL,
    
    -- Metadatos del embedding
    embedding_id VARCHAR(255) UNIQUE NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(50),
    dimension INTEGER NOT NULL DEFAULT 1024,
    
    -- Metadatos de generación
    generation_method VARCHAR(50) DEFAULT 'huggingface_api', -- huggingface_api, local_model
    api_endpoint VARCHAR(255),
    generation_time_ms INTEGER,
    batch_id VARCHAR(255), -- Para tracking de lotes
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_dimension CHECK (dimension = 1024),
    CONSTRAINT chk_generation_method CHECK (generation_method IN ('huggingface_api', 'local_model'))
);

-- =====================================================
-- TABLA DE METADATOS ENRICHECIDOS
-- =====================================================
CREATE TABLE IF NOT EXISTS siem_metadata (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES siem_documents(id) ON DELETE CASCADE,
    
    -- Categorización automática
    category VARCHAR(100), -- ANAM, AWB, BL, CP, Marco_jurídico, etc.
    subcategory VARCHAR(100),
    legal_type VARCHAR(100), -- Ley, Reglamento, Decreto, Acuerdo, etc.
    
    -- Extracción de entidades
    entities JSONB, -- Entidades nombradas extraídas
    keywords TEXT[], -- Palabras clave extraídas
    topics TEXT[], -- Temas identificados
    
    -- Resumen y análisis
    summary TEXT,
    key_points TEXT[],
    legal_references TEXT[], -- Referencias a otras leyes/artículos
    
    -- Metadatos de calidad
    confidence_score DECIMAL(3,2), -- 0.00 a 1.00
    quality_indicators JSONB,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- TABLA DE PROCESAMIENTO Y MONITOREO
-- =====================================================
CREATE TABLE IF NOT EXISTS siem_processing_logs (
    id SERIAL PRIMARY KEY,
    batch_id VARCHAR(255) NOT NULL,
    document_id INTEGER REFERENCES siem_documents(id) ON DELETE CASCADE,
    
    -- Estado del procesamiento
    stage VARCHAR(50) NOT NULL, -- extraction, chunking, embedding, storage
    status VARCHAR(20) NOT NULL, -- started, completed, failed, retrying
    error_message TEXT,
    
    -- Métricas de rendimiento
    processing_time_ms INTEGER,
    tokens_processed INTEGER,
    chunks_created INTEGER,
    embeddings_generated INTEGER,
    
    -- Metadatos de API
    api_calls_made INTEGER DEFAULT 0,
    api_rate_limit_remaining INTEGER,
    api_response_time_ms INTEGER,
    
    -- Timestamps
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Constraints
    CONSTRAINT chk_stage CHECK (stage IN ('extraction', 'chunking', 'embedding', 'storage')),
    CONSTRAINT chk_status CHECK (status IN ('started', 'completed', 'failed', 'retrying'))
);

-- =====================================================
-- ÍNDICES OPTIMIZADOS PARA RENDIMIENTO
-- =====================================================

-- Índices para búsqueda por metadatos
CREATE INDEX IF NOT EXISTS idx_documents_file_name ON siem_documents(file_name);
CREATE INDEX IF NOT EXISTS idx_documents_file_extension ON siem_documents(file_extension);
CREATE INDEX IF NOT EXISTS idx_documents_processing_status ON siem_documents(processing_status);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON siem_documents(created_at);
CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON siem_documents(file_hash);

-- Índices para chunks
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON siem_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_type ON siem_chunks(chunk_type);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_id ON siem_chunks(embedding_id);
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_hash ON siem_chunks(chunk_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_legal_hierarchy ON siem_chunks USING GIN(legal_hierarchy);

-- Índices para embeddings (optimizados para pgvector)
CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id ON siem_embeddings(chunk_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_embedding_id ON siem_embeddings(embedding_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_model_name ON siem_embeddings(model_name);
CREATE INDEX IF NOT EXISTS idx_embeddings_batch_id ON siem_embeddings(batch_id);

-- Índice vectorial para búsqueda por similitud (IVFFlat para mejor rendimiento)
CREATE INDEX IF NOT EXISTS idx_embeddings_vector_cosine 
ON siem_embeddings USING ivfflat (embedding_vector vector_cosine_ops) 
WITH (lists = 100);

-- Índice vectorial para búsqueda por distancia euclidiana
CREATE INDEX IF NOT EXISTS idx_embeddings_vector_l2 
ON siem_embeddings USING ivfflat (embedding_vector vector_l2_ops) 
WITH (lists = 100);

-- Índices para metadatos
CREATE INDEX IF NOT EXISTS idx_metadata_document_id ON siem_metadata(document_id);
CREATE INDEX IF NOT EXISTS idx_metadata_category ON siem_metadata(category);
CREATE INDEX IF NOT EXISTS idx_metadata_legal_type ON siem_metadata(legal_type);
CREATE INDEX IF NOT EXISTS idx_metadata_keywords ON siem_metadata USING GIN(keywords);
CREATE INDEX IF NOT EXISTS idx_metadata_topics ON siem_metadata USING GIN(topics);
CREATE INDEX IF NOT EXISTS idx_metadata_entities ON siem_metadata USING GIN(entities);

-- Índices para logs de procesamiento
CREATE INDEX IF NOT EXISTS idx_processing_logs_batch_id ON siem_processing_logs(batch_id);
CREATE INDEX IF NOT EXISTS idx_processing_logs_document_id ON siem_processing_logs(document_id);
CREATE INDEX IF NOT EXISTS idx_processing_logs_stage ON siem_processing_logs(stage);
CREATE INDEX IF NOT EXISTS idx_processing_logs_status ON siem_processing_logs(status);
CREATE INDEX IF NOT EXISTS idx_processing_logs_started_at ON siem_processing_logs(started_at);

-- =====================================================
-- FUNCIONES RPC PARA BÚSQUEDA HÍBRIDA
-- =====================================================

-- Función para búsqueda vectorial por similitud
CREATE OR REPLACE FUNCTION search_embeddings_cosine(
    query_embedding VECTOR(1024),
    match_threshold FLOAT DEFAULT 0.7,
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    chunk_id INT,
    chunk_text TEXT,
    similarity FLOAT,
    document_id INT,
    file_name VARCHAR(255)
)
LANGUAGE SQL STABLE
AS $$
    SELECT 
        c.id as chunk_id,
        c.chunk_text,
        1 - (e.embedding_vector <=> query_embedding) as similarity,
        c.document_id,
        d.file_name
    FROM siem_embeddings e
    JOIN siem_chunks c ON e.chunk_id = c.id
    JOIN siem_documents d ON c.document_id = d.id
    WHERE 1 - (e.embedding_vector <=> query_embedding) > match_threshold
    ORDER BY e.embedding_vector <=> query_embedding
    LIMIT match_count;
$$;

-- Función para búsqueda híbrida (vectorial + texto)
CREATE OR REPLACE FUNCTION search_hybrid(
    query_embedding VECTOR(1024),
    query_text TEXT,
    match_threshold FLOAT DEFAULT 0.7,
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    chunk_id INT,
    chunk_text TEXT,
    similarity FLOAT,
    text_rank FLOAT,
    combined_score FLOAT,
    document_id INT,
    file_name VARCHAR(255),
    category VARCHAR(100)
)
LANGUAGE SQL STABLE
AS $$
    WITH vector_search AS (
        SELECT 
            c.id as chunk_id,
            c.chunk_text,
            1 - (e.embedding_vector <=> query_embedding) as similarity,
            c.document_id,
            d.file_name,
            m.category
        FROM siem_embeddings e
        JOIN siem_chunks c ON e.chunk_id = c.id
        JOIN siem_documents d ON c.document_id = d.id
        LEFT JOIN siem_metadata m ON d.id = m.document_id
        WHERE 1 - (e.embedding_vector <=> query_embedding) > match_threshold
    ),
    text_search AS (
        SELECT 
            c.id as chunk_id,
            ts_rank(to_tsvector('spanish', c.chunk_text), plainto_tsquery('spanish', query_text)) as text_rank,
            c.document_id
        FROM siem_chunks c
        WHERE to_tsvector('spanish', c.chunk_text) @@ plainto_tsquery('spanish', query_text)
    )
    SELECT 
        vs.chunk_id,
        vs.chunk_text,
        vs.similarity,
        COALESCE(ts.text_rank, 0) as text_rank,
        (vs.similarity * 0.7 + COALESCE(ts.text_rank, 0) * 0.3) as combined_score,
        vs.document_id,
        vs.file_name,
        vs.category
    FROM vector_search vs
    LEFT JOIN text_search ts ON vs.chunk_id = ts.chunk_id
    ORDER BY combined_score DESC
    LIMIT match_count;
$$;

-- Función para búsqueda por metadatos
CREATE OR REPLACE FUNCTION search_by_metadata(
    category_filter VARCHAR(100) DEFAULT NULL,
    legal_type_filter VARCHAR(100) DEFAULT NULL,
    keywords_filter TEXT[] DEFAULT NULL,
    date_from TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    date_to TIMESTAMP WITH TIME ZONE DEFAULT NULL
)
RETURNS TABLE (
    document_id INT,
    file_name VARCHAR(255),
    category VARCHAR(100),
    legal_type VARCHAR(100),
    keywords TEXT[],
    created_at TIMESTAMP WITH TIME ZONE
)
LANGUAGE SQL STABLE
AS $$
    SELECT 
        d.id as document_id,
        d.file_name,
        m.category,
        m.legal_type,
        m.keywords,
        d.created_at
    FROM siem_documents d
    LEFT JOIN siem_metadata m ON d.id = m.document_id
    WHERE 
        (category_filter IS NULL OR m.category = category_filter)
        AND (legal_type_filter IS NULL OR m.legal_type = legal_type_filter)
        AND (keywords_filter IS NULL OR m.keywords && keywords_filter)
        AND (date_from IS NULL OR d.created_at >= date_from)
        AND (date_to IS NULL OR d.created_at <= date_to)
    ORDER BY d.created_at DESC;
$$;

-- =====================================================
-- TRIGGERS PARA MANTENIMIENTO AUTOMÁTICO
-- =====================================================

-- Trigger para actualizar updated_at automáticamente
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Aplicar trigger a todas las tablas
CREATE TRIGGER update_siem_documents_updated_at BEFORE UPDATE ON siem_documents FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_siem_chunks_updated_at BEFORE UPDATE ON siem_chunks FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_siem_embeddings_updated_at BEFORE UPDATE ON siem_embeddings FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_siem_metadata_updated_at BEFORE UPDATE ON siem_metadata FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- VISTAS PARA CONSULTAS FRECUENTES
-- =====================================================

-- Vista para estadísticas de procesamiento
CREATE OR REPLACE VIEW processing_stats AS
SELECT 
    COUNT(*) as total_documents,
    COUNT(CASE WHEN processing_status = 'completed' THEN 1 END) as completed_documents,
    COUNT(CASE WHEN processing_status = 'failed' THEN 1 END) as failed_documents,
    COUNT(CASE WHEN processing_status = 'processing' THEN 1 END) as processing_documents,
    AVG(processing_duration_seconds) as avg_processing_time_seconds,
    SUM(chunk_count) as total_chunks,
    SUM(token_count) as total_tokens
FROM siem_documents;

-- Vista para documentos con metadatos completos
CREATE OR REPLACE VIEW documents_with_metadata AS
SELECT 
    d.*,
    m.category,
    m.subcategory,
    m.legal_type,
    m.keywords,
    m.topics,
    m.summary,
    m.confidence_score
FROM siem_documents d
LEFT JOIN siem_metadata m ON d.id = m.document_id;

-- =====================================================
-- COMENTARIOS Y DOCUMENTACIÓN
-- =====================================================

COMMENT ON TABLE siem_documents IS 'Documentos principales del sistema SIEM con metadatos de procesamiento';
COMMENT ON TABLE siem_chunks IS 'Chunks de texto con estructura legal preservada';
COMMENT ON TABLE siem_embeddings IS 'Vectores de embeddings optimizados para búsqueda semántica';
COMMENT ON TABLE siem_metadata IS 'Metadatos enriquecidos y categorización automática';
COMMENT ON TABLE siem_processing_logs IS 'Logs detallados de procesamiento para monitoreo';

COMMENT ON FUNCTION search_embeddings_cosine IS 'Búsqueda vectorial por similitud coseno';
COMMENT ON FUNCTION search_hybrid IS 'Búsqueda híbrida combinando vectorial y texto';
COMMENT ON FUNCTION search_by_metadata IS 'Búsqueda por metadatos y filtros';

-- =====================================================
-- FIN DEL ESQUEMA
-- =====================================================
