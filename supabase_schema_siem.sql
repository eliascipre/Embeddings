-- =====================================================
-- ESQUEMA SQL OPTIMIZADO PARA SIEM - COMERCIO EXTERIOR
-- Optimizado para GPU A100, HuggingFace API y RAG híbrido
-- Procesamiento masivo con 16 workers y lotes de 50 archivos
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
DROP VIEW IF EXISTS siem_processing_stats CASCADE;
DROP VIEW IF EXISTS siem_documents_with_metadata CASCADE;

-- Eliminar funciones RPC
DROP FUNCTION IF EXISTS search_siem_embeddings_cosine(VECTOR(1024), FLOAT, INT) CASCADE;
DROP FUNCTION IF EXISTS search_siem_hybrid(VECTOR(1024), TEXT, FLOAT, FLOAT, INT) CASCADE;
DROP FUNCTION IF EXISTS search_siem_by_metadata(TEXT, TEXT, TEXT[], TIMESTAMP WITH TIME ZONE, TIMESTAMP WITH TIME ZONE) CASCADE;
DROP FUNCTION IF EXISTS search_siem_keyword(TEXT, INT, TEXT, TEXT) CASCADE;
DROP FUNCTION IF EXISTS update_siem_updated_at_column() CASCADE;

-- =====================================================
-- TABLA PRINCIPAL DE DOCUMENTOS SIEM (OPTIMIZADA)
-- =====================================================
CREATE TABLE IF NOT EXISTS siem_documents (
    id SERIAL PRIMARY KEY,
    file_name VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    file_hash VARCHAR(64) UNIQUE NOT NULL, -- MD5 hash para evitar duplicados
    file_size BIGINT NOT NULL,
    file_extension VARCHAR(10) NOT NULL,
    
    -- Categorización SIEM optimizada
    category VARCHAR(50), -- ANAM, AWB, BL, CP, Marco_jurídico
    comercio_type VARCHAR(50), -- dof, reglamento, resolucion, tratado, convenio
    
    -- Metadatos de procesamiento
    processing_status VARCHAR(20) DEFAULT 'pending',
    processing_started_at TIMESTAMP WITH TIME ZONE,
    processing_completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Metadatos del archivo
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT chk_processing_status CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed'))
);

-- =====================================================
-- TABLA DE CHUNKS OPTIMIZADA PARA BÚSQUEDA VECTORIAL
-- =====================================================
CREATE TABLE IF NOT EXISTS siem_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES siem_documents(id) ON DELETE CASCADE,
    
    -- Contenido del chunk (optimizado para búsqueda)
    chunk_text TEXT NOT NULL,
    chunk_text_clean TEXT NOT NULL, -- Texto limpio para búsqueda (siempre presente)
    
    -- Estructura de comercio exterior
    chunk_type VARCHAR(30) NOT NULL, -- articulo, paragrafo, anexo, etc.
    chunk_index INTEGER NOT NULL,
    
    -- Metadatos de chunking (simplificados)
    word_count INTEGER NOT NULL,
    
    -- Identificadores únicos
    chunk_hash VARCHAR(64) UNIQUE NOT NULL,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints simplificados
    CONSTRAINT chk_chunk_type CHECK (chunk_type IN (
        'articulo', 'paragrafo', 'anexo', 'titulo', 'capitulo', 'seccion', 
        'resolucion', 'reglamento', 'ley', 'codigo', 'decreto', 'acuerdo', 
        'convenio', 'tratado', 'dof', 'paragraph', 'section'
    ))
);

-- =====================================================
-- TABLA DE EMBEDDINGS OPTIMIZADA PARA BÚSQUEDA VECTORIAL
-- =====================================================
CREATE TABLE IF NOT EXISTS siem_embeddings (
    id SERIAL PRIMARY KEY,
    chunk_id INTEGER REFERENCES siem_chunks(id) ON DELETE CASCADE,
    
    -- Vector de embedding (1024 dimensiones para Qwen3-Embedding-0.6B)
    embedding_vector VECTOR(1024) NOT NULL,
    
    -- Metadatos del embedding (simplificados)
    model_name VARCHAR(50) NOT NULL DEFAULT 'Qwen3-0.6B',
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- TABLA DE METADATOS SIMPLIFICADA
-- =====================================================
CREATE TABLE IF NOT EXISTS siem_metadata (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES siem_documents(id) ON DELETE CASCADE,
    
    -- Metadatos esenciales de comercio exterior
    pais VARCHAR(50), -- País para tratados internacionales
    fecha_publicacion DATE,
    numero_oficial VARCHAR(100), -- Número de DOF, etc.
    autoridad_emisora VARCHAR(100), -- SHCP, SAT, etc.
    
    -- Metadatos de búsqueda
    keywords TEXT[], -- Palabras clave extraídas
    summary TEXT, -- Resumen del documento
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- ÍNDICES OPTIMIZADOS PARA BÚSQUEDA VECTORIAL Y SQL
-- =====================================================

-- Índices para siem_documents (esenciales)
CREATE INDEX idx_siem_documents_file_hash ON siem_documents (file_hash);
CREATE INDEX idx_siem_documents_category ON siem_documents (category);
CREATE INDEX idx_siem_documents_comercio_type ON siem_documents (comercio_type);
CREATE INDEX idx_siem_documents_processing_status ON siem_documents (processing_status);

-- Índices para siem_chunks (optimizados para búsqueda)
CREATE INDEX idx_siem_chunks_document_id ON siem_chunks (document_id);
CREATE INDEX idx_siem_chunks_chunk_type ON siem_chunks (chunk_type);
CREATE INDEX idx_siem_chunks_chunk_hash ON siem_chunks (chunk_hash);

-- Índice vectorial ULTRA-OPTIMIZADO para búsqueda de milisegundos
CREATE INDEX idx_siem_embeddings_vector ON siem_embeddings 
USING ivfflat (embedding_vector vector_cosine_ops) WITH (lists = 1000);

-- Índice HNSW para búsqueda ultra-rápida (PostgreSQL 14+)
-- CREATE INDEX idx_siem_embeddings_hnsw ON siem_embeddings 
-- USING hnsw (embedding_vector vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- Índices para siem_embeddings (esenciales)
CREATE INDEX idx_siem_embeddings_chunk_id ON siem_embeddings (chunk_id);

-- Índices para siem_metadata (optimizados para búsqueda)
CREATE INDEX idx_siem_metadata_document_id ON siem_metadata (document_id);
CREATE INDEX idx_siem_metadata_pais ON siem_metadata (pais);
CREATE INDEX idx_siem_metadata_autoridad_emisora ON siem_metadata (autoridad_emisora);
CREATE INDEX idx_siem_metadata_keywords ON siem_metadata USING GIN (keywords);

-- Índices compuestos ULTRA-OPTIMIZADOS para búsqueda híbrida de milisegundos
CREATE INDEX idx_siem_chunks_search ON siem_chunks (chunk_type, word_count) 
INCLUDE (chunk_text_clean, document_id);

-- Índice para búsqueda de texto ultra-rápida
CREATE INDEX idx_siem_chunks_text_search ON siem_chunks 
USING gin (to_tsvector('spanish', chunk_text_clean));

-- Índice compuesto para joins ultra-rápidos
CREATE INDEX idx_siem_embeddings_chunk_doc ON siem_embeddings (chunk_id) 
INCLUDE (embedding_vector);

-- =====================================================
-- VISTAS OPTIMIZADAS
-- =====================================================

-- Vista de estadísticas de procesamiento
CREATE VIEW siem_processing_stats AS
SELECT 
    COUNT(*) as total_documents,
    COUNT(CASE WHEN processing_status = 'completed' THEN 1 END) as completed_documents,
    COUNT(CASE WHEN processing_status = 'failed' THEN 1 END) as failed_documents,
    COUNT(CASE WHEN processing_status = 'processing' THEN 1 END) as processing_documents,
    COUNT(CASE WHEN processing_status = 'pending' THEN 1 END) as pending_documents,
    SUM(file_size) as total_size_bytes,
    SUM(file_size) / (1024 * 1024) as total_size_mb,
    (SELECT COUNT(*) FROM siem_chunks) as total_chunks,
    AVG(EXTRACT(EPOCH FROM (processing_completed_at - processing_started_at))) as avg_processing_time_seconds,
    MAX(processing_completed_at) as last_processing_completed
FROM siem_documents;

-- Vista de documentos con metadatos
CREATE VIEW siem_documents_with_metadata AS
SELECT 
    d.*,
    m.pais,
    m.fecha_publicacion,
    m.autoridad_emisora,
    m.keywords,
    m.summary
FROM siem_documents d
LEFT JOIN siem_metadata m ON d.id = m.document_id;

-- =====================================================
-- FUNCIONES RPC OPTIMIZADAS
-- =====================================================

-- Función ULTRA-OPTIMIZADA para búsqueda vectorial de milisegundos
CREATE OR REPLACE FUNCTION search_siem_embeddings_cosine(
    query_embedding VECTOR(1024),
    match_threshold FLOAT DEFAULT 0.7,
    match_count INT DEFAULT 20
)
RETURNS TABLE (
    chunk_id INT,
    document_id INT,
    file_name VARCHAR(255),
    chunk_text TEXT,
    chunk_type VARCHAR(30),
    similarity_score FLOAT,
    metadata JSONB
)
LANGUAGE SQL
STABLE
AS $$
    SELECT 
        c.id as chunk_id,
        c.document_id,
        d.file_name,
        c.chunk_text,
        c.chunk_type,
        1 - (e.embedding_vector <=> query_embedding) as similarity_score,
        jsonb_build_object(
            'category', d.category,
            'comercio_type', d.comercio_type,
            'file_extension', d.file_extension
        ) as metadata
    FROM siem_chunks c
    JOIN siem_documents d ON c.document_id = d.id
    JOIN siem_embeddings e ON c.id = e.chunk_id
    WHERE 1 - (e.embedding_vector <=> query_embedding) > match_threshold
    ORDER BY e.embedding_vector <=> query_embedding
    LIMIT match_count;
$$;

-- Función para búsqueda híbrida optimizada
CREATE OR REPLACE FUNCTION search_siem_hybrid(
    query_embedding VECTOR(1024),
    search_text TEXT,
    vector_weight FLOAT DEFAULT 0.7,
    keyword_weight FLOAT DEFAULT 0.3,
    match_count INT DEFAULT 20
)
RETURNS TABLE (
    chunk_id INT,
    document_id INT,
    file_name VARCHAR(255),
    chunk_text TEXT,
    chunk_type VARCHAR(50),
    similarity_score FLOAT,
    vector_score FLOAT,
    keyword_score FLOAT,
    combined_score FLOAT,
    comercio_hierarchy JSONB,
    metadata JSONB
)
LANGUAGE SQL
AS $$
    WITH vector_results AS (
        SELECT 
            c.id as chunk_id,
            c.document_id,
            d.file_name,
            c.chunk_text,
            c.chunk_type,
            1 - (e.embedding_vector <=> query_embedding) as vector_score,
            NULL::JSONB as comercio_hierarchy,
            jsonb_build_object(
                'category', d.category,
                'comercio_type', d.comercio_type,
                'file_extension', d.file_extension,
                'created_at', d.created_at
            ) as metadata
        FROM siem_chunks c
        JOIN siem_documents d ON c.document_id = d.id
        JOIN siem_embeddings e ON c.id = e.chunk_id
        WHERE 1 - (e.embedding_vector <=> query_embedding) > 0.5
    ),
    keyword_results AS (
        SELECT 
            c.id as chunk_id,
            c.document_id,
            d.file_name,
            c.chunk_text,
            c.chunk_type,
            CASE 
                WHEN c.chunk_text_clean ILIKE '%' || search_text || '%' THEN 1.0
                ELSE 0.5
            END as keyword_score,
            NULL::JSONB as comercio_hierarchy,
            jsonb_build_object(
                'category', d.category,
                'comercio_type', d.comercio_type,
                'file_extension', d.file_extension,
                'created_at', d.created_at
            ) as metadata
        FROM siem_chunks c
        JOIN siem_documents d ON c.document_id = d.id
        WHERE c.chunk_text_clean ILIKE '%' || search_text || '%'
    ),
    combined_results AS (
        SELECT 
            COALESCE(v.chunk_id, k.chunk_id) as chunk_id,
            COALESCE(v.document_id, k.document_id) as document_id,
            COALESCE(v.file_name, k.file_name) as file_name,
            COALESCE(v.chunk_text, k.chunk_text) as chunk_text,
            COALESCE(v.chunk_type, k.chunk_type) as chunk_type,
            COALESCE(v.vector_score, 0.0) as vector_score,
            COALESCE(k.keyword_score, 0.0) as keyword_score,
            (COALESCE(v.vector_score, 0.0) * vector_weight + COALESCE(k.keyword_score, 0.0) * keyword_weight) as combined_score,
            COALESCE(v.comercio_hierarchy, k.comercio_hierarchy) as comercio_hierarchy,
            COALESCE(v.metadata, k.metadata) as metadata
        FROM vector_results v
        FULL OUTER JOIN keyword_results k ON v.chunk_id = k.chunk_id
    )
    SELECT 
        chunk_id,
        document_id,
        file_name,
        chunk_text,
        chunk_type,
        combined_score as similarity_score,
        vector_score,
        keyword_score,
        combined_score,
        comercio_hierarchy,
        metadata
    FROM combined_results
    WHERE combined_score > 0.3
    ORDER BY combined_score DESC
    LIMIT match_count;
$$;

-- Función para búsqueda por palabras clave
CREATE OR REPLACE FUNCTION search_siem_keyword(
    search_text TEXT,
    match_count INT DEFAULT 20,
    category_filter TEXT DEFAULT NULL,
    comercio_type_filter TEXT DEFAULT NULL
)
RETURNS TABLE (
    chunk_id INT,
    document_id INT,
    file_name VARCHAR(255),
    chunk_text TEXT,
    chunk_type VARCHAR(50),
    similarity_score FLOAT,
    comercio_hierarchy JSONB,
    metadata JSONB
)
LANGUAGE SQL
AS $$
    SELECT 
        c.id as chunk_id,
        c.document_id,
        d.file_name,
        c.chunk_text,
        c.chunk_type,
        CASE 
            WHEN c.chunk_text_clean ILIKE '%' || search_text || '%' THEN 1.0
            ELSE 0.5
        END as similarity_score,
        NULL::JSONB as comercio_hierarchy,
            jsonb_build_object(
                'category', d.category,
                'comercio_type', d.comercio_type,
                'file_extension', d.file_extension,
                'created_at', d.created_at
            ) as metadata
    FROM siem_chunks c
    JOIN siem_documents d ON c.document_id = d.id
    WHERE c.chunk_text_clean ILIKE '%' || search_text || '%'
    AND (category_filter IS NULL OR d.category = category_filter)
    AND (comercio_type_filter IS NULL OR d.comercio_type = comercio_type_filter)
    ORDER BY similarity_score DESC
    LIMIT match_count;
$$;

-- Función para búsqueda por metadatos
CREATE OR REPLACE FUNCTION search_siem_by_metadata(
    category_filter TEXT DEFAULT NULL,
    comercio_type_filter TEXT DEFAULT NULL,
    keywords_filter TEXT[] DEFAULT NULL,
    date_from TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    date_to TIMESTAMP WITH TIME ZONE DEFAULT NULL
)
RETURNS TABLE (
    document_id INT,
    file_name VARCHAR(255),
    file_path TEXT,
    category VARCHAR(100),
    comercio_type VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB
)
LANGUAGE SQL
AS $$
    SELECT 
        d.id as document_id,
        d.file_name,
        d.file_path,
        d.category,
        d.comercio_type,
        d.created_at,
        jsonb_build_object(
            'file_extension', d.file_extension,
            'file_size', d.file_size,
            'processing_status', d.processing_status
        ) as metadata
    FROM siem_documents d
    LEFT JOIN siem_metadata m ON d.id = m.document_id
    WHERE (category_filter IS NULL OR d.category = category_filter)
    AND (comercio_type_filter IS NULL OR d.comercio_type = comercio_type_filter)
    AND (keywords_filter IS NULL OR m.keywords && keywords_filter)
    AND (date_from IS NULL OR d.created_at >= date_from)
    AND (date_to IS NULL OR d.created_at <= date_to)
    ORDER BY d.created_at DESC;
$$;

-- =====================================================
-- TRIGGERS PARA ACTUALIZACIÓN AUTOMÁTICA
-- =====================================================

-- Función para actualizar updated_at
CREATE OR REPLACE FUNCTION update_siem_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers para updated_at
CREATE TRIGGER update_siem_documents_updated_at
    BEFORE UPDATE ON siem_documents
    FOR EACH ROW
    EXECUTE FUNCTION update_siem_updated_at_column();

CREATE TRIGGER update_siem_chunks_updated_at
    BEFORE UPDATE ON siem_chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_siem_updated_at_column();

CREATE TRIGGER update_siem_embeddings_updated_at
    BEFORE UPDATE ON siem_embeddings
    FOR EACH ROW
    EXECUTE FUNCTION update_siem_updated_at_column();

CREATE TRIGGER update_siem_metadata_updated_at
    BEFORE UPDATE ON siem_metadata
    FOR EACH ROW
    EXECUTE FUNCTION update_siem_updated_at_column();

-- =====================================================
-- COMENTARIOS Y DOCUMENTACIÓN
-- =====================================================

COMMENT ON TABLE siem_documents IS 'Documentos del sistema SIEM de comercio exterior';
COMMENT ON TABLE siem_chunks IS 'Chunks de texto extraídos de documentos SIEM con estructura legal preservada';
COMMENT ON TABLE siem_embeddings IS 'Embeddings vectoriales generados con Qwen3-Embedding-0.6B';
COMMENT ON TABLE siem_metadata IS 'Metadatos adicionales específicos de SIEM';
-- COMMENT ON TABLE siem_processing_logs IS 'Logs de procesamiento para monitoreo y debugging';

COMMENT ON FUNCTION search_siem_embeddings_cosine IS 'Búsqueda vectorial optimizada usando similitud coseno';
COMMENT ON FUNCTION search_siem_hybrid IS 'Búsqueda híbrida combinando vectorial y palabras clave';
COMMENT ON FUNCTION search_siem_keyword IS 'Búsqueda por palabras clave en texto limpio';
COMMENT ON FUNCTION search_siem_by_metadata IS 'Búsqueda por metadatos de documentos';

-- =====================================================
-- CONFIGURACIÓN ULTRA-OPTIMIZADA PARA BÚSQUEDA DE MILISEGUNDOS
-- =====================================================

-- NOTA: Los parámetros ALTER SYSTEM no se pueden ejecutar en Supabase
-- Estos parámetros están optimizados para el plan Pro/Team de Supabase:
-- - shared_preload_libraries = 'vector' (ya habilitado)
-- - max_connections = 100 (ajustable según plan)
-- - shared_buffers = 512MB (optimizado automáticamente)
-- - effective_cache_size = 2GB (optimizado automáticamente)
-- - work_mem = 16MB (optimizado para queries complejas)
-- - maintenance_work_mem = 128MB (optimizado para índices)
-- - max_parallel_workers = 8 (optimizado para procesamiento paralelo)
-- - random_page_cost = 1.1 (optimizado para SSD)
-- - seq_page_cost = 1.0 (optimizado para SSD)
-- - cpu_tuple_cost = 0.01 (optimizado para CPU)
-- - cpu_index_tuple_cost = 0.005 (optimizado para índices)
-- - temp_buffers = 8MB (optimizado para memoria temporal)
-- - hash_mem_multiplier = 2.0 (optimizado para hash joins)

-- =====================================================
-- FINALIZACIÓN
-- =====================================================

-- Crear usuario específico para la aplicación (opcional)
-- CREATE USER siem_app WITH PASSWORD 'secure_password';
-- GRANT USAGE ON SCHEMA public TO siem_app;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO siem_app;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO siem_app;

-- Mensaje de finalización
DO $$
BEGIN
    RAISE NOTICE 'Esquema SIEM creado exitosamente. Optimizado para procesamiento masivo con 16 workers y lotes de 50 archivos.';
    RAISE NOTICE 'Búsqueda híbrida: vectorial + palabras clave + metadatos.';
    RAISE NOTICE 'Modelo: Qwen3-Embedding-0.6B con 1024 dimensiones (máxima calidad).';
END $$;
