-- =====================================================
-- ACTUALIZACIÓN DEL ESQUEMA PARA CHUNKING INTELIGENTE
-- Agregar columnas para metadatos avanzados y compresión
-- =====================================================

-- Agregar columnas a la tabla siem_chunks para chunking inteligente
ALTER TABLE siem_chunks 
ADD COLUMN IF NOT EXISTS chunk_id VARCHAR(100),
ADD COLUMN IF NOT EXISTS hierarchy_level INTEGER DEFAULT 1,
ADD COLUMN IF NOT EXISTS parent_chunk_id VARCHAR(100),
ADD COLUMN IF NOT EXISTS article_number VARCHAR(20),
ADD COLUMN IF NOT EXISTS paragraph_number VARCHAR(20),
ADD COLUMN IF NOT EXISTS inciso_number VARCHAR(20),
ADD COLUMN IF NOT EXISTS chapter_title VARCHAR(255),
ADD COLUMN IF NOT EXISTS section_title VARCHAR(255),
ADD COLUMN IF NOT EXISTS law_title VARCHAR(255),
ADD COLUMN IF NOT EXISTS page_number INTEGER,
ADD COLUMN IF NOT EXISTS word_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS char_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS is_compressed BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS compression_ratio FLOAT DEFAULT 1.0,
ADD COLUMN IF NOT EXISTS metadata JSONB;

-- Actualizar constraint de tipos de chunk para incluir nuevos tipos
ALTER TABLE siem_chunks DROP CONSTRAINT IF EXISTS chk_chunk_type;

ALTER TABLE siem_chunks ADD CONSTRAINT chk_chunk_type CHECK (chunk_type IN (
    'articulo', 'paragrafo', 'anexo', 'titulo', 'capitulo', 'seccion', 
    'resolucion', 'reglamento', 'ley', 'codigo', 'decreto', 'acuerdo', 
    'convenio', 'tratado', 'dof', 'paragraph', 'section', 'inciso',
    'fraccion', 'transitorio', 'definicion', 'disposicion'
));

-- Crear índices para optimizar búsquedas
CREATE INDEX IF NOT EXISTS idx_siem_chunks_chunk_id ON siem_chunks(chunk_id);
CREATE INDEX IF NOT EXISTS idx_siem_chunks_hierarchy_level ON siem_chunks(hierarchy_level);
CREATE INDEX IF NOT EXISTS idx_siem_chunks_parent_chunk_id ON siem_chunks(parent_chunk_id);
CREATE INDEX IF NOT EXISTS idx_siem_chunks_article_number ON siem_chunks(article_number);
CREATE INDEX IF NOT EXISTS idx_siem_chunks_page_number ON siem_chunks(page_number);
CREATE INDEX IF NOT EXISTS idx_siem_chunks_word_count ON siem_chunks(word_count);
CREATE INDEX IF NOT EXISTS idx_siem_chunks_is_compressed ON siem_chunks(is_compressed);
CREATE INDEX IF NOT EXISTS idx_siem_chunks_metadata ON siem_chunks USING GIN(metadata);

-- Crear índice compuesto para búsquedas jerárquicas
CREATE INDEX IF NOT EXISTS idx_siem_chunks_hierarchy_search 
ON siem_chunks(document_id, hierarchy_level, chunk_type);

-- Crear índice para búsquedas por contenido comprimido
CREATE INDEX IF NOT EXISTS idx_siem_chunks_compressed_content 
ON siem_chunks(chunk_text) WHERE is_compressed = TRUE;

-- Agregar columna de duración de procesamiento a siem_documents
ALTER TABLE siem_documents 
ADD COLUMN IF NOT EXISTS processing_duration_seconds INTEGER,
ADD COLUMN IF NOT EXISTS total_chunks INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS compressed_chunks INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS avg_chunk_size FLOAT DEFAULT 0,
ADD COLUMN IF NOT EXISTS hierarchy_levels JSONB;

-- Crear vista para estadísticas de chunking inteligente
CREATE OR REPLACE VIEW siem_chunking_stats AS
SELECT 
    d.id as document_id,
    d.file_name,
    d.file_size,
    d.processing_duration_seconds,
    COUNT(c.id) as total_chunks,
    COUNT(CASE WHEN c.is_compressed THEN 1 END) as compressed_chunks,
    ROUND(AVG(c.char_count)::NUMERIC, 2) as avg_chunk_size,
    ROUND(AVG(c.word_count)::NUMERIC, 2) as avg_word_count,
    COUNT(CASE WHEN c.hierarchy_level = 1 THEN 1 END) as level_1_chunks,
    COUNT(CASE WHEN c.hierarchy_level = 2 THEN 1 END) as level_2_chunks,
    COUNT(CASE WHEN c.hierarchy_level = 3 THEN 1 END) as level_3_chunks,
    COUNT(CASE WHEN c.chunk_type = 'articulo' THEN 1 END) as articles_count,
    COUNT(CASE WHEN c.chunk_type = 'capitulo' THEN 1 END) as chapters_count,
    COUNT(CASE WHEN c.chunk_type = 'paragrafo' THEN 1 END) as paragraphs_count,
    COUNT(CASE WHEN c.chunk_type = 'inciso' THEN 1 END) as incisos_count,
    ROUND(AVG(c.compression_ratio)::NUMERIC, 3) as avg_compression_ratio
FROM siem_documents d
LEFT JOIN siem_chunks c ON d.id = c.document_id
GROUP BY d.id, d.file_name, d.file_size, d.processing_duration_seconds;

-- Crear función para descomprimir chunks
CREATE OR REPLACE FUNCTION decompress_chunk_content(chunk_id_param VARCHAR)
RETURNS TEXT AS $$
DECLARE
    chunk_content TEXT;
    is_compressed_flag BOOLEAN;
BEGIN
    SELECT chunk_text, is_compressed INTO chunk_content, is_compressed_flag
    FROM siem_chunks 
    WHERE chunk_id = chunk_id_param;
    
    IF NOT FOUND THEN
        RETURN NULL;
    END IF;
    
    IF is_compressed_flag THEN
        -- Aquí se implementaría la lógica de descompresión
        -- Por ahora retornamos el contenido tal como está
        RETURN chunk_content;
    ELSE
        RETURN chunk_content;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Crear función para buscar chunks por jerarquía
CREATE OR REPLACE FUNCTION search_chunks_by_hierarchy(
    doc_id_param INTEGER,
    hierarchy_level_param INTEGER DEFAULT NULL,
    chunk_type_param VARCHAR DEFAULT NULL
)
RETURNS TABLE(
    chunk_id VARCHAR,
    chunk_text TEXT,
    chunk_type VARCHAR,
    hierarchy_level INTEGER,
    parent_chunk_id VARCHAR,
    article_number VARCHAR,
    word_count INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.chunk_id,
        c.chunk_text,
        c.chunk_type,
        c.hierarchy_level,
        c.parent_chunk_id,
        c.article_number,
        c.word_count
    FROM siem_chunks c
    WHERE c.document_id = doc_id_param
    AND (hierarchy_level_param IS NULL OR c.hierarchy_level = hierarchy_level_param)
    AND (chunk_type_param IS NULL OR c.chunk_type = chunk_type_param)
    ORDER BY c.hierarchy_level, c.article_number, c.paragraph_number, c.inciso_number;
END;
$$ LANGUAGE plpgsql;

-- Crear función para obtener estadísticas de compresión
CREATE OR REPLACE FUNCTION get_compression_stats()
RETURNS TABLE(
    total_chunks BIGINT,
    compressed_chunks BIGINT,
    compression_ratio FLOAT,
    avg_compression_ratio FLOAT,
    total_size_before_compression BIGINT,
    total_size_after_compression BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*) as total_chunks,
        COUNT(CASE WHEN is_compressed THEN 1 END) as compressed_chunks,
        ROUND((COUNT(CASE WHEN is_compressed THEN 1 END)::FLOAT / COUNT(*))::NUMERIC, 3) as compression_ratio,
        ROUND(AVG(compression_ratio)::NUMERIC, 3) as avg_compression_ratio,
        SUM(char_count) as total_size_before_compression,
        SUM(CASE WHEN is_compressed THEN char_count * compression_ratio ELSE char_count END) as total_size_after_compression
    FROM siem_chunks;
END;
$$ LANGUAGE plpgsql;

-- Crear trigger para actualizar estadísticas de documento
CREATE OR REPLACE FUNCTION update_document_chunk_stats()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' OR TG_OP = 'UPDATE' THEN
        UPDATE siem_documents 
        SET 
            total_chunks = (
                SELECT COUNT(*) FROM siem_chunks WHERE document_id = NEW.document_id
            ),
            compressed_chunks = (
                SELECT COUNT(*) FROM siem_chunks WHERE document_id = NEW.document_id AND is_compressed = TRUE
            ),
            avg_chunk_size = (
                SELECT AVG(char_count) FROM siem_chunks WHERE document_id = NEW.document_id
            ),
            hierarchy_levels = (
                SELECT jsonb_build_object(
                    'level_1', COUNT(CASE WHEN hierarchy_level = 1 THEN 1 END),
                    'level_2', COUNT(CASE WHEN hierarchy_level = 2 THEN 1 END),
                    'level_3', COUNT(CASE WHEN hierarchy_level = 3 THEN 1 END)
                ) FROM siem_chunks WHERE document_id = NEW.document_id
            )
        WHERE id = NEW.document_id;
    END IF;
    
    IF TG_OP = 'DELETE' THEN
        UPDATE siem_documents 
        SET 
            total_chunks = (
                SELECT COUNT(*) FROM siem_chunks WHERE document_id = OLD.document_id
            ),
            compressed_chunks = (
                SELECT COUNT(*) FROM siem_chunks WHERE document_id = OLD.document_id AND is_compressed = TRUE
            ),
            avg_chunk_size = (
                SELECT AVG(char_count) FROM siem_chunks WHERE document_id = OLD.document_id
            ),
            hierarchy_levels = (
                SELECT jsonb_build_object(
                    'level_1', COUNT(CASE WHEN hierarchy_level = 1 THEN 1 END),
                    'level_2', COUNT(CASE WHEN hierarchy_level = 2 THEN 1 END),
                    'level_3', COUNT(CASE WHEN hierarchy_level = 3 THEN 1 END)
                ) FROM siem_chunks WHERE document_id = OLD.document_id
            )
        WHERE id = OLD.document_id;
    END IF;
    
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Crear trigger
DROP TRIGGER IF EXISTS trigger_update_document_chunk_stats ON siem_chunks;
CREATE TRIGGER trigger_update_document_chunk_stats
    AFTER INSERT OR UPDATE OR DELETE ON siem_chunks
    FOR EACH ROW EXECUTE FUNCTION update_document_chunk_stats();

-- Comentarios para documentación
COMMENT ON COLUMN siem_chunks.chunk_id IS 'Identificador único del chunk generado por el sistema de chunking inteligente';
COMMENT ON COLUMN siem_chunks.hierarchy_level IS 'Nivel jerárquico: 1=artículo, 2=párrafo, 3=inciso, etc.';
COMMENT ON COLUMN siem_chunks.parent_chunk_id IS 'ID del chunk padre en la jerarquía';
COMMENT ON COLUMN siem_chunks.article_number IS 'Número del artículo al que pertenece el chunk';
COMMENT ON COLUMN siem_chunks.paragraph_number IS 'Número del párrafo al que pertenece el chunk';
COMMENT ON COLUMN siem_chunks.inciso_number IS 'Número del inciso al que pertenece el chunk';
COMMENT ON COLUMN siem_chunks.chapter_title IS 'Título del capítulo al que pertenece el chunk';
COMMENT ON COLUMN siem_chunks.section_title IS 'Título de la sección al que pertenece el chunk';
COMMENT ON COLUMN siem_chunks.law_title IS 'Título de la ley al que pertenece el chunk';
COMMENT ON COLUMN siem_chunks.page_number IS 'Número de página del documento';
COMMENT ON COLUMN siem_chunks.word_count IS 'Número de palabras en el chunk';
COMMENT ON COLUMN siem_chunks.char_count IS 'Número de caracteres en el chunk';
COMMENT ON COLUMN siem_chunks.is_compressed IS 'Indica si el chunk está comprimido';
COMMENT ON COLUMN siem_chunks.compression_ratio IS 'Ratio de compresión (1.0 = sin comprimir)';
COMMENT ON COLUMN siem_chunks.metadata IS 'Metadatos adicionales del chunk en formato JSON';

COMMENT ON VIEW siem_chunking_stats IS 'Vista con estadísticas detalladas del chunking inteligente por documento';
COMMENT ON FUNCTION decompress_chunk_content IS 'Función para descomprimir contenido de chunks comprimidos';
COMMENT ON FUNCTION search_chunks_by_hierarchy IS 'Función para buscar chunks por jerarquía y tipo';
COMMENT ON FUNCTION get_compression_stats IS 'Función para obtener estadísticas de compresión globales';
