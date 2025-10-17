-- =====================================================
-- SCRIPT PARA AUMENTAR LÍMITES DE BASE DE DATOS
-- Soluciona errores de "index row size" y "maximum size"
-- =====================================================

-- 1. AUMENTAR TAMAÑO DE PÁGINA DE BUFFER
-- Esto permite índices más grandes
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET work_mem = '64MB';
ALTER SYSTEM SET maintenance_work_mem = '256MB';

-- 2. ELIMINAR ÍNDICES PROBLEMÁTICOS
-- Los índices que están causando problemas de tamaño
DROP INDEX IF EXISTS idx_siem_chunks_search;
DROP INDEX IF EXISTS idx_siem_chunks_chunk_text;
DROP INDEX IF EXISTS idx_siem_chunks_chunk_text_clean;

-- 3. CREAR ÍNDICES OPTIMIZADOS CON HASH
-- Usar hash en lugar de btree para campos grandes
CREATE INDEX IF NOT EXISTS idx_siem_chunks_chunk_text_hash 
ON siem_chunks USING hash (chunk_text);

CREATE INDEX IF NOT EXISTS idx_siem_chunks_chunk_text_clean_hash 
ON siem_chunks USING hash (chunk_text_clean);

-- 4. CREAR ÍNDICE DE TEXTO COMPLETO
-- Para búsquedas de texto completo más eficientes
CREATE INDEX IF NOT EXISTS idx_siem_chunks_fulltext 
ON siem_chunks USING gin (to_tsvector('spanish', chunk_text_clean));

-- 5. CREAR ÍNDICES PARCIALES PARA CAMPOS PEQUEÑOS
-- Solo indexar campos que realmente necesitamos buscar
CREATE INDEX IF NOT EXISTS idx_siem_chunks_type 
ON siem_chunks (chunk_type) WHERE chunk_type IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_siem_chunks_document_id 
ON siem_chunks (document_id) WHERE document_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_siem_chunks_hierarchy_level 
ON siem_chunks (hierarchy_level) WHERE hierarchy_level IS NOT NULL;

-- 6. OPTIMIZAR TABLA DE EMBEDDINGS
-- Asegurar que el índice de embeddings funcione correctamente
DROP INDEX IF EXISTS idx_siem_embeddings_chunk_doc;
CREATE INDEX IF NOT EXISTS idx_siem_embeddings_chunk_id 
ON siem_embeddings (chunk_id) WHERE chunk_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_siem_embeddings_document_id 
ON siem_embeddings (document_id) WHERE document_id IS NOT NULL;

-- 7. CREAR ÍNDICE PARA BÚSQUEDA HÍBRIDA
-- Combinar búsqueda de texto y metadatos
CREATE INDEX IF NOT EXISTS idx_siem_chunks_hybrid_search 
ON siem_chunks (chunk_type, hierarchy_level, document_id) 
WHERE chunk_text_clean IS NOT NULL;

-- 8. OPTIMIZAR CONFIGURACIÓN DE POSTGRESQL
-- Aumentar límites para operaciones grandes
ALTER SYSTEM SET max_wal_size = '2GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';

-- 9. CREAR VISTA PARA BÚSQUEDAS RÁPIDAS
-- Vista optimizada para consultas frecuentes
CREATE OR REPLACE VIEW siem_chunks_search_view AS
SELECT 
    id,
    document_id,
    chunk_type,
    hierarchy_level,
    chunk_text,
    chunk_text_clean,
    word_count,
    char_count,
    page_number,
    created_at,
    -- Campos calculados para búsqueda
    CASE 
        WHEN chunk_text_clean IS NOT NULL THEN to_tsvector('spanish', chunk_text_clean)
        ELSE NULL
    END as search_vector
FROM siem_chunks
WHERE chunk_text IS NOT NULL 
  AND length(chunk_text) > 10;

-- 10. CREAR FUNCIÓN DE BÚSQUEDA OPTIMIZADA
CREATE OR REPLACE FUNCTION search_chunks_optimized(
    search_query TEXT,
    chunk_type_filter TEXT DEFAULT NULL,
    limit_results INTEGER DEFAULT 20
)
RETURNS TABLE (
    id INTEGER,
    document_id INTEGER,
    chunk_type VARCHAR,
    chunk_text TEXT,
    similarity REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.id,
        c.document_id,
        c.chunk_type,
        c.chunk_text,
        ts_rank(c.search_vector, plainto_tsquery('spanish', search_query)) as similarity
    FROM siem_chunks_search_view c
    WHERE 
        c.search_vector @@ plainto_tsquery('spanish', search_query)
        AND (chunk_type_filter IS NULL OR c.chunk_type = chunk_type_filter)
    ORDER BY similarity DESC
    LIMIT limit_results;
END;
$$ LANGUAGE plpgsql;

-- 11. CREAR ÍNDICE PARA LA FUNCIÓN DE BÚSQUEDA
CREATE INDEX IF NOT EXISTS idx_siem_chunks_search_vector 
ON siem_chunks_search_view USING gin (search_vector);

-- 12. VACUUM Y ANALYZE
-- Optimizar estadísticas de la base de datos
VACUUM ANALYZE siem_chunks;
VACUUM ANALYZE siem_embeddings;
VACUUM ANALYZE siem_documents;

-- 13. MOSTRAR ESTADÍSTICAS FINALES
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats 
WHERE tablename IN ('siem_chunks', 'siem_embeddings', 'siem_documents')
ORDER BY tablename, attname;

-- =====================================================
-- INSTRUCCIONES DE USO:
-- 1. Ejecutar este script en Supabase SQL Editor
-- 2. Reiniciar la base de datos si es necesario
-- 3. Verificar que los índices se crearon correctamente
-- =====================================================
