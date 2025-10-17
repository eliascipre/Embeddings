-- =====================================================
-- SCRIPT SIMPLE PARA ARREGLAR ÍNDICES PROBLEMÁTICOS
-- =====================================================

-- 1. ELIMINAR ÍNDICES QUE CAUSAN ERRORES DE TAMAÑO
DROP INDEX IF EXISTS idx_siem_chunks_search;
DROP INDEX IF EXISTS idx_siem_chunks_chunk_text;
DROP INDEX IF EXISTS idx_siem_chunks_chunk_text_clean;

-- 2. CREAR ÍNDICES SIMPLES Y EFICIENTES
-- Solo para campos pequeños y esenciales
CREATE INDEX IF NOT EXISTS idx_siem_chunks_document_id 
ON siem_chunks (document_id);

CREATE INDEX IF NOT EXISTS idx_siem_chunks_type 
ON siem_chunks (chunk_type);

CREATE INDEX IF NOT EXISTS idx_siem_chunks_hierarchy_level 
ON siem_chunks (hierarchy_level);

-- 3. CREAR ÍNDICE DE HASH PARA BÚSQUEDAS RÁPIDAS
-- Hash es más eficiente para campos grandes
CREATE INDEX IF NOT EXISTS idx_siem_chunks_text_hash 
ON siem_chunks USING hash (chunk_text_clean);

-- 4. OPTIMIZAR TABLA DE EMBEDDINGS
DROP INDEX IF EXISTS idx_siem_embeddings_chunk_doc;
CREATE INDEX IF NOT EXISTS idx_siem_embeddings_chunk_id 
ON siem_embeddings (chunk_id);

CREATE INDEX IF NOT EXISTS idx_siem_embeddings_document_id 
ON siem_embeddings (document_id);

-- 5. VACUUM PARA OPTIMIZAR
VACUUM ANALYZE siem_chunks;
VACUUM ANALYZE siem_embeddings;

-- =====================================================
-- ESTE SCRIPT SOLUCIONA:
-- - "index row size exceeds btree maximum"
-- - "index row requires X bytes, maximum size is 8191"
-- - "Values larger than 1/3 of a buffer page cannot be indexed"
-- =====================================================
