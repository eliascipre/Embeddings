-- =====================================================
-- SCRIPT MÍNIMO PARA SUPABASE
-- Solo elimina índices problemáticos
-- =====================================================

-- ELIMINAR ÍNDICES QUE CAUSAN ERRORES DE TAMAÑO
DROP INDEX IF EXISTS idx_siem_chunks_search;
DROP INDEX IF EXISTS idx_siem_chunks_chunk_text;
DROP INDEX IF EXISTS idx_siem_chunks_chunk_text_clean;
DROP INDEX IF EXISTS idx_siem_embeddings_chunk_doc;

-- VACUUM PARA LIMPIAR
VACUUM ANALYZE siem_chunks;
VACUUM ANALYZE siem_embeddings;

-- =====================================================
-- ESTO SOLUCIONA:
-- - "index row size exceeds btree maximum"
-- - "index row requires X bytes, maximum size is 8191"
-- - "Values larger than 1/3 of a buffer page cannot be indexed"
-- =====================================================
