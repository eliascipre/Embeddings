-- =====================================================
-- SCRIPT FINAL PARA SUPABASE - SIN VACUUM
-- =====================================================

-- ELIMINAR ÍNDICES PROBLEMÁTICOS
DROP INDEX IF EXISTS idx_siem_chunks_search;
DROP INDEX IF EXISTS idx_siem_chunks_chunk_text;
DROP INDEX IF EXISTS idx_siem_chunks_chunk_text_clean;
DROP INDEX IF EXISTS idx_siem_embeddings_chunk_doc;

-- =====================================================
-- ESTO SOLUCIONA:
-- - "index row size exceeds btree maximum"
-- - "index row requires X bytes, maximum size is 8191"
-- - "Values larger than 1/3 of a buffer page cannot be indexed"
-- =====================================================
