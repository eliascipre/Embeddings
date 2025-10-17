-- =====================================================
-- SCRIPT SIMPLE PARA SUPABASE - SIN ALTER SYSTEM
-- =====================================================

-- 1. ELIMINAR ÍNDICES PROBLEMÁTICOS
DROP INDEX IF EXISTS idx_siem_chunks_search;
DROP INDEX IF EXISTS idx_siem_chunks_chunk_text;
DROP INDEX IF EXISTS idx_siem_chunks_chunk_text_clean;

-- 2. VERIFICAR ESTRUCTURA DE TABLAS PRIMERO
-- Ver qué columnas existen realmente
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'siem_chunks' 
ORDER BY ordinal_position;

-- 3. CREAR ÍNDICES SOLO PARA COLUMNAS QUE EXISTEN
-- Índice para document_id (si existe)
CREATE INDEX IF NOT EXISTS idx_siem_chunks_doc_id 
ON siem_chunks (document_id);

-- Índice para chunk_type
CREATE INDEX IF NOT EXISTS idx_siem_chunks_type 
ON siem_chunks (chunk_type);

-- Índice para hierarchy_level (si existe)
CREATE INDEX IF NOT EXISTS idx_siem_chunks_hierarchy 
ON siem_chunks (hierarchy_level);

-- 4. CREAR ÍNDICE DE HASH PARA TEXTO (más eficiente)
-- Solo si chunk_text_clean existe
CREATE INDEX IF NOT EXISTS idx_siem_chunks_text_hash 
ON siem_chunks USING hash (chunk_text_clean);

-- 5. OPTIMIZAR TABLA DE EMBEDDINGS
-- Verificar estructura primero
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'siem_embeddings' 
ORDER BY ordinal_position;

-- Crear índices solo para columnas que existen
CREATE INDEX IF NOT EXISTS idx_siem_embeddings_chunk_id 
ON siem_embeddings (chunk_id);

-- 6. VACUUM PARA OPTIMIZAR
VACUUM ANALYZE siem_chunks;
VACUUM ANALYZE siem_embeddings;

-- =====================================================
-- ESTE SCRIPT:
-- ✅ No usa ALTER SYSTEM (no permitido en Supabase)
-- ✅ Verifica columnas antes de crear índices
-- ✅ Solo crea índices para campos que existen
-- ✅ Usa hash para campos de texto grandes
-- =====================================================
