-- Script para arreglar el índice de Supabase que excede el límite de tamaño
-- Error: index row size 2832 exceeds btree version 4 maximum 2704

-- 1. Primero verificar la estructura de la tabla
SELECT column_name, data_type, character_maximum_length
FROM information_schema.columns 
WHERE table_name = 'legal_chunks'
ORDER BY ordinal_position;

-- 2. Eliminar el índice problemático
DROP INDEX IF EXISTS idx_legal_chunks_state_law_type;

-- 3. Crear índices básicos sin especificar columnas que pueden no existir
CREATE INDEX IF NOT EXISTS idx_legal_chunks_document_id 
ON legal_chunks (document_id);

CREATE INDEX IF NOT EXISTS idx_legal_chunks_chunk_number 
ON legal_chunks (chunk_number);

-- 4. Crear índice funcional usando hash para law_title (si existe)
CREATE INDEX IF NOT EXISTS idx_legal_chunks_law_title_hash 
ON legal_chunks USING btree (md5(law_title));

-- 5. Crear índice de texto completo para law_title (si existe)
CREATE INDEX IF NOT EXISTS idx_legal_chunks_law_title_text 
ON legal_chunks USING gin (to_tsvector('spanish', law_title));

-- 6. Verificar que los índices se crearon correctamente
SELECT 
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes 
WHERE tablename = 'legal_chunks'
ORDER BY indexname;