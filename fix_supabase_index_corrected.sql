-- Script corregido para arreglar el índice de Supabase
-- Error: index row size 2832 exceeds btree version 4 maximum 2704

-- 1. Eliminar el índice problemático que usa columnas incorrectas
DROP INDEX IF EXISTS idx_legal_chunks_state_law_type;

-- 2. Crear índices correctos para legal_chunks usando las columnas que realmente existen
CREATE INDEX IF NOT EXISTS idx_legal_chunks_document_id 
ON legal_chunks (document_id);

CREATE INDEX IF NOT EXISTS idx_legal_chunks_chunk_type 
ON legal_chunks (chunk_type);

CREATE INDEX IF NOT EXISTS idx_legal_chunks_page_number 
ON legal_chunks (page_number);

-- 3. Crear índice funcional usando hash para law_title (evita límite de tamaño)
CREATE INDEX IF NOT EXISTS idx_legal_chunks_law_title_hash 
ON legal_chunks USING btree (md5(law_title));

-- 4. Crear índice de texto completo para law_title
CREATE INDEX IF NOT EXISTS idx_legal_chunks_law_title_text 
ON legal_chunks USING gin (to_tsvector('spanish', law_title));

-- 5. Crear índice para chapter_title
CREATE INDEX IF NOT EXISTS idx_legal_chunks_chapter_title 
ON legal_chunks (chapter_title);

-- 6. Crear índice para section_title
CREATE INDEX IF NOT EXISTS idx_legal_chunks_section_title 
ON legal_chunks (section_title);

-- 7. Crear índice compuesto para búsquedas eficientes
CREATE INDEX IF NOT EXISTS idx_legal_chunks_document_chunk_type 
ON legal_chunks (document_id, chunk_type);

-- 8. Crear índices para legal_documents (donde sí está la columna state)
CREATE INDEX IF NOT EXISTS idx_legal_documents_state 
ON legal_documents (state);

CREATE INDEX IF NOT EXISTS idx_legal_documents_law_type 
ON legal_documents (law_type);

CREATE INDEX IF NOT EXISTS idx_legal_documents_state_law_type 
ON legal_documents (state, law_type);

-- 9. Crear índice funcional para law_type en legal_documents (evita límite de tamaño)
CREATE INDEX IF NOT EXISTS idx_legal_documents_law_type_hash 
ON legal_documents USING btree (state, md5(law_type));

-- 10. Verificar que los índices se crearon correctamente
SELECT 
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes 
WHERE tablename IN ('legal_chunks', 'legal_documents')
ORDER BY tablename, indexname;
