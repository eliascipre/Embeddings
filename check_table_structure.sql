-- Verificar la estructura de las tablas
-- Ejecutar en el SQL Editor de Supabase

-- 1. Ver estructura de siem_documents
SELECT column_name, data_type, is_nullable 
FROM information_schema.columns 
WHERE table_name = 'siem_documents' 
ORDER BY ordinal_position;

-- 2. Ver estructura de siem_chunks
SELECT column_name, data_type, is_nullable 
FROM information_schema.columns 
WHERE table_name = 'siem_chunks' 
ORDER BY ordinal_position;

-- 3. Ver estructura de siem_embeddings
SELECT column_name, data_type, is_nullable 
FROM information_schema.columns 
WHERE table_name = 'siem_embeddings' 
ORDER BY ordinal_position;

-- 4. Ver algunos registros de ejemplo
SELECT * FROM siem_documents LIMIT 3;
SELECT * FROM siem_chunks LIMIT 3;
SELECT * FROM siem_embeddings LIMIT 3;
