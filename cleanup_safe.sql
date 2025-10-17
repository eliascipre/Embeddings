-- Limpieza segura de documentos en estado "processing"
-- Ejecutar paso a paso en el SQL Editor de Supabase

-- PASO 1: Ver cuántos documentos están en processing
SELECT COUNT(*) as documentos_processing 
FROM siem_documents 
WHERE processing_status = 'processing';

-- PASO 2: Ver IDs de documentos en processing
SELECT id, file_name, processing_status 
FROM siem_documents 
WHERE processing_status = 'processing' 
ORDER BY created_at DESC 
LIMIT 10;

-- PASO 3: Verificar si hay chunks (cambiar 'document_id' por la columna correcta)
SELECT COUNT(*) as total_chunks FROM siem_chunks;

-- PASO 4: Verificar si hay embeddings (cambiar 'document_id' por la columna correcta)  
SELECT COUNT(*) as total_embeddings FROM siem_embeddings;

-- PASO 5: Si las tablas están vacías, solo eliminar documentos
-- (Descomenta la siguiente línea solo si las tablas chunks y embeddings están vacías)
-- DELETE FROM siem_documents WHERE processing_status = 'processing';
