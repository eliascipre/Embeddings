-- Limpieza corregida basada en la estructura real de las tablas
-- Ejecutar en el SQL Editor de Supabase

-- PASO 1: Ver cuántos documentos están en processing
SELECT COUNT(*) as documentos_processing 
FROM siem_documents 
WHERE processing_status = 'processing';

-- PASO 2: Verificar si hay chunks (necesito ver la estructura de siem_chunks)
SELECT COUNT(*) as total_chunks FROM siem_chunks;

-- PASO 3: Verificar si hay embeddings
SELECT COUNT(*) as total_embeddings FROM siem_embeddings;

-- PASO 4: Si las tablas están vacías, eliminar solo documentos
-- (Ejecutar solo si chunks y embeddings están vacíos)
DELETE FROM siem_documents WHERE processing_status = 'processing';

-- PASO 5: Verificar limpieza
SELECT 'siem_documents' as tabla, COUNT(*) as registros FROM siem_documents
UNION ALL
SELECT 'siem_chunks' as tabla, COUNT(*) as registros FROM siem_chunks
UNION ALL
SELECT 'siem_embeddings' as tabla, COUNT(*) as registros FROM siem_embeddings;
