-- Limpiar documentos en estado "processing" para empezar limpio
-- Ejecutar en el SQL Editor de Supabase

-- 1. Eliminar chunks relacionados con documentos en processing
DELETE FROM siem_chunks 
WHERE document_id IN (
    SELECT id FROM siem_documents 
    WHERE processing_status = 'processing'
);

-- 2. Eliminar embeddings relacionados con documentos en processing
DELETE FROM siem_embeddings 
WHERE document_id IN (
    SELECT id FROM siem_documents 
    WHERE processing_status = 'processing'
);

-- 3. Eliminar documentos en estado processing
DELETE FROM siem_documents 
WHERE processing_status = 'processing';

-- 4. Verificar que las tablas estén limpias
SELECT 'siem_documents' as tabla, COUNT(*) as registros FROM siem_documents
UNION ALL
SELECT 'siem_chunks' as tabla, COUNT(*) as registros FROM siem_chunks
UNION ALL
SELECT 'siem_embeddings' as tabla, COUNT(*) as registros FROM siem_embeddings;
