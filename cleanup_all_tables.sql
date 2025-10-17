-- Limpiar TODAS las tablas de la base de datos
-- ⚠️ ADVERTENCIA: Este script eliminará TODOS los datos de las tablas
-- Ejecutar en el SQL Editor de Supabase

-- PASO 1: Verificar contenido actual
SELECT 'siem_documents' as tabla, COUNT(*) as registros FROM siem_documents
UNION ALL
SELECT 'siem_chunks' as tabla, COUNT(*) as registros FROM siem_chunks
UNION ALL
SELECT 'siem_embeddings' as tabla, COUNT(*) as registros FROM siem_embeddings
UNION ALL
SELECT 'siem_metadata' as tabla, COUNT(*) as registros FROM siem_metadata;

-- PASO 2: Eliminar en orden correcto (respetando foreign keys)
-- Primero eliminar embeddings (depende de chunks)
DELETE FROM siem_embeddings;

-- Luego eliminar chunks (depende de documents)
DELETE FROM siem_chunks;

-- Luego eliminar metadata (depende de documents)
DELETE FROM siem_metadata;

-- Finalmente eliminar documents
DELETE FROM siem_documents;

-- PASO 3: Verificar limpieza
SELECT 'siem_documents' as tabla, COUNT(*) as registros FROM siem_documents
UNION ALL
SELECT 'siem_chunks' as tabla, COUNT(*) as registros FROM siem_chunks
UNION ALL
SELECT 'siem_embeddings' as tabla, COUNT(*) as registros FROM siem_embeddings
UNION ALL
SELECT 'siem_metadata' as tabla, COUNT(*) as registros FROM siem_metadata;

-- PASO 4: Resetear secuencias (opcional, para que los IDs empiecen desde 1)
-- ALTER SEQUENCE siem_documents_id_seq RESTART WITH 1;
-- ALTER SEQUENCE siem_chunks_id_seq RESTART WITH 1;
-- ALTER SEQUENCE siem_embeddings_id_seq RESTART WITH 1;
