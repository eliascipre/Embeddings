-- =====================================================
-- SCRIPT DE LIMPIEZA DE BASE DE DATOS SIEM
-- Elimina datos duplicados y reinicia contadores
-- =====================================================

-- Deshabilitar triggers temporalmente
SET session_replication_role = replica;

-- =====================================================
-- ELIMINAR DATOS DUPLICADOS Y REINICIAR CONTADORES
-- =====================================================

-- 1. Eliminar datos de tablas dependientes primero
DELETE FROM siem_processing_logs;
DELETE FROM siem_embeddings;
DELETE FROM siem_chunks;
DELETE FROM siem_metadata;

-- 2. Eliminar documentos
DELETE FROM siem_documents;

-- 3. Reiniciar secuencias
ALTER SEQUENCE siem_documents_id_seq RESTART WITH 1;
ALTER SEQUENCE siem_chunks_id_seq RESTART WITH 1;
ALTER SEQUENCE siem_embeddings_id_seq RESTART WITH 1;
ALTER SEQUENCE siem_metadata_id_seq RESTART WITH 1;
ALTER SEQUENCE siem_processing_logs_id_seq RESTART WITH 1;

-- 4. Las estadísticas se regeneran automáticamente (processing_stats es una vista)

-- 5. Rehabilitar triggers
SET session_replication_role = DEFAULT;

-- =====================================================
-- VERIFICAR LIMPIEZA
-- =====================================================

-- Verificar que las tablas están vacías
SELECT 'siem_documents' as tabla, COUNT(*) as registros FROM siem_documents
UNION ALL
SELECT 'siem_chunks', COUNT(*) FROM siem_chunks
UNION ALL
SELECT 'siem_embeddings', COUNT(*) FROM siem_embeddings
UNION ALL
SELECT 'siem_metadata', COUNT(*) FROM siem_metadata
UNION ALL
SELECT 'siem_processing_logs', COUNT(*) FROM siem_processing_logs;

-- =====================================================
-- OPTIMIZAR TABLAS DESPUÉS DE LIMPIEZA
-- =====================================================

-- ANALYZE para optimizar (VACUUM se ejecuta automáticamente en Supabase)
ANALYZE siem_documents;
ANALYZE siem_chunks;
ANALYZE siem_embeddings;
ANALYZE siem_metadata;
ANALYZE siem_processing_logs;

-- =====================================================
-- MENSAJE DE CONFIRMACIÓN
-- =====================================================
DO $$
BEGIN
    RAISE NOTICE 'Base de datos SIEM limpiada exitosamente';
    RAISE NOTICE 'Contadores reiniciados';
    RAISE NOTICE 'Tablas optimizadas';
END $$;