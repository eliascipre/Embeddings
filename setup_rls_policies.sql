-- =====================================================
-- POLÍTICAS RLS PARA ANON KEY - SIEM COMERCIO EXTERIOR
-- =====================================================

-- Habilitar RLS en todas las tablas
ALTER TABLE siem_documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE siem_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE siem_embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE siem_metadata ENABLE ROW LEVEL SECURITY;

-- =====================================================
-- POLÍTICAS PARA SIEM_DOCUMENTS
-- =====================================================

-- Permitir lectura completa para anon
CREATE POLICY "anon_read_documents" ON siem_documents
    FOR SELECT USING (true);

-- Permitir inserción para anon
CREATE POLICY "anon_insert_documents" ON siem_documents
    FOR INSERT WITH CHECK (true);

-- Permitir actualización para anon
CREATE POLICY "anon_update_documents" ON siem_documents
    FOR UPDATE USING (true);

-- Permitir eliminación para anon (opcional)
CREATE POLICY "anon_delete_documents" ON siem_documents
    FOR DELETE USING (true);

-- =====================================================
-- POLÍTICAS PARA SIEM_CHUNKS
-- =====================================================

-- Permitir lectura completa para anon
CREATE POLICY "anon_read_chunks" ON siem_chunks
    FOR SELECT USING (true);

-- Permitir inserción para anon
CREATE POLICY "anon_insert_chunks" ON siem_chunks
    FOR INSERT WITH CHECK (true);

-- Permitir actualización para anon
CREATE POLICY "anon_update_chunks" ON siem_chunks
    FOR UPDATE USING (true);

-- Permitir eliminación para anon
CREATE POLICY "anon_delete_chunks" ON siem_chunks
    FOR DELETE USING (true);

-- =====================================================
-- POLÍTICAS PARA SIEM_EMBEDDINGS
-- =====================================================

-- Permitir lectura completa para anon
CREATE POLICY "anon_read_embeddings" ON siem_embeddings
    FOR SELECT USING (true);

-- Permitir inserción para anon
CREATE POLICY "anon_insert_embeddings" ON siem_embeddings
    FOR INSERT WITH CHECK (true);

-- Permitir actualización para anon
CREATE POLICY "anon_update_embeddings" ON siem_embeddings
    FOR UPDATE USING (true);

-- Permitir eliminación para anon
CREATE POLICY "anon_delete_embeddings" ON siem_embeddings
    FOR DELETE USING (true);

-- =====================================================
-- POLÍTICAS PARA SIEM_METADATA
-- =====================================================

-- Permitir lectura completa para anon
CREATE POLICY "anon_read_metadata" ON siem_metadata
    FOR SELECT USING (true);

-- Permitir inserción para anon
CREATE POLICY "anon_insert_metadata" ON siem_metadata
    FOR INSERT WITH CHECK (true);

-- Permitir actualización para anon
CREATE POLICY "anon_update_metadata" ON siem_metadata
    FOR UPDATE USING (true);

-- Permitir eliminación para anon
CREATE POLICY "anon_delete_metadata" ON siem_metadata
    FOR DELETE USING (true);

-- =====================================================
-- VERIFICACIÓN DE POLÍTICAS
-- =====================================================

-- Verificar que las políticas están activas
SELECT schemaname, tablename, rowsecurity 
FROM pg_tables 
WHERE schemaname = 'public' 
AND tablename LIKE 'siem_%';

-- Verificar políticas creadas
SELECT schemaname, tablename, policyname, permissive, roles, cmd, qual
FROM pg_policies 
WHERE schemaname = 'public' 
AND tablename LIKE 'siem_%'
ORDER BY tablename, policyname;

-- =====================================================
-- MENSAJE DE CONFIRMACIÓN
-- =====================================================

DO $$
BEGIN
    RAISE NOTICE 'Políticas RLS configuradas correctamente para anon key';
    RAISE NOTICE 'Ahora puedes usar la anon key para procesamiento masivo';
END $$;
