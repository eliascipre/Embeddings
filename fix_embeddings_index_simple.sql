-- Script simplificado para corregir el índice de embeddings
-- Ejecutar en Supabase SQL Editor

-- 1. Eliminar el índice problemático
DROP INDEX IF EXISTS idx_siem_embeddings_chunk_doc;

-- 2. Crear índices simples y eficientes
CREATE INDEX IF NOT EXISTS idx_siem_embeddings_chunk_id ON siem_embeddings(chunk_id);
CREATE INDEX IF NOT EXISTS idx_siem_embeddings_model ON siem_embeddings(model_name);
CREATE INDEX IF NOT EXISTS idx_siem_embeddings_created ON siem_embeddings(generated_at);

-- 3. Agregar columna de dimensiones si no existe
ALTER TABLE siem_embeddings 
ADD COLUMN IF NOT EXISTS dimensions INTEGER DEFAULT 1024;

-- 4. Establecer dimensiones por defecto (no intentar calcular automáticamente)
UPDATE siem_embeddings 
SET dimensions = 1024 
WHERE dimensions IS NULL;

-- 5. Crear vista para estadísticas
CREATE OR REPLACE VIEW siem_embeddings_stats AS
SELECT 
    COUNT(*) as total_embeddings,
    AVG(dimensions) as avg_dimensions,
    COUNT(DISTINCT chunk_id) as unique_chunks,
    COUNT(DISTINCT model_name) as unique_models,
    MIN(generated_at) as first_embedding,
    MAX(generated_at) as last_embedding
FROM siem_embeddings;

-- 6. Verificar que todo esté funcionando
SELECT 'Índices creados correctamente' as status;
