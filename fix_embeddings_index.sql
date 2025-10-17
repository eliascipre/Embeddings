-- Script para corregir el índice de embeddings y permitir vectores de 1024 dimensiones
-- Ejecutar en Supabase SQL Editor

-- 1. Eliminar el índice problemático
DROP INDEX IF EXISTS idx_siem_embeddings_chunk_doc;

-- 2. Crear un índice más eficiente usando hash del vector
CREATE INDEX IF NOT EXISTS idx_siem_embeddings_chunk_id ON siem_embeddings(chunk_id);

-- 3. Crear un índice para búsquedas vectoriales usando pgvector (si está disponible)
-- Si tienes la extensión pgvector instalada:
-- CREATE INDEX IF NOT EXISTS idx_siem_embeddings_vector_cosine 
-- ON siem_embeddings USING ivfflat (embedding_vector vector_cosine_ops) 
-- WITH (lists = 100);

-- 4. Alternativa: Crear un índice funcional con hash del vector
CREATE INDEX IF NOT EXISTS idx_siem_embeddings_vector_hash 
ON siem_embeddings USING hash (md5(embedding_vector::text));

-- 5. Verificar que la tabla tenga las columnas correctas
ALTER TABLE siem_embeddings 
ADD COLUMN IF NOT EXISTS dimensions INTEGER DEFAULT 1024;

-- 6. Actualizar dimensiones existentes (usando array_length para float[])
UPDATE siem_embeddings 
SET dimensions = array_length(embedding_vector::float[], 1) 
WHERE dimensions IS NULL;

-- 7. Crear vista para estadísticas de embeddings
CREATE OR REPLACE VIEW siem_embeddings_stats AS
SELECT 
    COUNT(*) as total_embeddings,
    AVG(dimensions) as avg_dimensions,
    MIN(dimensions) as min_dimensions,
    MAX(dimensions) as max_dimensions,
    COUNT(DISTINCT chunk_id) as unique_chunks,
    COUNT(DISTINCT model_name) as unique_models
FROM siem_embeddings;

-- 8. Función para búsqueda vectorial (si pgvector está disponible)
CREATE OR REPLACE FUNCTION search_similar_embeddings(
    query_vector FLOAT[],
    similarity_threshold FLOAT DEFAULT 0.7,
    max_results INTEGER DEFAULT 10
)
RETURNS TABLE(
    chunk_id INTEGER,
    similarity FLOAT,
    model_name TEXT
) AS $$
BEGIN
    -- Esta función requiere la extensión pgvector
    -- Si no está disponible, usar búsqueda por palabras clave
    RETURN QUERY
    SELECT 
        se.chunk_id,
        1.0 - (se.embedding_vector <=> query_vector) as similarity,
        se.model_name
    FROM siem_embeddings se
    WHERE 1.0 - (se.embedding_vector <=> query_vector) > similarity_threshold
    ORDER BY similarity DESC
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;
