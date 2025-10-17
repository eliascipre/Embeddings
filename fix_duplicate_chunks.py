#!/usr/bin/env python3
"""
Script para limpiar chunks duplicados y optimizar la base de datos
"""

import logging
from config_supabase import get_supabase_url, get_supabase_key, get_database_tables
from supabase import create_client

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_duplicate_chunks():
    """Limpia chunks duplicados manteniendo solo el más reciente"""
    
    # Conectar a Supabase
    supabase_url = get_supabase_url()
    supabase_key = get_supabase_key()
    tables = get_database_tables()
    
    supabase = create_client(supabase_url, supabase_key)
    
    try:
        logger.info("Iniciando limpieza de chunks duplicados...")
        
        # 1. Identificar chunks duplicados
        logger.info("Identificando chunks duplicados...")
        duplicate_query = """
        SELECT chunk_hash, COUNT(*) as count
        FROM siem_chunks 
        GROUP BY chunk_hash 
        HAVING COUNT(*) > 1
        ORDER BY count DESC
        """
        
        # Ejecutar consulta SQL personalizada
        result = supabase.rpc('execute_sql', {'query': duplicate_query}).execute()
        
        if not result.data:
            logger.info("No se encontraron chunks duplicados")
            return
        
        logger.info(f"Encontrados {len(result.data)} hashes duplicados")
        
        # 2. Eliminar duplicados manteniendo el más reciente
        for duplicate in result.data:
            chunk_hash = duplicate['chunk_hash']
            count = duplicate['count']
            
            logger.info(f"Procesando hash {chunk_hash[:8]}... ({count} duplicados)")
            
            # Obtener todos los chunks con este hash
            chunks = supabase.table(tables['chunks']).select('*').eq('chunk_hash', chunk_hash).order('created_at', desc=True).execute()
            
            if len(chunks.data) > 1:
                # Mantener el más reciente, eliminar los demás
                chunks_to_delete = chunks.data[1:]  # Todos excepto el primero (más reciente)
                
                for chunk in chunks_to_delete:
                    # Eliminar embeddings asociados
                    supabase.table(tables['embeddings']).delete().eq('chunk_id', chunk['id']).execute()
                    
                    # Eliminar el chunk
                    supabase.table(tables['chunks']).delete().eq('id', chunk['id']).execute()
                
                logger.info(f"Eliminados {len(chunks_to_delete)} chunks duplicados para hash {chunk_hash[:8]}...")
        
        logger.info("Limpieza de duplicados completada")
        
        # 3. Optimizar tablas
        logger.info("Optimizando tablas...")
        optimize_queries = [
            "ANALYZE siem_chunks",
            "ANALYZE siem_embeddings", 
            "ANALYZE siem_documents"
        ]
        
        for query in optimize_queries:
            try:
                supabase.rpc('execute_sql', {'query': query}).execute()
                logger.info(f"Ejecutado: {query}")
            except Exception as e:
                logger.warning(f"Error ejecutando {query}: {e}")
        
        logger.info("Optimización completada")
        
    except Exception as e:
        logger.error(f"Error durante la limpieza: {e}")
        raise

def verify_cleanup():
    """Verifica que la limpieza fue exitosa"""
    
    supabase_url = get_supabase_url()
    supabase_key = get_supabase_key()
    tables = get_database_tables()
    
    supabase = create_client(supabase_url, supabase_key)
    
    try:
        # Verificar duplicados restantes
        duplicate_check = """
        SELECT COUNT(*) as duplicate_count
        FROM (
            SELECT chunk_hash, COUNT(*) as count
            FROM siem_chunks 
            GROUP BY chunk_hash 
            HAVING COUNT(*) > 1
        ) as duplicates
        """
        
        result = supabase.rpc('execute_sql', {'query': duplicate_check}).execute()
        duplicate_count = result.data[0]['duplicate_count'] if result.data else 0
        
        if duplicate_count == 0:
            logger.info("✅ Verificación exitosa: No hay chunks duplicados")
        else:
            logger.warning(f"⚠️  Aún existen {duplicate_count} grupos de chunks duplicados")
        
        # Estadísticas generales
        stats_query = """
        SELECT 
            'siem_documents' as tabla, COUNT(*) as registros FROM siem_documents
        UNION ALL
        SELECT 'siem_chunks', COUNT(*) FROM siem_chunks
        UNION ALL
        SELECT 'siem_embeddings', COUNT(*) FROM siem_embeddings
        """
        
        stats_result = supabase.rpc('execute_sql', {'query': stats_query}).execute()
        
        logger.info("📊 Estadísticas de la base de datos:")
        for stat in stats_result.data:
            logger.info(f"  {stat['tabla']}: {stat['registros']} registros")
            
    except Exception as e:
        logger.error(f"Error durante la verificación: {e}")

if __name__ == "__main__":
    try:
        clean_duplicate_chunks()
        verify_cleanup()
        logger.info("🎉 Proceso de limpieza completado exitosamente")
    except Exception as e:
        logger.error(f"❌ Error en el proceso de limpieza: {e}")
        exit(1)
