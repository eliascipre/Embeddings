#!/usr/bin/env python3
"""
Ejemplo de uso del sistema RAG optimizado con Hugging Face A100
"""
import asyncio
import logging
from pathlib import Path
import os

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def ejemplo_procesamiento_masivo():
    """Ejemplo de procesamiento masivo de documentos"""
    from src.optimized_rag_system import OptimizedRAGSystem
    
    # Configuración
    SUPABASE_URL = "https://zcxqxrgtmnfixkgeaurj.supabase.co"
    SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InpjeHF4cmd0bW5maXhrZ2VhdXJqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA2MTk4MTgsImV4cCI6MjA3NjE5NTgxOH0.YqseZLeQDohHdc-s9QduefPqy5SOSyWJysK_q8cMtio"
    HF_TOKEN = os.getenv("HF_TOKEN", "tu_token_aqui")
    
    # Directorio con documentos
    source_dir = Path("leyes_de_todos_los_estados")
    
    logger.info("🚀 Iniciando procesamiento masivo...")
    
    async with OptimizedRAGSystem(SUPABASE_URL, SUPABASE_KEY, HF_TOKEN) as rag_system:
        # Procesar todos los documentos
        results = await rag_system.process_documents_massive(
            source_directory=source_dir,
            max_workers=8,  # Para A100
            batch_size=50   # PDFs por lote
        )
        
        logger.info(f"✅ Procesamiento completado: {results['processed_files']} archivos")

async def ejemplo_busqueda():
    """Ejemplo de búsqueda de documentos"""
    from src.optimized_rag_system import OptimizedRAGSystem
    
    # Configuración
    SUPABASE_URL = "https://zcxqxrgtmnfixkgeaurj.supabase.co"
    SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InpjeHF4cmd0bW5maXhrZ2VhdXJqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA2MTk4MTgsImV4cCI6MjA3NjE5NTgxOH0.YqseZLeQDohHdc-s9QduefPqy5SOSyWJysK_q8cMtio"
    HF_TOKEN = os.getenv("HF_TOKEN", "tu_token_aqui")
    
    logger.info("🔍 Iniciando búsqueda...")
    
    async with OptimizedRAGSystem(SUPABASE_URL, SUPABASE_KEY, HF_TOKEN) as rag_system:
        # Búsqueda híbrida
        results = await rag_system.search_documents_hybrid(
            query="derechos de las mujeres",
            search_type="hybrid",
            filters={"state": "jalisco"},
            top_k=10
        )
        
        logger.info(f"📊 Encontrados {results['total_found']} resultados")
        for i, result in enumerate(results['results'][:5], 1):
            logger.info(f"{i}. {result['metadata']['title']} - {result['similarity']:.3f}")

async def ejemplo_busqueda_por_articulo():
    """Ejemplo de búsqueda por artículo específico"""
    from src.optimized_rag_system import OptimizedRAGSystem
    
    # Configuración
    SUPABASE_URL = "https://zcxqxrgtmnfixkgeaurj.supabase.co"
    SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InpjeHF4cmd0bW5maXhrZ2VhdXJqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA2MTk4MTgsImV4cCI6MjA3NjE5NTgxOH0.YqseZLeQDohHdc-s9QduefPqy5SOSyWJysK_q8cMtio"
    HF_TOKEN = os.getenv("HF_TOKEN", "tu_token_aqui")
    
    logger.info("🔍 Buscando artículo específico...")
    
    async with OptimizedRAGSystem(SUPABASE_URL, SUPABASE_KEY, HF_TOKEN) as rag_system:
        # Búsqueda por metadatos
        results = await rag_system.search_documents_hybrid(
            query="artículo 1",
            search_type="metadata",
            filters={"article_number": "1", "state": "jalisco"},
            top_k=5
        )
        
        logger.info(f"📊 Encontrados {results['total_found']} artículos")
        for result in results['results']:
            logger.info(f"Artículo {result['metadata']['article_number']}: {result['content'][:100]}...")

async def ejemplo_estadisticas():
    """Ejemplo de obtención de estadísticas"""
    from src.optimized_rag_system import OptimizedRAGSystem
    
    # Configuración
    SUPABASE_URL = "https://zcxqxrgtmnfixkgeaurj.supabase.co"
    SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InpjeHF4cmd0bW5maXhrZ2VhdXJqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA2MTk4MTgsImV4cCI6MjA3NjE5NTgxOH0.YqseZLeQDohHdc-s9QduefPqy5SOSyWJysK_q8cMtio"
    HF_TOKEN = os.getenv("HF_TOKEN", "tu_token_aqui")
    
    logger.info("📊 Obteniendo estadísticas...")
    
    async with OptimizedRAGSystem(SUPABASE_URL, SUPABASE_KEY, HF_TOKEN) as rag_system:
        stats = await rag_system.get_database_stats()
        
        logger.info("📈 ESTADÍSTICAS:")
        logger.info(f"   📄 Documentos: {stats.get('total_documents', 0)}")
        logger.info(f"   📝 Chunks: {stats.get('total_chunks', 0)}")
        logger.info(f"   🏛️ Estados: {stats.get('total_states', 0)}")
        logger.info(f"   📚 Tipos de ley: {stats.get('total_law_types', 0)}")
        logger.info(f"   💾 Tamaño: {stats.get('total_size_mb', 0)} MB")

def main():
    """Función principal con ejemplos"""
    print("🚀 SISTEMA RAG OPTIMIZADO CON HUGGING FACE A100")
    print("=" * 60)
    print("1. Procesamiento masivo de documentos")
    print("2. Búsqueda híbrida")
    print("3. Búsqueda por artículo específico")
    print("4. Estadísticas de la base de datos")
    print("5. Ejecutar todos los ejemplos")
    
    opcion = input("\nSelecciona una opción (1-5): ")
    
    if opcion == "1":
        asyncio.run(ejemplo_procesamiento_masivo())
    elif opcion == "2":
        asyncio.run(ejemplo_busqueda())
    elif opcion == "3":
        asyncio.run(ejemplo_busqueda_por_articulo())
    elif opcion == "4":
        asyncio.run(ejemplo_estadisticas())
    elif opcion == "5":
        print("🔄 Ejecutando todos los ejemplos...")
        asyncio.run(ejemplo_estadisticas())
        asyncio.run(ejemplo_busqueda())
        asyncio.run(ejemplo_busqueda_por_articulo())
    else:
        print("❌ Opción no válida")

if __name__ == "__main__":
    main()
