#!/usr/bin/env python3
"""
Script de prueba para el sistema RAG SIEM
Verifica que todos los componentes funcionen correctamente
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.optimized_siem_rag_system import OptimizedSIEMRAGSystem
from src.comercio_exterior_processor import ComercioExteriorProcessor
from src.qwen3_embedding_client import Qwen3EmbeddingClient, Qwen3Config
from config_supabase import get_supabase_url, get_supabase_key

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_document_processor():
    """Probar el procesador de documentos de comercio exterior"""
    logger.info("Probando procesador de documentos de comercio exterior...")
    
    processor = ComercioExteriorProcessor()
    
    # Crear un texto de prueba de comercio exterior
    test_text = """
    DOF 241221 - Reglas de Comercio Exterior 2022
    
    CAPÍTULO I
    DISPOSICIONES GENERALES
    
    ARTÍCULO 1. Las presentes Reglas tienen por objeto regular el comercio exterior de México.
    
    ARTÍCULO 2. Para los efectos de estas Reglas se entiende por:
    I. Comercio exterior: la actividad comercial que se realiza entre residentes de México y residentes del extranjero;
    II. Exportación: la salida de mercancías del territorio nacional;
    III. Importación: la entrada de mercancías al territorio nacional;
    IV. Valor en aduana: el valor de las mercancías para efectos aduaneros.
    
    ANEXO 1
    CLASIFICACIÓN ARANCELARIA
    
    La clasificación arancelaria se realizará conforme al Sistema Armonizado de Designación y Codificación de Mercancías.
    """
    
    # Procesar texto
    chunks = processor.create_comercio_chunks(test_text, Path("test_document.pdf"))
    
    logger.info(f"✓ Chunks creados: {len(chunks)}")
    for i, chunk in enumerate(chunks[:3]):  # Mostrar solo los primeros 3
        logger.info(f"  Chunk {i+1}: {chunk['type']} - {chunk['text'][:100]}...")
    
    return len(chunks) > 0

async def test_embedding_client():
    """Probar el cliente de embeddings"""
    logger.info("Probando cliente de embeddings Qwen3...")
    
    # Verificar token
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        logger.error("HF_TOKEN no configurado")
        return False
    
    try:
        config = Qwen3Config(token=hf_token)
        client = Qwen3EmbeddingClient(config)
        
        async with client:
            # Probar embedding simple
            test_text = "Este es un texto de prueba para generar embeddings"
            embedding = await client.generate_single_embedding(test_text)
            
            logger.info(f"✓ Embedding generado: {len(embedding)} dimensiones")
            logger.info(f"✓ Primeros 5 valores: {embedding[:5]}")
            
            # Probar lote de embeddings
            test_texts = [
                "Comercio exterior de México",
                "Ley Aduanera",
                "Reglamento de Comercio Exterior"
            ]
            
            embeddings = await client.generate_embeddings_batch(test_texts)
            logger.info(f"✓ Lote de embeddings generado: {len(embeddings)} embeddings")
            
            return True
            
    except Exception as e:
        logger.error(f"✗ Error probando cliente de embeddings: {e}")
        return False

async def test_supabase_connection():
    """Probar conexión con Supabase"""
    logger.info("Probando conexión con Supabase...")
    
    try:
        from supabase import create_client
        
        url = get_supabase_url()
        key = get_supabase_key()
        
        supabase = create_client(url, key)
        
        # Probar consulta simple
        result = supabase.table('siem_documents').select('id').limit(1).execute()
        logger.info("✓ Conexión con Supabase establecida")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Error conectando con Supabase: {e}")
        return False

async def test_rag_system():
    """Probar el sistema RAG completo"""
    logger.info("Probando sistema RAG completo...")
    
    try:
        # Verificar variables de entorno
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            logger.error("HF_TOKEN no configurado")
            return False
        
        # Crear componentes
        processor = ComercioExteriorProcessor()
        
        config = Qwen3Config(token=hf_token)
        embedding_client = Qwen3EmbeddingClient(config)
        
        rag_system = OptimizedSIEMRAGSystem(
            supabase_url=get_supabase_url(),
            supabase_key=get_supabase_key(),
            embedding_client=embedding_client,
            document_processor=processor
        )
        
        # Inicializar sistema
        await rag_system.initialize()
        logger.info("✓ Sistema RAG inicializado")
        
        # Obtener estadísticas
        stats = await rag_system.get_database_stats()
        logger.info(f"✓ Estadísticas de BD: {stats}")
        
        # Limpiar recursos
        await rag_system.cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Error probando sistema RAG: {e}")
        return False

async def test_file_processing():
    """Probar procesamiento de archivos reales"""
    logger.info("Probando procesamiento de archivos reales...")
    
    siem_path = Path("SIEM")
    if not siem_path.exists():
        logger.warning("Directorio SIEM no encontrado, saltando prueba de archivos")
        return True
    
    # Buscar algunos archivos de prueba
    test_files = []
    for ext in ['.pdf', '.docx', '.txt']:
        files = list(siem_path.rglob(f"*{ext}"))[:2]  # Tomar solo 2 archivos por tipo
        test_files.extend(files)
    
    if not test_files:
        logger.warning("No se encontraron archivos de prueba")
        return True
    
    logger.info(f"Encontrados {len(test_files)} archivos de prueba")
    
    processor = ComercioExteriorProcessor()
    
    for file_path in test_files[:3]:  # Probar solo los primeros 3
        try:
            logger.info(f"Procesando: {file_path.name}")
            
            # Extraer texto
            text = processor.extract_text(file_path)
            if not text.strip():
                logger.warning(f"  No se pudo extraer texto de {file_path.name}")
                continue
            
            # Crear chunks
            chunks = processor.create_comercio_chunks(text, file_path)
            logger.info(f"  ✓ {len(chunks)} chunks creados")
            
        except Exception as e:
            logger.error(f"  ✗ Error procesando {file_path.name}: {e}")
    
    return True

async def main():
    """Función principal de pruebas"""
    logger.info("=" * 60)
    logger.info("INICIANDO PRUEBAS DEL SISTEMA RAG SIEM")
    logger.info("=" * 60)
    
    tests = [
        ("Procesador de documentos", test_document_processor),
        ("Cliente de embeddings", test_embedding_client),
        ("Conexión Supabase", test_supabase_connection),
        ("Sistema RAG", test_rag_system),
        ("Procesamiento de archivos", test_file_processing),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = await test_func()
            results[test_name] = result
            if result:
                logger.info(f"✓ {test_name}: PASÓ")
            else:
                logger.error(f"✗ {test_name}: FALLÓ")
        except Exception as e:
            logger.error(f"✗ {test_name}: ERROR - {e}")
            results[test_name] = False
    
    # Resumen de resultados
    logger.info("\n" + "=" * 60)
    logger.info("RESUMEN DE PRUEBAS")
    logger.info("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASÓ" if result else "✗ FALLÓ"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nTotal: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        logger.info("🎉 ¡Todas las pruebas pasaron! El sistema está listo para usar.")
        logger.info("\nPara procesar documentos SIEM, ejecuta:")
        logger.info("python massive_processing_main.py process SIEM --supabase-url $SUPABASE_URL --supabase-key $SUPABASE_KEY --hf-token $HF_TOKEN --max-workers 16 --batch-size 50")
    else:
        logger.error("❌ Algunas pruebas fallaron. Revisa la configuración.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
