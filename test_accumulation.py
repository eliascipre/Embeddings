#!/usr/bin/env python3
"""
Script de prueba para verificar la acumulación de chunks
"""
import asyncio
import logging
import sys
from pathlib import Path

# Agregar src al path
sys.path.append(str(Path(__file__).parent / "src"))

from optimized_gpu_processing import OptimizedGPURAGSystem

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_accumulation():
    """Probar la acumulación de chunks"""
    
    logger.info("🧪 PROBANDO ACUMULACIÓN DE CHUNKS")
    logger.info("="*50)
    
    # Crear sistema optimizado
    rag_system = OptimizedGPURAGSystem(
        "https://rygrdlradxyykzuudgtu.supabase.co",
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJ5Z3JkbHJhZHh5eWt6dXVkZ3R1Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MDcxODQwNiwiZXhwIjoyMDc2Mjk0NDA2fQ.jbnFbBjlq_NsLaNamGJ98a4uXay4lAZX_BVWQCobpCw",
        "https://wfuosp4mnqkimde9.us-east-1.aws.endpoints.huggingface.cloud"
    )
    
    try:
        rag_system.initialize()
        
        # Procesar solo los primeros 5 archivos para prueba
        source_path = Path("SIEM/SIEM")
        files = []
        for ext in ['.pdf', '.txt', '.md']:
            files.extend(list(source_path.glob(f"**/*{ext}")))
        
        # Limitar a 5 archivos para prueba
        test_files = files[:5]
        logger.info(f"📁 Probando con {len(test_files)} archivos:")
        for f in test_files:
            logger.info(f"   - {f.name}")
        
        # Simular el procesamiento de chunks
        all_chunks = []
        for file_path in test_files:
            try:
                logger.info(f"📄 Procesando: {file_path.name}")
                chunks = rag_system.document_processor.process_document(file_path)
                if chunks:
                    all_chunks.extend(chunks)
                    logger.info(f"   ✅ {len(chunks)} chunks generados")
                else:
                    logger.warning(f"   ⚠️ No se generaron chunks")
            except Exception as e:
                logger.error(f"   ❌ Error: {e}")
        
        logger.info(f"📊 TOTAL CHUNKS ACUMULADOS: {len(all_chunks)}")
        
        # Calcular lotes
        batch_size = 32
        num_batches = (len(all_chunks) + batch_size - 1) // batch_size
        logger.info(f"📦 LOTES CALCULADOS:")
        logger.info(f"   - Tamaño de lote: {batch_size}")
        logger.info(f"   - Número de lotes: {num_batches}")
        logger.info(f"   - Chunks por lote promedio: {len(all_chunks) / num_batches:.1f}")
        
        # Mostrar distribución de lotes
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(all_chunks))
            batch_chunks = all_chunks[start_idx:end_idx]
            logger.info(f"   - Lote {i+1}: {len(batch_chunks)} chunks (índices {start_idx}-{end_idx-1})")
        
        logger.info("✅ PRUEBA DE ACUMULACIÓN COMPLETADA")
        
    except Exception as e:
        logger.error(f"❌ Error en prueba: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(test_accumulation())
