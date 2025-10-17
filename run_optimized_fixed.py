#!/usr/bin/env python3
"""
Script principal para ejecutar el procesamiento optimizado corregido
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
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('optimized_processing.log')
    ]
)

logger = logging.getLogger(__name__)

async def main():
    """Función principal"""
    try:
        # Inicializar sistema
        logger.info("🚀 Inicializando sistema RAG optimizado...")
        system = OptimizedGPURAGSystem(
            supabase_url="https://rygrdlradxyykzuudgtu.supabase.co",
            supabase_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJ5Z3JkbHJhZHh5eWt6dXVkZ3R1Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MDcxODQwNiwiZXhwIjoyMDc2Mjk0NDA2fQ.jbnFbBjlq_NsLaNamGJ98a4uXay4lAZX_BVWQCobpCw",
            hf_endpoint="https://wfuosp4mnqkimde9.us-east-1.aws.endpoints.huggingface.cloud"
        )
        
        # Procesar todos los archivos
        results = await system.process_documents_optimized("SIEM/SIEM", max_workers=20)
        
        logger.info(f"✅ Procesamiento completado: {results}")
        
    except Exception as e:
        logger.error(f"❌ Error en procesamiento: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(main())
