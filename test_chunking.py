#!/usr/bin/env python3
"""
Script de prueba para verificar el chunking mejorado
"""
import logging
import sys
from pathlib import Path

# Agregar src al path
sys.path.append(str(Path(__file__).parent / "src"))

from simple_document_processor import SimpleDocumentProcessor

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_chunking():
    """Probar el chunking con diferentes tamaños"""
    
    logger.info("🧪 PROBANDO CHUNKING MEJORADO")
    logger.info("="*50)
    
    # Probar con diferentes tamaños de chunk
    chunk_sizes = [500, 1000, 2000, 3000]
    
    # Usar un PDF de prueba
    test_file = Path("SIEM/SIEM/convencion_viena.pdf")
    
    if not test_file.exists():
        logger.error(f"❌ Archivo de prueba no encontrado: {test_file}")
        return
    
    for chunk_size in chunk_sizes:
        logger.info(f"\n📦 Probando chunk size: {chunk_size}")
        logger.info("-" * 30)
        
        # Crear procesador con el tamaño de chunk
        processor = SimpleDocumentProcessor(
            max_chunk_size=chunk_size, 
            chunk_overlap=chunk_size // 10  # 10% de overlap
        )
        
        # Procesar documento
        chunks = processor.process_document(test_file)
        
        logger.info(f"✅ Chunks generados: {len(chunks)}")
        
        if chunks:
            # Mostrar estadísticas de los chunks
            chunk_lengths = [len(chunk['text']) for chunk in chunks]
            avg_length = sum(chunk_lengths) / len(chunk_lengths)
            min_length = min(chunk_lengths)
            max_length = max(chunk_lengths)
            
            logger.info(f"   - Longitud promedio: {avg_length:.0f} caracteres")
            logger.info(f"   - Longitud mínima: {min_length} caracteres")
            logger.info(f"   - Longitud máxima: {max_length} caracteres")
            
            # Mostrar los primeros 3 chunks
            for i, chunk in enumerate(chunks[:3]):
                preview = chunk['text'][:100].replace('\n', ' ')
                logger.info(f"   - Chunk {i+1}: {preview}...")
        
        logger.info("")

if __name__ == "__main__":
    test_chunking()
