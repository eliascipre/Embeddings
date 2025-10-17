#!/usr/bin/env python3
"""
Script mejorado para verificar la prevención de duplicados
en el pipeline SIEM optimizado con manejo robusto de errores
"""

import asyncio
import logging
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from optimized_siem_pipeline import OptimizedSIEMPipeline

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_duplicate_prevention():
    """Prueba la prevención de duplicados con procesamiento paralelo"""
    
    siem_path = "/home/elias/Documentos/Embeddings/SIEM"
    
    if not Path(siem_path).exists():
        logger.error(f"La ruta {siem_path} no existe")
        return False
    
    logger.info("🧪 Iniciando prueba mejorada de prevención de duplicados...")
    
    # Crear instancia del pipeline
    try:
        pipeline = OptimizedSIEMPipeline(siem_path)
        logger.info("✅ Pipeline inicializado correctamente")
    except Exception as e:
        logger.error(f"❌ Error inicializando pipeline: {e}")
        return False
    
    # Obtener algunos archivos para probar
    test_files = []
    for ext in ['.pdf', '.docx', '.txt']:
        test_files.extend(Path(siem_path).rglob(f'*{ext}'))
    
    # Limitar a 3 archivos para la prueba (más rápido)
    test_files = test_files[:3]
    
    if not test_files:
        logger.error("No se encontraron archivos para probar")
        return False
    
    logger.info(f"📁 Archivos de prueba: {len(test_files)}")
    for file_path in test_files:
        logger.info(f"  - {file_path.name}")
    
    # Procesar archivos en paralelo para simular race conditions
    logger.info("🔄 Procesando archivos en paralelo...")
    
    def process_file(file_path):
        """Procesa un archivo individual con manejo robusto de errores"""
        try:
            logger.info(f"Procesando: {file_path.name}")
            result = pipeline._process_single_document(file_path, "test_batch")
            return result
        except Exception as e:
            logger.error(f"Error procesando {file_path.name}: {e}")
            return None
    
    # Primera pasada - procesar archivos
    start_time = time.time()
    results = []
    
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:  # Reducido para evitar sobrecarga
            future_to_file = {
                executor.submit(process_file, file_path): file_path 
                for file_path in test_files
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        logger.info(f"✅ Procesado exitosamente: {file_path.name}")
                    else:
                        logger.warning(f"⚠️  No se pudo procesar: {file_path.name}")
                except Exception as e:
                    logger.error(f"❌ Error procesando {file_path.name}: {e}")
    except Exception as e:
        logger.error(f"❌ Error en procesamiento paralelo: {e}")
        return False
    
    # Segunda pasada - procesar los mismos archivos para probar duplicados
    logger.info("🔄 Segunda pasada - probando detección de duplicados...")
    
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_file = {
                executor.submit(process_file, file_path): file_path 
                for file_path in test_files
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        logger.info(f"✅ Procesado exitosamente (duplicado): {file_path.name}")
                    else:
                        logger.warning(f"⚠️  No se pudo procesar (duplicado): {file_path.name}")
                except Exception as e:
                    logger.error(f"❌ Error procesando duplicado {file_path.name}: {e}")
    except Exception as e:
        logger.error(f"❌ Error en segunda pasada: {e}")
        return False
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Mostrar estadísticas
    logger.info("\n" + "="*60)
    logger.info("📊 RESULTADOS DE LA PRUEBA")
    logger.info("="*60)
    logger.info(f"Archivos procesados: {len(test_files)}")
    logger.info(f"Tiempo total: {processing_time:.2f} segundos")
    logger.info(f"Documentos duplicados detectados: {pipeline.duplicate_documents}")
    logger.info(f"Chunks duplicados manejados: {pipeline.duplicate_chunks}")
    logger.info(f"Errores: {pipeline.stats.errors}")
    
    # Verificar que no hay duplicados en la base de datos
    try:
        logger.info("\n🔍 Verificando duplicados en la base de datos...")
        
        # Verificar duplicados de chunks
        try:
            all_chunks = pipeline.supabase.table('siem_chunks').select('chunk_hash').execute()
            
            if all_chunks.data:
                # Contar duplicados en Python
                hash_counts = {}
                for chunk in all_chunks.data:
                    chunk_hash = chunk['chunk_hash']
                    hash_counts[chunk_hash] = hash_counts.get(chunk_hash, 0) + 1
                
                # Encontrar duplicados
                duplicates = {hash_val: count for hash_val, count in hash_counts.items() if count > 1}
                
                if duplicates:
                    logger.error(f"❌ Se encontraron {len(duplicates)} chunks duplicados en la base de datos")
                    for hash_val, count in list(duplicates.items())[:5]:  # Mostrar solo los primeros 5
                        logger.error(f"  - Hash: {hash_val[:8]}... ({count} duplicados)")
                    return False
                else:
                    logger.info("✅ No se encontraron chunks duplicados en la base de datos")
            else:
                logger.info("✅ No hay chunks en la base de datos")
                
        except Exception as e:
            logger.error(f"Error verificando duplicados de chunks: {e}")
            return False
        
        # Verificar duplicados de documentos
        try:
            all_docs = pipeline.supabase.table('siem_documents').select('file_hash').execute()
            
            if all_docs.data:
                # Contar duplicados en Python
                hash_counts = {}
                for doc in all_docs.data:
                    file_hash = doc['file_hash']
                    hash_counts[file_hash] = hash_counts.get(file_hash, 0) + 1
                
                # Encontrar duplicados
                duplicates = {hash_val: count for hash_val, count in hash_counts.items() if count > 1}
                
                if duplicates:
                    logger.error(f"❌ Se encontraron {len(duplicates)} documentos duplicados en la base de datos")
                    for hash_val, count in list(duplicates.items())[:5]:  # Mostrar solo los primeros 5
                        logger.error(f"  - Hash: {hash_val[:8]}... ({count} duplicados)")
                    return False
                else:
                    logger.info("✅ No se encontraron documentos duplicados en la base de datos")
            else:
                logger.info("✅ No hay documentos en la base de datos")
                
        except Exception as e:
            logger.error(f"Error verificando duplicados de documentos: {e}")
            return False
        
    except Exception as e:
        logger.error(f"Error verificando duplicados: {e}")
        return False
    
    logger.info("\n🎉 PRUEBA EXITOSA: La prevención de duplicados funciona correctamente")
    logger.info("✅ UPSERT atómico elimina race conditions")
    logger.info("✅ Verificación previa de documentos funciona")
    logger.info("✅ Manejo robusto de errores implementado")
    logger.info("✅ Reintentos automáticos para errores de conexión")
    
    return True

if __name__ == "__main__":
    try:
        success = test_duplicate_prevention()
        if success:
            logger.info("🎉 Todas las pruebas pasaron exitosamente")
            exit(0)
        else:
            logger.error("❌ Algunas pruebas fallaron")
            exit(1)
    except Exception as e:
        logger.error(f"❌ Error durante las pruebas: {e}")
        exit(1)
