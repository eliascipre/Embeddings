#!/usr/bin/env python3
"""
Script de prueba para verificar las correcciones
"""
import asyncio
import logging
from pathlib import Path
import sys

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent / "src"))

from src.legal_chunking_processor import LegalChunkingProcessor

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_date_conversion():
    """Probar conversión de fechas"""
    processor = LegalChunkingProcessor()
    
    test_dates = [
        "9 de Abril de 2022",
        "14 DE MARZO DE 2014",
        "18 DE MAYO DE 2022",
        "5 DE JULIO DE 2012",
        "23 de septiembre de 1944",
        "22 de agosto de 2009",
        "19 DE DICIEMBRE DE 2019",
        "29 de mayo de 2020",
        "20 de agosto de 2022",
        "17 DE ENERO DE 2025"
    ]
    
    logger.info("🧪 Probando conversión de fechas:")
    for date_str in test_dates:
        iso_date = processor._convert_spanish_date_to_iso(date_str)
        logger.info(f"   '{date_str}' -> '{iso_date}'")

def test_batch_size():
    """Probar tamaño de lote"""
    from src.huggingface_api_client import HFConfig
    
    config = HFConfig()
    logger.info(f"📦 Tamaño de lote configurado: {config.batch_size}")
    
    if config.batch_size <= 32:
        logger.info("✅ Tamaño de lote correcto (≤32)")
    else:
        logger.error("❌ Tamaño de lote demasiado grande (>32)")

if __name__ == "__main__":
    logger.info("🚀 Iniciando pruebas de correcciones...")
    
    test_date_conversion()
    test_batch_size()
    
    logger.info("✅ Pruebas completadas")
