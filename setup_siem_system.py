#!/usr/bin/env python3
"""
Script de configuración para el sistema RAG SIEM
Configura la base de datos y prepara el entorno
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from supabase import create_client, Client

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Configurar variables de entorno"""
    # Leer variables de entorno del archivo .env si existe
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    
    # Verificar variables requeridas
    required_vars = ['SUPABASE_URL', 'SUPABASE_KEY', 'HF_TOKEN']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Variables de entorno faltantes: {', '.join(missing_vars)}")
        logger.error("Por favor, configura las siguientes variables:")
        logger.error("export SUPABASE_URL='tu_url_supabase'")
        logger.error("export SUPABASE_KEY='tu_clave_supabase'")
        logger.error("export HF_TOKEN='tu_token_huggingface'")
        sys.exit(1)
    
    logger.info("Variables de entorno configuradas correctamente")

def create_directories():
    """Crear directorios necesarios"""
    directories = [
        'logs',
        'resultados_procesamiento',
        'temp',
        'src'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Directorio creado: {directory}")

def test_supabase_connection():
    """Probar conexión con Supabase"""
    try:
        url = os.getenv('SUPABASE_URL')
        key = os.getenv('SUPABASE_KEY')
        
        supabase = create_client(url, key)
        
        # Probar conexión
        result = supabase.table('siem_documents').select('id').limit(1).execute()
        logger.info("Conexión con Supabase establecida correctamente")
        return True
        
    except Exception as e:
        logger.error(f"Error conectando con Supabase: {e}")
        return False

def create_env_file():
    """Crear archivo .env de ejemplo"""
    env_content = """# Configuración del Sistema RAG SIEM
# Reemplaza los valores con tus credenciales reales

# Supabase
SUPABASE_URL=https://tu-proyecto.supabase.co
SUPABASE_KEY=tu_clave_supabase_aqui

# Hugging Face
HF_TOKEN=tu_token_huggingface_aqui

# Configuración opcional
MAX_WORKERS=16
BATCH_SIZE=50
EMBEDDING_DIMENSIONS=1024
"""
    
    env_file = Path('.env.example')
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    logger.info(f"Archivo de ejemplo creado: {env_file}")

def main():
    """Función principal de configuración"""
    logger.info("Configurando sistema RAG SIEM...")
    
    # 1. Crear directorios
    create_directories()
    
    # 2. Crear archivo .env de ejemplo
    create_env_file()
    
    # 3. Configurar variables de entorno
    setup_environment()
    
    # 4. Probar conexión con Supabase
    if test_supabase_connection():
        logger.info("✓ Sistema configurado correctamente")
        logger.info("✓ Puedes ejecutar: python massive_processing_main.py process SIEM --supabase-url $SUPABASE_URL --supabase-key $SUPABASE_KEY --hf-token $HF_TOKEN --max-workers 16 --batch-size 50")
    else:
        logger.error("✗ Error en la configuración")
        sys.exit(1)

if __name__ == "__main__":
    main()
