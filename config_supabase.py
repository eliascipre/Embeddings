"""
Configuración para Supabase - VectorDB_Comercio_Exterior
"""

import os
from typing import Dict, Any

# Configuración de Supabase
SUPABASE_CONFIG = {
    'url': 'https://rygrdlradxyykzuudgtu.supabase.co',
    'api_key': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJ5Z3JkbHJhZHh5eWt6dXVkZ3R1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA3MTg0MDYsImV4cCI6MjA3NjI5NDQwNn0.ZeFX5rK3kfqupVX_yHKluIsCB1beCfCLsS1fdqEU_10',
    'project_id': 'rygrdlradxyykzuudgtu'
}

# Configuración de la base de datos
DATABASE_CONFIG = {
    'schema': 'public',
    'tables': {
        'documents': 'siem_documents',
        'chunks': 'siem_chunks',
        'embeddings': 'siem_embeddings',
        'metadata': 'siem_metadata'
    }
}

# Configuración de embeddings
EMBEDDING_CONFIG = {
    'model_name': 'Qwen3:0.6b',
    'huggingface_endpoint': 'https://wfuosp4mnqkimde9.us-east-1.aws.endpoints.huggingface.cloud',
    'max_tokens': 512,
    'chunk_overlap': 50,
    'min_chunk_size': 100,
    'dimension': 1024,  # Qwen3-Embedding-0.6B soporta hasta 1024 dimensiones (máxima calidad)
    'api_type': 'custom_endpoint'  # Indica que es un endpoint personalizado
}

# Configuración de chunking para documentos legales
CHUNKING_CONFIG = {
    'max_chunk_size': 512,
    'chunk_overlap': 50,
    'min_chunk_size': 100,
    'sentence_splitter': r'[.!?]+(?:\s|$)',
    'paragraph_splitter': r'\n\s*\n',
    'section_splitter': r'\n\s*(?:Artículo|CAPÍTULO|TÍTULO|SECCIÓN|Art\.|Cap\.|Tit\.|Sec\.)',
    'legal_patterns': {
        'articulo': r'Artículo\s+\d+',
        'capitulo': r'CAPÍTULO\s+[IVX]+',
        'titulo': r'TÍTULO\s+[IVX]+',
        'seccion': r'SECCIÓN\s+[IVX]+',
        'inciso': r'[a-z]\)',
        'fraccion': r'\d+\.',
        'parrafo': r'[A-Z]\.',
        'ley': r'Ley\s+[A-Za-z\s]+',
        'reglamento': r'Reglamento\s+[A-Za-z\s]+',
        'decreto': r'Decreto\s+[A-Za-z\s]+',
        'acuerdo': r'Acuerdo\s+[A-Za-z\s]+'
    }
}

# Configuración de procesamiento
PROCESSING_CONFIG = {
    'batch_size': 100,
    'max_files_per_batch': 50,
    'retry_attempts': 3,
    'timeout_seconds': 300,
    'supported_extensions': ['.pdf', '.docx', '.doc', '.txt'],
    'output_dir': 'resultados_procesamiento',
    'logs_dir': 'logs'
}

def get_supabase_url() -> str:
    """Retorna la URL de Supabase"""
    return SUPABASE_CONFIG['url']

def get_supabase_key() -> str:
    """Retorna la API key de Supabase"""
    return SUPABASE_CONFIG['api_key']

def get_database_tables() -> Dict[str, str]:
    """Retorna la configuración de tablas"""
    return DATABASE_CONFIG['tables']

def get_embedding_config() -> Dict[str, Any]:
    """Retorna la configuración de embeddings"""
    return EMBEDDING_CONFIG

def get_chunking_config() -> Dict[str, Any]:
    """Retorna la configuración de chunking"""
    return CHUNKING_CONFIG

def get_processing_config() -> Dict[str, Any]:
    """Retorna la configuración de procesamiento"""
    return PROCESSING_CONFIG

# Variables de entorno (opcional)
def setup_environment():
    """Configura las variables de entorno"""
    os.environ['SUPABASE_URL'] = SUPABASE_CONFIG['url']
    os.environ['SUPABASE_KEY'] = SUPABASE_CONFIG['api_key']
    os.environ['SUPABASE_PROJECT_ID'] = SUPABASE_CONFIG['project_id']
