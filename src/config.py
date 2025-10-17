"""
Configuración unificada del sistema RAG optimizado con Hugging Face A100
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

class OptimizedRAGConfig(BaseSettings):
    """Configuración optimizada para RAG masivo con Hugging Face A100"""
    
    # Hugging Face Configuration
    hf_token: str = Field(default="", description="Token de Hugging Face")
    hf_api_url: str = Field(
        default="https://wfuosp4mnqkimde9.us-east-1.aws.endpoints.huggingface.cloud",
        description="URL de la API de Hugging Face"
    )
    
    # Supabase Configuration
    supabase_url: str = Field(
        default="https://zcxqxrgtmnfixkgeaurj.supabase.co",
        description="URL de Supabase"
    )
    supabase_key: str = Field(
        default="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InpjeHF4cmd0bW5maXhrZ2VhdXJqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA2MTk4MTgsImV4cCI6MjA3NjE5NTgxOH0.YqseZLeQDohHdc-s9QduefPqy5SOSyWJysK_q8cMtio",
        description="Clave de Supabase"
    )
    
    # Processing Configuration
    max_workers: int = Field(default=8, description="Número máximo de workers paralelos")
    batch_size: int = Field(default=50, description="Tamaño del lote para procesamiento")
    hf_batch_size: int = Field(default=32, description="Tamaño del lote para Hugging Face")
    max_concurrent_requests: int = Field(default=50, description="Máximo de requests concurrentes a HF")
    
    # Chunking Configuration
    chunk_size: int = Field(default=1200, description="Tamaño de chunk")
    chunk_overlap: int = Field(default=200, description="Solapamiento entre chunks")
    
    # Search Configuration
    default_top_k: int = Field(default=20, description="Número por defecto de resultados")
    vector_weight: float = Field(default=0.7, description="Peso para búsqueda vectorial")
    keyword_weight: float = Field(default=0.3, description="Peso para búsqueda por palabras clave")
    match_threshold: float = Field(default=0.6, description="Umbral de similitud")
    
    # Database Configuration
    supabase_batch_size: int = Field(default=1000, description="Tamaño de lote para Supabase")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Nivel de logging")
    log_file: str = Field(default="massive_processing.log", description="Archivo de log")
    
    # Paths
    project_root: Path = Field(default=Path(__file__).parent.parent)
    data_dir: Path = Field(default=Path(__file__).parent.parent / "data")
    logs_dir: Path = Field(default=Path(__file__).parent.parent / "logs")
    cache_dir: Path = Field(default=Path(__file__).parent.parent / "cache")
    
    # Performance Configuration
    timeout: int = Field(default=300, description="Timeout para requests")
    retry_attempts: int = Field(default=3, description="Intentos de reintento")
    retry_delay: float = Field(default=1.0, description="Delay entre reintentos")
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore"
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Crear directorios necesarios
        self.data_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)

# Instancia global de configuración
config = OptimizedRAGConfig()

# Configurar variables de entorno si no están definidas
if not config.hf_token:
    config.hf_token = os.getenv("HF_TOKEN", "")
if not config.supabase_url:
    config.supabase_url = os.getenv("SUPABASE_URL", config.supabase_url)
if not config.supabase_key:
    config.supabase_key = os.getenv("SUPABASE_KEY", config.supabase_key)
