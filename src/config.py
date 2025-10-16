"""
Configuración del sistema de embeddings legales
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

class LegalRAGConfig(BaseSettings):
    """Configuración del sistema RAG Legal"""
    
    # Supabase Configuration
    supabase_url: str = Field(default="https://zcxqxrgtmnfixkgeaurj.supabase.co")
    supabase_key: str = Field(default="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InpjeHF4cmd0bW5maXhrZ2VhdXJqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA2MTk4MTgsImV4cCI6MjA3NjE5NTgxOH0.YqseZLeQDohHdc-s9QduefPqy5SOSyWJysK_q8cMtio")
    supabase_service_key: Optional[str] = Field(default=None)
    
    # Database Configuration
    database_url: str = Field(default="postgresql://postgres:password@localhost:5432/legal_embeddings")
    pgvector_extension: str = Field(default="vector")
    
    # OpenAI Configuration (para RAG-Anything)
    openai_api_key: Optional[str] = Field(default=None)
    openai_base_url: Optional[str] = Field(default=None)
    
    # Embedding Model Configuration
    embedding_model: str = Field(default="Qwen/Qwen3-Embedding-0.6B")
    embedding_device: str = Field(default="cuda")
    embedding_dimensions: int = Field(default=1024)
    
    # RAG Configuration
    chunk_size: int = Field(default=1200)
    chunk_overlap: int = Field(default=200)
    batch_size: int = Field(default=100)
    similarity_threshold: float = Field(default=0.7)
    
    # GPU Configuration
    cuda_visible_devices: str = Field(default="0")
    torch_device: str = Field(default="cuda")
    
    # Logging Configuration
    log_level: str = Field(default="INFO")
    log_file: str = Field(default="logs/legal_rag.log")
    
    # Processing Configuration
    max_documents: int = Field(default=10000)
    parallel_workers: int = Field(default=4)
    cache_dir: str = Field(default="./cache")
    
    # Paths
    project_root: Path = Field(default=Path(__file__).parent.parent)
    data_dir: Path = Field(default=Path(__file__).parent.parent / "data")
    logs_dir: Path = Field(default=Path(__file__).parent.parent / "logs")
    cache_directory: Path = Field(default=Path(__file__).parent.parent / "cache")
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore"  # Ignorar campos extra
    }

# Instancia global de configuración
config = LegalRAGConfig()

# Crear directorios necesarios
config.data_dir.mkdir(exist_ok=True)
config.logs_dir.mkdir(exist_ok=True)
config.cache_directory.mkdir(exist_ok=True)
