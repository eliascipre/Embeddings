#!/usr/bin/env python3
"""
Prueba rápida del sistema corregido
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.optimized_siem_rag_system import OptimizedSIEMRAGSystem
from src.comercio_exterior_processor import ComercioExteriorProcessor
from src.qwen3_embedding_client import Qwen3EmbeddingClient, Qwen3Config

async def test_system():
    """Probar el sistema corregido"""
    print("🧪 Probando sistema corregido...")
    
    try:
        # Configurar cliente de embeddings
        qwen3_config = Qwen3Config(
            endpoint="https://wfuosp4mnqkimde9.us-east-1.aws.endpoints.huggingface.cloud",
            token="",
            max_batch_size=32,
            rate_limit_per_minute=100,
            timeout_seconds=300,
            retry_attempts=5
        )
        embedding_client = Qwen3EmbeddingClient(qwen3_config)
        
        # Inicializar cliente
        await embedding_client.__aenter__()
        print("✅ Cliente de embeddings inicializado")
        
        # Probar generación de embedding
        test_text = "Esta es una prueba de embedding"
        embedding = await embedding_client.generate_single_embedding(test_text)
        print(f"✅ Embedding generado: {len(embedding)} dimensiones")
        
        # Limpiar
        await embedding_client.__aexit__(None, None, None)
        print("✅ Cliente de embeddings cerrado correctamente")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_system())
