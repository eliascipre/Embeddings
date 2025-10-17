#!/usr/bin/env python3
"""
Prueba simple del cliente de embeddings
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_processing import SimpleEmbeddingClient

def test_embedding():
    """Probar el cliente de embeddings"""
    print("🧪 Probando cliente de embeddings...")
    
    client = SimpleEmbeddingClient()
    
    # Texto de prueba
    test_text = "ARTÍCULO 1. Esta es una prueba de embedding para documentos legales."
    
    print(f"📝 Texto de prueba: {test_text}")
    
    # Generar embedding
    embedding = client.generate_embedding(test_text)
    
    if embedding:
        print(f"✅ Embedding generado exitosamente")
        print(f"📊 Dimensiones: {len(embedding)}")
        print(f"🔢 Primeros 5 valores: {embedding[:5]}")
        print(f"📏 Tamaño total: {len(str(embedding))} caracteres")
    else:
        print("❌ Error generando embedding")

if __name__ == "__main__":
    test_embedding()
