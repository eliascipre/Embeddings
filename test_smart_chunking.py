#!/usr/bin/env python3
"""
Script de prueba para el chunking inteligente
Verifica que el sistema filtre contenido inútil y genere chunks de calidad
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from comercio_exterior_processor import ComercioExteriorProcessor
from smart_legal_chunker import SmartLegalChunker
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_content_filtering():
    """Probar filtrado de contenido inútil"""
    processor = ComercioExteriorProcessor()
    
    # Contenido inútil que debería ser filtrado
    useless_content = [
        "PÁGINA 1 (SIN TEXTO) ---",
        "PÁGINA 2 (SIN TEXTO) ---",
        "Nota: El presente documento se da a conocer en el Portal del SAT",
        "DOF - Diario Oficial de la Federación",
        "FE de erratas al Decreto",
        "Página 1 de 1",
        "--- PÁGINA 1 ---",
        "123",
        "IV",
        "",
        "   ",
    ]
    
    # Contenido útil que debería pasar
    useful_content = [
        "ARTÍCULO 1. Esta Ley tiene por objeto regular el comercio exterior",
        "CAPÍTULO I. De las Disposiciones Generales",
        "TÍTULO PRIMERO. De los Derechos y Obligaciones",
        "SECCIÓN PRIMERA. De los Derechos Fundamentales",
        "PÁRRAFO PRIMERO. Los derechos fundamentales son inviolables",
        "FRACCIÓN I. El derecho a la vida",
        "TRANSITORIO PRIMERO. Esta Ley entrará en vigor al día siguiente",
        "DEFINICIONES. Para efectos de esta Ley se entenderá por:",
        "DISPOSICIONES FINALES. Esta Ley se publicará en el Diario Oficial",
        "La mercancía de procedencia extranjera deberá cumplir con los requisitos",
        "El valor en aduana se determinará conforme a las reglas establecidas",
    ]
    
    print("=== PRUEBA DE FILTRADO DE CONTENIDO ===")
    
    # Probar contenido inútil
    print("\n📋 Contenido inútil (debería ser filtrado):")
    for content in useless_content:
        is_useful = processor._is_useful_content(content)
        status = "✅ FILTRADO" if not is_useful else "❌ NO FILTRADO"
        print(f"  '{content[:50]}...' -> {status}")
    
    # Probar contenido útil
    print("\n📋 Contenido útil (debería pasar):")
    for content in useful_content:
        is_useful = processor._is_useful_content(content)
        status = "✅ PASA" if is_useful else "❌ FILTRADO INCORRECTAMENTE"
        print(f"  '{content[:50]}...' -> {status}")
    
    return True

def test_smart_chunking():
    """Probar chunking inteligente"""
    chunker = SmartLegalChunker(max_chunk_size=1000, compression_threshold=500)
    
    # Texto de prueba con estructura legal
    test_text = """
    LEY DE COMERCIO EXTERIOR
    
    CAPÍTULO I
    DISPOSICIONES GENERALES
    
    ARTÍCULO 1. Esta Ley tiene por objeto regular el comercio exterior de los Estados Unidos Mexicanos, 
    así como establecer las bases para el desarrollo de las actividades de comercio exterior.
    
    ARTÍCULO 2. Para efectos de esta Ley se entenderá por:
    I. Comercio exterior: el intercambio de mercancías entre el territorio nacional y el extranjero;
    II. Mercancía: todo objeto material que pueda ser objeto de comercio;
    III. Aduana: la oficina pública encargada de controlar el tráfico de mercancías.
    
    PÁRRAFO PRIMERO. Las definiciones contenidas en este artículo son de carácter general.
    
    PÁRRAFO SEGUNDO. Para casos específicos se aplicarán las definiciones contenidas en los tratados internacionales.
    
    CAPÍTULO II
    DE LAS AUTORIDADES COMPETENTES
    
    ARTÍCULO 3. La Secretaría de Economía será la autoridad competente para la aplicación de esta Ley.
    
    ARTÍCULO 4. La Secretaría de Hacienda y Crédito Público será la autoridad competente para el control aduanero.
    """
    
    print("\n=== PRUEBA DE CHUNKING INTELIGENTE ===")
    
    # Crear chunks
    chunks = chunker.chunk_document(test_text, Path("test_document.pdf"))
    
    print(f"\n📊 Chunks generados: {len(chunks)}")
    
    # Mostrar cada chunk
    for i, chunk in enumerate(chunks):
        print(f"\n--- CHUNK {i+1} ---")
        print(f"Tipo: {chunk.chunk_type}")
        print(f"ID: {chunk.chunk_id}")
        print(f"Nivel jerárquico: {chunk.hierarchy_level}")
        print(f"Artículo: {chunk.article_number}")
        print(f"Párrafo: {chunk.paragraph_number}")
        print(f"Inciso: {chunk.inciso_number}")
        print(f"Palabras: {chunk.word_count}")
        print(f"Caracteres: {chunk.char_count}")
        print(f"Comprimido: {chunk.is_compressed}")
        print(f"Contenido: {chunk.content[:100]}...")
    
    # Obtener estadísticas
    stats = chunker.get_chunk_statistics(chunks)
    print(f"\n📈 ESTADÍSTICAS:")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Chunks comprimidos: {stats['compressed_chunks']}")
    print(f"Ratio de compresión: {stats['compression_ratio']:.2%}")
    print(f"Tamaño promedio: {stats['avg_chunk_size']:.0f} caracteres")
    print(f"Tipos de chunk: {stats['chunk_types']}")
    print(f"Niveles jerárquicos: {stats['hierarchy_levels']}")
    
    return len(chunks) > 0

def test_chunk_validation():
    """Probar validación de tipos de chunk"""
    from src.smart_legal_chunker import LegalChunk
    chunker = SmartLegalChunker()
    
    # Tipos permitidos
    allowed_types = chunker.allowed_chunk_types
    
    print("\n=== PRUEBA DE VALIDACIÓN DE TIPOS ===")
    print(f"Tipos permitidos: {sorted(allowed_types)}")
    
    # Crear chunks con diferentes tipos (contenido más largo para pasar validación)
    test_chunks = [
        LegalChunk(content="ARTÍCULO 1. Esta es una prueba de validación de tipos de chunk para artículos.", chunk_type="articulo", chunk_id="test1", hierarchy_level=1),
        LegalChunk(content="PÁRRAFO PRIMERO. Este es un párrafo de prueba para validar el tipo de chunk correspondiente.", chunk_type="paragrafo", chunk_id="test2", hierarchy_level=2),
        LegalChunk(content="INCISO A). Este es un inciso de prueba para validar el tipo de chunk correspondiente.", chunk_type="inciso", chunk_id="test3", hierarchy_level=3),
        LegalChunk(content="CONTENIDO INVÁLIDO. Este chunk tiene un tipo inválido que debería ser mapeado.", chunk_type="invalid_type", chunk_id="test4", hierarchy_level=1),
        LegalChunk(content="ANEXO I. Este es un anexo de prueba para validar el tipo de chunk correspondiente.", chunk_type="anexo", chunk_id="test5", hierarchy_level=1),
    ]
    
    # Validar chunks
    validated_chunks = chunker._validate_chunks(test_chunks)
    
    print(f"\nChunks originales: {len(test_chunks)}")
    print(f"Chunks validados: {len(validated_chunks)}")
    
    for chunk in validated_chunks:
        print(f"  {chunk.chunk_id}: {chunk.chunk_type}")
    
    return len(validated_chunks) > 0

def test_compression():
    """Probar compresión de chunks"""
    from src.smart_legal_chunker import LegalChunk
    chunker = SmartLegalChunker(max_chunk_size=1000, compression_threshold=500)
    
    # Crear chunk largo
    long_content = "Este es un contenido muy largo. " * 100  # ~3000 caracteres
    long_chunk = LegalChunk(
        content=long_content,
        chunk_type="articulo",
        chunk_id="long_test",
        hierarchy_level=1,
        char_count=len(long_content)
    )
    
    print("\n=== PRUEBA DE COMPRESIÓN ===")
    print(f"Contenido original: {len(long_content)} caracteres")
    
    # Comprimir chunks
    compressed_chunks = chunker._compress_large_chunks([long_chunk])
    
    if compressed_chunks[0].is_compressed:
        print(f"✅ Chunk comprimido")
        print(f"Ratio de compresión: {compressed_chunks[0].compression_ratio:.2f}")
        
        # Probar descompresión
        decompressed = chunker.decompress_chunk(compressed_chunks[0])
        print(f"Descomprimido: {len(decompressed)} caracteres")
        print(f"Contenido idéntico: {decompressed == long_content}")
    else:
        print("❌ Chunk no fue comprimido")
    
    return True

def main():
    """Función principal de pruebas"""
    print("🧪 INICIANDO PRUEBAS DEL CHUNKING INTELIGENTE\n")
    
    tests = [
        ("Filtrado de contenido", test_content_filtering),
        ("Chunking inteligente", test_smart_chunking),
        ("Validación de tipos", test_chunk_validation),
        ("Compresión de chunks", test_compression),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"PRUEBA: {test_name}")
        print(f"{'='*60}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASÓ" if result else "❌ FALLÓ"
            print(f"\n{status}: {test_name}")
        except Exception as e:
            print(f"\n❌ ERROR en {test_name}: {e}")
            results.append((test_name, False))
    
    # Resumen final
    print(f"\n{'='*60}")
    print("RESUMEN DE PRUEBAS")
    print(f"{'='*60}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASÓ" if result else "❌ FALLÓ"
        print(f"{status}: {test_name}")
    
    print(f"\n📊 RESULTADO FINAL: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print("🎉 ¡TODAS LAS PRUEBAS PASARON! El chunking inteligente está funcionando correctamente.")
        return 0
    else:
        print("⚠️  ALGUNAS PRUEBAS FALLARON. Revisar la implementación.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
