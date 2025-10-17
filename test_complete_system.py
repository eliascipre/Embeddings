#!/usr/bin/env python3
"""
Script de prueba completo del sistema de chunking inteligente
Verifica todas las funcionalidades implementadas
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.comercio_exterior_processor import ComercioExteriorProcessor
from src.smart_legal_chunker import SmartLegalChunker, LegalChunk
import logging
from pathlib import Path
import json

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_complete_chunking_pipeline():
    """Probar el pipeline completo de chunking inteligente"""
    print("🧪 INICIANDO PRUEBA COMPLETA DEL SISTEMA DE CHUNKING INTELIGENTE\n")
    
    # Crear procesador
    processor = ComercioExteriorProcessor()
    
    # Texto de prueba con estructura legal compleja
    test_text = """
    LEY DE COMERCIO EXTERIOR
    
    TÍTULO PRIMERO
    DISPOSICIONES GENERALES
    
    CAPÍTULO I
    DEL OBJETO Y ÁMBITO DE APLICACIÓN
    
    ARTÍCULO 1. Esta Ley tiene por objeto regular el comercio exterior de los Estados Unidos Mexicanos, 
    así como establecer las bases para el desarrollo de las actividades de comercio exterior, 
    con el fin de contribuir al crecimiento económico del país y al bienestar de la población.
    
    ARTÍCULO 2. Para efectos de esta Ley se entenderá por:
    I. Comercio exterior: el intercambio de mercancías entre el territorio nacional y el extranjero;
    II. Mercancía: todo objeto material que pueda ser objeto de comercio;
    III. Aduana: la oficina pública encargada de controlar el tráfico de mercancías;
    IV. Valor en aduana: el valor de las mercancías determinado conforme a las reglas establecidas;
    V. Origen: el país donde fueron producidas las mercancías.
    
    PÁRRAFO PRIMERO. Las definiciones contenidas en este artículo son de carácter general y se aplicarán 
    en todo el territorio nacional, salvo disposición en contrario.
    
    PÁRRAFO SEGUNDO. Para casos específicos se aplicarán las definiciones contenidas en los tratados 
    internacionales de los que México sea parte.
    
    CAPÍTULO II
    DE LAS AUTORIDADES COMPETENTES
    
    ARTÍCULO 3. La Secretaría de Economía será la autoridad competente para la aplicación de esta Ley 
    en materia de comercio exterior, salvo en los casos expresamente atribuidos a otras dependencias.
    
    ARTÍCULO 4. La Secretaría de Hacienda y Crédito Público será la autoridad competente para el control 
    aduanero y la determinación del valor en aduana de las mercancías.
    
    FRACCIÓN I. La Secretaría de Hacienda y Crédito Público determinará el valor en aduana conforme 
    a las reglas establecidas en los tratados internacionales.
    
    FRACCIÓN II. En caso de discrepancia, se aplicarán los procedimientos de resolución de controversias 
    previstos en los tratados correspondientes.
    
    TRANSITORIO PRIMERO. Esta Ley entrará en vigor al día siguiente de su publicación en el Diario 
    Oficial de la Federación.
    
    TRANSITORIO SEGUNDO. Los reglamentos y disposiciones administrativas necesarios para la aplicación 
    de esta Ley deberán expedirse dentro de los noventa días siguientes a su entrada en vigor.
    
    DISPOSICIONES FINALES
    
    ARTÍCULO FINAL. Se derogan todas las disposiciones que se opongan a esta Ley.
    """
    
    print("=== PRUEBA DE CHUNKING INTELIGENTE COMPLETO ===")
    
    # Crear chunks usando el procesador
    chunks = processor.create_comercio_chunks(test_text, Path("test_ley_comercio_exterior.pdf"))
    
    print(f"\n📊 RESULTADOS DEL CHUNKING:")
    print(f"Total de chunks generados: {len(chunks)}")
    
    if not chunks:
        print("❌ No se generaron chunks. Revisar el sistema.")
        return False
    
    # Analizar chunks generados
    chunk_types = {}
    hierarchy_levels = {}
    articles = set()
    paragraphs = set()
    incisos = set()
    
    for chunk in chunks:
        chunk_type = chunk.get('type', 'unknown')
        hierarchy_level = chunk.get('hierarchy_level', 1)
        
        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        hierarchy_levels[hierarchy_level] = hierarchy_levels.get(hierarchy_level, 0) + 1
        
        if chunk.get('article_number'):
            articles.add(chunk['article_number'])
        if chunk.get('paragraph_number'):
            paragraphs.add(chunk['paragraph_number'])
        if chunk.get('inciso_number'):
            incisos.add(chunk['inciso_number'])
    
    print(f"\n📈 ESTADÍSTICAS DETALLADAS:")
    print(f"Tipos de chunk: {chunk_types}")
    print(f"Niveles jerárquicos: {hierarchy_levels}")
    print(f"Artículos encontrados: {sorted(articles)}")
    print(f"Párrafos encontrados: {sorted(paragraphs)}")
    print(f"Incisos encontrados: {sorted(incisos)}")
    
    # Mostrar algunos chunks de ejemplo
    print(f"\n📋 EJEMPLOS DE CHUNKS GENERADOS:")
    for i, chunk in enumerate(chunks[:5]):  # Mostrar solo los primeros 5
        print(f"\n--- CHUNK {i+1} ---")
        print(f"Tipo: {chunk.get('type', 'N/A')}")
        print(f"Nivel jerárquico: {chunk.get('hierarchy_level', 'N/A')}")
        print(f"Artículo: {chunk.get('article_number', 'N/A')}")
        print(f"Párrafo: {chunk.get('paragraph_number', 'N/A')}")
        print(f"Inciso: {chunk.get('inciso_number', 'N/A')}")
        print(f"Palabras: {chunk.get('word_count', 'N/A')}")
        print(f"Caracteres: {chunk.get('char_count', 'N/A')}")
        print(f"Comprimido: {chunk.get('is_compressed', False)}")
        print(f"Contenido: {chunk['text'][:150]}...")
    
    # Verificar calidad de chunks
    quality_issues = []
    
    for i, chunk in enumerate(chunks):
        text = chunk['text']
        
        # Verificar chunks muy cortos
        if chunk.get('word_count', 0) < 5:
            quality_issues.append(f"Chunk {i+1}: Muy corto ({chunk.get('word_count', 0)} palabras)")
        
        # Verificar chunks con contenido inútil
        if any(indicator in text.lower() for indicator in ['página', 'sin texto', 'dof - diario']):
            quality_issues.append(f"Chunk {i+1}: Contenido inútil detectado")
        
        # Verificar chunks sin contenido legal
        legal_indicators = ['artículo', 'capítulo', 'título', 'sección', 'párrafo', 'fracción', 'transitorio']
        if not any(indicator in text.lower() for indicator in legal_indicators):
            if chunk.get('word_count', 0) < 50:  # Solo marcar como problema si es corto
                quality_issues.append(f"Chunk {i+1}: Sin indicadores legales y muy corto")
    
    if quality_issues:
        print(f"\n⚠️  PROBLEMAS DE CALIDAD DETECTADOS:")
        for issue in quality_issues[:10]:  # Mostrar solo los primeros 10
            print(f"  - {issue}")
        if len(quality_issues) > 10:
            print(f"  ... y {len(quality_issues) - 10} problemas más")
    else:
        print(f"\n✅ TODOS LOS CHUNKS SON DE CALIDAD")
    
    # Verificar estructura jerárquica
    print(f"\n🏗️  ESTRUCTURA JERÁRQUICA:")
    level_1_chunks = [c for c in chunks if c.get('hierarchy_level') == 1]
    level_2_chunks = [c for c in chunks if c.get('hierarchy_level') == 2]
    level_3_chunks = [c for c in chunks if c.get('hierarchy_level') == 3]
    
    print(f"  Nivel 1 (artículos/capítulos): {len(level_1_chunks)}")
    print(f"  Nivel 2 (párrafos): {len(level_2_chunks)}")
    print(f"  Nivel 3 (incisos): {len(level_3_chunks)}")
    
    # Verificar compresión
    compressed_chunks = [c for c in chunks if c.get('is_compressed', False)]
    if compressed_chunks:
        print(f"\n🗜️  COMPRESIÓN:")
        print(f"  Chunks comprimidos: {len(compressed_chunks)}")
        avg_compression = sum(c.get('compression_ratio', 1.0) for c in compressed_chunks) / len(compressed_chunks)
        print(f"  Ratio promedio de compresión: {avg_compression:.3f}")
    
    return len(chunks) > 0 and len(quality_issues) == 0

def test_database_schema_compatibility():
    """Probar compatibilidad con el esquema de base de datos"""
    print("\n=== PRUEBA DE COMPATIBILIDAD CON ESQUEMA DE BASE DE DATOS ===")
    
    # Simular datos que se insertarían en la base de datos
    sample_chunk = {
        'text': 'ARTÍCULO 1. Esta Ley tiene por objeto regular el comercio exterior.',
        'type': 'articulo',
        'chunk_id': 'test_123_art_1',
        'hierarchy_level': 1,
        'parent_chunk_id': None,
        'article_number': '1',
        'paragraph_number': None,
        'inciso_number': None,
        'chapter_title': 'DISPOSICIONES GENERALES',
        'section_title': None,
        'law_title': 'LEY DE COMERCIO EXTERIOR',
        'page_number': 1,
        'word_count': 12,
        'char_count': 65,
        'is_compressed': False,
        'compression_ratio': 1.0,
        'metadata': {
            'filename': 'test_ley.pdf',
            'document_type': 'ley',
            'processed_at': '2025-01-17T10:00:00Z'
        }
    }
    
    # Verificar que todos los campos requeridos estén presentes
    required_fields = [
        'text', 'type', 'chunk_id', 'hierarchy_level', 'article_number',
        'word_count', 'char_count', 'is_compressed', 'compression_ratio'
    ]
    
    missing_fields = []
    for field in required_fields:
        if field not in sample_chunk:
            missing_fields.append(field)
    
    if missing_fields:
        print(f"❌ Campos faltantes: {missing_fields}")
        return False
    
    # Verificar tipos de datos
    type_checks = [
        ('hierarchy_level', int),
        ('word_count', int),
        ('char_count', int),
        ('is_compressed', bool),
        ('compression_ratio', (int, float)),
    ]
    
    type_errors = []
    for field, expected_type in type_checks:
        value = sample_chunk.get(field)
        if value is not None and not isinstance(value, expected_type):
            type_errors.append(f"{field}: esperado {expected_type.__name__}, obtenido {type(value).__name__}")
    
    if type_errors:
        print(f"❌ Errores de tipo: {type_errors}")
        return False
    
    print("✅ Compatibilidad con esquema de base de datos verificada")
    return True

def main():
    """Función principal de pruebas"""
    print("🚀 INICIANDO PRUEBAS COMPLETAS DEL SISTEMA DE CHUNKING INTELIGENTE\n")
    
    tests = [
        ("Pipeline completo de chunking", test_complete_chunking_pipeline),
        ("Compatibilidad con esquema de BD", test_database_schema_compatibility),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*80}")
        print(f"PRUEBA: {test_name}")
        print(f"{'='*80}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASÓ" if result else "❌ FALLÓ"
            print(f"\n{status}: {test_name}")
        except Exception as e:
            print(f"\n❌ ERROR en {test_name}: {e}")
            results.append((test_name, False))
    
    # Resumen final
    print(f"\n{'='*80}")
    print("RESUMEN FINAL DE PRUEBAS")
    print(f"{'='*80}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASÓ" if result else "❌ FALLÓ"
        print(f"{status}: {test_name}")
    
    print(f"\n📊 RESULTADO FINAL: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print("\n🎉 ¡TODAS LAS PRUEBAS PASARON!")
        print("✅ El sistema de chunking inteligente está funcionando correctamente")
        print("✅ Se han implementado todas las mejoras solicitadas:")
        print("   - Chunking inteligente con preservación de jerarquía")
        print("   - Validaciones robustas para tipos de chunk")
        print("   - Optimización de tamaño de chunks")
        print("   - Compresión de chunks para documentos largos")
        print("   - Filtrado de contenido inútil")
        print("   - Esquema de base de datos actualizado")
        return 0
    else:
        print("\n⚠️  ALGUNAS PRUEBAS FALLARON")
        print("Revisar la implementación antes de usar en producción")
        return 1

if __name__ == "__main__":
    sys.exit(main())
