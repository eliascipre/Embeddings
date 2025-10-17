#!/usr/bin/env python3
"""
Script de prueba para verificar las correcciones realizadas
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from comercio_exterior_processor import ComercioExteriorProcessor
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_chunk_type_validation():
    """Probar validación de tipos de chunk"""
    processor = ComercioExteriorProcessor()
    
    # Texto de prueba con diferentes tipos de secciones
    test_text = """
    ARTÍCULO 1. Este es un artículo de prueba.
    
    CAPÍTULO I. Este es un capítulo de prueba.
    
    ANEXO A. Este es un anexo de prueba.
    
    INCISO 1. Este es un inciso que debería mapearse a 'section'.
    
    PÁRRAFO 1. Este es un párrafo de prueba.
    """
    
    # Probar detección de tipos
    sections = processor._split_by_sections(test_text)
    
    print("=== PRUEBA DE VALIDACIÓN DE TIPOS DE CHUNK ===")
    for i, section in enumerate(sections):
        if section.strip():
            section_type = processor._detect_section_type(section)
            print(f"Sección {i+1}: '{section[:50]}...' -> Tipo: {section_type}")
    
    # Probar creación de chunks
    chunks = processor.create_comercio_chunks(test_text, "test_document.pdf")
    
    print(f"\n=== CHUNKS CREADOS ({len(chunks)}) ===")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: Tipo='{chunk['type']}', Texto='{chunk['text'][:100]}...'")
    
    # Verificar que todos los tipos están permitidos
    allowed_types = ['articulo', 'paragrafo', 'anexo', 'titulo', 'capitulo', 'seccion', 
                    'resolucion', 'reglamento', 'ley', 'codigo', 'decreto', 'acuerdo', 
                    'convenio', 'tratado', 'dof', 'paragraph', 'section']
    
    invalid_types = []
    for chunk in chunks:
        if chunk['type'] not in allowed_types:
            invalid_types.append(chunk['type'])
    
    if invalid_types:
        print(f"\n❌ ERROR: Tipos no permitidos encontrados: {invalid_types}")
        return False
    else:
        print(f"\n✅ ÉXITO: Todos los tipos de chunk son válidos")
        return True

def test_unstructured_import():
    """Probar que el import de unstructured funciona"""
    try:
        from unstructured.partition.pdf import partition_pdf
        print("✅ ÉXITO: Import de partition_pdf funciona correctamente")
        return True
    except ImportError as e:
        print(f"❌ ERROR: No se puede importar partition_pdf: {e}")
        return False

if __name__ == "__main__":
    print("=== PRUEBAS DE CORRECCIONES ===\n")
    
    # Prueba 1: Validación de tipos de chunk
    test1_passed = test_chunk_type_validation()
    
    # Prueba 2: Import de unstructured
    test2_passed = test_unstructured_import()
    
    print(f"\n=== RESUMEN ===")
    print(f"Validación de tipos de chunk: {'✅ PASÓ' if test1_passed else '❌ FALLÓ'}")
    print(f"Import de unstructured: {'✅ PASÓ' if test2_passed else '❌ FALLÓ'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 TODAS LAS PRUEBAS PASARON - Las correcciones funcionan correctamente")
        sys.exit(0)
    else:
        print("\n⚠️  ALGUNAS PRUEBAS FALLARON - Revisar las correcciones")
        sys.exit(1)

