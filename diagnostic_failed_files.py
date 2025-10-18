#!/usr/bin/env python3
"""
Script de diagnóstico para analizar archivos que fallaron en el procesamiento
Identifica exactamente por qué no se pudieron procesar los 18 archivos restantes
"""
import fitz
import logging
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PDFDiagnostic:
    """Diagnóstico completo de archivos PDF"""
    
    def __init__(self):
        self.results = []
    
    def analyze_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Analizar un archivo PDF en detalle"""
        result = {
            'file_name': file_path.name,
            'file_path': str(file_path),
            'file_size': 0,
            'exists': False,
            'is_readable': False,
            'page_count': 0,
            'total_text_length': 0,
            'pages_with_text': 0,
            'pages_without_text': 0,
            'text_samples': [],
            'error': None,
            'diagnosis': '',
            'recommendation': ''
        }
        
        try:
            # Verificar existencia y tamaño
            if file_path.exists():
                result['exists'] = True
                result['file_size'] = file_path.stat().st_size
                logger.info(f"📄 Analizando: {file_path.name} ({result['file_size']:,} bytes)")
            else:
                result['error'] = "Archivo no encontrado"
                result['diagnosis'] = "ARCHIVO_NO_ENCONTRADO"
                result['recommendation'] = "Verificar ruta del archivo"
                return result
            
            # Intentar abrir el PDF
            try:
                doc = fitz.open(str(file_path))
                result['is_readable'] = True
                result['page_count'] = len(doc)
                
                # Analizar cada página
                for page_num in range(len(doc)):
                    try:
                        page = doc.load_page(page_num)
                        page_text = page.get_text()
                        
                        if page_text.strip():
                            result['pages_with_text'] += 1
                            result['total_text_length'] += len(page_text)
                            
                            # Guardar muestra de texto de las primeras páginas
                            if len(result['text_samples']) < 3:
                                result['text_samples'].append({
                                    'page': page_num + 1,
                                    'text_length': len(page_text),
                                    'text_preview': page_text[:200].replace('\n', ' ').strip()
                                })
                        else:
                            result['pages_without_text'] += 1
                            
                    except Exception as page_error:
                        logger.warning(f"⚠️ Error en página {page_num + 1}: {page_error}")
                        result['pages_without_text'] += 1
                
                doc.close()
                
                # Diagnosticar el problema
                if result['total_text_length'] == 0:
                    result['diagnosis'] = "PDF_SIN_TEXTO_EXTRAIBLE"
                    result['recommendation'] = "PDF probablemente escaneado (imagen). Necesita OCR."
                elif result['total_text_length'] < 50:
                    result['diagnosis'] = "PDF_CON_POCO_TEXTO"
                    result['recommendation'] = "PDF con muy poco texto. Verificar si es legible."
                elif result['pages_without_text'] > result['pages_with_text']:
                    result['diagnosis'] = "PDF_MAYORIA_PAGINAS_SIN_TEXTO"
                    result['recommendation'] = "PDF con mayoría de páginas sin texto extraíble. Posible PDF escaneado."
                else:
                    result['diagnosis'] = "PDF_CON_TEXTO_NORMAL"
                    result['recommendation'] = "PDF normal con texto. Revisar configuración de procesamiento."
                
            except Exception as doc_error:
                result['error'] = str(doc_error)
                result['diagnosis'] = "PDF_CORRUPTO_O_PROTEGIDO"
                result['recommendation'] = "PDF corrupto o protegido. Verificar integridad del archivo."
                
        except Exception as e:
            result['error'] = str(e)
            result['diagnosis'] = "ERROR_GENERAL"
            result['recommendation'] = "Error inesperado. Revisar logs detallados."
        
        return result
    
    def analyze_all_failed_files(self, siem_directory: str) -> List[Dict[str, Any]]:
        """Analizar todos los archivos que fallaron según el log"""
        
        # Lista de archivos que fallaron según el log
        failed_files = [
            "DOF_1703_22_Circunscripción_Aduanas_mod.pdf",
            "DOF_190122_Vehiculos_procedencia_extr.pdf", 
            "DOF_181217_Reglas_Comercio_Exterior_2018.pdf",
            "DOF_240619_Reglas_Comercio_Exterior_2019.pdf",
            "DOF_110621_Reglas_Comercio_Exterior_2021.pdf",
            "DOF_140721_ANAM.pdf",
            "DOF_270222_Vehiculos_procedencia_extr_ref.pdf",
            "DOF_241221_Reglas_Comercio_Exterior_2022.pdf",
            "Poblalines_ANAM.pdf",
            "DOF_220224_Codigo_de_conducta_ANAM.pdf",
            "DOF_300620_Reglas_Comercio_Exterior_2020.pdf",
            "DOF_210122_Vehiculos_procedencia_extr.pdf",
            "DOF_211221_Reglamento_ANAM.pdf",
            "DOF_010322_Circunscripción_Aduanas.pdf"
        ]
        
        siem_path = Path(siem_directory)
        results = []
        
        logger.info(f"🔍 Analizando {len(failed_files)} archivos que fallaron...")
        
        for file_name in failed_files:
            # Buscar el archivo en toda la estructura de SIEM
            file_path = None
            for file_path in siem_path.rglob(file_name):
                break
            
            if file_path:
                result = self.analyze_pdf(file_path)
                results.append(result)
            else:
                logger.warning(f"⚠️ Archivo no encontrado: {file_name}")
                results.append({
                    'file_name': file_name,
                    'file_path': 'NO_ENCONTRADO',
                    'file_size': 0,
                    'exists': False,
                    'is_readable': False,
                    'page_count': 0,
                    'total_text_length': 0,
                    'pages_with_text': 0,
                    'pages_without_text': 0,
                    'text_samples': [],
                    'error': 'Archivo no encontrado en el sistema de archivos',
                    'diagnosis': 'ARCHIVO_NO_ENCONTRADO',
                    'recommendation': 'Verificar que el archivo existe en la carpeta SIEM'
                })
        
        return results
    
    def generate_report(self, results: List[Dict[str, Any]]) -> str:
        """Generar reporte detallado de los resultados"""
        
        # Estadísticas generales
        total_files = len(results)
        files_found = sum(1 for r in results if r['exists'])
        files_readable = sum(1 for r in results if r['is_readable'])
        files_with_text = sum(1 for r in results if r['total_text_length'] > 0)
        
        # Agrupar por diagnóstico
        diagnosis_groups = {}
        for result in results:
            diagnosis = result['diagnosis']
            if diagnosis not in diagnosis_groups:
                diagnosis_groups[diagnosis] = []
            diagnosis_groups[diagnosis].append(result)
        
        report = f"""
# REPORTE DE DIAGNÓSTICO - ARCHIVOS FALLIDOS
# Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## RESUMEN GENERAL
- Total de archivos analizados: {total_files}
- Archivos encontrados: {files_found}
- Archivos legibles: {files_readable}
- Archivos con texto extraíble: {files_with_text}
- Tasa de éxito: {(files_with_text/total_files)*100:.1f}%

## DIAGNÓSTICOS POR CATEGORÍA
"""
        
        for diagnosis, files in diagnosis_groups.items():
            report += f"\n### {diagnosis} ({len(files)} archivos)\n"
            for file_info in files:
                report += f"- **{file_info['file_name']}**\n"
                report += f"  - Tamaño: {file_info['file_size']:,} bytes\n"
                report += f"  - Páginas: {file_info['page_count']}\n"
                report += f"  - Texto extraído: {file_info['total_text_length']} caracteres\n"
                report += f"  - Recomendación: {file_info['recommendation']}\n"
                if file_info['error']:
                    report += f"  - Error: {file_info['error']}\n"
                report += "\n"
        
        # Detalles específicos de archivos con texto
        report += "\n## ARCHIVOS CON TEXTO EXTRAÍBLE\n"
        for result in results:
            if result['total_text_length'] > 0:
                report += f"\n### {result['file_name']}\n"
                report += f"- Páginas con texto: {result['pages_with_text']}/{result['page_count']}\n"
                report += f"- Total de caracteres: {result['total_text_length']:,}\n"
                report += f"- Muestras de texto:\n"
                for sample in result['text_samples']:
                    report += f"  - Página {sample['page']}: {sample['text_preview']}...\n"
        
        return report

def main():
    """Función principal"""
    try:
        logger.info("🚀 Iniciando diagnóstico de archivos fallidos...")
        
        # Crear instancia del diagnosticador
        diagnostic = PDFDiagnostic()
        
        # Analizar archivos fallidos
        results = diagnostic.analyze_all_failed_files("SIEM")
        
        # Generar reporte
        report = diagnostic.generate_report(results)
        
        # Guardar reporte
        report_file = f"diagnostic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Guardar resultados detallados en JSON
        json_file = f"diagnostic_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Mostrar resumen en consola
        print("\n" + "="*80)
        print("RESUMEN DEL DIAGNÓSTICO")
        print("="*80)
        print(report)
        print("="*80)
        print(f"📄 Reporte detallado guardado en: {report_file}")
        print(f"📊 Resultados JSON guardados en: {json_file}")
        
        # Estadísticas rápidas
        total_files = len(results)
        files_with_text = sum(1 for r in results if r['total_text_length'] > 0)
        pdfs_scanned = sum(1 for r in results if r['diagnosis'] == 'PDF_SIN_TEXTO_EXTRAIBLE')
        
        print(f"\n📈 ESTADÍSTICAS RÁPIDAS:")
        print(f"   - Archivos analizados: {total_files}")
        print(f"   - Con texto extraíble: {files_with_text}")
        print(f"   - PDFs escaneados (sin texto): {pdfs_scanned}")
        print(f"   - Tasa de éxito: {(files_with_text/total_files)*100:.1f}%")
        
        logger.info("✅ Diagnóstico completado")
        
    except Exception as e:
        logger.error(f"❌ Error en diagnóstico: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
