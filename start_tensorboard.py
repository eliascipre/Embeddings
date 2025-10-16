#!/usr/bin/env python3
"""
Script para iniciar TensorBoard y monitorear el progreso
"""
import subprocess
import sys
from pathlib import Path

def start_tensorboard():
    """Iniciar TensorBoard"""
    try:
        # Directorio de logs de TensorBoard
        log_dir = Path(__file__).parent / "logs" / "tensorboard"
        
        if not log_dir.exists():
            print(f"❌ Directorio de TensorBoard no encontrado: {log_dir}")
            print("💡 Ejecuta primero el procesamiento para generar logs")
            return
        
        print(f"🚀 Iniciando TensorBoard en: {log_dir}")
        print("📊 Abre tu navegador en: http://localhost:6006")
        print("⏹️ Presiona Ctrl+C para detener")
        
        # Iniciar TensorBoard
        subprocess.run([
            sys.executable, "-m", "tensorboard.main",
            "--logdir", str(log_dir),
            "--port", "6006",
            "--host", "0.0.0.0"
        ])
        
    except KeyboardInterrupt:
        print("\n⏹️ TensorBoard detenido")
    except Exception as e:
        print(f"❌ Error iniciando TensorBoard: {e}")

if __name__ == "__main__":
    start_tensorboard()
