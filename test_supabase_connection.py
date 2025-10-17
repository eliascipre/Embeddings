#!/usr/bin/env python3
"""
Script de prueba para verificar la conexión con Supabase
"""

import asyncio
import aiohttp
import json

async def test_supabase_connection():
    """Probar la conexión con Supabase"""
    
    supabase_url = "https://rygrdlradxyykzuudgtu.supabase.co"
    supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJ5Z3JkbHJhZHh5eWt6dXVkZ3R1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA3MTg0MDYsImV4cCI6MjA3NjI5NDQwNn0.ZeFX5rK3kfqupVX_yHKluIsCB1beCfCLsS1fdqEU_10"
    
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "application/json"
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            # Probar conexión básica
            print("🔍 Probando conexión básica...")
            async with session.get(f"{supabase_url}/rest/v1/", headers=headers) as response:
                print(f"Status: {response.status}")
                if response.status == 200:
                    print("✅ Conexión básica exitosa")
                else:
                    text = await response.text()
                    print(f"❌ Error: {text}")
                    return False
            
            # Probar acceso a tabla siem_documents
            print("\n🔍 Probando acceso a tabla siem_documents...")
            async with session.get(f"{supabase_url}/rest/v1/siem_documents?select=id&limit=1", headers=headers) as response:
                print(f"Status: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Acceso a tabla exitoso: {data}")
                else:
                    text = await response.text()
                    print(f"❌ Error accediendo a tabla: {text}")
                    return False
            
            # Probar inserción de prueba
            print("\n🔍 Probando inserción de prueba...")
            test_data = {
                "file_name": "test_connection.pdf",
                "file_path": "/test/connection.pdf",
                "file_hash": "test_hash_123",
                "file_size": 1024,
                "file_extension": "pdf",
                "category": "test",
                "comercio_type": "test"
            }
            
            async with session.post(f"{supabase_url}/rest/v1/siem_documents", 
                                  headers=headers, 
                                  json=test_data) as response:
                print(f"Status: {response.status}")
                if response.status == 201:
                    # Supabase devuelve 201 sin JSON en algunos casos
                    print("✅ Inserción exitosa (201 Created)")
                    
                    # Buscar el documento insertado para obtener el ID
                    async with session.get(f"{supabase_url}/rest/v1/siem_documents?file_hash=eq.test_hash_123&select=id", 
                                         headers=headers) as get_response:
                        if get_response.status == 200:
                            data = await get_response.json()
                            if data:
                                document_id = data[0]['id']
                                # Limpiar datos de prueba
                                async with session.delete(f"{supabase_url}/rest/v1/siem_documents?id=eq.{document_id}", 
                                                        headers=headers) as delete_response:
                                    print(f"✅ Datos de prueba eliminados: {delete_response.status}")
                            else:
                                print("⚠️ No se pudo encontrar el documento insertado")
                        else:
                            print("⚠️ No se pudo verificar la inserción")
                else:
                    text = await response.text()
                    print(f"❌ Error en inserción: {text}")
                    return False
            
            print("\n🎉 ¡Todas las pruebas pasaron! La API key funciona correctamente.")
            return True
            
        except Exception as e:
            print(f"❌ Error inesperado: {e}")
            return False

if __name__ == "__main__":
    asyncio.run(test_supabase_connection())
