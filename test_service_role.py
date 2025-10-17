#!/usr/bin/env python3
"""
Script de prueba para verificar la service_role key
"""

import asyncio
import aiohttp
import json

async def test_service_role():
    """Probar la service_role key"""
    
    supabase_url = "https://rygrdlradxyykzuudgtu.supabase.co"
    service_role_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJ5Z3JkbHJhZHh5eWt6dXVkZ3R1Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MDcxODQwNiwiZXhwIjoyMDc2Mjk0NDA2fQ.jbnFbBjlq_NsLaNamGJ98a4uXay4lAZX_BVWQCobpCw"
    
    headers = {
        "apikey": service_role_key,
        "Authorization": f"Bearer {service_role_key}",
        "Content-Type": "application/json"
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            # Probar conexión básica
            print("🔍 Probando conexión con service_role key...")
            async with session.get(f"{supabase_url}/rest/v1/siem_documents?select=id&limit=1", 
                                 headers=headers) as response:
                print(f"Status: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Conexión exitosa: {data}")
                    
                    # Probar inserción
                    print("\n🔍 Probando inserción...")
                    test_data = {
                        "file_name": "test_service_role.pdf",
                        "file_path": "/test/service_role.pdf",
                        "file_hash": "test_service_role_123",
                        "file_size": 2048,
                        "file_extension": "pdf",
                        "category": "test",
                        "comercio_type": "test"
                    }
                    
                    async with session.post(f"{supabase_url}/rest/v1/siem_documents", 
                                          headers=headers, 
                                          json=test_data) as post_response:
                        print(f"Status: {post_response.status}")
                        if post_response.status == 201:
                            print("✅ Inserción exitosa")
                            
                            # Limpiar datos de prueba
                            async with session.delete(f"{supabase_url}/rest/v1/siem_documents?file_hash=eq.test_service_role_123", 
                                                    headers=headers) as delete_response:
                                print(f"✅ Datos de prueba eliminados: {delete_response.status}")
                        else:
                            text = await post_response.text()
                            print(f"❌ Error en inserción: {text}")
                    
                    return True
                else:
                    text = await response.text()
                    print(f"❌ Error: {text}")
                    return False
                    
        except Exception as e:
            print(f"❌ Error inesperado: {e}")
            return False

if __name__ == "__main__":
    result = asyncio.run(test_service_role())
    if result:
        print("\n🎉 ¡Service role key funciona correctamente!")
    else:
        print("\n💥 Service role key no funciona")
