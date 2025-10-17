#!/usr/bin/env python3
"""
Script de debug para verificar la autenticación con Supabase
"""

import asyncio
import aiohttp
import json

async def debug_supabase_auth():
    """Debug de autenticación con Supabase"""
    
    supabase_url = "https://rygrdlradxyykzuudgtu.supabase.co"
    supabase_key = "eyJhbGciOiJIUzI1NiIsInRcCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJ5Z3JkbHJhZHh5eWt6dXVkZ3R1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA3MTg0MDYsImV4cCI6MjA3NjI5NDQwNn0.ZeFX5rK3kfqupVX_yHKluIsCB1beCfCLsS1fdqEU_10"
    
    print("🔍 Debug de autenticación con Supabase")
    print(f"URL: {supabase_url}")
    print(f"Key: {supabase_key[:20]}...")
    
    # Probar diferentes configuraciones de headers
    headers_configs = [
        {
            "name": "Configuración 1: Solo apikey",
            "headers": {
                "apikey": supabase_key,
                "Content-Type": "application/json"
            }
        },
        {
            "name": "Configuración 2: Solo Authorization",
            "headers": {
                "Authorization": f"Bearer {supabase_key}",
                "Content-Type": "application/json"
            }
        },
        {
            "name": "Configuración 3: Ambos (como en el test)",
            "headers": {
                "apikey": supabase_key,
                "Authorization": f"Bearer {supabase_key}",
                "Content-Type": "application/json"
            }
        },
        {
            "name": "Configuración 4: Con Accept",
            "headers": {
                "apikey": supabase_key,
                "Authorization": f"Bearer {supabase_key}",
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
        }
    ]
    
    async with aiohttp.ClientSession() as session:
        for config in headers_configs:
            print(f"\n🧪 Probando {config['name']}...")
            print(f"Headers: {config['headers']}")
            
            try:
                async with session.get(f"{supabase_url}/rest/v1/siem_documents?select=id&limit=1", 
                                     headers=config['headers']) as response:
                    print(f"Status: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ ÉXITO: {data}")
                        return config['headers']  # Devolver la configuración que funciona
                    else:
                        text = await response.text()
                        print(f"❌ Error: {text}")
                        
            except Exception as e:
                print(f"❌ Excepción: {e}")
    
    print("\n❌ Ninguna configuración funcionó")
    return None

if __name__ == "__main__":
    result = asyncio.run(debug_supabase_auth())
    if result:
        print(f"\n🎉 Configuración que funciona: {result}")
    else:
        print("\n💥 Todas las configuraciones fallaron")
