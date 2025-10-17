#!/usr/bin/env python3
"""
Sistema de búsqueda híbrida para documentos SIEM
Combina búsqueda vectorial, textual y por metadatos
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
import json

# Configuración local
from config_supabase import (
    get_supabase_url, get_supabase_key, get_database_tables,
    get_embedding_config
)

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Resultado de búsqueda híbrida"""
    chunk_id: int
    chunk_text: str
    similarity_score: float
    text_rank: float
    combined_score: float
    document_id: int
    file_name: str
    category: Optional[str]
    legal_type: Optional[str]
    metadata: Dict[str, Any]

@dataclass
class SearchFilters:
    """Filtros para búsqueda"""
    category: Optional[str] = None
    legal_type: Optional[str] = None
    keywords: Optional[List[str]] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    min_similarity: float = 0.7
    max_results: int = 10

class HybridSearchSystem:
    """Sistema de búsqueda híbrida"""
    
    def __init__(self):
        self.supabase = self._init_supabase()
        self.tables = get_database_tables()
        self.embedding_config = get_embedding_config()
        self.embedding_model = None
        self._load_embedding_model()
    
    def _init_supabase(self) -> Client:
        """Inicializa Supabase"""
        try:
            url = get_supabase_url()
            key = get_supabase_key()
            supabase = create_client(url, key)
            logger.info("Conexión con Supabase establecida")
            return supabase
        except Exception as e:
            logger.error(f"Error conectando con Supabase: {e}")
            raise
    
    def _load_embedding_model(self):
        """Carga el modelo de embeddings para consultas"""
        try:
            model_name = self.embedding_config['model_name']
            logger.info(f"Cargando modelo para búsqueda: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
            logger.info("Modelo de embeddings cargado")
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            raise
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Genera embedding para la consulta usando Qwen3"""
        try:
            # Usar el cliente Qwen3 para generar embedding de consulta
            from qwen3_embedding_client import Qwen3EmbeddingClient, Qwen3Config
            
            config = Qwen3Config(embedding_dimensions=1024)
            client = Qwen3EmbeddingClient(config)
            
            # Generar embedding de forma síncrona
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                embedding = loop.run_until_complete(client.generate_single_embedding(query))
                return embedding
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Error generando embedding de consulta con Qwen3: {e}")
            raise
    
    async def search_vectorial(self, query: str, filters: SearchFilters) -> List[SearchResult]:
        """Búsqueda vectorial por similitud semántica"""
        try:
            # Generar embedding de la consulta
            query_embedding = self.generate_query_embedding(query)
            
            # Llamar a la función RPC de Supabase
            result = self.supabase.rpc(
                'search_embeddings_cosine',
                {
                    'query_embedding': query_embedding,
                    'match_threshold': filters.min_similarity,
                    'match_count': filters.max_results
                }
            ).execute()
            
            # Convertir resultados
            results = []
            for row in result.data:
                results.append(SearchResult(
                    chunk_id=row['chunk_id'],
                    chunk_text=row['chunk_text'],
                    similarity_score=row['similarity'],
                    text_rank=0.0,  # No aplicable en búsqueda vectorial pura
                    combined_score=row['similarity'],
                    document_id=row['document_id'],
                    file_name=row['file_name'],
                    category=None,  # Se puede obtener con join adicional
                    legal_type=None,
                    metadata={}
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error en búsqueda vectorial: {e}")
            return []
    
    async def search_textual(self, query: str, filters: SearchFilters) -> List[SearchResult]:
        """Búsqueda textual por palabras clave"""
        try:
            # Construir consulta SQL con full-text search
            query_sql = f"""
            SELECT 
                c.id as chunk_id,
                c.chunk_text,
                ts_rank(to_tsvector('spanish', c.chunk_text), plainto_tsquery('spanish', '{query}')) as text_rank,
                c.document_id,
                d.file_name,
                m.category,
                m.legal_type
            FROM {self.tables['chunks']} c
            JOIN {self.tables['documents']} d ON c.document_id = d.id
            LEFT JOIN {self.tables['metadata']} m ON d.id = m.document_id
            WHERE to_tsvector('spanish', c.chunk_text) @@ plainto_tsquery('spanish', '{query}')
            """
            
            # Aplicar filtros
            if filters.category:
                query_sql += f" AND m.category = '{filters.category}'"
            if filters.legal_type:
                query_sql += f" AND m.legal_type = '{filters.legal_type}'"
            
            query_sql += f" ORDER BY text_rank DESC LIMIT {filters.max_results}"
            
            result = self.supabase.rpc('exec_sql', {'sql': query_sql}).execute()
            
            # Convertir resultados
            results = []
            for row in result.data:
                results.append(SearchResult(
                    chunk_id=row['chunk_id'],
                    chunk_text=row['chunk_text'],
                    similarity_score=0.0,  # No aplicable en búsqueda textual pura
                    text_rank=row['text_rank'],
                    combined_score=row['text_rank'],
                    document_id=row['document_id'],
                    file_name=row['file_name'],
                    category=row.get('category'),
                    legal_type=row.get('legal_type'),
                    metadata={}
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error en búsqueda textual: {e}")
            return []
    
    async def search_hybrid(self, query: str, filters: SearchFilters) -> List[SearchResult]:
        """Búsqueda híbrida combinando vectorial y textual"""
        try:
            # Generar embedding de la consulta
            query_embedding = self.generate_query_embedding(query)
            
            # Llamar a la función RPC híbrida de Supabase
            result = self.supabase.rpc(
                'search_hybrid',
                {
                    'query_embedding': query_embedding,
                    'query_text': query,
                    'match_threshold': filters.min_similarity,
                    'match_count': filters.max_results
                }
            ).execute()
            
            # Convertir resultados
            results = []
            for row in result.data:
                results.append(SearchResult(
                    chunk_id=row['chunk_id'],
                    chunk_text=row['chunk_text'],
                    similarity_score=row['similarity'],
                    text_rank=row['text_rank'],
                    combined_score=row['combined_score'],
                    document_id=row['document_id'],
                    file_name=row['file_name'],
                    category=row.get('category'),
                    legal_type=row.get('legal_type'),
                    metadata={}
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error en búsqueda híbrida: {e}")
            return []
    
    async def search_by_metadata(self, filters: SearchFilters) -> List[SearchResult]:
        """Búsqueda por metadatos"""
        try:
            # Llamar a la función RPC de metadatos
            result = self.supabase.rpc(
                'search_by_metadata',
                {
                    'category_filter': filters.category,
                    'legal_type_filter': filters.legal_type,
                    'keywords_filter': filters.keywords,
                    'date_from': filters.date_from.isoformat() if filters.date_from else None,
                    'date_to': filters.date_to.isoformat() if filters.date_to else None
                }
            ).execute()
            
            # Convertir resultados
            results = []
            for row in result.data:
                results.append(SearchResult(
                    chunk_id=0,  # No aplicable en búsqueda por metadatos
                    chunk_text="",  # Se puede obtener con join adicional
                    similarity_score=0.0,
                    text_rank=0.0,
                    combined_score=1.0,  # Score alto para resultados de metadatos
                    document_id=row['document_id'],
                    file_name=row['file_name'],
                    category=row.get('category'),
                    legal_type=row.get('legal_type'),
                    metadata={
                        'keywords': row.get('keywords', []),
                        'created_at': row.get('created_at')
                    }
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error en búsqueda por metadatos: {e}")
            return []
    
    async def search_combined(self, query: str, filters: SearchFilters) -> List[SearchResult]:
        """Búsqueda combinada que integra todos los métodos"""
        try:
            # Ejecutar búsquedas en paralelo
            vectorial_task = asyncio.create_task(self.search_vectorial(query, filters))
            textual_task = asyncio.create_task(self.search_textual(query, filters))
            hybrid_task = asyncio.create_task(self.search_hybrid(query, filters))
            
            # Esperar resultados
            vectorial_results = await vectorial_task
            textual_results = await textual_task
            hybrid_results = await hybrid_task
            
            # Combinar y rankear resultados
            all_results = self._combine_search_results(
                vectorial_results, textual_results, hybrid_results
            )
            
            # Aplicar filtros adicionales
            filtered_results = self._apply_filters(all_results, filters)
            
            # Ordenar por score combinado
            filtered_results.sort(key=lambda x: x.combined_score, reverse=True)
            
            return filtered_results[:filters.max_results]
            
        except Exception as e:
            logger.error(f"Error en búsqueda combinada: {e}")
            return []
    
    def _combine_search_results(self, vectorial: List[SearchResult], 
                              textual: List[SearchResult], 
                              hybrid: List[SearchResult]) -> List[SearchResult]:
        """Combina resultados de diferentes métodos de búsqueda"""
        # Crear diccionario para evitar duplicados
        results_dict = {}
        
        # Agregar resultados híbridos (tienen mejor score)
        for result in hybrid:
            key = result.chunk_id
            if key not in results_dict or result.combined_score > results_dict[key].combined_score:
                results_dict[key] = result
        
        # Agregar resultados vectoriales si no existen
        for result in vectorial:
            key = result.chunk_id
            if key not in results_dict:
                results_dict[key] = result
        
        # Agregar resultados textuales si no existen
        for result in textual:
            key = result.chunk_id
            if key not in results_dict:
                results_dict[key] = result
        
        return list(results_dict.values())
    
    def _apply_filters(self, results: List[SearchResult], filters: SearchFilters) -> List[SearchResult]:
        """Aplica filtros adicionales a los resultados"""
        filtered = []
        
        for result in results:
            # Filtro por categoría
            if filters.category and result.category != filters.category:
                continue
            
            # Filtro por tipo legal
            if filters.legal_type and result.legal_type != filters.legal_type:
                continue
            
            # Filtro por palabras clave
            if filters.keywords:
                text_lower = result.chunk_text.lower()
                if not any(keyword.lower() in text_lower for keyword in filters.keywords):
                    continue
            
            # Filtro por similitud mínima
            if result.similarity_score < filters.min_similarity:
                continue
            
            filtered.append(result)
        
        return filtered
    
    def get_search_suggestions(self, query: str, limit: int = 5) -> List[str]:
        """Obtiene sugerencias de búsqueda basadas en el contenido"""
        try:
            # Buscar términos similares en los metadatos
            result = self.supabase.table(self.tables['metadata']).select('keywords').limit(100).execute()
            
            # Extraer todas las palabras clave
            all_keywords = []
            for row in result.data:
                if row.get('keywords'):
                    all_keywords.extend(row['keywords'])
            
            # Encontrar palabras similares a la consulta
            query_words = query.lower().split()
            suggestions = []
            
            for keyword in set(all_keywords):
                keyword_lower = keyword.lower()
                for word in query_words:
                    if word in keyword_lower or keyword_lower in word:
                        suggestions.append(keyword)
                        break
            
            return suggestions[:limit]
            
        except Exception as e:
            logger.error(f"Error obteniendo sugerencias: {e}")
            return []
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de búsqueda"""
        try:
            # Obtener estadísticas de la base de datos
            result = self.supabase.table('processing_stats').select('*').execute()
            
            if result.data:
                stats = result.data[0]
                return {
                    'total_documents': stats.get('total_documents', 0),
                    'completed_documents': stats.get('completed_documents', 0),
                    'total_chunks': stats.get('total_chunks', 0),
                    'total_tokens': stats.get('total_tokens', 0),
                    'avg_processing_time': stats.get('avg_processing_time_seconds', 0)
                }
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {}

class SearchAPI:
    """API para búsqueda híbrida"""
    
    def __init__(self):
        self.search_system = HybridSearchSystem()
    
    async def search(self, query: str, search_type: str = "combined", 
                    filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """API principal de búsqueda"""
        try:
            # Crear filtros
            search_filters = SearchFilters()
            if filters:
                search_filters.category = filters.get('category')
                search_filters.legal_type = filters.get('legal_type')
                search_filters.keywords = filters.get('keywords')
                search_filters.min_similarity = filters.get('min_similarity', 0.7)
                search_filters.max_results = filters.get('max_results', 10)
            
            # Ejecutar búsqueda según el tipo
            if search_type == "vectorial":
                results = await self.search_system.search_vectorial(query, search_filters)
            elif search_type == "textual":
                results = await self.search_system.search_textual(query, search_filters)
            elif search_type == "hybrid":
                results = await self.search_system.search_hybrid(query, search_filters)
            elif search_type == "metadata":
                results = await self.search_system.search_by_metadata(search_filters)
            else:  # combined
                results = await self.search_system.search_combined(query, search_filters)
            
            # Convertir resultados a diccionario
            results_dict = []
            for result in results:
                results_dict.append({
                    'chunk_id': result.chunk_id,
                    'chunk_text': result.chunk_text,
                    'similarity_score': result.similarity_score,
                    'text_rank': result.text_rank,
                    'combined_score': result.combined_score,
                    'document_id': result.document_id,
                    'file_name': result.file_name,
                    'category': result.category,
                    'legal_type': result.legal_type,
                    'metadata': result.metadata
                })
            
            return {
                'query': query,
                'search_type': search_type,
                'total_results': len(results_dict),
                'results': results_dict,
                'filters_applied': filters
            }
            
        except Exception as e:
            logger.error(f"Error en API de búsqueda: {e}")
            return {
                'error': str(e),
                'query': query,
                'search_type': search_type,
                'total_results': 0,
                'results': []
            }
    
    async def get_suggestions(self, query: str) -> List[str]:
        """Obtiene sugerencias de búsqueda"""
        return self.search_system.get_search_suggestions(query)
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema"""
        return self.search_system.get_search_statistics()

# Ejemplo de uso
async def main():
    """Ejemplo de uso del sistema de búsqueda"""
    api = SearchAPI()
    
    # Búsqueda combinada
    results = await api.search(
        query="comercio exterior aduanas",
        search_type="combined",
        filters={
            'category': 'ANAM',
            'min_similarity': 0.6,
            'max_results': 5
        }
    )
    
    print(f"Búsqueda: {results['query']}")
    print(f"Resultados encontrados: {results['total_results']}")
    
    for i, result in enumerate(results['results'], 1):
        print(f"\n{i}. {result['file_name']}")
        print(f"   Score: {result['combined_score']:.3f}")
        print(f"   Texto: {result['chunk_text'][:200]}...")

if __name__ == "__main__":
    asyncio.run(main())
