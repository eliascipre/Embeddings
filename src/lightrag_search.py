"""
Sistema de búsqueda inteligente con LightRAG
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime
import json

# LightRAG imports
try:
    from lightrag import LightRAG, QueryParam
    from lightrag.llm import gpt_4o_mini_complete, gpt_4o_complete
    LIGHTRAG_AVAILABLE = True
except ImportError:
    LIGHTRAG_AVAILABLE = False
    logging.warning("LightRAG no disponible")

# Embeddings
from sentence_transformers import SentenceTransformer
import torch

from config import config
from supabase_database import SupabaseVectorDatabase

logger = logging.getLogger(__name__)

class LightRAGSearchEngine:
    """Motor de búsqueda inteligente con LightRAG"""
    
    def __init__(self, vector_db: SupabaseVectorDatabase):
        self.vector_db = vector_db
        self.lightrag = None
        self.embeddings_model = None
        self._setup_lightrag()
        self._setup_embeddings()
        
    def _setup_lightrag(self):
        """Configurar LightRAG"""
        if not LIGHTRAG_AVAILABLE:
            logger.warning("⚠️ LightRAG no disponible, usando búsqueda vectorial básica")
            return
            
        try:
            # Configurar LightRAG
            self.lightrag = LightRAG(
                working_dir=str(config.cache_directory / "lightrag"),
                llm_model_func=gpt_4o_mini_complete,  # Usar modelo más económico para búsquedas
                llm_model_name="gpt-4o-mini",
                embedding_dim=config.embedding_dimensions,
                max_token=4000,
                mode="local",  # Modo local para mejor rendimiento
                enable_rag_search_rewrite=True,
                enable_rag_search_hybrid=True,
                enable_rag_search_hybrid_fusion=True
            )
            logger.info("✅ LightRAG configurado correctamente")
            
        except Exception as e:
            logger.error(f"❌ Error configurando LightRAG: {e}")
            self.lightrag = None
    
    def _setup_embeddings(self):
        """Configurar modelo de embeddings"""
        try:
            self.embeddings_model = SentenceTransformer(
                config.embedding_model,
                device=config.embedding_device,
                trust_remote_code=True
            )
            logger.info(f"✅ Modelo de embeddings cargado: {config.embedding_model}")
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo de embeddings: {e}")
            raise
    
    async def add_documents_to_lightrag(self, documents: List[Dict[str, Any]]) -> bool:
        """Agregar documentos a LightRAG para indexación inteligente"""
        if not self.lightrag:
            logger.warning("⚠️ LightRAG no disponible, saltando indexación inteligente")
            return False
            
        try:
            logger.info(f"🔄 Agregando {len(documents)} documentos a LightRAG...")
            
            for doc in documents:
                # Preparar documento para LightRAG
                doc_content = {
                    "content": doc.get("content", ""),
                    "metadata": {
                        "source": doc.get("metadata", {}).get("source", ""),
                        "title": doc.get("metadata", {}).get("title", ""),
                        "state": doc.get("metadata", {}).get("state", ""),
                        "document_type": doc.get("metadata", {}).get("document_type", ""),
                        "chunk_id": doc.get("metadata", {}).get("chunk_id", 0)
                    }
                }
                
                # Agregar a LightRAG
                await self.lightrag.ainsert(
                    content=doc_content["content"],
                    metadata=doc_content["metadata"]
                )
            
            logger.info(f"✅ {len(documents)} documentos agregados a LightRAG")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error agregando documentos a LightRAG: {e}")
            return False
    
    async def search_with_lightrag(
        self, 
        query: str, 
        mode: str = "hybrid",
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Buscar con LightRAG usando diferentes modos"""
        if not self.lightrag:
            logger.warning("⚠️ LightRAG no disponible, usando búsqueda vectorial básica")
            return await self._fallback_vector_search(query, top_k)
        
        try:
            logger.info(f"🔍 Buscando con LightRAG: '{query}' (modo: {mode})")
            
            # Configurar parámetros de búsqueda
            query_param = QueryParam(
                query=query,
                mode=mode,  # "naive", "local", "global", "hybrid"
                top_k=top_k,
                score_threshold=config.similarity_threshold
            )
            
            # Realizar búsqueda
            results = await self.lightrag.aquery(query_param)
            
            # Procesar resultados
            processed_results = []
            for result in results:
                processed_result = {
                    "content": result.get("content", ""),
                    "metadata": result.get("metadata", {}),
                    "score": result.get("score", 0.0),
                    "source": "lightrag",
                    "search_mode": mode
                }
                processed_results.append(processed_result)
            
            logger.info(f"✅ LightRAG encontró {len(processed_results)} resultados")
            return processed_results
            
        except Exception as e:
            logger.error(f"❌ Error buscando con LightRAG: {e}")
            return await self._fallback_vector_search(query, top_k)
    
    async def _fallback_vector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Búsqueda vectorial de fallback cuando LightRAG no está disponible"""
        try:
            logger.info(f"🔄 Usando búsqueda vectorial de fallback para: '{query}'")
            
            # Generar embedding de la consulta
            query_embedding = self.embeddings_model.encode(
                [query], 
                normalize_embeddings=True,
                convert_to_tensor=False
            )[0].tolist()
            
            # Buscar en la base de datos vectorial
            results = await self.vector_db.search_similar_chunks(
                query_embedding=query_embedding,
                limit=top_k,
                similarity_threshold=config.similarity_threshold
            )
            
            # Procesar resultados
            processed_results = []
            for result in results:
                processed_result = {
                    "content": result.get("content", ""),
                    "metadata": result.get("metadata", {}),
                    "score": result.get("similarity", 0.0),
                    "source": "vector_search",
                    "search_mode": "cosine_similarity"
                }
                processed_results.append(processed_result)
            
            logger.info(f"✅ Búsqueda vectorial encontró {len(processed_results)} resultados")
            return processed_results
            
        except Exception as e:
            logger.error(f"❌ Error en búsqueda vectorial de fallback: {e}")
            return []
    
    async def hybrid_search(
        self, 
        query: str, 
        top_k: int = 10,
        lightrag_weight: float = 0.7,
        vector_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Búsqueda híbrida combinando LightRAG y búsqueda vectorial"""
        try:
            logger.info(f"🔍 Búsqueda híbrida para: '{query}'")
            
            # Búsqueda con LightRAG
            lightrag_results = []
            if self.lightrag:
                lightrag_results = await self.search_with_lightrag(
                    query, 
                    mode="hybrid", 
                    top_k=top_k
                )
            
            # Búsqueda vectorial
            vector_results = await self._fallback_vector_search(query, top_k)
            
            # Combinar resultados
            combined_results = self._combine_search_results(
                lightrag_results, 
                vector_results, 
                lightrag_weight, 
                vector_weight
            )
            
            # Ordenar por score combinado
            combined_results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
            
            logger.info(f"✅ Búsqueda híbrida completada: {len(combined_results)} resultados")
            return combined_results[:top_k]
            
        except Exception as e:
            logger.error(f"❌ Error en búsqueda híbrida: {e}")
            return await self._fallback_vector_search(query, top_k)
    
    def _combine_search_results(
        self, 
        lightrag_results: List[Dict], 
        vector_results: List[Dict],
        lightrag_weight: float,
        vector_weight: float
    ) -> List[Dict[str, Any]]:
        """Combinar resultados de LightRAG y búsqueda vectorial"""
        combined = {}
        
        # Procesar resultados de LightRAG
        for result in lightrag_results:
            content = result.get("content", "")
            if content in combined:
                combined[content]["lightrag_score"] = result.get("score", 0)
            else:
                combined[content] = {
                    "content": content,
                    "metadata": result.get("metadata", {}),
                    "lightrag_score": result.get("score", 0),
                    "vector_score": 0,
                    "source": "lightrag"
                }
        
        # Procesar resultados vectoriales
        for result in vector_results:
            content = result.get("content", "")
            if content in combined:
                combined[content]["vector_score"] = result.get("score", 0)
            else:
                combined[content] = {
                    "content": content,
                    "metadata": result.get("metadata", {}),
                    "lightrag_score": 0,
                    "vector_score": result.get("score", 0),
                    "source": "vector"
                }
        
        # Calcular score combinado
        for content, result in combined.items():
            combined_score = (
                result["lightrag_score"] * lightrag_weight + 
                result["vector_score"] * vector_weight
            )
            result["combined_score"] = combined_score
        
        return list(combined.values())
    
    async def get_lightrag_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de LightRAG"""
        if not self.lightrag:
            return {"available": False, "error": "LightRAG no disponible"}
        
        try:
            # Obtener estadísticas básicas
            stats = {
                "available": True,
                "working_dir": str(config.cache_directory / "lightrag"),
                "llm_model": "gpt-4o-mini",
                "embedding_dimensions": config.embedding_dimensions,
                "modes_available": ["naive", "local", "global", "hybrid"],
                "features": {
                    "rag_search_rewrite": True,
                    "rag_search_hybrid": True,
                    "rag_search_hybrid_fusion": True
                }
            }
            
            return stats
            
        except Exception as e:
            return {"available": False, "error": str(e)}
    
    def get_search_capabilities(self) -> Dict[str, Any]:
        """Obtener capacidades de búsqueda disponibles"""
        return {
            "lightrag_available": LIGHTRAG_AVAILABLE and self.lightrag is not None,
            "vector_search_available": True,
            "hybrid_search_available": LIGHTRAG_AVAILABLE and self.lightrag is not None,
            "search_modes": {
                "lightrag": ["naive", "local", "global", "hybrid"] if self.lightrag else [],
                "vector": ["cosine_similarity"],
                "hybrid": ["lightrag_vector_combination"]
            },
            "embedding_model": config.embedding_model,
            "similarity_threshold": config.similarity_threshold
        }
