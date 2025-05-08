"""Vector Search Service implementation.

This service is responsible for vector embedding and retrieval operations.
It provides a unified interface for vector operations regardless of the 
underlying vector database (Qdrant, FAISS, etc.)
"""
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import httpx
import numpy as np
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from microservices.service_interfaces import (
    ServiceRequest, 
    ServiceResponse,
    VectorSearchServiceInterface
)

# Configure logging
logger = logging.getLogger(__name__)


class VectorSearchRequest(ServiceRequest):
    """Vector search request model."""
    query_vector: List[float]
    collection: str
    k: int = 10
    filter_obj: Optional[Dict[str, Any]] = None


class EmbeddingRequest(ServiceRequest):
    """Embedding request model."""
    text: str
    model: Optional[str] = None


class VectorSearchResponse(ServiceResponse):
    """Vector search response model."""
    results: Optional[List[Dict[str, Any]]] = None


class VectorSearchService(VectorSearchServiceInterface):
    """Implementation of the Vector Search Service."""
    
    def __init__(
        self,
        default_embedding_model: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        cache_service_url: Optional[str] = None,
        use_faiss: bool = False,
        faiss_index_path: Optional[str] = None
    ):
        """Initialize the vector search service.
        
        Args:
            default_embedding_model: Name of the default embedding model to use
            qdrant_url: URL for Qdrant vector database (if using Qdrant)
            qdrant_api_key: API key for Qdrant (if using Qdrant)
            cache_service_url: Optional URL for the Cache Service
            use_faiss: Whether to use FAISS for vector search
            faiss_index_path: Path to the FAISS index file (if using FAISS)
        """
        self.default_embedding_model = default_embedding_model
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.cache_service_url = cache_service_url
        self.use_faiss = use_faiss
        self.faiss_index_path = faiss_index_path
        
        # Load embedding models
        self.embedding_models = {}
        self._load_default_embedding_model()
        
        # Initialize FAISS if needed
        if use_faiss:
            self._initialize_faiss()
            
        # Initialize httpx client
        self.client = httpx.AsyncClient(timeout=30.0)
        self.is_running = True
        
        logger.info(f"Vector Search Service initialized with model: {default_embedding_model}")
    
    def _load_default_embedding_model(self):
        """Load the default embedding model."""
        try:
            logger.info(f"Loading embedding model: {self.default_embedding_model}")
            self.embedding_models[self.default_embedding_model] = SentenceTransformer(self.default_embedding_model)
            logger.info(f"Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
    
    def _initialize_faiss(self):
        """Initialize FAISS index if using FAISS."""
        if self.use_faiss:
            try:
                import faiss
                logger.info(f"Loading FAISS index from: {self.faiss_index_path}")
                self.faiss_index = faiss.read_index(self.faiss_index_path)
                logger.info(f"FAISS index loaded successfully")
            except Exception as e:
                logger.error(f"Error loading FAISS index: {str(e)}")
                raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if the service is healthy and return status information."""
        status = {
            "status": "healthy",
            "embedding_models": list(self.embedding_models.keys()),
            "version": "1.0.0"
        }
        
        # Check Qdrant connection if using Qdrant
        if self.qdrant_url:
            try:
                headers = {}
                if self.qdrant_api_key:
                    headers["api-key"] = self.qdrant_api_key
                
                response = await self.client.get(
                    f"{self.qdrant_url}/collections",
                    headers=headers
                )
                
                status["qdrant_status"] = "connected" if response.status_code == 200 else "error"
                
            except Exception as e:
                status["qdrant_status"] = f"error: {str(e)}"
                status["status"] = "degraded"
        
        # Check cache service if configured
        if self.cache_service_url:
            try:
                response = await self.client.get(f"{self.cache_service_url}/health")
                status["cache_status"] = "connected" if response.status_code == 200 else "error"
            except Exception as e:
                status["cache_status"] = f"error: {str(e)}"
        
        return status
    
    async def process_request(self, request: ServiceRequest) -> ServiceResponse:
        """Process a service request and return a response."""
        if isinstance(request, VectorSearchRequest):
            try:
                results = await self.search(
                    query_vector=request.query_vector,
                    collection=request.collection,
                    k=request.k,
                    filter_obj=request.filter_obj
                )
                
                return VectorSearchResponse(
                    request_id=request.request_id,
                    status="success",
                    data={"results": results}
                )
                
            except Exception as e:
                logger.error(f"Error in vector search: {str(e)}")
                return VectorSearchResponse(
                    request_id=request.request_id,
                    status="error",
                    message=f"Error in vector search: {str(e)}"
                )
                
        elif isinstance(request, EmbeddingRequest):
            try:
                embedding = await self.get_embedding(
                    text=request.text,
                    model=request.model or self.default_embedding_model
                )
                
                return ServiceResponse(
                    request_id=request.request_id,
                    status="success",
                    data={"embedding": embedding}
                )
                
            except Exception as e:
                logger.error(f"Error generating embedding: {str(e)}")
                return ServiceResponse(
                    request_id=request.request_id,
                    status="error",
                    message=f"Error generating embedding: {str(e)}"
                )
                
        else:
            return ServiceResponse(
                request_id=getattr(request, "request_id", "unknown"),
                status="error",
                message="Invalid request type"
            )
    
    async def search(
        self, 
        query_vector: List[float], 
        collection: str, 
        k: int = 10, 
        filter_obj: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform vector search with the provided query vector.
        
        Args:
            query_vector: The query vector to search with
            collection: The collection to search in
            k: The number of results to return
            filter_obj: Optional filter for the search
            
        Returns:
            List of search results
        """
        # Check cache first if cache service is configured
        if self.cache_service_url:
            cache_key = f"vector_search:{collection}:{str(query_vector[:5])}:{k}:{str(filter_obj)}"
            
            try:
                cache_response = await self.client.get(
                    f"{self.cache_service_url}/get",
                    params={"key": cache_key}
                )
                
                if cache_response.status_code == 200:
                    cached_data = cache_response.json().get("data")
                    if cached_data:
                        logger.info(f"Cache hit for vector search")
                        return cached_data
            except Exception as e:
                logger.warning(f"Error checking cache: {str(e)}")
        
        # If using FAISS
        if self.use_faiss:
            return await self._search_faiss(query_vector, k)
            
        # If using Qdrant
        elif self.qdrant_url:
            return await self._search_qdrant(query_vector, collection, k, filter_obj)
            
        # Fallback to simulated vector search
        else:
            return await self._search_simulated(query_vector, collection, k, filter_obj)
    
    async def _search_qdrant(
        self, 
        query_vector: List[float], 
        collection: str, 
        k: int, 
        filter_obj: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Search using Qdrant vector database."""
        headers = {"Content-Type": "application/json"}
        if self.qdrant_api_key:
            headers["api-key"] = self.qdrant_api_key
        
        search_params = {
            "vector": query_vector,
            "limit": k
        }
        
        if filter_obj:
            search_params["filter"] = filter_obj
        
        response = await self.client.post(
            f"{self.qdrant_url}/collections/{collection}/points/search",
            json=search_params,
            headers=headers
        )
        
        if response.status_code != 200:
            raise Exception(f"Qdrant search failed: {response.text}")
        
        results = response.json().get("result", [])
        
        # Format results to standard format
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result.get("id"),
                "score": result.get("score", 0.0),
                "metadata": result.get("payload", {}),
                "vector": result.get("vector", [])
            })
        
        # Cache results if cache service is configured
        if self.cache_service_url:
            cache_key = f"vector_search:{collection}:{str(query_vector[:5])}:{k}:{str(filter_obj)}"
            
            try:
                await self.client.post(
                    f"{self.cache_service_url}/set",
                    json={
                        "key": cache_key,
                        "value": formatted_results,
                        "ttl": 300  # 5 minutes TTL
                    }
                )
            except Exception as e:
                logger.warning(f"Error caching results: {str(e)}")
        
        return formatted_results
    
    async def _search_faiss(self, query_vector: List[float], k: int) -> List[Dict[str, Any]]:
        """Search using FAISS index."""
        if not hasattr(self, 'faiss_index'):
            raise Exception("FAISS index not initialized")
        
        # Convert query_vector to numpy array
        query_np = np.array(query_vector).reshape(1, -1).astype('float32')
        
        # Search the index
        distances, indices = self.faiss_index.search(query_np, k)
        
        # Format results (limited metadata since FAISS only stores indices)
        results = []
        for i in range(len(indices[0])):
            if indices[0][i] >= 0:  # Skip invalid indices
                results.append({
                    "id": str(indices[0][i]),
                    "score": float(1.0 - distances[0][i]) if distances[0][i] <= 1.0 else 0.0,
                    "metadata": {"index": int(indices[0][i])},
                    "vector": []  # FAISS doesn't return vectors
                })
        
        return results
    
    async def _search_simulated(
        self, 
        query_vector: List[float], 
        collection: str, 
        k: int, 
        filter_obj: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Simulated vector search for testing/development."""
        import json
        import random
        from pathlib import Path
        
        # Load simulated vector database
        try:
            db_path = Path("simulated_vector_db.json")
            if db_path.exists():
                with open(db_path, "r") as f:
                    vector_db = json.load(f)
            else:
                # Generate dummy database
                vector_db = {
                    "documents": [
                        {
                            "id": f"doc_{i}",
                            "vector": [random.random() for _ in range(len(query_vector))],
                            "metadata": {
                                "title": f"Document {i}",
                                "text": f"This is document {i} content."
                            }
                        }
                        for i in range(100)
                    ]
                }
        except Exception as e:
            logger.error(f"Error loading simulated vector DB: {str(e)}")
            vector_db = {"documents": []}
        
        # Calculate similarity (dot product) between query and all documents
        results = []
        query_np = np.array(query_vector)
        
        for doc in vector_db.get("documents", []):
            if "vector" in doc:
                doc_vector = np.array(doc["vector"])
                # Calculate dot product similarity
                similarity = np.dot(query_np, doc_vector) / (np.linalg.norm(query_np) * np.linalg.norm(doc_vector))
                
                # Apply filter if provided
                if filter_obj:
                    include = True
                    for key, value in filter_obj.items():
                        if key not in doc.get("metadata", {}) or doc["metadata"][key] != value:
                            include = False
                            break
                    
                    if not include:
                        continue
                
                results.append({
                    "id": doc.get("id", ""),
                    "score": float(similarity),
                    "metadata": doc.get("metadata", {}),
                    "vector": doc.get("vector", [])
                })
        
        # Sort by score and limit to k results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]
    
    async def get_embedding(self, text: str, model: str) -> List[float]:
        """Get embedding vector for text.
        
        Args:
            text: The text to embed
            model: The name of the embedding model to use
            
        Returns:
            Embedding vector as a list of floats
        """
        # Check cache first if cache service is configured
        if self.cache_service_url:
            # Use a hash of the text as the cache key to handle long texts
            import hashlib
            text_hash = hashlib.md5(text.encode()).hexdigest()
            cache_key = f"embedding:{model}:{text_hash}"
            
            try:
                cache_response = await self.client.get(
                    f"{self.cache_service_url}/get",
                    params={"key": cache_key}
                )
                
                if cache_response.status_code == 200:
                    cached_data = cache_response.json().get("data")
                    if cached_data:
                        logger.info(f"Cache hit for embedding")
                        return cached_data
            except Exception as e:
                logger.warning(f"Error checking cache: {str(e)}")
        
        # Load model if not already loaded
        if model not in self.embedding_models:
            logger.info(f"Loading embedding model: {model}")
            try:
                self.embedding_models[model] = SentenceTransformer(model)
                logger.info(f"Model {model} loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model {model}: {str(e)}")
                # Fallback to default model
                logger.info(f"Falling back to default model: {self.default_embedding_model}")
                model = self.default_embedding_model
        
        # Generate embedding
        try:
            start_time = time.time()
            embedding = self.embedding_models[model].encode(text)
            embedding_list = embedding.tolist()
            logger.info(f"Embedding generated in {time.time() - start_time:.2f}s")
            
            # Cache result if cache service is configured
            if self.cache_service_url:
                import hashlib
                text_hash = hashlib.md5(text.encode()).hexdigest()
                cache_key = f"embedding:{model}:{text_hash}"
                
                try:
                    await self.client.post(
                        f"{self.cache_service_url}/set",
                        json={
                            "key": cache_key,
                            "value": embedding_list,
                            "ttl": 3600  # 1 hour TTL
                        }
                    )
                except Exception as e:
                    logger.warning(f"Error caching embedding: {str(e)}")
            
            return embedding_list
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the service."""
        logger.info("Shutting down Vector Search Service")
        self.is_running = False
        # Clean up resources
        self.embedding_models.clear()
        await self.client.aclose()


# FastAPI specific code
def create_fastapi_app():
    """Create a FastAPI app for the Vector Search Service."""
    from fastapi import FastAPI, HTTPException
    import os
    
    app = FastAPI(title="RAG Vector Search Service", version="1.0.0")
    
    # Initialize service with environment variables or defaults
    service = VectorSearchService(
        default_embedding_model=os.getenv("DEFAULT_EMBEDDING_MODEL", "sentence-transformers/multi-qa-mpnet-base-dot-v1"),
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        cache_service_url=os.getenv("CACHE_SERVICE_URL"),
        use_faiss=os.getenv("USE_FAISS", "false").lower() == "true",
        faiss_index_path=os.getenv("FAISS_INDEX_PATH")
    )
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return await service.health_check()
    
    @app.post("/search")
    async def search(request: VectorSearchRequest):
        """Process a vector search request."""
        response = await service.process_request(request)
        if response.status == "error":
            raise HTTPException(status_code=400, detail=response.message)
        return response
    
    @app.post("/embed")
    async def embed(request: EmbeddingRequest):
        """Generate embedding for text."""
        response = await service.process_request(request)
        if response.status == "error":
            raise HTTPException(status_code=400, detail=response.message)
        return response
    
    @app.get("/shutdown")
    async def shutdown():
        """Shutdown the service."""
        await service.shutdown()
        return {"status": "shutting down"}
    
    return app