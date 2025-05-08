"""Service interfaces for RAG microservices architecture.

This module defines the base interfaces that all services must implement
to ensure consistent API behavior across the microservices architecture.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel


class ServiceRequest(BaseModel):
    """Base model for service requests."""
    request_id: str


class ServiceResponse(BaseModel):
    """Base model for service responses."""
    request_id: str
    status: str
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class ServiceInterface(ABC):
    """Base interface that all RAG microservices must implement."""
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check if the service is healthy and return status information."""
        pass
    
    @abstractmethod
    async def process_request(self, request: ServiceRequest) -> ServiceResponse:
        """Process a service request and return a response."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Gracefully shutdown the service."""
        pass


class QueryServiceInterface(ServiceInterface):
    """Interface for the Query Analysis and Orchestration Service."""
    
    @abstractmethod
    async def analyze_query(self, query_text: str, query_id: str) -> Dict[str, Any]:
        """Analyze a query to determine retrieval strategy."""
        pass
    
    @abstractmethod
    async def orchestrate_retrieval(self, query_text: str, strategy: Dict[str, Any], 
                                  query_id: str) -> Dict[str, Any]:
        """Orchestrate the retrieval process based on query analysis."""
        pass


class VectorSearchServiceInterface(ServiceInterface):
    """Interface for the Vector Search Service."""
    
    @abstractmethod
    async def search(self, query_vector: List[float], collection: str, 
                   k: int, filter_obj: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Perform vector search with the provided query vector."""
        pass
    
    @abstractmethod
    async def get_embedding(self, text: str, model: str) -> List[float]:
        """Get embedding vector for text."""
        pass


class BM25ServiceInterface(ServiceInterface):
    """Interface for the BM25 Lexical Search Service."""
    
    @abstractmethod
    async def search(self, query_text: str, collection: str, 
                   k: int, filter_obj: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Perform BM25 search with the provided query text."""
        pass
    
    @abstractmethod
    async def update_index(self, collection: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Update the BM25 index with new documents."""
        pass


class FusionServiceInterface(ServiceInterface):
    """Interface for the Fusion Service."""
    
    @abstractmethod
    async def fuse_results(self, vector_results: List[Dict[str, Any]], 
                         bm25_results: List[Dict[str, Any]],
                         strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fuse results from different retrievers based on strategy."""
        pass
    
    @abstractmethod
    async def rerank(self, results: List[Dict[str, Any]], query_text: str,
                   query_vector: Optional[List[float]] = None,
                   strategy: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Rerank results based on strategy."""
        pass


class CacheServiceInterface(ServiceInterface):
    """Interface for the Distributed Cache Service."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in the cache with optional TTL."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a value from the cache."""
        pass
    
    @abstractmethod
    async def flush(self) -> bool:
        """Flush the entire cache."""
        pass


class ColBERTServiceInterface(ServiceInterface):
    """Interface for the ColBERT late interaction service."""
    
    @abstractmethod
    async def encode_query(self, query_text: str) -> Dict[str, Any]:
        """Encode query into token-level embeddings."""
        pass
    
    @abstractmethod
    async def score(self, query_embeddings: Dict[str, Any], 
                  document_vectors: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """Score documents using late interaction with query tokens."""
        pass


class SPLADEServiceInterface(ServiceInterface):
    """Interface for the SPLADE sparse retrieval service."""
    
    @abstractmethod
    async def encode_query(self, query_text: str) -> Dict[str, float]:
        """Encode query into a sparse vector representation."""
        pass
    
    @abstractmethod
    async def encode_document(self, document_text: str) -> Dict[str, float]:
        """Encode document into a sparse vector representation."""
        pass
    
    @abstractmethod
    async def search(self, query_vector: Dict[str, float], collection: str,
                   k: int) -> List[Dict[str, Any]]:
        """Perform search using sparse vectors."""
        pass