"""BM25 Service implementation.

This service is responsible for lexical search using the BM25 algorithm.
It provides efficient keyword-based retrieval capabilities.
"""
import asyncio
import json
import logging
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import httpx
import numpy as np
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from microservices.service_interfaces import (
    BM25ServiceInterface,
    ServiceRequest,
    ServiceResponse
)

# Configure logging
logger = logging.getLogger(__name__)


class BM25SearchRequest(ServiceRequest):
    """BM25 search request model."""
    query_text: str
    collection: str
    k: int = 10
    filter_obj: Optional[Dict[str, Any]] = None


class UpdateIndexRequest(ServiceRequest):
    """Update index request model."""
    collection: str
    documents: List[Dict[str, Any]]


class BM25SearchResponse(ServiceResponse):
    """BM25 search response model."""
    results: Optional[List[Dict[str, Any]]] = None


class BM25Service(BM25ServiceInterface):
    """Implementation of the BM25 Lexical Search Service."""
    
    def __init__(
        self,
        index_dir: str = "bm25_indices",
        cache_service_url: Optional[str] = None,
        tokenizer: Optional[Any] = None
    ):
        """Initialize the BM25 service.
        
        Args:
            index_dir: Directory to store BM25 indices
            cache_service_url: Optional URL for the Cache Service
            tokenizer: Optional custom tokenizer function
        """
        self.index_dir = Path(index_dir)
        self.cache_service_url = cache_service_url
        self.tokenizer = tokenizer or self._default_tokenizer
        self.indices = {}
        self.document_lookup = {}
        
        # Create index directory if it doesn't exist
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing indices
        self._load_existing_indices()
        
        # Initialize httpx client
        self.client = httpx.AsyncClient(timeout=30.0)
        self.is_running = True
        
        logger.info(f"BM25 Service initialized with index dir: {index_dir}")
    
    def _default_tokenizer(self, text: str) -> List[str]:
        """Default tokenization function.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        import re
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        # Tokenize
        return text.split()
    
    def _load_existing_indices(self):
        """Load existing BM25 indices from disk."""
        for index_file in self.index_dir.glob("*.bm25"):
            try:
                collection_name = index_file.stem
                logger.info(f"Loading BM25 index for collection: {collection_name}")
                
                with open(index_file, 'rb') as f:
                    index_data = pickle.load(f)
                    
                self.indices[collection_name] = index_data['index']
                self.document_lookup[collection_name] = index_data['documents']
                
                logger.info(f"Loaded BM25 index for {collection_name} with {len(self.document_lookup[collection_name])} documents")
                
            except Exception as e:
                logger.error(f"Error loading index {index_file}: {str(e)}")
    
    def _save_index(self, collection: str):
        """Save a BM25 index to disk.
        
        Args:
            collection: Collection name
        """
        if collection in self.indices:
            index_path = self.index_dir / f"{collection}.bm25"
            
            index_data = {
                'index': self.indices[collection],
                'documents': self.document_lookup[collection]
            }
            
            try:
                with open(index_path, 'wb') as f:
                    pickle.dump(index_data, f)
                    
                logger.info(f"Saved BM25 index for {collection} with {len(self.document_lookup[collection])} documents")
                
            except Exception as e:
                logger.error(f"Error saving index {collection}: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if the service is healthy and return status information."""
        status = {
            "status": "healthy",
            "indices": list(self.indices.keys()),
            "document_counts": {k: len(v) for k, v in self.document_lookup.items()},
            "version": "1.0.0"
        }
        
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
        if isinstance(request, BM25SearchRequest):
            try:
                results = await self.search(
                    query_text=request.query_text,
                    collection=request.collection,
                    k=request.k,
                    filter_obj=request.filter_obj
                )
                
                return BM25SearchResponse(
                    request_id=request.request_id,
                    status="success",
                    data={"results": results}
                )
                
            except Exception as e:
                logger.error(f"Error in BM25 search: {str(e)}")
                return BM25SearchResponse(
                    request_id=request.request_id,
                    status="error",
                    message=f"Error in BM25 search: {str(e)}"
                )
                
        elif isinstance(request, UpdateIndexRequest):
            try:
                result = await self.update_index(
                    collection=request.collection,
                    documents=request.documents
                )
                
                return ServiceResponse(
                    request_id=request.request_id,
                    status="success",
                    data=result
                )
                
            except Exception as e:
                logger.error(f"Error updating index: {str(e)}")
                return ServiceResponse(
                    request_id=request.request_id,
                    status="error",
                    message=f"Error updating index: {str(e)}"
                )
                
        else:
            return ServiceResponse(
                request_id=getattr(request, "request_id", "unknown"),
                status="error",
                message="Invalid request type"
            )
    
    async def search(
        self, 
        query_text: str, 
        collection: str, 
        k: int = 10, 
        filter_obj: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform BM25 search with the provided query text.
        
        Args:
            query_text: The query text to search with
            collection: The collection to search in
            k: The number of results to return
            filter_obj: Optional filter for the search
            
        Returns:
            List of search results
        """
        # Check if the collection exists
        if collection not in self.indices:
            logger.warning(f"Collection {collection} not found, creating empty index")
            await self.update_index(collection, [])
            return []
        
        # Check cache first if cache service is configured
        if self.cache_service_url:
            import hashlib
            query_hash = hashlib.md5(query_text.encode()).hexdigest()
            cache_key = f"bm25_search:{collection}:{query_hash}:{k}:{str(filter_obj)}"
            
            try:
                cache_response = await self.client.get(
                    f"{self.cache_service_url}/get",
                    params={"key": cache_key}
                )
                
                if cache_response.status_code == 200:
                    cached_data = cache_response.json().get("data")
                    if cached_data:
                        logger.info(f"Cache hit for BM25 search")
                        return cached_data
            except Exception as e:
                logger.warning(f"Error checking cache: {str(e)}")
        
        # Tokenize query
        tokenized_query = self.tokenizer(query_text)
        
        # Perform BM25 search
        bm25_index = self.indices[collection]
        doc_scores = bm25_index.get_scores(tokenized_query)
        
        # Get document indices sorted by score
        doc_indices = np.argsort(doc_scores)[::-1].tolist()
        
        # Prepare results
        results = []
        documents = self.document_lookup[collection]
        
        for idx in doc_indices:
            doc = documents[idx]
            score = float(doc_scores[idx])
            
            # Skip documents with zero score
            if score <= 0:
                continue
            
            # Apply filter if provided
            if filter_obj:
                include = True
                for key, value in filter_obj.items():
                    if key not in doc.get("metadata", {}) or doc["metadata"][key] != value:
                        include = False
                        break
                
                if not include:
                    continue
            
            # Add document to results
            results.append({
                "id": doc.get("id", str(idx)),
                "score": score,
                "metadata": doc.get("metadata", {}),
                "text": doc.get("text", "")
            })
            
            # Stop after finding k results
            if len(results) >= k:
                break
        
        # Cache results if cache service is configured
        if self.cache_service_url:
            import hashlib
            query_hash = hashlib.md5(query_text.encode()).hexdigest()
            cache_key = f"bm25_search:{collection}:{query_hash}:{k}:{str(filter_obj)}"
            
            try:
                await self.client.post(
                    f"{self.cache_service_url}/set",
                    json={
                        "key": cache_key,
                        "value": results,
                        "ttl": 300  # 5 minutes TTL
                    }
                )
            except Exception as e:
                logger.warning(f"Error caching results: {str(e)}")
        
        return results
    
    async def update_index(
        self, 
        collection: str, 
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Update the BM25 index with new documents.
        
        This method can be used to:
        1. Create a new index if the collection doesn't exist
        2. Add new documents to an existing index
        3. Replace an existing index entirely
        
        Args:
            collection: Collection name
            documents: List of documents to index
            
        Returns:
            Status information about the update
        """
        start_time = time.time()
        
        # Check if we're creating a new index or updating an existing one
        creating_new = collection not in self.indices
        
        # Process documents
        processed_docs = []
        corpus = []
        
        for doc in documents:
            # Ensure the document has the required fields
            if "text" not in doc:
                logger.warning(f"Document is missing 'text' field, skipping")
                continue
            
            # Tokenize the document text
            tokenized_text = self.tokenizer(doc["text"])
            
            # Add to corpus for BM25
            corpus.append(tokenized_text)
            
            # Add to processed documents
            processed_docs.append({
                "id": doc.get("id", str(len(processed_docs))),
                "text": doc["text"],
                "metadata": doc.get("metadata", {})
            })
        
        # Create or update the index
        if creating_new or len(documents) > 0:
            # For a new index or full replacement
            if creating_new or len(documents) >= len(self.document_lookup.get(collection, [])):
                logger.info(f"Creating new BM25 index for collection {collection}")
                self.document_lookup[collection] = processed_docs
                self.indices[collection] = BM25Okapi(corpus) if corpus else None
            
            # For adding documents to an existing index
            else:
                logger.info(f"Adding {len(processed_docs)} documents to existing index for {collection}")
                
                # Get existing corpus
                existing_corpus = []
                for doc in self.document_lookup[collection]:
                    existing_corpus.append(self.tokenizer(doc["text"]))
                
                # Combine with new documents
                combined_corpus = existing_corpus + corpus
                self.document_lookup[collection].extend(processed_docs)
                self.indices[collection] = BM25Okapi(combined_corpus)
            
            # Save the updated index
            self._save_index(collection)
        
        # Return status information
        elapsed_time = time.time() - start_time
        return {
            "collection": collection,
            "documents_processed": len(processed_docs),
            "total_documents": len(self.document_lookup.get(collection, [])),
            "elapsed_time": elapsed_time,
            "status": "created" if creating_new else "updated"
        }
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the service."""
        logger.info("Shutting down BM25 Service")
        self.is_running = False
        
        # Save all indices
        for collection in self.indices:
            self._save_index(collection)
        
        await self.client.aclose()


# FastAPI specific code
def create_fastapi_app():
    """Create a FastAPI app for the BM25 Service."""
    from fastapi import FastAPI, HTTPException
    import os
    
    app = FastAPI(title="RAG BM25 Service", version="1.0.0")
    
    # Initialize service with environment variables or defaults
    service = BM25Service(
        index_dir=os.getenv("BM25_INDEX_DIR", "bm25_indices"),
        cache_service_url=os.getenv("CACHE_SERVICE_URL")
    )
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return await service.health_check()
    
    @app.post("/search")
    async def search(request: BM25SearchRequest):
        """Process a BM25 search request."""
        response = await service.process_request(request)
        if response.status == "error":
            raise HTTPException(status_code=400, detail=response.message)
        return response
    
    @app.post("/update")
    async def update_index(request: UpdateIndexRequest):
        """Update a BM25 index."""
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