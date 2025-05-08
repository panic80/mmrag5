"""ColBERT Service implementation.

This service implements the ColBERT late interaction model for token-level 
interaction between queries and documents. ColBERT creates contextualized 
embeddings for each token and performs maximum similarity matching between
query tokens and document tokens.
"""
import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import torch
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModel

from microservices.service_interfaces import (
    ColBERTServiceInterface,
    ServiceRequest,
    ServiceResponse
)

# Configure logging
logger = logging.getLogger(__name__)


class EncodeQueryRequest(ServiceRequest):
    """Encode query request model."""
    query_text: str
    max_query_tokens: Optional[int] = 32


class ScoreRequest(ServiceRequest):
    """Score request model."""
    query_embeddings: Dict[str, Any]
    document_vectors: List[Dict[str, Any]]


class ColBERTResponse(ServiceResponse):
    """ColBERT response model."""
    embeddings: Optional[Dict[str, Any]] = None
    scores: Optional[Dict[str, float]] = None


class ColBERTService(ColBERTServiceInterface):
    """Implementation of the ColBERT late interaction service."""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_query_tokens: int = 32,
        max_doc_tokens: int = 180,
        similarity_metric: str = "cosine",
        use_gpu: bool = False,
        cache_service_url: Optional[str] = None
    ):
        """Initialize the ColBERT service.
        
        Args:
            model_name: Name of the pre-trained model to use
            max_query_tokens: Maximum number of tokens for queries
            max_doc_tokens: Maximum number of tokens for documents
            similarity_metric: Similarity metric to use ('cosine' or 'l2')
            use_gpu: Whether to use GPU for inference
            cache_service_url: Optional URL for the Cache Service
        """
        self.model_name = model_name
        self.max_query_tokens = max_query_tokens
        self.max_doc_tokens = max_doc_tokens
        self.similarity_metric = similarity_metric
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.cache_service_url = cache_service_url
        
        # Initialize tokenizer and model
        self._initialize_model()
        
        # Initialize httpx client
        self.client = httpx.AsyncClient(timeout=30.0)
        self.is_running = True
        
        # Device information
        device_info = f"{'GPU' if self.use_gpu else 'CPU'}"
        
        logger.info(f"ColBERT Service initialized with model: {model_name}, device: {device_info}")
    
    def _initialize_model(self):
        """Initialize the tokenizer and model."""
        try:
            logger.info(f"Loading tokenizer: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            logger.info(f"Loading model: {self.model_name}")
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Move model to GPU if available
            self.device = torch.device("cuda" if self.use_gpu else "cpu")
            self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if the service is healthy and return status information."""
        status = {
            "status": "healthy",
            "version": "1.0.0",
            "model_name": self.model_name,
            "device": str(self.device),
            "max_query_tokens": self.max_query_tokens,
            "max_doc_tokens": self.max_doc_tokens,
            "similarity_metric": self.similarity_metric
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
        if isinstance(request, EncodeQueryRequest):
            try:
                embeddings = await self.encode_query(
                    query_text=request.query_text,
                    max_query_tokens=request.max_query_tokens or self.max_query_tokens
                )
                
                return ColBERTResponse(
                    request_id=request.request_id,
                    status="success",
                    data=embeddings
                )
                
            except Exception as e:
                logger.error(f"Error encoding query: {str(e)}")
                return ColBERTResponse(
                    request_id=request.request_id,
                    status="error",
                    message=f"Error encoding query: {str(e)}"
                )
                
        elif isinstance(request, ScoreRequest):
            try:
                scores = await self.score(
                    query_embeddings=request.query_embeddings,
                    document_vectors=request.document_vectors
                )
                
                return ColBERTResponse(
                    request_id=request.request_id,
                    status="success",
                    data={"scores": scores}
                )
                
            except Exception as e:
                logger.error(f"Error scoring documents: {str(e)}")
                return ColBERTResponse(
                    request_id=request.request_id,
                    status="error",
                    message=f"Error scoring documents: {str(e)}"
                )
                
        else:
            return ServiceResponse(
                request_id=getattr(request, "request_id", "unknown"),
                status="error",
                message="Invalid request type"
            )
    
    async def encode_query(
        self, 
        query_text: str, 
        max_query_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Encode query into token-level embeddings.
        
        Args:
            query_text: The query text to encode
            max_query_tokens: Maximum number of tokens to encode
            
        Returns:
            Dictionary containing token-level embeddings and token info
        """
        # Check cache first if cache service is configured
        if self.cache_service_url:
            import hashlib
            query_hash = hashlib.md5(query_text.encode()).hexdigest()
            cache_key = f"colbert_query:{query_hash}:{max_query_tokens or self.max_query_tokens}"
            
            try:
                cache_response = await self.client.get(
                    f"{self.cache_service_url}/get",
                    params={"key": cache_key}
                )
                
                if cache_response.status_code == 200:
                    cached_data = cache_response.json().get("data")
                    if cached_data:
                        logger.info(f"Cache hit for query encoding")
                        return cached_data
            except Exception as e:
                logger.warning(f"Error checking cache: {str(e)}")
        
        # Use the specified max_query_tokens or the default
        max_tokens = max_query_tokens or self.max_query_tokens
        
        # Tokenize the query
        encoded_query = self.tokenizer(
            query_text,
            max_length=max_tokens,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move tensors to device
        input_ids = encoded_query['input_ids'].to(self.device)
        attention_mask = encoded_query['attention_mask'].to(self.device)
        
        # Execute the model
        with torch.no_grad():
            start_time = time.time()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            logger.info(f"Query encoded in {time.time() - start_time:.2f}s")
        
        # Get the last hidden state for token embeddings
        last_hidden_state = outputs.last_hidden_state
        
        # Create token-level embeddings using the attention mask to filter padding
        token_embeddings = []
        filtered_token_ids = []
        
        # Extract only the embeddings for actual tokens (excluding padding)
        for i in range(input_ids.size(1)):
            if attention_mask[0, i]:
                token_embeddings.append(last_hidden_state[0, i].cpu().numpy().tolist())
                filtered_token_ids.append(int(input_ids[0, i]))
        
        # Get token strings from token IDs
        token_strings = self.tokenizer.convert_ids_to_tokens(filtered_token_ids)
        
        # Create result
        result = {
            "embeddings": token_embeddings,
            "token_ids": filtered_token_ids,
            "tokens": token_strings,
            "query_text": query_text,
            "embedding_dim": len(token_embeddings[0]) if token_embeddings else 0,
            "num_tokens": len(token_embeddings)
        }
        
        # Cache result if cache service is configured
        if self.cache_service_url:
            import hashlib
            query_hash = hashlib.md5(query_text.encode()).hexdigest()
            cache_key = f"colbert_query:{query_hash}:{max_query_tokens or self.max_query_tokens}"
            
            try:
                await self.client.post(
                    f"{self.cache_service_url}/set",
                    json={
                        "key": cache_key,
                        "value": result,
                        "ttl": 3600  # 1 hour TTL
                    }
                )
            except Exception as e:
                logger.warning(f"Error caching query encoding: {str(e)}")
        
        return result
    
    async def encode_document(
        self, 
        document_text: str, 
        max_doc_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Encode document into token-level embeddings.
        
        Args:
            document_text: The document text to encode
            max_doc_tokens: Maximum number of tokens to encode
            
        Returns:
            Dictionary containing token-level embeddings and token info
        """
        # Check cache first if cache service is configured
        if self.cache_service_url:
            import hashlib
            doc_hash = hashlib.md5(document_text.encode()).hexdigest()
            cache_key = f"colbert_doc:{doc_hash}:{max_doc_tokens or self.max_doc_tokens}"
            
            try:
                cache_response = await self.client.get(
                    f"{self.cache_service_url}/get",
                    params={"key": cache_key}
                )
                
                if cache_response.status_code == 200:
                    cached_data = cache_response.json().get("data")
                    if cached_data:
                        logger.info(f"Cache hit for document encoding")
                        return cached_data
            except Exception as e:
                logger.warning(f"Error checking cache: {str(e)}")
        
        # Use the specified max_doc_tokens or the default
        max_tokens = max_doc_tokens or self.max_doc_tokens
        
        # Tokenize the document
        encoded_doc = self.tokenizer(
            document_text,
            max_length=max_tokens,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move tensors to device
        input_ids = encoded_doc['input_ids'].to(self.device)
        attention_mask = encoded_doc['attention_mask'].to(self.device)
        
        # Execute the model
        with torch.no_grad():
            start_time = time.time()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            logger.info(f"Document encoded in {time.time() - start_time:.2f}s")
        
        # Get the last hidden state for token embeddings
        last_hidden_state = outputs.last_hidden_state
        
        # Create token-level embeddings using the attention mask to filter padding
        token_embeddings = []
        filtered_token_ids = []
        
        # Extract only the embeddings for actual tokens (excluding padding)
        for i in range(input_ids.size(1)):
            if attention_mask[0, i]:
                token_embeddings.append(last_hidden_state[0, i].cpu().numpy().tolist())
                filtered_token_ids.append(int(input_ids[0, i]))
        
        # Get token strings from token IDs
        token_strings = self.tokenizer.convert_ids_to_tokens(filtered_token_ids)
        
        # Create result
        result = {
            "embeddings": token_embeddings,
            "token_ids": filtered_token_ids,
            "tokens": token_strings,
            "document_text": document_text,
            "embedding_dim": len(token_embeddings[0]) if token_embeddings else 0,
            "num_tokens": len(token_embeddings)
        }
        
        # Cache result if cache service is configured
        if self.cache_service_url:
            import hashlib
            doc_hash = hashlib.md5(document_text.encode()).hexdigest()
            cache_key = f"colbert_doc:{doc_hash}:{max_doc_tokens or self.max_doc_tokens}"
            
            try:
                await self.client.post(
                    f"{self.cache_service_url}/set",
                    json={
                        "key": cache_key,
                        "value": result,
                        "ttl": 3600  # 1 hour TTL
                    }
                )
            except Exception as e:
                logger.warning(f"Error caching document encoding: {str(e)}")
        
        return result
    
    async def score(
        self, 
        query_embeddings: Dict[str, Any], 
        document_vectors: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Score documents using late interaction with query tokens.
        
        Args:
            query_embeddings: Query token embeddings
            document_vectors: List of document vectors to score
            
        Returns:
            Dictionary mapping document IDs to scores
        """
        # Check cache first if cache service is configured
        if self.cache_service_url:
            import hashlib
            import json
            
            # Generate a deterministic representation of inputs for caching
            query_hash = hashlib.md5(json.dumps(query_embeddings, sort_keys=True).encode()).hexdigest()
            doc_ids = sorted([doc.get("id", "") for doc in document_vectors])
            docs_hash = hashlib.md5(json.dumps(doc_ids, sort_keys=True).encode()).hexdigest()
            
            cache_key = f"colbert_scores:{query_hash}:{docs_hash}"
            
            try:
                cache_response = await self.client.get(
                    f"{self.cache_service_url}/get",
                    params={"key": cache_key}
                )
                
                if cache_response.status_code == 200:
                    cached_data = cache_response.json().get("data")
                    if cached_data:
                        logger.info(f"Cache hit for ColBERT scores")
                        return cached_data
            except Exception as e:
                logger.warning(f"Error checking cache: {str(e)}")
        
        start_time = time.time()
        
        # Extract query token embeddings
        q_embeddings = query_embeddings.get("embeddings", [])
        if not q_embeddings:
            logger.warning("No query embeddings provided")
            return {}
        
        # Convert to numpy array for efficient computation
        q_emb_array = np.array(q_embeddings)
        
        # Dictionary to store scores
        scores = {}
        
        # Process each document
        for doc in document_vectors:
            doc_id = doc.get("id", "")
            
            # Get document text if available, otherwise encode the vectors directly
            if "text" in doc:
                # Need to encode the document first
                doc_data = await self.encode_document(doc["text"])
                d_emb_array = np.array(doc_data.get("embeddings", []))
            elif "vector" in doc:
                # If doc vector is a single vector (not token-level), expand to token level
                vector = doc.get("vector", [])
                
                # Check if it's token-level (list of lists) or document-level (single list)
                if isinstance(vector, list) and len(vector) > 0 and isinstance(vector[0], list):
                    # Already token-level
                    d_emb_array = np.array(vector)
                else:
                    # Create a single "token" for the entire document
                    d_emb_array = np.array([vector])
            else:
                logger.warning(f"Document {doc_id} has no text or vector, skipping")
                continue
            
            # Skip empty embeddings
            if d_emb_array.size == 0:
                logger.warning(f"Document {doc_id} has empty embeddings, skipping")
                continue
            
            # Calculate similarity matrix between query tokens and document tokens
            if self.similarity_metric == "cosine":
                # Normalize vectors for cosine similarity
                q_emb_norm = q_emb_array / np.linalg.norm(q_emb_array, axis=1, keepdims=True)
                d_emb_norm = d_emb_array / np.linalg.norm(d_emb_array, axis=1, keepdims=True)
                
                # Calculate cosine similarity
                sim_matrix = np.matmul(q_emb_norm, d_emb_norm.T)
            else:
                # L2 distance (convert to similarity)
                sim_matrix = -np.sqrt(np.sum((q_emb_array[:, np.newaxis, :] - d_emb_array[np.newaxis, :, :]) ** 2, axis=2))
            
            # MaxSim operation: for each query token, find the maximum similarity with any document token
            max_sim = np.max(sim_matrix, axis=1)
            
            # Sum over all query tokens to get the final score
            score = float(np.sum(max_sim))
            
            # Store score
            scores[doc_id] = score
        
        logger.info(f"Scored {len(document_vectors)} documents in {time.time() - start_time:.2f}s")
        
        # Cache results if cache service is configured
        if self.cache_service_url:
            import hashlib
            import json
            
            query_hash = hashlib.md5(json.dumps(query_embeddings, sort_keys=True).encode()).hexdigest()
            doc_ids = sorted([doc.get("id", "") for doc in document_vectors])
            docs_hash = hashlib.md5(json.dumps(doc_ids, sort_keys=True).encode()).hexdigest()
            
            cache_key = f"colbert_scores:{query_hash}:{docs_hash}"
            
            try:
                await self.client.post(
                    f"{self.cache_service_url}/set",
                    json={
                        "key": cache_key,
                        "value": scores,
                        "ttl": 3600  # 1 hour TTL
                    }
                )
            except Exception as e:
                logger.warning(f"Error caching ColBERT scores: {str(e)}")
        
        return scores
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the service."""
        logger.info("Shutting down ColBERT Service")
        self.is_running = False
        await self.client.aclose()


# FastAPI specific code
def create_fastapi_app():
    """Create a FastAPI app for the ColBERT Service."""
    from fastapi import FastAPI, HTTPException
    import os
    
    app = FastAPI(title="RAG ColBERT Service", version="1.0.0")
    
    # Initialize service with environment variables or defaults
    service = ColBERTService(
        model_name=os.getenv("COLBERT_MODEL_NAME", "bert-base-uncased"),
        max_query_tokens=int(os.getenv("MAX_QUERY_TOKENS", "32")),
        max_doc_tokens=int(os.getenv("MAX_DOC_TOKENS", "180")),
        similarity_metric=os.getenv("SIMILARITY_METRIC", "cosine"),
        use_gpu=os.getenv("USE_GPU", "false").lower() == "true",
        cache_service_url=os.getenv("CACHE_SERVICE_URL")
    )
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return await service.health_check()
    
    @app.post("/encode_query")
    async def encode_query(request: EncodeQueryRequest):
        """Encode a query into token-level embeddings."""
        response = await service.process_request(request)
        if response.status == "error":
            raise HTTPException(status_code=400, detail=response.message)
        return response
    
    @app.post("/encode_document")
    async def encode_document(document_text: str, max_doc_tokens: Optional[int] = None):
        """Encode a document into token-level embeddings."""
        result = await service.encode_document(document_text, max_doc_tokens)
        return {"status": "success", "data": result}
    
    @app.post("/score")
    async def score(request: ScoreRequest):
        """Score documents using late interaction with query tokens."""
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