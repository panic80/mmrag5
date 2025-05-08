"""SPLADE Service implementation.

This service implements SPLADE (SParse Lexical AnD Expansion) for improved 
sparse retrieval. SPLADE produces sparse representations where each dimension
corresponds to a term in the vocabulary, allowing for efficient retrieval
while capturing both lexical matching and semantic expansion.
"""
import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import httpx
import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForMaskedLM

from microservices.service_interfaces import (
    SPLADEServiceInterface,
    ServiceRequest,
    ServiceResponse
)

# Configure logging
logger = logging.getLogger(__name__)


class EncodeQueryRequest(ServiceRequest):
    """Encode query request model."""
    query_text: str


class EncodeDocumentRequest(ServiceRequest):
    """Encode document request model."""
    document_text: str


class SPLADESearchRequest(ServiceRequest):
    """SPLADE search request model."""
    query_vector: Dict[str, float]
    collection: str
    k: int = 10


class SPLADEResponse(ServiceResponse):
    """SPLADE response model."""
    sparse_vector: Optional[Dict[str, float]] = None
    results: Optional[List[Dict[str, Any]]] = None


class SPLADEService(SPLADEServiceInterface):
    """Implementation of the SPLADE sparse retrieval service."""
    
    def __init__(
        self,
        model_name: str = "naver/splade-cocondenser-ensembledistil",
        max_seq_length: int = 256,
        use_gpu: bool = False,
        lambda_d: float = 0.0001,  # Regularization parameter
        index_dir: str = "splade_indices",
        cache_service_url: Optional[str] = None
    ):
        """Initialize the SPLADE service.
        
        Args:
            model_name: Name of the pre-trained SPLADE model to use
            max_seq_length: Maximum sequence length
            use_gpu: Whether to use GPU for inference
            lambda_d: Regularization parameter for SPLADE
            index_dir: Directory to store SPLADE indices
            cache_service_url: Optional URL for the Cache Service
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.lambda_d = lambda_d
        self.index_dir = os.path.join(os.getcwd(), index_dir)
        self.cache_service_url = cache_service_url
        
        # Make sure index directory exists
        os.makedirs(self.index_dir, exist_ok=True)
        
        # Initialize tokenizer and model
        self._initialize_model()
        
        # Initialize index storage
        self.indices = {}
        self._load_indices()
        
        # Initialize httpx client
        self.client = httpx.AsyncClient(timeout=30.0)
        self.is_running = True
        
        # Device information
        device_info = f"{'GPU' if self.use_gpu else 'CPU'}"
        
        logger.info(f"SPLADE Service initialized with model: {model_name}, device: {device_info}")
    
    def _initialize_model(self):
        """Initialize the tokenizer and model."""
        try:
            logger.info(f"Loading tokenizer: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            logger.info(f"Loading model: {self.model_name}")
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            
            # Move model to GPU if available
            self.device = torch.device("cuda" if self.use_gpu else "cpu")
            self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Get vocabulary size
            self.vocab_size = len(self.tokenizer.get_vocab())
            
            logger.info(f"Model loaded successfully on {self.device} with vocabulary size: {self.vocab_size}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _load_indices(self):
        """Load existing SPLADE indices from disk."""
        logger.info(f"Loading SPLADE indices from {self.index_dir}")
        
        # Check for index files
        for filename in os.listdir(self.index_dir):
            if filename.endswith('.splade.index'):
                collection_name = filename.split('.')[0]
                index_path = os.path.join(self.index_dir, filename)
                
                try:
                    # Load index
                    self.indices[collection_name] = self._load_index(index_path)
                    logger.info(f"Loaded SPLADE index for collection: {collection_name}")
                except Exception as e:
                    logger.error(f"Error loading SPLADE index: {str(e)}")
    
    def _load_index(self, index_path: str) -> Dict[str, Any]:
        """Load a SPLADE index from disk.
        
        Args:
            index_path: Path to the index file
            
        Returns:
            Dictionary containing the index data
        """
        import pickle
        with open(index_path, 'rb') as f:
            index_data = pickle.load(f)
        
        return index_data
    
    def _save_index(self, collection: str, index_data: Dict[str, Any]):
        """Save a SPLADE index to disk.
        
        Args:
            collection: Collection name
            index_data: Index data to save
        """
        import pickle
        index_path = os.path.join(self.index_dir, f"{collection}.splade.index")
        
        with open(index_path, 'wb') as f:
            pickle.dump(index_data, f)
            
        logger.info(f"Saved SPLADE index for collection: {collection}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if the service is healthy and return status information."""
        status = {
            "status": "healthy",
            "version": "1.0.0",
            "model_name": self.model_name,
            "device": str(self.device),
            "max_seq_length": self.max_seq_length,
            "collections": list(self.indices.keys())
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
                sparse_vector = await self.encode_query(request.query_text)
                
                return SPLADEResponse(
                    request_id=request.request_id,
                    status="success",
                    data={"sparse_vector": sparse_vector}
                )
                
            except Exception as e:
                logger.error(f"Error encoding query: {str(e)}")
                return SPLADEResponse(
                    request_id=request.request_id,
                    status="error",
                    message=f"Error encoding query: {str(e)}"
                )
                
        elif isinstance(request, EncodeDocumentRequest):
            try:
                sparse_vector = await self.encode_document(request.document_text)
                
                return SPLADEResponse(
                    request_id=request.request_id,
                    status="success",
                    data={"sparse_vector": sparse_vector}
                )
                
            except Exception as e:
                logger.error(f"Error encoding document: {str(e)}")
                return SPLADEResponse(
                    request_id=request.request_id,
                    status="error",
                    message=f"Error encoding document: {str(e)}"
                )
                
        elif isinstance(request, SPLADESearchRequest):
            try:
                results = await self.search(
                    query_vector=request.query_vector,
                    collection=request.collection,
                    k=request.k
                )
                
                return SPLADEResponse(
                    request_id=request.request_id,
                    status="success",
                    data={"results": results}
                )
                
            except Exception as e:
                logger.error(f"Error in SPLADE search: {str(e)}")
                return SPLADEResponse(
                    request_id=request.request_id,
                    status="error",
                    message=f"Error in SPLADE search: {str(e)}"
                )
                
        else:
            return ServiceResponse(
                request_id=getattr(request, "request_id", "unknown"),
                status="error",
                message="Invalid request type"
            )
    
    async def encode_query(self, query_text: str) -> Dict[str, float]:
        """Encode query into a sparse vector representation.
        
        Args:
            query_text: The query text to encode
            
        Returns:
            Dictionary mapping term IDs to weights
        """
        # Check cache first if cache service is configured
        if self.cache_service_url:
            import hashlib
            query_hash = hashlib.md5(query_text.encode()).hexdigest()
            cache_key = f"splade_query:{query_hash}"
            
            try:
                cache_response = await self.client.get(
                    f"{self.cache_service_url}/get",
                    params={"key": cache_key}
                )
                
                if cache_response.status_code == 200:
                    cached_data = cache_response.json().get("data")
                    if cached_data:
                        logger.info(f"Cache hit for SPLADE query encoding")
                        return cached_data
            except Exception as e:
                logger.warning(f"Error checking cache: {str(e)}")
        
        start_time = time.time()
        
        # Tokenize and convert to tensors
        inputs = self.tokenizer(
            query_text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move tensors to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get SPLADE activations
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Apply log(1 + ReLU(logits))
            activations = torch.log(1 + F.relu(logits))
            
            # Apply max pooling over sequence dimension
            weights, _ = torch.max(activations, dim=1)
            
            # Apply L1 regularization if lambda_d > 0
            if self.lambda_d > 0:
                l1_penalty = self.lambda_d * torch.sum(weights)
                # Note: in training this would be added to the loss
                # but in inference we just log it
                logger.debug(f"L1 penalty: {l1_penalty.item()}")
        
        # Convert to sparse dictionary
        weights = weights[0].cpu().numpy()
        
        # Create sparse vector (only include non-zero weights)
        sparse_vector = {}
        vocab = self.tokenizer.get_vocab()
        reverse_vocab = {v: k for k, v in vocab.items()}
        
        for idx, weight in enumerate(weights):
            if weight > 0:
                token = reverse_vocab.get(idx)
                if token:
                    sparse_vector[token] = float(weight)
        
        logger.info(f"Query encoded in {time.time() - start_time:.2f}s with {len(sparse_vector)} non-zero dimensions")
        
        # Cache result if cache service is configured
        if self.cache_service_url:
            import hashlib
            query_hash = hashlib.md5(query_text.encode()).hexdigest()
            cache_key = f"splade_query:{query_hash}"
            
            try:
                await self.client.post(
                    f"{self.cache_service_url}/set",
                    json={
                        "key": cache_key,
                        "value": sparse_vector,
                        "ttl": 3600  # 1 hour TTL
                    }
                )
            except Exception as e:
                logger.warning(f"Error caching SPLADE query: {str(e)}")
        
        return sparse_vector
    
    async def encode_document(self, document_text: str) -> Dict[str, float]:
        """Encode document into a sparse vector representation.
        
        Args:
            document_text: The document text to encode
            
        Returns:
            Dictionary mapping term IDs to weights
        """
        # Check cache first if cache service is configured
        if self.cache_service_url:
            import hashlib
            doc_hash = hashlib.md5(document_text.encode()).hexdigest()
            cache_key = f"splade_doc:{doc_hash}"
            
            try:
                cache_response = await self.client.get(
                    f"{self.cache_service_url}/get",
                    params={"key": cache_key}
                )
                
                if cache_response.status_code == 200:
                    cached_data = cache_response.json().get("data")
                    if cached_data:
                        logger.info(f"Cache hit for SPLADE document encoding")
                        return cached_data
            except Exception as e:
                logger.warning(f"Error checking cache: {str(e)}")
        
        start_time = time.time()
        
        # For longer documents, split into chunks and aggregate
        max_chunk_size = self.max_seq_length - 20  # Leave room for special tokens
        
        # Simple chunking by sentences
        import re
        sentences = re.split(r'(?<=[.!?])\s+', document_text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        # If no chunks (unlikely), use the whole document
        if not chunks:
            chunks = [document_text]
        
        # Process each chunk
        all_weights = []
        
        for chunk in chunks:
            # Tokenize and convert to tensors
            inputs = self.tokenizer(
                chunk,
                max_length=self.max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Move tensors to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get SPLADE activations
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Apply log(1 + ReLU(logits))
                activations = torch.log(1 + F.relu(logits))
                
                # Apply max pooling over sequence dimension
                weights, _ = torch.max(activations, dim=1)
            
            # Store chunk weights
            all_weights.append(weights[0].cpu().numpy())
        
        # Aggregate weights from all chunks using max pooling
        if len(all_weights) > 1:
            aggregated_weights = np.max(np.stack(all_weights), axis=0)
        else:
            aggregated_weights = all_weights[0]
        
        # Create sparse vector (only include non-zero weights)
        sparse_vector = {}
        vocab = self.tokenizer.get_vocab()
        reverse_vocab = {v: k for k, v in vocab.items()}
        
        for idx, weight in enumerate(aggregated_weights):
            if weight > 0:
                token = reverse_vocab.get(idx)
                if token:
                    sparse_vector[token] = float(weight)
        
        logger.info(f"Document encoded in {time.time() - start_time:.2f}s with {len(sparse_vector)} non-zero dimensions")
        
        # Cache result if cache service is configured
        if self.cache_service_url:
            import hashlib
            doc_hash = hashlib.md5(document_text.encode()).hexdigest()
            cache_key = f"splade_doc:{doc_hash}"
            
            try:
                await self.client.post(
                    f"{self.cache_service_url}/set",
                    json={
                        "key": cache_key,
                        "value": sparse_vector,
                        "ttl": 3600  # 1 hour TTL
                    }
                )
            except Exception as e:
                logger.warning(f"Error caching SPLADE document: {str(e)}")
        
        return sparse_vector
    
    async def index_documents(
        self, 
        collection: str, 
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Index documents for SPLADE search.
        
        Args:
            collection: Collection name
            documents: List of documents to index
            
        Returns:
            Status information for the indexing operation
        """
        start_time = time.time()
        
        # Initialize collection index if it doesn't exist
        if collection not in self.indices:
            self.indices[collection] = {
                "document_vectors": {},
                "document_metadata": {},
                "inverted_index": {}
            }
        
        # Process each document
        doc_count = 0
        
        for doc in documents:
            doc_id = doc.get("id")
            if not doc_id:
                continue
                
            text = doc.get("text")
            if not text:
                continue
            
            # Encode document
            sparse_vector = await self.encode_document(text)
            
            # Add to document vectors
            self.indices[collection]["document_vectors"][doc_id] = sparse_vector
            
            # Add to document metadata
            self.indices[collection]["document_metadata"][doc_id] = {
                "id": doc_id,
                "metadata": doc.get("metadata", {}),
                "text": text[:1000]  # Store preview of text
            }
            
            # Update inverted index
            for term, weight in sparse_vector.items():
                if term not in self.indices[collection]["inverted_index"]:
                    self.indices[collection]["inverted_index"][term] = {}
                
                self.indices[collection]["inverted_index"][term][doc_id] = weight
            
            doc_count += 1
        
        # Save index
        self._save_index(collection, self.indices[collection])
        
        elapsed_time = time.time() - start_time
        
        return {
            "collection": collection,
            "documents_indexed": doc_count,
            "total_documents": len(self.indices[collection]["document_vectors"]),
            "unique_terms": len(self.indices[collection]["inverted_index"]),
            "elapsed_time": elapsed_time
        }
    
    async def search(
        self, 
        query_vector: Dict[str, float], 
        collection: str,
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """Perform search using sparse vectors.
        
        Args:
            query_vector: Sparse query vector
            collection: Collection to search in
            k: Number of results to return
            
        Returns:
            List of search results
        """
        # Check if collection exists
        if collection not in self.indices:
            logger.warning(f"Collection {collection} not found")
            return []
        
        # Check cache first if cache service is configured
        if self.cache_service_url:
            import hashlib
            import json
            
            # Generate a deterministic representation of inputs for caching
            query_hash = hashlib.md5(json.dumps(sorted(query_vector.items())).encode()).hexdigest()
            
            cache_key = f"splade_search:{collection}:{query_hash}:{k}"
            
            try:
                cache_response = await self.client.get(
                    f"{self.cache_service_url}/get",
                    params={"key": cache_key}
                )
                
                if cache_response.status_code == 200:
                    cached_data = cache_response.json().get("data")
                    if cached_data:
                        logger.info(f"Cache hit for SPLADE search")
                        return cached_data
            except Exception as e:
                logger.warning(f"Error checking cache: {str(e)}")
        
        start_time = time.time()
        
        # Access collection index components
        inverted_index = self.indices[collection]["inverted_index"]
        document_metadata = self.indices[collection]["document_metadata"]
        
        # Score documents
        doc_scores = {}
        
        # For each query term
        for term, q_weight in query_vector.items():
            # Skip if term not in index
            if term not in inverted_index:
                continue
                
            # For each document containing the term
            for doc_id, d_weight in inverted_index[term].items():
                # Accumulate score (simple dot product)
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0
                
                doc_scores[doc_id] += q_weight * d_weight
        
        # Sort by score
        sorted_docs = sorted(
            [(doc_id, score) for doc_id, score in doc_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Prepare results
        results = []
        
        for doc_id, score in sorted_docs[:k]:
            # Get document metadata
            doc_data = document_metadata.get(doc_id, {})
            
            # Add to results
            results.append({
                "id": doc_id,
                "score": float(score),
                "metadata": doc_data.get("metadata", {}),
                "text": doc_data.get("text", "")
            })
        
        logger.info(f"SPLADE search completed in {time.time() - start_time:.2f}s with {len(results)} results")
        
        # Cache results if cache service is configured
        if self.cache_service_url:
            import hashlib
            import json
            
            query_hash = hashlib.md5(json.dumps(sorted(query_vector.items())).encode()).hexdigest()
            
            cache_key = f"splade_search:{collection}:{query_hash}:{k}"
            
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
                logger.warning(f"Error caching SPLADE search results: {str(e)}")
        
        return results
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the service."""
        logger.info("Shutting down SPLADE Service")
        self.is_running = False
        await self.client.aclose()


# FastAPI specific code
def create_fastapi_app():
    """Create a FastAPI app for the SPLADE Service."""
    from fastapi import FastAPI, HTTPException
    import os
    
    app = FastAPI(title="RAG SPLADE Service", version="1.0.0")
    
    # Initialize service with environment variables or defaults
    service = SPLADEService(
        model_name=os.getenv("SPLADE_MODEL_NAME", "naver/splade-cocondenser-ensembledistil"),
        max_seq_length=int(os.getenv("MAX_SEQ_LENGTH", "256")),
        use_gpu=os.getenv("USE_GPU", "false").lower() == "true",
        lambda_d=float(os.getenv("LAMBDA_D", "0.0001")),
        index_dir=os.getenv("SPLADE_INDEX_DIR", "splade_indices"),
        cache_service_url=os.getenv("CACHE_SERVICE_URL")
    )
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return await service.health_check()
    
    @app.post("/encode_query")
    async def encode_query(request: EncodeQueryRequest):
        """Encode a query into a sparse vector."""
        response = await service.process_request(request)
        if response.status == "error":
            raise HTTPException(status_code=400, detail=response.message)
        return response
    
    @app.post("/encode_document")
    async def encode_document(request: EncodeDocumentRequest):
        """Encode a document into a sparse vector."""
        response = await service.process_request(request)
        if response.status == "error":
            raise HTTPException(status_code=400, detail=response.message)
        return response
    
    @app.post("/index")
    async def index_documents(collection: str, documents: List[Dict[str, Any]]):
        """Index documents for SPLADE search."""
        result = await service.index_documents(collection, documents)
        return {"status": "success", "data": result}
    
    @app.post("/search")
    async def search(request: SPLADESearchRequest):
        """Search using SPLADE sparse vectors."""
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