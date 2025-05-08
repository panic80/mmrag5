"""Fusion Service implementation.

This service is responsible for combining search results from different retrieval methods
and reranking them based on various strategies.
"""
import asyncio
import logging
import math
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import httpx
import numpy as np
from pydantic import BaseModel, Field

from microservices.service_interfaces import (
    FusionServiceInterface,
    ServiceRequest,
    ServiceResponse
)

# Configure logging
logger = logging.getLogger(__name__)


class FusionRequest(ServiceRequest):
    """Fusion request model."""
    vector_results: List[Dict[str, Any]]
    bm25_results: List[Dict[str, Any]]
    splade_results: Optional[List[Dict[str, Any]]] = None
    strategy: Dict[str, Any] = {"alpha": 0.5, "method": "linear"}


class RerankRequest(ServiceRequest):
    """Rerank request model."""
    results: List[Dict[str, Any]]
    query_text: str
    query_vector: Optional[List[float]] = None
    strategy: Optional[Dict[str, Any]] = None


class FusionResponse(ServiceResponse):
    """Fusion response model."""
    results: Optional[List[Dict[str, Any]]] = None


class FusionService(FusionServiceInterface):
    """Implementation of the Fusion Service."""
    
    def __init__(
        self,
        cache_service_url: Optional[str] = None,
        vector_service_url: Optional[str] = None
    ):
        """Initialize the fusion service.
        
        Args:
            cache_service_url: Optional URL for the Cache Service
            vector_service_url: Optional URL for the Vector Service (for embeddings)
        """
        self.cache_service_url = cache_service_url
        self.vector_service_url = vector_service_url
        
        # Initialize httpx client
        self.client = httpx.AsyncClient(timeout=30.0)
        self.is_running = True
        
        logger.info(f"Fusion Service initialized")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if the service is healthy and return status information."""
        status = {
            "status": "healthy",
            "version": "1.0.0",
            "fusion_methods": ["linear", "rrf", "softmax", "logit", "max", "min", "harmonic"],
            "rerank_methods": ["mmr", "context_aware", "diversity", "relevance"]
        }
        
        # Check cache service if configured
        if self.cache_service_url:
            try:
                response = await self.client.get(f"{self.cache_service_url}/health")
                status["cache_status"] = "connected" if response.status_code == 200 else "error"
            except Exception as e:
                status["cache_status"] = f"error: {str(e)}"
        
        # Check vector service if configured
        if self.vector_service_url:
            try:
                response = await self.client.get(f"{self.vector_service_url}/health")
                status["vector_status"] = "connected" if response.status_code == 200 else "error"
            except Exception as e:
                status["vector_status"] = f"error: {str(e)}"
        
        return status
    
    async def process_request(self, request: ServiceRequest) -> ServiceResponse:
        """Process a service request and return a response."""
        if isinstance(request, FusionRequest):
            try:
                results = await self.fuse_results(
                    vector_results=request.vector_results,
                    bm25_results=request.bm25_results,
                    strategy=request.strategy,
                    splade_results=request.splade_results
                )
                
                return FusionResponse(
                    request_id=request.request_id,
                    status="success",
                    data={"results": results}
                )
                
            except Exception as e:
                logger.error(f"Error in fusion: {str(e)}")
                return FusionResponse(
                    request_id=request.request_id,
                    status="error",
                    message=f"Error in fusion: {str(e)}"
                )
                
        elif isinstance(request, RerankRequest):
            try:
                results = await self.rerank(
                    results=request.results,
                    query_text=request.query_text,
                    query_vector=request.query_vector,
                    strategy=request.strategy
                )
                
                return FusionResponse(
                    request_id=request.request_id,
                    status="success",
                    data={"results": results}
                )
                
            except Exception as e:
                logger.error(f"Error in reranking: {str(e)}")
                return FusionResponse(
                    request_id=request.request_id,
                    status="error",
                    message=f"Error in reranking: {str(e)}"
                )
                
        else:
            return ServiceResponse(
                request_id=getattr(request, "request_id", "unknown"),
                status="error",
                message="Invalid request type"
            )
    
    async def fuse_results(
        self, 
        vector_results: List[Dict[str, Any]], 
        bm25_results: List[Dict[str, Any]],
        strategy: Dict[str, Any],
        splade_results: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Fuse results from different retrievers based on strategy.
        
        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            strategy: Fusion strategy parameters
            splade_results: Optional results from SPLADE search
            
        Returns:
            List of fused search results
        """
        # Check cache first if cache service is configured
        if self.cache_service_url:
            # Create a cache key based on the input data
            import hashlib
            import json
            
            # Generate deterministic string representation of inputs
            cache_input = {
                "vector_ids": [r.get("id") for r in vector_results],
                "bm25_ids": [r.get("id") for r in bm25_results],
                "splade_ids": [r.get("id") for r in (splade_results or [])],
                "strategy": strategy
            }
            
            cache_str = json.dumps(cache_input, sort_keys=True)
            cache_hash = hashlib.md5(cache_str.encode()).hexdigest()
            cache_key = f"fusion:{cache_hash}"
            
            try:
                cache_response = await self.client.get(
                    f"{self.cache_service_url}/get",
                    params={"key": cache_key}
                )
                
                if cache_response.status_code == 200:
                    cached_data = cache_response.json().get("data")
                    if cached_data:
                        logger.info(f"Cache hit for fusion")
                        return cached_data
            except Exception as e:
                logger.warning(f"Error checking cache: {str(e)}")
        
        # Extract strategy parameters
        alpha = strategy.get("alpha", 0.5)
        method = strategy.get("method", "linear")
        
        # Normalize alpha to be between 0 and 1
        alpha = max(0, min(1, alpha))
        
        # Create a mapping of document IDs to their combined scores
        doc_scores = {}
        
        # Helper function to normalize scores in a list of results
        def normalize_scores(results):
            if not results:
                return {}
            
            # Extract scores
            scores = [r.get("score", 0) for r in results]
            
            # Handle the case where all scores are the same
            if len(set(scores)) <= 1:
                return {r.get("id"): 1.0 for r in results}
            
            # Min-max normalization
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score
            
            # Create a mapping of document IDs to normalized scores
            return {
                r.get("id"): (r.get("score", 0) - min_score) / score_range
                for r in results
            }
        
        # Normalize scores for each result set
        vector_scores = normalize_scores(vector_results)
        bm25_scores = normalize_scores(bm25_results)
        splade_scores = normalize_scores(splade_results) if splade_results else {}
        
        # Create a set of all document IDs
        all_doc_ids = set(vector_scores.keys()) | set(bm25_scores.keys())
        if splade_results:
            all_doc_ids |= set(splade_scores.keys())
        
        # Apply fusion method
        if method == "linear":
            # Linear interpolation: alpha * vector_score + (1-alpha) * bm25_score
            for doc_id in all_doc_ids:
                vector_score = vector_scores.get(doc_id, 0)
                bm25_score = bm25_scores.get(doc_id, 0)
                splade_score = splade_scores.get(doc_id, 0)
                
                if splade_results:
                    # With SPLADE: alpha * vector_score + beta * bm25_score + gamma * splade_score
                    beta = (1 - alpha) * 0.6  # Allocate 60% of the remaining weight to BM25
                    gamma = (1 - alpha) * 0.4  # Allocate 40% of the remaining weight to SPLADE
                    score = alpha * vector_score + beta * bm25_score + gamma * splade_score
                else:
                    # Without SPLADE: alpha * vector_score + (1-alpha) * bm25_score
                    score = alpha * vector_score + (1 - alpha) * bm25_score
                
                doc_scores[doc_id] = score
        
        elif method == "rrf":
            # Reciprocal Rank Fusion
            
            # Create rank dictionaries for each result set
            vector_ranks = {doc_id: i+1 for i, doc_id in enumerate(sorted(
                vector_scores.keys(), key=lambda k: vector_scores[k], reverse=True))}
            
            bm25_ranks = {doc_id: i+1 for i, doc_id in enumerate(sorted(
                bm25_scores.keys(), key=lambda k: bm25_scores[k], reverse=True))}
            
            splade_ranks = {}
            if splade_results:
                splade_ranks = {doc_id: i+1 for i, doc_id in enumerate(sorted(
                    splade_scores.keys(), key=lambda k: splade_scores[k], reverse=True))}
            
            # RRF constant (typically 60)
            k = 60
            
            for doc_id in all_doc_ids:
                vector_rank = vector_ranks.get(doc_id, len(vector_scores) + 1)
                bm25_rank = bm25_ranks.get(doc_id, len(bm25_scores) + 1)
                
                if splade_results:
                    splade_rank = splade_ranks.get(doc_id, len(splade_scores) + 1)
                    # With SPLADE: combine all three ranks with weights
                    vector_rrf = 1 / (k + vector_rank)
                    bm25_rrf = 1 / (k + bm25_rank)
                    splade_rrf = 1 / (k + splade_rank)
                    
                    # Apply weights
                    score = alpha * vector_rrf + (1 - alpha) * 0.6 * bm25_rrf + (1 - alpha) * 0.4 * splade_rrf
                else:
                    # Without SPLADE: weighted combination of vector and BM25 RRF scores
                    vector_rrf = 1 / (k + vector_rank)
                    bm25_rrf = 1 / (k + bm25_rank)
                    score = alpha * vector_rrf + (1 - alpha) * bm25_rrf
                
                doc_scores[doc_id] = score
        
        elif method == "softmax":
            # Softmax fusion (emphasizes high scores)
            # Convert normalized scores to softmax probabilities
            
            def softmax(scores):
                # Temperature parameter (controls the "sharpness" of the distribution)
                temp = 10
                exp_scores = [math.exp(score * temp) for score in scores.values()]
                sum_exp = sum(exp_scores)
                if sum_exp > 0:
                    return {doc_id: math.exp(score * temp) / sum_exp 
                            for doc_id, score in scores.items()}
                else:
                    return {doc_id: 1.0 / len(scores) for doc_id in scores.keys()}
            
            vector_sm = softmax(vector_scores)
            bm25_sm = softmax(bm25_scores)
            splade_sm = softmax(splade_scores) if splade_results else {}
            
            for doc_id in all_doc_ids:
                vector_score = vector_sm.get(doc_id, 0)
                bm25_score = bm25_sm.get(doc_id, 0)
                
                if splade_results:
                    splade_score = splade_sm.get(doc_id, 0)
                    beta = (1 - alpha) * 0.6
                    gamma = (1 - alpha) * 0.4
                    score = alpha * vector_score + beta * bm25_score + gamma * splade_score
                else:
                    score = alpha * vector_score + (1 - alpha) * bm25_score
                
                doc_scores[doc_id] = score
        
        elif method == "max":
            # Take the maximum score for each document
            for doc_id in all_doc_ids:
                vector_score = vector_scores.get(doc_id, 0)
                bm25_score = bm25_scores.get(doc_id, 0)
                splade_score = splade_scores.get(doc_id, 0)
                
                scores = [vector_score, bm25_score]
                if splade_results:
                    scores.append(splade_score)
                
                doc_scores[doc_id] = max(scores)
        
        elif method == "harmonic":
            # Harmonic mean of scores (emphasizes documents that score well in multiple retrievers)
            for doc_id in all_doc_ids:
                vector_score = vector_scores.get(doc_id, 0.0001)  # Avoid division by zero
                bm25_score = bm25_scores.get(doc_id, 0.0001)
                
                if splade_results:
                    splade_score = splade_scores.get(doc_id, 0.0001)
                    # Weighted harmonic mean
                    inverse_sum = (alpha / vector_score) + ((1 - alpha) * 0.6 / bm25_score) + ((1 - alpha) * 0.4 / splade_score)
                    score = 1 / inverse_sum if inverse_sum > 0 else 0
                else:
                    # Weighted harmonic mean of vector and BM25
                    inverse_sum = (alpha / vector_score) + ((1 - alpha) / bm25_score)
                    score = 1 / inverse_sum if inverse_sum > 0 else 0
                
                doc_scores[doc_id] = score
        
        else:
            # Default to linear interpolation
            logger.warning(f"Unknown fusion method: {method}, using linear interpolation")
            for doc_id in all_doc_ids:
                vector_score = vector_scores.get(doc_id, 0)
                bm25_score = bm25_scores.get(doc_id, 0)
                doc_scores[doc_id] = alpha * vector_score + (1 - alpha) * bm25_score
        
        # Collect all document data
        doc_data = {}
        
        # Process vector results
        for result in vector_results:
            doc_id = result.get("id")
            if doc_id in all_doc_ids:
                doc_data[doc_id] = {
                    "id": doc_id,
                    "metadata": result.get("metadata", {}),
                    "text": result.get("text", ""),
                    "vector_score": vector_scores.get(doc_id, 0),
                    "scores": {"vector": vector_scores.get(doc_id, 0)}
                }
        
        # Process BM25 results
        for result in bm25_results:
            doc_id = result.get("id")
            if doc_id in all_doc_ids:
                if doc_id in doc_data:
                    doc_data[doc_id]["bm25_score"] = bm25_scores.get(doc_id, 0)
                    doc_data[doc_id]["scores"]["bm25"] = bm25_scores.get(doc_id, 0)
                    # Prefer text from BM25 as it might be the original text
                    if "text" in result:
                        doc_data[doc_id]["text"] = result["text"]
                else:
                    doc_data[doc_id] = {
                        "id": doc_id,
                        "metadata": result.get("metadata", {}),
                        "text": result.get("text", ""),
                        "bm25_score": bm25_scores.get(doc_id, 0),
                        "scores": {"bm25": bm25_scores.get(doc_id, 0)}
                    }
        
        # Process SPLADE results
        if splade_results:
            for result in splade_results:
                doc_id = result.get("id")
                if doc_id in all_doc_ids:
                    if doc_id in doc_data:
                        doc_data[doc_id]["splade_score"] = splade_scores.get(doc_id, 0)
                        doc_data[doc_id]["scores"]["splade"] = splade_scores.get(doc_id, 0)
                    else:
                        doc_data[doc_id] = {
                            "id": doc_id,
                            "metadata": result.get("metadata", {}),
                            "text": result.get("text", ""),
                            "splade_score": splade_scores.get(doc_id, 0),
                            "scores": {"splade": splade_scores.get(doc_id, 0)}
                        }
        
        # Create the final result list with combined scores
        results = []
        for doc_id, score in doc_scores.items():
            if doc_id in doc_data:
                doc = doc_data[doc_id]
                doc["score"] = score
                doc["method"] = method
                results.append(doc)
        
        # Sort results by score, highest first
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Cache results if cache service is configured
        if self.cache_service_url:
            import hashlib
            import json
            
            cache_input = {
                "vector_ids": [r.get("id") for r in vector_results],
                "bm25_ids": [r.get("id") for r in bm25_results],
                "splade_ids": [r.get("id") for r in (splade_results or [])],
                "strategy": strategy
            }
            
            cache_str = json.dumps(cache_input, sort_keys=True)
            cache_hash = hashlib.md5(cache_str.encode()).hexdigest()
            cache_key = f"fusion:{cache_hash}"
            
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
                logger.warning(f"Error caching fusion results: {str(e)}")
        
        return results
    
    async def rerank(
        self, 
        results: List[Dict[str, Any]], 
        query_text: str,
        query_vector: Optional[List[float]] = None,
        strategy: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Rerank results based on strategy.
        
        Args:
            results: List of search results to rerank
            query_text: The original query text
            query_vector: Optional query vector embedding
            strategy: Reranking strategy parameters
            
        Returns:
            List of reranked search results
        """
        if not results:
            return []
        
        # Use default strategy if none provided
        if not strategy:
            strategy = {"method": "mmr"}
        
        method = strategy.get("method", "mmr")
        
        # Check cache first if cache service is configured
        if self.cache_service_url:
            import hashlib
            import json
            
            # Generate deterministic string representation of inputs
            cache_input = {
                "result_ids": [r.get("id") for r in results],
                "query_text": query_text,
                "method": method
            }
            
            cache_str = json.dumps(cache_input, sort_keys=True)
            cache_hash = hashlib.md5(cache_str.encode()).hexdigest()
            cache_key = f"rerank:{cache_hash}"
            
            try:
                cache_response = await self.client.get(
                    f"{self.cache_service_url}/get",
                    params={"key": cache_key}
                )
                
                if cache_response.status_code == 200:
                    cached_data = cache_response.json().get("data")
                    if cached_data:
                        logger.info(f"Cache hit for reranking")
                        return cached_data
            except Exception as e:
                logger.warning(f"Error checking cache: {str(e)}")
        
        # Make a copy of results to avoid modifying the original
        reranked_results = results.copy()
        
        # Get query embedding if not provided and needed
        if method in ["mmr", "context_aware"] and not query_vector and self.vector_service_url:
            try:
                response = await self.client.post(
                    f"{self.vector_service_url}/embed",
                    json={"text": query_text}
                )
                
                if response.status_code == 200:
                    query_vector = response.json().get("embedding")
                else:
                    logger.warning(f"Failed to get query embedding: {response.text}")
            except Exception as e:
                logger.warning(f"Error getting query embedding: {str(e)}")
        
        # Apply reranking method
        if method == "mmr":
            # Maximum Marginal Relevance reranking
            lambda_param = strategy.get("lambda", 0.7)  # Balance between relevance and diversity
            
            # We need document vectors for MMR
            doc_vectors = []
            has_vectors = True
            
            for result in results:
                if "vector" in result:
                    doc_vectors.append(np.array(result["vector"]))
                else:
                    has_vectors = False
                    break
            
            if has_vectors and query_vector and len(doc_vectors) > 0:
                # Convert query vector to numpy array
                query_np = np.array(query_vector)
                
                # Calculate MMR scores
                mmr_indices = self._calculate_mmr(
                    query_vector=query_np,
                    doc_vectors=doc_vectors,
                    lambda_param=lambda_param
                )
                
                # Reorder results based on MMR indices
                reranked_results = [results[i] for i in mmr_indices]
                
                # Update MMR score
                for i, result in enumerate(reranked_results):
                    result["mmr_rank"] = i
                    # Blend original score with MMR rank
                    result["score"] = 0.7 * result["score"] + 0.3 * (1.0 - (i / len(results)))
            else:
                logger.warning("Document vectors not available for MMR reranking")
        
        elif method == "diversity":
            # Simple diversity-based reranking (ensure different document types appear)
            # Group documents by type or other metadata field
            grouped_docs = {}
            
            # Try to find a suitable field to group by (e.g., doc_type, source, category)
            group_field = None
            for field in ["doc_type", "source", "category", "content_type", "file_type"]:
                if any(field in r.get("metadata", {}) for r in results):
                    group_field = field
                    break
            
            if group_field:
                # Group documents
                for result in results:
                    group = result.get("metadata", {}).get(group_field, "unknown")
                    if group not in grouped_docs:
                        grouped_docs[group] = []
                    grouped_docs[group].append(result)
                
                # Interleave results from different groups
                reranked_results = []
                remaining = {g: docs.copy() for g, docs in grouped_docs.items()}
                
                while any(len(docs) > 0 for docs in remaining.values()):
                    for group, docs in sorted(remaining.items()):
                        if docs:
                            # Take the highest scored document from this group
                            reranked_results.append(docs.pop(0))
                
                # Update score based on new rank
                for i, result in enumerate(reranked_results):
                    result["diversity_rank"] = i
                    # Blend original score with diversity rank
                    result["score"] = 0.8 * result["score"] + 0.2 * (1.0 - (i / len(results)))
            else:
                logger.warning("No suitable field found for diversity reranking")
        
        elif method == "context_aware":
            # Context-aware reranking based on query and document text
            import re
            
            # Extract keywords from query
            query_words = set(re.findall(r'\b\w+\b', query_text.lower()))
            
            # Calculate context scores
            for result in reranked_results:
                text = result.get("text", "").lower()
                
                # Calculate keyword match score
                keyword_matches = sum(1 for word in query_words if word in text)
                keyword_score = keyword_matches / max(1, len(query_words))
                
                # Check for exact phrase matches
                phrases = re.findall(r'"([^"]*)"', query_text)
                phrase_score = 0
                if phrases:
                    phrase_matches = sum(1 for phrase in phrases if phrase.lower() in text)
                    phrase_score = phrase_matches / len(phrases)
                
                # Paragraph relevance (if text contains paragraphs)
                para_score = 0
                paragraphs = text.split("\n\n")
                if len(paragraphs) > 1:
                    para_scores = []
                    for para in paragraphs:
                        para_keyword_matches = sum(1 for word in query_words if word in para)
                        para_scores.append(para_keyword_matches / max(1, len(query_words)))
                    para_score = max(para_scores)
                
                # Calculate context score
                context_score = 0.5 * keyword_score + 0.3 * phrase_score + 0.2 * para_score
                
                # Update result score
                result["context_score"] = context_score
                result["score"] = 0.6 * result["score"] + 0.4 * context_score
            
            # Sort by updated score
            reranked_results.sort(key=lambda x: x["score"], reverse=True)
        
        elif method == "relevance":
            # Just resort by the original relevance score (no actual reranking)
            reranked_results.sort(key=lambda x: x["score"], reverse=True)
        
        else:
            logger.warning(f"Unknown reranking method: {method}, using original order")
        
        # Cache results if cache service is configured
        if self.cache_service_url:
            import hashlib
            import json
            
            cache_input = {
                "result_ids": [r.get("id") for r in results],
                "query_text": query_text,
                "method": method
            }
            
            cache_str = json.dumps(cache_input, sort_keys=True)
            cache_hash = hashlib.md5(cache_str.encode()).hexdigest()
            cache_key = f"rerank:{cache_hash}"
            
            try:
                await self.client.post(
                    f"{self.cache_service_url}/set",
                    json={
                        "key": cache_key,
                        "value": reranked_results,
                        "ttl": 300  # 5 minutes TTL
                    }
                )
            except Exception as e:
                logger.warning(f"Error caching reranking results: {str(e)}")
        
        return reranked_results
    
    def _calculate_mmr(
        self, 
        query_vector: np.ndarray, 
        doc_vectors: List[np.ndarray], 
        lambda_param: float = 0.7, 
        selected_indices: Optional[List[int]] = None
    ) -> List[int]:
        """Calculate Maximum Marginal Relevance (MMR).
        
        Args:
            query_vector: Query vector
            doc_vectors: List of document vectors
            lambda_param: Trade-off between relevance and diversity (higher means more relevance)
            selected_indices: Indices of documents already selected
            
        Returns:
            List of indices ordered by MMR score
        """
        if selected_indices is None:
            selected_indices = []
        
        # Initialize remaining indices
        remaining_indices = [i for i in range(len(doc_vectors)) if i not in selected_indices]
        
        # If no documents left to select, return already selected indices
        if not remaining_indices:
            return selected_indices
        
        # Normalize vectors for cosine similarity
        query_norm = query_vector / np.linalg.norm(query_vector)
        doc_norms = [v / np.linalg.norm(v) for v in doc_vectors]
        
        # Calculate relevance scores (cosine similarity to query)
        relevance_scores = [np.dot(query_norm, doc_norms[i]) for i in remaining_indices]
        
        # If no documents selected yet, select the most relevant one
        if not selected_indices:
            most_relevant_idx = remaining_indices[np.argmax(relevance_scores)]
            selected_indices.append(most_relevant_idx)
            return self._calculate_mmr(query_vector, doc_vectors, lambda_param, selected_indices)
        
        # Calculate diversity scores
        mmr_scores = []
        for i, idx in enumerate(remaining_indices):
            relevance = relevance_scores[i]
            
            # Calculate diversity (minimum similarity to already selected documents)
            diversity = float('inf')
            for sel_idx in selected_indices:
                similarity = np.dot(doc_norms[idx], doc_norms[sel_idx])
                diversity = min(diversity, 1 - similarity)
            
            # Calculate MMR score
            mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity
            mmr_scores.append(mmr_score)
        
        # Select the document with the highest MMR score
        max_mmr_idx = np.argmax(mmr_scores)
        selected_indices.append(remaining_indices[max_mmr_idx])
        
        # Recursively select the next document until all are selected
        if len(selected_indices) < len(doc_vectors):
            return self._calculate_mmr(query_vector, doc_vectors, lambda_param, selected_indices)
        else:
            return selected_indices
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the service."""
        logger.info("Shutting down Fusion Service")
        self.is_running = False
        await self.client.aclose()


# FastAPI specific code
def create_fastapi_app():
    """Create a FastAPI app for the Fusion Service."""
    from fastapi import FastAPI, HTTPException
    import os
    
    app = FastAPI(title="RAG Fusion Service", version="1.0.0")
    
    # Initialize service with environment variables or defaults
    service = FusionService(
        cache_service_url=os.getenv("CACHE_SERVICE_URL"),
        vector_service_url=os.getenv("VECTOR_SERVICE_URL")
    )
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return await service.health_check()
    
    @app.post("/fuse")
    async def fuse(request: FusionRequest):
        """Process a fusion request."""
        response = await service.process_request(request)
        if response.status == "error":
            raise HTTPException(status_code=400, detail=response.message)
        return response
    
    @app.post("/rerank")
    async def rerank(request: RerankRequest):
        """Process a rerank request."""
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