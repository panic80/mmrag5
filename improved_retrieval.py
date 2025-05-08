#!/usr/bin/env python3
"""
improved_retrieval.py

Enhanced retrieval components for the RAG pipeline with improved performance,
relevance, and architecture.

This module implements the high-priority improvements from the comprehensive
improvement plan, including:
1. Tiered caching
2. Parallel retrieval execution
3. Query-dependent fusion strategies
4. Incremental BM25 index building
5. Enhanced tokenization and preprocessing
6. Query analysis module
7. Modular component interfaces

Architecture:
- Retrieval components have clear interfaces (RetrieverInterface)
- Fusion components have dedicated interfaces (FusionInterface)
- Reranking components have dedicated interfaces (RerankerInterface)
- Factory classes provide consistent component creation
- High-level orchestration through RetrievalOrchestrator
"""
from __future__ import annotations

import os
import re
import json
import time
import uuid
import asyncio
import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union, Callable
from datetime import datetime

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, HasIdCondition

# Setup logging
logger = logging.getLogger("improved_retrieval")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

#############################################################################
# Caching Components
#############################################################################

class ResultsCache:
    """Multi-level caching system for RAG retrieval results with TTL support."""
    
    def __init__(self, ttl: int = 300):  # 5 minutes TTL by default
        """Initialize cache with specified TTL.
        
        Args:
            ttl: Time-to-live in seconds for cache entries
        """
        self.cache = {}
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
        logger.debug(f"Initialized ResultsCache with TTL={ttl} seconds")
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached result if available and not expired.
        
        Args:
            key: Cache key to look up
            
        Returns:
            Cached value if found and valid, None otherwise
        """
        if key in self.cache:
            result, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                self.hits += 1
                logger.debug(f"Cache hit for key: {key}")
                return result
            else:
                # Expired entry
                del self.cache[key]
                logger.debug(f"Cache entry expired for key: {key}")
        
        self.misses += 1
        logger.debug(f"Cache miss for key: {key}")
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Store value in cache with current timestamp.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        self.cache[key] = (value, time.time())
        logger.debug(f"Cached result for key: {key}")
    
    def invalidate(self, key_pattern: str = None) -> int:
        """Invalidate cache entries matching pattern.
        
        Args:
            key_pattern: Regex pattern to match keys for invalidation
                         If None, all entries are invalidated
        
        Returns:
            Number of invalidated entries
        """
        if key_pattern is None:
            count = len(self.cache)
            self.cache.clear()
            logger.info(f"Invalidated all {count} cache entries")
            return count
        
        pattern = re.compile(key_pattern)
        invalidated = []
        for key in list(self.cache.keys()):
            if pattern.search(key):
                invalidated.append(key)
                del self.cache[key]
        
        logger.info(f"Invalidated {len(invalidated)} cache entries matching '{key_pattern}'")
        return len(invalidated)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        total_requests = self.hits + self.misses
        hit_ratio = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": hit_ratio,
            "ttl": self.ttl
        }

# Global cache instances
results_cache = ResultsCache()
embedding_cache = {}  # Will be used with the lru_cache decorator

#############################################################################
# Vector Index Optimization Components
#############################################################################

class VectorIndexOptimizer:
    """Optimizer for vector index parameters and dimensionality."""
    
    def __init__(self, client: QdrantClient):
        """Initialize the vector index optimizer.
        
        Args:
            client: Qdrant client instance
        """
        self.client = client
        self.default_hnsw_params = {
            "m": 16,              # Number of connections per node (higher = better recall but more memory)
            "ef_construct": 100,  # Construction parameter (higher = better recall but slower indexing)
            "full_scan_threshold": 10000,  # When to switch from HNSW to full scan (small collections)
            "max_indexing_threads": 0,     # Number of threads (0 = auto)
        }
        logger.info("Initialized VectorIndexOptimizer")
    
    async def configure_index_parameters(
        self,
        collection: str,
        hnsw_params: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Configure HNSW parameters for a collection.
        
        Args:
            collection: Collection name
            hnsw_params: Custom HNSW parameters, or None for defaults
            
        Returns:
            Success flag
        """
        try:
            # Get current configuration
            collection_info = self.client.get_collection(collection_name=collection)
            logger.info(f"Current vector configuration for {collection}: {collection_info.config}")
            
            # Use defaults if not specified
            params = hnsw_params or self.default_hnsw_params
            logger.info(f"Setting HNSW parameters for {collection}: {params}")
            
            # Reconfigure vector params if supported by client
            if hasattr(self.client, "update_collection"):
                config_params = {f"hnsw_config.{k}": v for k, v in params.items()}
                self.client.update_collection(
                    collection_name=collection,
                    **config_params
                )
                logger.info(f"Successfully updated HNSW parameters for {collection}")
                return True
            else:
                logger.warning("Client does not support updating collection parameters")
                return False
        except Exception as e:
            logger.error(f"Failed to configure index parameters: {e}")
            return False
    
    async def apply_vector_quantization(
        self,
        collection: str,
        scalar_quantization: bool = True,
        quantization_config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Apply vector quantization to improve memory efficiency.
        
        Args:
            collection: Collection name
            scalar_quantization: Whether to use scalar quantization
            quantization_config: Custom quantization configuration
            
        Returns:
            Success flag
        """
        try:
            # Default scalar quantization config
            default_config = {
                "type": "scalar",
                "quantile": 0.99,      # Trim outliers
                "always_ram": True,    # Keep in RAM for better performance
            }
            
            # Use defaults if not specified
            config = quantization_config or default_config
            logger.info(f"Applying vector quantization to {collection}: {config}")
            
            # Apply quantization if supported by client
            if hasattr(self.client, "update_collection"):
                self.client.update_collection(
                    collection_name=collection,
                    quantization_config=config
                )
                logger.info(f"Successfully applied quantization to {collection}")
                return True
            else:
                logger.warning("Client does not support quantization")
                return False
        except Exception as e:
            logger.error(f"Failed to apply vector quantization: {e}")
            return False
    
    async def reduce_dimensions(
        self,
        vectors: List[List[float]],
        target_dim: int = 256,
        method: str = "pca"
    ) -> List[List[float]]:
        """Reduce dimensionality of vectors for efficient storage and retrieval.
        
        Args:
            vectors: List of embedding vectors
            target_dim: Target dimensionality
            method: Dimensionality reduction method ('pca' or 'random')
            
        Returns:
            List of reduced vectors
        """
        try:
            import numpy as np
            from sklearn.decomposition import PCA
            from sklearn.random_projection import GaussianRandomProjection
            
            # Convert to numpy array
            vectors_array = np.array(vectors)
            original_dim = vectors_array.shape[1]
            
            # Don't reduce if already smaller than target
            if original_dim <= target_dim:
                logger.info(f"Vectors already at {original_dim} dimensions, no reduction needed")
                return vectors
            
            logger.info(f"Reducing dimensions from {original_dim} to {target_dim} using {method}")
            
            # Apply dimensionality reduction
            if method == "pca":
                reducer = PCA(n_components=target_dim)
            else:  # random projection
                reducer = GaussianRandomProjection(n_components=target_dim)
                
            reduced_vectors = reducer.fit_transform(vectors_array)
            
            logger.info(f"Successfully reduced dimensions from {original_dim} to {target_dim}")
            return reduced_vectors.tolist()
        except ImportError:
            logger.error("Required packages not installed: scikit-learn, numpy")
            return vectors
        except Exception as e:
            logger.error(f"Dimension reduction failed: {e}")
            return vectors

# Instantiate optimizer (will be initialized when needed)
vector_optimizer = None

@lru_cache(maxsize=100)
def get_cached_embedding(text: str, model_name: str, client: Any) -> List[float]:
    """Get cached embedding or create a new one.
    
    Args:
        text: Text to embed
        model_name: Embedding model name
        client: OpenAI client
        
    Returns:
        Text embedding vector
    """
    # Create cache key based on text and model
    cache_key = f"{hash(text)}:{model_name}"
    
    # Check in-memory cache first for fastest retrieval
    if cache_key in embedding_cache:
        logger.debug(f"Found embedding in memory cache for text (length={len(text)})")
        return embedding_cache[cache_key]
    
    logger.debug(f"Getting embedding for text (length={len(text)}) with model {model_name}")
    
    # Generate embedding based on client type
    try:
        if hasattr(client, "embeddings"):  # OpenAI >=1.0 style
            resp = client.embeddings.create(model=model_name, input=[text])
            embedding = resp.data[0].embedding
        else:  # Legacy style
            resp = client.Embedding.create(model=model_name, input=[text])
            embedding = resp["data"][0]["embedding"]
        
        # Store in memory cache for future use
        embedding_cache[cache_key] = embedding
        
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        # Return empty embedding as fallback
        embedding_dim = 1536  # Default for most OpenAI models
        if "ada" in model_name:
            embedding_dim = 1024
        elif "3-small" in model_name:
            embedding_dim = 1536
        elif "3-large" in model_name:
            embedding_dim = 3072
            
        logger.warning(f"Returning zero vector with {embedding_dim} dimensions as fallback")
        return [0.0] * embedding_dim

#############################################################################
# Query Analysis Components
#############################################################################

class QueryAnalyzer:
    """Query Analysis Module for determining optimal retrieval strategies.
    
    This class encapsulates all query analysis logic to determine:
    - Query type and characteristics
    - Optimal retrieval strategy
    - Fusion parameters
    - Entity extraction for boosting
    """
    
    def __init__(self):
        # Define indicator words for different query types
        self.factual_indicators = [
            'who', 'what', 'when', 'where', 'which', 'how many',
            'list', 'name', 'find', 'identify', 'locate'
        ]
        
        self.conceptual_indicators = [
            'why', 'how', 'explain', 'describe', 'compare',
            'analyze', 'evaluate', 'summarize', 'define',
            'discuss', 'elaborate', 'examine'
        ]
        
        # Stopwords to ignore in entity detection
        self.stopwords = {
            'and', 'the', 'is', 'in', 'to', 'of', 'a', 'for',
            'with', 'on', 'at', 'by', 'an', 'this', 'that'
        }
    
    def detect_entities(self, query_text: str) -> List[str]:
        """Extract key entities from query for potential boosting.
        
        Args:
            query_text: The query text
            
        Returns:
            List of detected entities
        """
        entities = []
        
        # Extract quoted phrases as exact match entities
        quoted = re.findall(r'"([^"]*)"', query_text)
        entities.extend(quoted)
        
        # Look for capitalized phrases (potential named entities)
        capitalized = re.findall(r'\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\b', query_text)
        entities.extend(capitalized)
        
        # Find potential technical terms
        technical_terms = re.findall(r'\b([a-zA-Z]+(?:[_\-][a-zA-Z]+)+)\b', query_text)
        entities.extend(technical_terms)
        
        # Find numbers and dates
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', query_text)
        entities.extend(numbers)
        
        # Remove any entities that are just stopwords
        entities = [e for e in entities if e.lower() not in self.stopwords]
        
        return list(set(entities))  # Remove duplicates
    
    def determine_query_complexity(self, query_text: str) -> Dict[str, Any]:
        """Analyze query complexity based on various factors.
        
        Args:
            query_text: The query text
            
        Returns:
            Dictionary with complexity metrics
        """
        words = query_text.split()
        word_count = len(words)
        
        # Average word length as a proxy for complexity
        avg_word_length = sum(len(word) for word in words) / max(1, word_count)
        
        # Count sentences
        sentences = re.split(r'[.!?]+', query_text)
        sentence_count = sum(1 for s in sentences if s.strip())
        
        # Estimate logical complexity by counting logical operators
        logical_operators = ['and', 'or', 'not', 'if', 'unless', 'except', 'but', 'between']
        logical_complexity = sum(1 for word in words if word.lower() in logical_operators)
        
        # Check for question complexity
        question_indicators = ['what', 'why', 'how', 'when', 'where', 'who', 'which']
        question_complexity = sum(1 for word in words if word.lower() in question_indicators)
        
        # Look for temporal indicators which might require time-sensitive retrieval
        temporal_indicators = ['today', 'yesterday', 'recent', 'latest', 'current', 'now', 'week', 'month', 'year']
        temporal_references = sum(1 for word in words if word.lower() in temporal_indicators)
        
        # Check if query has numerical references which might benefit from exact matching
        has_numbers = bool(re.search(r'\d+', query_text))
        
        return {
            "word_count": word_count,
            "avg_word_length": avg_word_length,
            "sentence_count": sentence_count,
            "logical_complexity": logical_complexity,
            "question_complexity": question_complexity,
            "temporal_references": temporal_references,
            "has_numbers": has_numbers,
            "is_complex": word_count > 12 or logical_complexity > 1 or sentence_count > 1,
            "is_temporal": temporal_references > 0,
            "is_question": question_complexity > 0
        }
    
    def analyze_query(self, query_text: str, default_alpha: float = 0.5) -> Dict[str, Any]:
        """Enhanced query analysis for optimal retrieval parameters.
        
        This function determines:
        - Optimal fusion weights (vector vs BM25)
        - Key entities for potential boosting
        - Retrieval strategy based on query characteristics
        - Whether query expansion would be beneficial
        
        Args:
            query_text: The query text to analyze
            default_alpha: Default alpha value to use if analysis is inconclusive
            
        Returns:
            Dictionary containing analysis results
        """
        query_lower = query_text.lower()
        
        # Check for factual indicators (who, what, when, where, etc.)
        factual_score = sum(1 for word in self.factual_indicators if word in query_lower.split())
        
        # Check for conceptual indicators (explanations, concepts, reasoning)
        conceptual_score = sum(1 for word in self.conceptual_indicators if word in query_lower.split())
        
        # Detect quoted phrases (often exact matches)
        quoted_phrases = len(re.findall(r'"([^"]*)"', query_text))
        if quoted_phrases > 0:
            factual_score += quoted_phrases
        
        # Analyze query complexity
        complexity = self.determine_query_complexity(query_text)
        query_words = complexity["word_count"]
        
        # Longer queries tend to be more conceptual
        if complexity["is_complex"]:
            conceptual_score += 1
        
        # Detect entities for potential boosting
        entities = self.detect_entities(query_text)
        
        # Determine if query expansion would be beneficial
        # Short queries with few specific entities often benefit from expansion
        expansions_needed = query_words < 8 and len(entities) < 2 and quoted_phrases == 0
        
        # Determine retrieval strategy
        strategy = self.determine_retrieval_strategy(
            factual_score,
            conceptual_score,
            complexity,
            len(entities),
            quoted_phrases
        )
        
        # Determine query type and optimal alpha
        if factual_score > conceptual_score:
            query_type = "factual"
            alpha = max(0.3, default_alpha - 0.2)  # More weight to BM25
            metric = "hybrid_bm25_heavy"
            fusion_method = "rrf"  # Reciprocal Rank Fusion works well for factual queries
        elif conceptual_score > factual_score:
            query_type = "conceptual"
            alpha = min(0.7, default_alpha + 0.2)  # More weight to vectors
            metric = "hybrid_vector_heavy"
            fusion_method = "softmax"  # Softmax balances scores better for conceptual
        else:
            query_type = "balanced"
            alpha = default_alpha
            metric = "hybrid_balanced"
            fusion_method = "linear"  # Linear works well for balanced queries
        
        # Return comprehensive analysis
        return {
            "query_type": query_type,
            "alpha": alpha,
            "entities": entities,
            "metric": metric,
            "expansions_needed": expansions_needed,
            "fusion_method": fusion_method,
            "factual_score": factual_score,
            "conceptual_score": conceptual_score,
            "word_count": query_words,
            "complexity": complexity,
            "strategy": strategy
        }
    
    def determine_retrieval_strategy(
        self,
        factual_score: int,
        conceptual_score: int,
        complexity: Dict[str, Any],
        entity_count: int,
        quoted_count: int
    ) -> Dict[str, Any]:
        """Determine the optimal retrieval strategy based on query characteristics.
        
        Args:
            factual_score: Score indicating how factual the query is
            conceptual_score: Score indicating how conceptual the query is
            complexity: Query complexity metrics
            entity_count: Number of entities in the query
            quoted_count: Number of quoted phrases
            
        Returns:
            Dictionary with strategy parameters
        """
        # For simple factual queries with specific entities or quotes
        if factual_score > conceptual_score and (entity_count > 0 or quoted_count > 0):
            return {
                "primary_retriever": "bm25",
                "secondary_retriever": "vector",
                "fusion_weight": 0.3,  # Vector weight
                "fusion_method": "rrf",
                "rerank_threshold": 0.6
            }
            
        # For complex conceptual queries
        elif conceptual_score > factual_score and complexity["is_complex"]:
            return {
                "primary_retriever": "vector",
                "secondary_retriever": "bm25",
                "fusion_weight": 0.7,  # Vector weight
                "fusion_method": "softmax",
                "rerank_threshold": 0.4
            }
            
        # For balanced, moderately complex queries
        elif complexity["word_count"] > 8 and complexity["word_count"] < 20:
            return {
                "primary_retriever": "hybrid",
                "secondary_retriever": None,
                "fusion_weight": 0.5,  # Balanced
                "fusion_method": "linear",
                "rerank_threshold": 0.5
            }
            
        # For short queries with few clues
        elif complexity["word_count"] < 8 and entity_count < 2:
            return {
                "primary_retriever": "hybrid",
                "secondary_retriever": "expansion",  # Suggest query expansion
                "fusion_weight": 0.4,  # Slightly favor BM25 for short queries
                "fusion_method": "rrf",
                "rerank_threshold": 0.7
            }
            
        # Default balanced strategy
        else:
            return {
                "primary_retriever": "hybrid",
                "secondary_retriever": None,
                "fusion_weight": 0.5,  # Balanced
                "fusion_method": "linear",
                "rerank_threshold": 0.5
            }

# Global instance of QueryAnalyzer
query_analyzer = QueryAnalyzer()

# For backward compatibility
def detect_entities(query_text: str) -> List[str]:
    """Backward-compatible wrapper for entity detection.
    
    Args:
        query_text: The query text
        
    Returns:
        List of detected entities
    """
    return query_analyzer.detect_entities(query_text)

def analyze_query(query_text: str, default_alpha: float = 0.5) -> Dict[str, Any]:
    """Backward-compatible wrapper for query analysis.
    
    Args:
        query_text: The query text to analyze
        default_alpha: Default alpha value to use if analysis is inconclusive
        
    Returns:
        Dictionary containing analysis results
    """
    return query_analyzer.analyze_query(query_text, default_alpha)

def preprocess_query(query_text: str) -> str:
    """Advanced query preprocessing to improve retrieval effectiveness.
    
    This function:
    1. Removes common filler words
    2. Normalizes punctuation
    3. Standardizes question formats
    
    Args:
        query_text: The original query text
        
    Returns:
        Preprocessed query text
    """
    # Convert to lowercase
    query = query_text.lower()
    
    # Remove filler words unless they're part of quoted phrases
    filler_words = r'\b(um|uh|well|so|like|you know|i mean|actually)\b'
    
    # Preserve content in quotes
    quoted_parts = re.findall(r'"([^"]*)"', query)
    
    # Replace quotes with placeholders
    for i, part in enumerate(quoted_parts):
        query = query.replace(f'"{part}"', f'QUOTED_PART_{i}')
    
    # Remove filler words
    query = re.sub(filler_words, '', query)
    
    # Standardize question formats
    query = re.sub(r'^(can you|could you|please|would you)?\s*(tell me|show me|let me know|find|search for|find out|explain)\s*', '', query)
    
    # Restore quoted parts
    for i, part in enumerate(quoted_parts):
        query = query.replace(f'QUOTED_PART_{i}', f'"{part}"')
    
    # Normalize whitespace
    query = re.sub(r'\s+', ' ', query).strip()
    
    return query

#############################################################################
# BM25 Components
#############################################################################

def improved_tokenize(text: str) -> List[str]:
    """Enhanced tokenization for BM25 with better language handling.
    
    Args:
        text: Text to tokenize
        
    Returns:
        List of tokens
    """
    # Clean text
    text = re.sub(r'[^\w\s-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    
    # Split by whitespace
    tokens = text.split()
    
    # Remove stopwords (can be expanded)
    stopwords = {'and', 'the', 'is', 'in', 'to', 'of', 'a', 'for', 'with', 'on', 'at'}
    tokens = [t for t in tokens if t not in stopwords]
    
    return tokens

def build_incremental_bm25_index(
    collection: str, 
    index_path: str, 
    client: QdrantClient,
    filter_obj: Optional[Filter] = None, 
    batch_size: int = 1000
) -> Dict[str, str]:
    """Build BM25 index incrementally, avoiding full collection scan when possible.
    
    Args:
        collection: Name of the Qdrant collection
        index_path: Path to the BM25 index file
        client: Qdrant client instance
        filter_obj: Optional filter to apply when scrolling collection
        batch_size: Number of records to fetch in each batch
        
    Returns:
        Dictionary mapping point IDs to text for BM25 indexing
    """
    # Default path if none provided
    if index_path is None:
        index_path = f"{collection}_bm25_index.json"
        logger.info(f"Using default BM25 index path: {index_path}")
        
    # Try to load existing index
    existing_index = {}
    if os.path.exists(index_path):
        try:
            with open(index_path, "r") as f:
                existing_index = json.load(f)
            logger.info(f"Loaded {len(existing_index)} entries from existing BM25 index")
        except Exception as e:
            logger.warning(f"Failed to load existing index: {e}")
    
    # Get collection info to determine if we need to update
    try:
        collection_info = client.get_collection(collection_name=collection)
        if hasattr(collection_info, 'vectors_count'):
            total_vectors = collection_info.vectors_count
        else:
            total_vectors = getattr(collection_info, 'points_count', 0)
        
        # Ensure total_vectors is not None
        total_vectors = 0 if total_vectors is None else total_vectors
        logger.info(f"Collection has {total_vectors} vectors/points")
    except Exception as e:
        logger.warning(f"Failed to get collection info: {e}")
        total_vectors = 0
    
    # Skip if existing index covers all points
    if total_vectors > 0 and len(existing_index) >= total_vectors:
        logger.info(f"Existing BM25 index is up-to-date ({len(existing_index)} entries)")
        return existing_index
    
    # Track new entries for incremental updates
    logger.info(f"Building incremental BM25 index for collection '{collection}'")
    new_entries = 0
    offset = None
    
    # Use same metadata filter if provided
    scroll_filter = filter_obj
    
    # Process in batches
    while True:
        records, offset = client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=batch_size,
            offset=offset,
            with_payload=True,
        )
        
        if not records:
            break
        
        # Process this batch
        batch_new = 0
        for rec in records:
            # Skip if already in index
            if rec.id in existing_index:
                continue
                
            payload = getattr(rec, 'payload', {}) or {}
            text = payload.get("chunk_text")
            if isinstance(text, str) and text:
                existing_index[rec.id] = text
                new_entries += 1
                batch_new += 1
        
        if batch_new > 0:
            logger.info(f"Processed batch, added {batch_new} new entries")
        
        # Periodically save progress for large collections
        if new_entries > 0 and new_entries % (batch_size * 5) == 0:
            with open(index_path, "w") as f:
                json.dump(existing_index, f)
            logger.info(f"Saved progress: {new_entries} new entries indexed")
        
        # Check if we're done
        if offset is None:
            break
    
    # Final save if we added any new entries
    if new_entries > 0:
        with open(index_path, "w") as f:
            json.dump(existing_index, f)
        logger.info(f"Added {new_entries} new entries to BM25 index (total: {len(existing_index)})")
    
    return existing_index

#############################################################################
# Fusion Components and Interfaces
#############################################################################

class FusionInterface:
    """Interface for fusion components that combine results from multiple retrievers."""
    
    def fuse(
        self,
        vec_scores: Dict[str, float],
        bm25_scores: Dict[str, float],
        alpha: float = 0.5,
        **kwargs
    ) -> Dict[str, float]:
        """Fuse scores from different retrievers into a single ranking.
        
        Args:
            vec_scores: Vector scores by point ID
            bm25_scores: BM25 scores by point ID
            alpha: Weight for vector scores (0.0-1.0)
            **kwargs: Additional parameters specific to fusion method
            
        Returns:
            Dictionary of fused scores by point ID
        """
        raise NotImplementedError("Subclasses must implement fuse()")


class RRFusion(FusionInterface):
    """Reciprocal Rank Fusion implementation."""
    
    def fuse(
        self,
        vec_scores: Dict[str, float],
        bm25_scores: Dict[str, float],
        alpha: float = 0.5,
        **kwargs
    ) -> Dict[str, float]:
        """Apply Reciprocal Rank Fusion with weights.
        
        Args:
            vec_scores: Vector scores by point ID
            bm25_scores: BM25 scores by point ID
            alpha: Weight for vector scores (0.0-1.0)
            **kwargs: Additional parameters (rrf_k)
            
        Returns:
            Dictionary of fused scores by point ID
        """
        rrf_k = kwargs.get('rrf_k', 60.0)
        
        # Convert scores to ranks
        vec_ranks = {pid: rank+1 for rank, (pid, _) in
                    enumerate(sorted(vec_scores.items(), key=lambda x: x[1], reverse=True))}
        
        bm25_ranks = {pid: rank+1 for rank, (pid, _) in
                     enumerate(sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True))}
        
        # Apply RRF with weights
        fused_scores = {}
        all_ids = set(vec_ranks) | set(bm25_ranks)
        
        for pid in all_ids:
            vec_score = alpha * (1.0 / (rrf_k + vec_ranks.get(pid, len(vec_ranks) + 1)))
            bm25_score = (1.0 - alpha) * (1.0 / (rrf_k + bm25_ranks.get(pid, len(bm25_ranks) + 1)))
            fused_scores[pid] = vec_score + bm25_score
        
        return fused_scores


class SoftmaxFusion(FusionInterface):
    """Softmax-based fusion implementation."""
    
    def fuse(
        self,
        vec_scores: Dict[str, float],
        bm25_scores: Dict[str, float],
        alpha: float = 0.5,
        **kwargs
    ) -> Dict[str, float]:
        """Apply softmax-normalized score fusion.
        
        Args:
            vec_scores: Vector scores by point ID
            bm25_scores: BM25 scores by point ID
            alpha: Weight for vector scores (0.0-1.0)
            **kwargs: Additional parameters (temperature)
            
        Returns:
            Dictionary of fused scores by point ID
        """
        temperature = kwargs.get('temperature', 0.1)
        
        # Apply softmax normalization to each set of scores
        def softmax_normalize(scores, temp):
            if not scores:
                return {}
            values = np.array(list(scores.values()))
            exp_scores = np.exp(values / temp)
            normalized = exp_scores / np.sum(exp_scores)
            return {pid: float(norm) for pid, norm in zip(scores.keys(), normalized)}
        
        # Normalize scores
        norm_vec = softmax_normalize(vec_scores, temperature)
        norm_bm25 = softmax_normalize(bm25_scores, temperature)
        
        # Combine with weights
        fused_scores = {}
        all_ids = set(norm_vec) | set(norm_bm25)
        
        for pid in all_ids:
            fused_scores[pid] = alpha * norm_vec.get(pid, 0.0) + (1.0 - alpha) * norm_bm25.get(pid, 0.0)
        
        return fused_scores


class LogisticFusion(FusionInterface):
    """Logistic normalization fusion implementation."""
    
    def fuse(
        self,
        vec_scores: Dict[str, float],
        bm25_scores: Dict[str, float],
        alpha: float = 0.5,
        **kwargs
    ) -> Dict[str, float]:
        """Apply logistic normalization for fusion.
        
        Args:
            vec_scores: Vector scores by point ID
            bm25_scores: BM25 scores by point ID
            alpha: Weight for vector scores (0.0-1.0)
            **kwargs: Additional parameters (k, x0)
            
        Returns:
            Dictionary of fused scores by point ID
        """
        # Logistic function parameters
        k = kwargs.get('k', 1.0)   # Steepness
        x0 = kwargs.get('x0', 0.5) # Midpoint
        
        # Apply logistic normalization to each set of scores
        def logistic_normalize(scores, k_val, x0_val):
            if not scores:
                return {}
                
            # Min-max normalize first
            values = np.array(list(scores.values()))
            if len(values) <= 1:
                return {pid: 0.5 for pid in scores.keys()}
                
            min_val, max_val = np.min(values), np.max(values)
            range_val = max_val - min_val
            
            if range_val == 0:
                # Avoid division by zero
                normalized = np.full_like(values, 0.5)
            else:
                # Scale to [0,1]
                normalized = (values - min_val) / range_val
                
                # Apply logistic function for smoother normalization
                normalized = 1 / (1 + np.exp(-k_val * (normalized - x0_val)))
                
            return {pid: float(norm) for pid, norm in zip(scores.keys(), normalized)}
        
        # Normalize scores
        norm_vec = logistic_normalize(vec_scores, k, x0)
        norm_bm25 = logistic_normalize(bm25_scores, k, x0)
        
        # Combine with weights
        fused_scores = {}
        all_ids = set(norm_vec) | set(norm_bm25)
        
        for pid in all_ids:
            fused_scores[pid] = alpha * norm_vec.get(pid, 0.0) + (1.0 - alpha) * norm_bm25.get(pid, 0.0)
        
        return fused_scores


class EnsembleFusion(FusionInterface):
    """Ensemble-based fusion that combines multiple fusion methods."""
    
    def fuse(
        self,
        vec_scores: Dict[str, float],
        bm25_scores: Dict[str, float],
        alpha: float = 0.5,
        **kwargs
    ) -> Dict[str, float]:
        """Apply multiple fusion methods and average results.
        
        Args:
            vec_scores: Vector scores by point ID
            bm25_scores: BM25 scores by point ID
            alpha: Weight for vector scores (0.0-1.0)
            **kwargs: Additional parameters for component fusers
            
        Returns:
            Dictionary of fused scores by point ID
        """
        methods = kwargs.get('methods', ['linear', 'softmax', 'rrf'])
        method_weights = kwargs.get('method_weights', None)
        
        # Default to equal weights if not specified
        if method_weights is None:
            method_weights = {method: 1.0/len(methods) for method in methods}
        
        # Create fusers
        fusers = {
            'linear': LinearFusion(),
            'softmax': SoftmaxFusion(),
            'rrf': RRFusion(),
            'logistic': LogisticFusion()
        }
        
        # Perform fusion with each method
        all_fused_scores = {}
        for method in methods:
            if method not in fusers:
                logger.warning(f"Unknown fusion method: {method}, skipping")
                continue
                
            method_params = kwargs.get(f'{method}_params', {})
            fused = fusers[method].fuse(vec_scores, bm25_scores, alpha, **method_params)
            all_fused_scores[method] = fused
        
        # Combine all methods using weights
        final_scores = {}
        for method, fused in all_fused_scores.items():
            weight = method_weights.get(method, 1.0/len(methods))
            for pid, score in fused.items():
                if pid not in final_scores:
                    final_scores[pid] = 0.0
                final_scores[pid] += weight * score
        
        return final_scores


class LinearFusion(FusionInterface):
    """Linear min-max normalized fusion implementation."""
    
    def fuse(
        self,
        vec_scores: Dict[str, float],
        bm25_scores: Dict[str, float],
        alpha: float = 0.5,
        **kwargs
    ) -> Dict[str, float]:
        """Apply min-max normalized linear fusion.
        
        Args:
            vec_scores: Vector scores by point ID
            bm25_scores: BM25 scores by point ID
            alpha: Weight for vector scores (0.0-1.0)
            **kwargs: Additional parameters (unused)
            
        Returns:
            Dictionary of fused scores by point ID
        """
        # Min-max normalize vector scores
        if vec_scores:
            vec_min = min(vec_scores.values())
            vec_max = max(vec_scores.values())
            vec_range = vec_max - vec_min
            
            norm_vec = {
                pid: (score - vec_min) / vec_range if vec_range > 0 else 0.5
                for pid, score in vec_scores.items()
            }
        else:
            norm_vec = {}
        
        # Min-max normalize BM25 scores
        if bm25_scores:
            bm25_min = min(bm25_scores.values())
            bm25_max = max(bm25_scores.values())
            bm25_range = bm25_max - bm25_min
            
            norm_bm25 = {
                pid: (score - bm25_min) / bm25_range if bm25_range > 0 else 0.5
                for pid, score in bm25_scores.items()
            }
        else:
            norm_bm25 = {}
        
        # Combine with weights
        fused_scores = {}
        all_ids = set(norm_vec) | set(norm_bm25)
        
        for pid in all_ids:
            fused_scores[pid] = alpha * norm_vec.get(pid, 0.0) + (1.0 - alpha) * norm_bm25.get(pid, 0.0)
        
        return fused_scores


class FusionFactory:
    """Factory for creating fusion method instances based on strategy."""
    
    @staticmethod
    def create_fusion(method_name: str) -> FusionInterface:
        """Create fusion method instance by name.
        
        Args:
            method_name: Name of fusion method
            
        Returns:
            Fusion method instance
        """
        methods = {
            "rrf": RRFusion(),
            "softmax": SoftmaxFusion(),
            "linear": LinearFusion(),
            "logistic": LogisticFusion(),
            "ensemble": EnsembleFusion(),
        }
        
        if method_name not in methods:
            logger.warning(f"Unknown fusion method: {method_name}, using default (linear)")
            
        return methods.get(method_name, LinearFusion())


def normalized_fusion(
    vec_scores: Dict[str, float],
    bm25_scores: Dict[str, float],
    alpha: float = 0.5,
    method: str = "softmax"
) -> Dict[str, float]:
    """Normalized fusion with multiple method options.
    
    Args:
        vec_scores: Vector scores by point ID
        bm25_scores: BM25 scores by point ID
        alpha: Weight for vector scores (0.0-1.0)
        method: Fusion method to use
        
    Returns:
        Dictionary of fused scores by point ID
    """
    # Generate normalized scores with the specified method
    fusion_component = FusionFactory.create_fusion(method)
    
    # Method-specific parameters
    fusion_params = {}
    if method == "rrf":
        fusion_params["rrf_k"] = 60.0
    elif method == "softmax":
        fusion_params["temperature"] = 0.1
    elif method == "ensemble":
        fusion_params["methods"] = ["linear", "softmax", "rrf"]
        
    return fusion_component.fuse(vec_scores, bm25_scores, alpha, **fusion_params)


# For backward compatibility
def rrf_fusion(
    vec_scores: Dict[str, float],
    bm25_scores: Dict[str, float],
    alpha: float = 0.5,
    rrf_k: float = 60.0
) -> Dict[str, float]:
    """Backward-compatible wrapper for RRF fusion.
    
    Args:
        vec_scores: Vector scores by point ID
        bm25_scores: BM25 scores by point ID
        alpha: Weight for vector scores (0.0-1.0)
        rrf_k: RRF k parameter
        
    Returns:
        Dictionary of fused scores by point ID
    """
    fusion = RRFusion()
    return fusion.fuse(vec_scores, bm25_scores, alpha, rrf_k=rrf_k)


def softmax_fusion(
    vec_scores: Dict[str, float],
    bm25_scores: Dict[str, float],
    alpha: float = 0.5,
    temperature: float = 0.1
) -> Dict[str, float]:
    """Backward-compatible wrapper for softmax fusion.
    
    Args:
        vec_scores: Vector scores by point ID
        bm25_scores: BM25 scores by point ID
        alpha: Weight for vector scores (0.0-1.0)
        temperature: Temperature parameter for softmax
        
    Returns:
        Dictionary of fused scores by point ID
    """
    fusion = SoftmaxFusion()
    return fusion.fuse(vec_scores, bm25_scores, alpha, temperature=temperature)


def linear_fusion(
    vec_scores: Dict[str, float],
    bm25_scores: Dict[str, float],
    alpha: float = 0.5
) -> Dict[str, float]:
    """Backward-compatible wrapper for linear fusion.
    
    Args:
        vec_scores: Vector scores by point ID
        bm25_scores: BM25 scores by point ID
        alpha: Weight for vector scores (0.0-1.0)
        
    Returns:
        Dictionary of fused scores by point ID
    """
    fusion = LinearFusion()
    return fusion.fuse(vec_scores, bm25_scores, alpha)


def get_fusion_method(method_name: str) -> Callable:
    """Get fusion method function by name (backward compatibility).
    
    Args:
        method_name: Name of fusion method
        
    Returns:
        Fusion method function
    """
    methods = {
        "rrf": rrf_fusion,
        "softmax": softmax_fusion,
        "linear": linear_fusion,
        "normalized": normalized_fusion,
    }
    
    if method_name not in methods:
        logger.warning(f"Unknown fusion method: {method_name}, using default (linear)")
    
    return methods.get(method_name, linear_fusion)

#############################################################################
# Parallel Retrieval Components
#############################################################################

async def vector_search(
    vector: List[float], 
    collection: str, 
    k: int, 
    filter_obj: Optional[Filter], 
    client: QdrantClient
) -> List[Any]:
    """Perform vector search asynchronously.
    
    Args:
        vector: Query vector
        collection: Collection name
        k: Number of results to return
        filter_obj: Optional filter
        client: Qdrant client
        
    Returns:
        List of search results
    """
    # Create cache key
    cache_key = f"vector:{collection}:{k}:{str(vector[:5])}...:{str(filter_obj)}"
    cached = results_cache.get(cache_key)
    if cached is not None:
        logger.debug("Using cached vector search results")
        return cached
    
    # Perform search based on client capabilities
    try:
        if hasattr(client, "query_points"):
            # New API
            resp = client.query_points(
                collection_name=collection,
                query=vector,
                limit=k,
                with_payload=True,
                with_vectors=True,
                query_filter=filter_obj,
            )
            results = getattr(resp, 'points', [])
        else:
            # Fallback to deprecated search()
            results = client.search(
                collection_name=collection,
                query_vector=vector,
                limit=k,
                with_payload=True,
                with_vectors=True,
                query_filter=filter_obj,
            )
        
        # Cache results
        results_cache.set(cache_key, results)
        return results
    
    except Exception as e:
        logger.error(f"Vector search error: {e}")
        return []

async def bm25_search(
    query_text: str, 
    collection: str, 
    k: int, 
    filter_obj: Optional[Filter], 
    client: QdrantClient,
    bm25_index: Dict[str, str] = None,
    index_path: str = None,
    batch_size: int = 1000
) -> List[Any]:
    """Perform BM25 search asynchronously.
    
    Args:
        query_text: Query text
        collection: Collection name
        k: Number of results to return
        filter_obj: Optional filter
        client: Qdrant client
        bm25_index: Optional pre-loaded BM25 index
        index_path: Path to BM25 index file
        batch_size: Batch size for index building
        
    Returns:
        List of search results with scores
    """
    from rank_bm25 import BM25Okapi
    
    # Create cache key
    cache_key = f"bm25:{collection}:{k}:{query_text}:{str(filter_obj)}"
    cached = results_cache.get(cache_key)
    if cached is not None:
        logger.debug("Using cached BM25 search results")
        return cached
    
    # Build or load BM25 index if not provided
    if bm25_index is None:
        bm25_index = build_incremental_bm25_index(
            collection, 
            index_path or f"{collection}_bm25_index.json", 
            client,
            filter_obj,
            batch_size
        )
    
    # Tokenize corpus and query
    ids = list(bm25_index.keys())
    
    # Perform BM25 search
    try:
        # Tokenize corpus
        tokenized_corpus = [improved_tokenize(bm25_index[_id]) for _id in ids]
        if not tokenized_corpus or all(len(tokens) == 0 for tokens in tokenized_corpus):
            logger.warning("BM25 tokenization produced an empty corpus")
            return []
        
        # Create BM25 model
        bm25 = BM25Okapi(tokenized_corpus)
        
        # Tokenize query
        tokenized_query = improved_tokenize(query_text)
        
        # Get scores
        scores = bm25.get_scores(tokenized_query)
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:k]
        
        # Prepare results
        from types import SimpleNamespace
        results = []
        
        # Get payload for these points from Qdrant
        top_ids = [ids[i] for i in top_indices if i < len(ids)]
        
        # Batch retrieval of point data from Qdrant
        fobj = Filter(must=[HasIdCondition(has_id=top_ids)])
        records, _ = await asyncio.to_thread(
            client.scroll,
            collection_name=collection,
            scroll_filter=fobj,
            with_payload=True,
            limit=len(top_ids)
        )
        
        # Create result objects with scores
        records_by_id = {rec.id: rec for rec in records}
        
        for i, idx in enumerate(top_indices):
            if idx >= len(ids):
                continue
                
            point_id = ids[idx]
            score = float(scores[idx])
            
            if point_id in records_by_id:
                record = records_by_id[point_id]
                result = SimpleNamespace(
                    id=point_id,
                    score=score,
                    payload=getattr(record, 'payload', {}) or {},
                    vector=getattr(record, 'vector', None),
                )
                results.append(result)
        
        # Cache results
        results_cache.set(cache_key, results)
        return results
        
    except Exception as e:
        logger.error(f"BM25 search error: {e}")
        return []

async def parallel_retrieval(
    query_text: str, 
    vector: List[float], 
    collection: str, 
    k: int, 
    filter_obj: Optional[Filter], 
    client: QdrantClient,
    bm25_index: Dict[str, str] = None,
    index_path: str = None
) -> Tuple[List[Any], List[Any]]:
    """Run vector and BM25 searches in parallel.
    
    Args:
        query_text: Query text
        vector: Query vector
        collection: Collection name
        k: Number of results to return
        filter_obj: Optional filter
        client: Qdrant client
        bm25_index: Optional pre-loaded BM25 index
        index_path: Path to BM25 index file
        
    Returns:
        Tuple of (vector_results, bm25_results)
    """
    # Create combined cache key
    cache_key = f"parallel:{collection}:{k}:{query_text}:{str(vector[:5])}...:{str(filter_obj)}"
    cached = results_cache.get(cache_key)
    if cached is not None:
        logger.debug("Using cached parallel retrieval results")
        return cached
    
    # Run both searches concurrently
    vector_results, bm25_results = await asyncio.gather(
        vector_search(vector, collection, k, filter_obj, client),
        bm25_search(query_text, collection, k, filter_obj, client, bm25_index, index_path)
    )
    
    # Cache combined results
    results = (vector_results, bm25_results)
    results_cache.set(cache_key, results)
    
    return results

#############################################################################
# Interfaces for Modular Component Architecture
#############################################################################

class RetrieverInterface:
    """Interface for retrieval components that fetch documents based on queries."""
    
    async def retrieve(
        self,
        query: str,
        query_vector: List[float],
        collection: str,
        k: int,
        filter_obj: Optional[Filter] = None
    ) -> List[Any]:
        """Retrieve documents based on query.
        
        Args:
            query: Query text
            query_vector: Query embedding vector
            collection: Collection name
            k: Number of results to return
            filter_obj: Optional filter
            
        Returns:
            List of retrieved documents
        """
        raise NotImplementedError("Subclasses must implement retrieve()")


class VectorRetriever(RetrieverInterface):
    """Vector-based retrieval component."""
    
    def __init__(self, client: QdrantClient):
        """Initialize vector retriever.
        
        Args:
            client: Qdrant client instance
        """
        self.client = client
        self.name = "vector"
        logger.info(f"Initialized {self.name} retriever")
    
    async def retrieve(
        self,
        query: str,
        query_vector: List[float],
        collection: str,
        k: int,
        filter_obj: Optional[Filter] = None
    ) -> List[Any]:
        """Perform vector retrieval.
        
        Args:
            query: Query text (not used in this retriever)
            query_vector: Query embedding vector
            collection: Collection name
            k: Number of results to return
            filter_obj: Optional filter
            
        Returns:
            List of retrieved documents
        """
        logger.debug(f"Vector retrieval for collection '{collection}', k={k}")
        return await vector_search(
            query_vector, collection, k, filter_obj, self.client
        )


class BM25Retriever(RetrieverInterface):
    """BM25-based retrieval component."""
    
    def __init__(self, client: QdrantClient, index_path: Optional[str] = None):
        """Initialize BM25 retriever.
        
        Args:
            client: Qdrant client instance
            index_path: Path to BM25 index file
        """
        self.client = client
        self.index_path = index_path
        self.bm25_index = None
        self.name = "bm25"
        logger.info(f"Initialized {self.name} retriever with index_path={index_path}")
    
    async def retrieve(
        self,
        query: str,
        query_vector: List[float],
        collection: str,
        k: int,
        filter_obj: Optional[Filter] = None
    ) -> List[Any]:
        """Perform BM25 retrieval.
        
        Args:
            query: Query text
            query_vector: Query embedding vector (not used in this retriever)
            collection: Collection name
            k: Number of results to return
            filter_obj: Optional filter
            
        Returns:
            List of retrieved documents
        """
        logger.debug(f"BM25 retrieval for collection '{collection}', k={k}")
        return await bm25_search(
            query, collection, k, filter_obj, self.client,
            self.bm25_index, self.index_path
        )


class HybridRetriever(RetrieverInterface):
    """Hybrid retrieval combining vector and BM25 approaches."""
    
    def __init__(
        self,
        client: QdrantClient,
        openai_client: Any,
        index_path: Optional[str] = None
    ):
        """Initialize hybrid retriever.
        
        Args:
            client: Qdrant client instance
            openai_client: OpenAI client instance
            index_path: Path to BM25 index file
        """
        self.client = client
        self.openai_client = openai_client
        self.index_path = index_path
        self.vector_retriever = VectorRetriever(client)
        self.bm25_retriever = BM25Retriever(client, index_path)
        self.name = "hybrid"
        logger.info(f"Initialized {self.name} retriever with index_path={index_path}")
    
    async def retrieve(
        self,
        query: str,
        query_vector: List[float],
        collection: str,
        k: int,
        filter_obj: Optional[Filter] = None
    ) -> List[Any]:
        """Perform hybrid retrieval.
        
        Args:
            query: Query text
            query_vector: Query embedding vector
            collection: Collection name
            k: Number of results to return
            filter_obj: Optional filter
            
        Returns:
            Tuple of (vector_results, bm25_results)
        """
        logger.debug(f"Hybrid retrieval for collection '{collection}', k={k}")
        return await parallel_retrieval(
            query, query_vector, collection, k, filter_obj,
            self.client, None, self.index_path
        )


class RetrieverFactory:
    """Factory for creating retriever instances based on strategy."""
    
    @staticmethod
    def create_retriever(
        strategy: str,
        client: QdrantClient,
        openai_client: Any = None,
        index_path: Optional[str] = None
    ) -> RetrieverInterface:
        """Create retriever instance based on strategy.
        
        Args:
            strategy: Retrieval strategy ('vector', 'bm25', or 'hybrid')
            client: Qdrant client instance
            openai_client: OpenAI client instance (required for hybrid)
            index_path: Path to BM25 index file
            
        Returns:
            Retriever instance
        """
        logger.info(f"Creating retriever for strategy: {strategy}")
        if strategy == "vector":
            return VectorRetriever(client)
        elif strategy == "bm25":
            return BM25Retriever(client, index_path)
        elif strategy == "hybrid":
            if openai_client is None:
                raise ValueError("OpenAI client required for hybrid retrieval")
            return HybridRetriever(client, openai_client, index_path)
        else:
            logger.warning(f"Unknown retrieval strategy: {strategy}, falling back to hybrid")
            if openai_client is None:
                raise ValueError("OpenAI client required for hybrid retrieval")
            return HybridRetriever(client, openai_client, index_path)


#############################################################################
# High-Level Orchestration
#############################################################################

class RetrievalOrchestrator:
    """Orchestrates the retrieval process using modular components."""
    
    def __init__(
        self,
        client: QdrantClient,
        openai_client: Any,
    ):
        """Initialize orchestrator.
        
        Args:
            client: Qdrant client instance
            openai_client: OpenAI client instance
        """
        self.client = client
        self.openai_client = openai_client
        self.query_analyzer = query_analyzer  # Use global instance
        self.fusion_factory = FusionFactory()
        self.retriever_factory = RetrieverFactory()
        self.reranker_factory = RerankerFactory()
        logger.info("Initialized RetrievalOrchestrator")
    
    async def retrieve_and_fuse(
        self,
        query_text: str,
        collection: str,
        model: str = "text-embedding-3-large",
        k: int = 20,
        filter_obj: Optional[Filter] = None,
        alpha: float = 0.5,
        rrf_k: float = 60.0,
        fusion_method: str = "auto",
        bm25_index: Dict[str, str] = None,
        index_path: str = None
    ) -> List[Any]:
        """Enhanced retrieval with parallel execution and smart fusion.
        
        Args:
            query_text: Query text
            collection: Collection name
            model: Embedding model name
            k: Number of results to return
            filter_obj: Optional filter
            alpha: Weight for vector scores in fusion
            rrf_k: RRF k parameter
            fusion_method: Fusion method (auto, rrf, linear, softmax)
            bm25_index: Optional pre-loaded BM25 index
            index_path: Path to BM25 index file
            
        Returns:
            List of fused search results
        """
        start_time = time.time()
        logger.info(f"Starting retrieval for query: '{query_text}'")
        
        # Create cache key for final results
        cache_key = (
            f"fused:{collection}:{k}:{query_text}:{str(filter_obj)}:"
            f"{alpha}:{rrf_k}:{fusion_method}"
        )
        cached = results_cache.get(cache_key)
        if cached is not None:
            logger.info("Using cached fused results")
            return cached
            
        # Preprocess query to improve retrieval effectiveness
        processed_query = preprocess_query(query_text)
        if processed_query != query_text:
            logger.debug(f"Preprocessed query: '{processed_query}' (original: '{query_text}')")
        
        # Analyze query to determine optimal parameters
        query_analysis = self.query_analyzer.analyze_query(processed_query, alpha)
        
        # Get query-dependent alpha and fusion method if auto selected
        query_alpha = query_analysis["alpha"]
        actual_fusion_method = fusion_method
        if fusion_method == "auto":
            actual_fusion_method = query_analysis["fusion_method"]
        
        logger.info(f"Query analysis: type={query_analysis['query_type']}, "
                    f"alpha={query_alpha:.2f}, fusion={actual_fusion_method}")
        
        # Log detected entities if any
        if query_analysis.get("entities"):
            entity_str = ", ".join(query_analysis["entities"])
            logger.debug(f"Detected entities: {entity_str}")
        
        # Select retrieval strategy based on query analysis
        strategy = query_analysis.get("strategy", {})
        primary_strategy = strategy.get("primary_retriever", "hybrid")
        logger.info(f"Selected primary retrieval strategy: {primary_strategy}")
        
        # Get query embedding
        vector = await asyncio.to_thread(
            get_cached_embedding,
            processed_query,
            model,
            self.openai_client
        )
        
        # Record embedding generation time
        embedding_time = time.time()
        logger.debug(f"Embedding generation took {embedding_time - start_time:.3f}s")
        
        # Create appropriate retriever based on strategy
        retriever = self.retriever_factory.create_retriever(
            primary_strategy,
            self.client,
            self.openai_client,
            index_path
        )
        
        # Execute retrieval with timing
        retrieval_start = time.time()
        if primary_strategy == "hybrid":
            vector_results, bm25_results = await retriever.retrieve(
                processed_query, vector, collection, k, filter_obj
            )
        else:
            # For non-hybrid strategies, we still run both retrievers for fusion
            vector_retriever = VectorRetriever(self.client)
            bm25_retriever = BM25Retriever(self.client, index_path)
            
            vector_results, bm25_results = await asyncio.gather(
                vector_retriever.retrieve(processed_query, vector, collection, k, filter_obj),
                bm25_retriever.retrieve(processed_query, vector, collection, k, filter_obj)
            )
        
        # Record retrieval time
        retrieval_time = time.time()
        logger.debug(f"Retrieval took {retrieval_time - retrieval_start:.3f}s")
        logger.info(f"Retrieved {len(vector_results)} vector results and {len(bm25_results)} BM25 results")
        
        # Extract scores for fusion
        vector_scores = {getattr(r, 'id', None): getattr(r, 'score', 0.0) for r in vector_results if hasattr(r, 'id')}
        bm25_scores = {getattr(r, 'id', None): getattr(r, 'score', 0.0) for r in bm25_results if hasattr(r, 'id')}
        
        # Create fusion component and perform fusion
        fusion_component = self.fusion_factory.create_fusion(actual_fusion_method)
        fusion_start = time.time()
        
        # Fusion parameters based on method
        fusion_params = {}
        if actual_fusion_method == "rrf":
            fusion_params["rrf_k"] = rrf_k
        
        fused_scores = fusion_component.fuse(
            vector_scores,
            bm25_scores,
            query_alpha,
            **fusion_params
        )
        
        # Record fusion time
        fusion_time = time.time()
        logger.debug(f"Fusion took {fusion_time - retrieval_time:.3f}s")
        
        # Create result map for lookups
        results_map = {}
        for r in vector_results + bm25_results:
            if not hasattr(r, 'id'):
                continue
            results_map[r.id] = r
        
        # Sort by fused score and create final results
        from types import SimpleNamespace
        fused_results = []
        
        for pid, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:k]:
            if pid in results_map:
                r = results_map[pid]
                result = SimpleNamespace(
                    id=pid,
                    score=score,
                    payload=getattr(r, 'payload', {}) or {},
                    vector=getattr(r, 'vector', None),
                )
                fused_results.append(result)
        
        # Cache fused results
        results_cache.set(cache_key, fused_results)
        
        return fused_results

# For backward compatibility - use orchestrator internally but present same interface
async def retrieve_and_fuse(
    query_text: str,
    collection: str,
    client: QdrantClient,
    openai_client: Any,
    model: str = "text-embedding-3-large",
    k: int = 20,
    filter_obj: Optional[Filter] = None,
    alpha: float = 0.5,
    rrf_k: float = 60.0,
    fusion_method: str = "auto",
    bm25_index: Dict[str, str] = None,
    index_path: str = None
) -> List[Any]:
    """Enhanced retrieval with parallel execution and smart fusion.
    
    Args:
        query_text: Query text
        collection: Collection name
        client: Qdrant client
        openai_client: OpenAI client
        model: Embedding model name
        k: Number of results to return
        filter_obj: Optional filter
        alpha: Weight for vector scores in fusion
        rrf_k: RRF k parameter
        fusion_method: Fusion method (auto, rrf, linear, softmax)
        bm25_index: Optional pre-loaded BM25 index
        index_path: Path to BM25 index file
        
    Returns:
        List of fused search results
    """
    orchestrator = RetrievalOrchestrator(client, openai_client)
    return await orchestrator.retrieve_and_fuse(
        query_text,
        collection,
        model,
        k,
        filter_obj,
        alpha,
        rrf_k,
        fusion_method,
        bm25_index,
        index_path
    )

#############################################################################
# Reranking Components and Interfaces
#############################################################################

#############################################################################
# Asynchronous Processing Pipeline
#############################################################################

class AsyncRetrievalPipeline:
    """Asynchronous processing pipeline for the entire retrieval process."""
    
    def __init__(
        self,
        client: QdrantClient,
        openai_client: Any,
        default_model: str = "text-embedding-3-large",
        default_collection: str = "rag_data",
        cache_ttl: int = 300,  # 5 minutes
        timeout: float = 10.0   # Default timeout in seconds
    ):
        """Initialize the asynchronous retrieval pipeline.
        
        Args:
            client: Qdrant client
            openai_client: OpenAI client
            default_model: Default embedding model
            default_collection: Default collection name
            cache_ttl: Cache TTL in seconds
            timeout: Default operation timeout in seconds
        """
        self.client = client
        self.openai_client = openai_client
        self.default_model = default_model
        self.default_collection = default_collection
        self.orchestrator = RetrievalOrchestrator(client, openai_client)
        self.query_analyzer = QueryAnalyzer()
        self.strategy_router = QueryStrategyRouter(self.query_analyzer)
        self.timeout = timeout
        self.performance_metrics = PerformanceMetrics()
        
        # Configure caches with provided TTL
        global results_cache
        results_cache = ResultsCache(ttl=cache_ttl)
        
        logger.info(f"Initialized AsyncRetrievalPipeline with timeout={timeout}s")
    
    async def process_query(
        self,
        query_text: str,
        collection: Optional[str] = None,
        model: Optional[str] = None,
        k: int = 20,
        filter_obj: Optional[Filter] = None,
        alpha: Optional[float] = None,
        fusion_method: Optional[str] = None,
        rerank_options: Optional[Dict[str, Any]] = None,
        max_retries: int = 2,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Process a query through the entire retrieval pipeline asynchronously.
        
        Args:
            query_text: The query text
            collection: Collection name (or default if None)
            model: Embedding model (or default if None)
            k: Number of results to return
            filter_obj: Optional filter
            alpha: Optional fusion weight (None for query-dependent)
            fusion_method: Optional fusion method (None for query-dependent)
            rerank_options: Optional reranking parameters
            max_retries: Maximum number of retries
            timeout: Operation timeout (or default if None)
            
        Returns:
            Dictionary with search results and performance metrics
        """
        # Set defaults
        actual_collection = collection or self.default_collection
        actual_model = model or self.default_model
        actual_timeout = timeout or self.timeout
        start_time = time.time()
        
        # Apply performance tracking
        process_id = self.performance_metrics.start_operation("process_query", {
            "query": query_text,
            "collection": actual_collection,
            "k": k
        })
        
        # Check cache first
        cache_key = (
            f"full_query:{query_text}:{actual_collection}:{k}:"
            f"{actual_model}:{alpha}:{fusion_method}:{hash(str(filter_obj))}"
        )
        cached_result = results_cache.get(cache_key)
        if cached_result:
            self.performance_metrics.end_operation(process_id, "cache_hit")
            logger.info(f"Cache hit for query: '{query_text}'")
            return cached_result
        
        try:
            # Process the query with timeout
            async with asyncio.timeout(actual_timeout):
                # Step 1: Preprocess query
                self.performance_metrics.start_operation("preprocess", {"process_id": process_id})
                processed_query = self.preprocess_query(query_text)
                self.performance_metrics.end_operation("preprocess")
                
                # Step 2: Run query analysis
                self.performance_metrics.start_operation("analyze", {"process_id": process_id})
                query_analysis = self.query_analyzer.analyze_query(processed_query)
                self.performance_metrics.end_operation("analyze")
                
                # Step 3: Determine retrieval strategy
                self.performance_metrics.start_operation("route", {"process_id": process_id})
                strategy = self.strategy_router.route_query(processed_query, query_analysis)
                self.performance_metrics.end_operation("route")
                
                # Step 4: Get query embedding
                self.performance_metrics.start_operation("embed", {"process_id": process_id})
                embedding_future = asyncio.to_thread(
                    get_cached_embedding,
                    processed_query,
                    actual_model,
                    self.openai_client
                )
                query_embedding = await embedding_future
                self.performance_metrics.end_operation("embed")
                
                # Step 5: Perform retrieval based on strategy
                self.performance_metrics.start_operation("retrieve", {"process_id": process_id})
                
                # Use determined values for fusion parameters
                actual_alpha = alpha if alpha is not None else strategy.get("fusion_weight", 0.5)
                actual_fusion_method = fusion_method or strategy.get("fusion_method", "linear")
                
                retrieval_options = {
                    "alpha": actual_alpha,
                    "fusion_method": actual_fusion_method,
                    "k": k,
                    "filter_obj": filter_obj
                }
                
                # Perform retrieval with retries
                for attempt in range(max_retries + 1):
                    try:
                        results = await self.orchestrator.retrieve_and_fuse(
                            query_text=processed_query,
                            collection=actual_collection,
                            model=actual_model,
                            **retrieval_options
                        )
                        break
                    except Exception as e:
                        if attempt < max_retries:
                            logger.warning(f"Retrieval attempt {attempt+1} failed: {e}, retrying...")
                            await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                        else:
                            raise
                
                self.performance_metrics.end_operation("retrieve")
                
                # Step 6: Apply reranking if requested
                if rerank_options:
                    self.performance_metrics.start_operation("rerank", {"process_id": process_id})
                    
                    rerank_method = rerank_options.get("method", "mmr")
                    reranker = RerankerFactory.create_reranker(rerank_method)
                    
                    results = await reranker.rerank(
                        results,
                        processed_query,
                        query_embedding,
                        **rerank_options
                    )
                    
                    self.performance_metrics.end_operation("rerank")
                
                # Prepare final result
                final_result = {
                    "results": results,
                    "query": query_text,
                    "processed_query": processed_query,
                    "strategy": strategy,
                    "metrics": self.performance_metrics.get_metrics(process_id),
                    "total_time": time.time() - start_time
                }
                
                # Cache result
                results_cache.set(cache_key, final_result)
                
                # Mark operation as complete
                self.performance_metrics.end_operation(process_id)
                
                return final_result
                
        except asyncio.TimeoutError:
            logger.error(f"Query processing timed out after {actual_timeout}s: '{query_text}'")
            self.performance_metrics.log_error(process_id, "timeout")
            return {
                "error": "timeout",
                "message": f"Query processing timed out after {actual_timeout}s",
                "query": query_text,
                "metrics": self.performance_metrics.get_metrics(process_id),
                "results": []
            }
            
        except Exception as e:
            logger.error(f"Error in query processing: {str(e)}")
            self.performance_metrics.log_error(process_id, str(e))
            return {
                "error": "processing_error",
                "message": str(e),
                "query": query_text,
                "metrics": self.performance_metrics.get_metrics(process_id),
                "results": []
            }
    
    def preprocess_query(self, query_text: str) -> str:
        """Apply query preprocessing.
        
        Args:
            query_text: Original query
            
        Returns:
            Preprocessed query
        """
        return preprocess_query(query_text)


class QueryStrategyRouter:
    """Router to determine the best retrieval strategy based on query analysis."""
    
    def __init__(self, query_analyzer: QueryAnalyzer):
        """Initialize the query strategy router.
        
        Args:
            query_analyzer: Query analyzer instance
        """
        self.query_analyzer = query_analyzer
        logger.info("Initialized QueryStrategyRouter")
    
    def route_query(
        self,
        query_text: str,
        query_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Determine the best retrieval path based on query analysis.
        
        Args:
            query_text: The query text
            query_analysis: Optional pre-computed query analysis
            
        Returns:
            Strategy configuration
        """
        # Get analysis if not provided
        if query_analysis is None:
            query_analysis = self.query_analyzer.analyze_query(query_text)
        
        query_type = query_analysis.get("query_type", "balanced")
        complexity = query_analysis.get("complexity", {})
        entities = query_analysis.get("entities", [])
        
        # For keyword-heavy factual queries with specific entities
        if query_type == "factual" and len(entities) > 0:
            logger.info(f"Routing query as factual with entities: {entities}")
            return {
                "primary_retriever": "bm25",
                "secondary_retriever": "vector",
                "fusion_weight": 0.3,  # Vector weight (lower for factual)
                "fusion_method": "rrf",
                "rerank_threshold": 0.6
            }
        
        # For complex conceptual queries
        elif query_type == "conceptual" and complexity.get("is_complex", False):
            logger.info("Routing query as complex conceptual")
            return {
                "primary_retriever": "vector",
                "secondary_retriever": "bm25",
                "fusion_weight": 0.7,  # Vector weight (higher for conceptual)
                "fusion_method": "softmax",
                "rerank_threshold": 0.4,
                "rerank_method": "cross_encoder"
            }
        
        # For queries with temporal references
        elif complexity.get("is_temporal", False):
            logger.info("Routing query as temporal")
            return {
                "primary_retriever": "hybrid",
                "secondary_retriever": "recency_boost",
                "fusion_weight": 0.5,
                "fusion_method": "linear",
                "rerank_threshold": 0.5,
                "boost_recent": True
            }
        
        # For queries with numerical references (numbers, dates, etc.)
        elif complexity.get("has_numbers", False):
            logger.info("Routing query with numerical references")
            return {
                "primary_retriever": "bm25",
                "secondary_retriever": "vector",
                "fusion_weight": 0.4,  # Slightly favor BM25 for numerical
                "fusion_method": "rrf",
                "rerank_threshold": 0.6
            }
        
        # For short queries with few clues
        elif complexity.get("word_count", 0) < 5:
            logger.info("Routing short query")
            return {
                "primary_retriever": "hybrid",
                "secondary_retriever": "expansion",
                "fusion_weight": 0.4,  # Slightly favor BM25 for short queries
                "fusion_method": "logistic",
                "rerank_threshold": 0.7,
                "expand_query": True
            }
            
        # For queries with high logical complexity
        elif complexity.get("logical_complexity", 0) > 1:
            logger.info("Routing logically complex query")
            return {
                "primary_retriever": "hybrid",
                "secondary_retriever": None,
                "fusion_weight": 0.6,  # Slightly favor vectors
                "fusion_method": "ensemble",
                "rerank_threshold": 0.5,
                "ensemble_methods": ["softmax", "linear", "rrf"]
            }
            
        # Default balanced strategy
        else:
            logger.info("Routing query with default balanced strategy")
            return {
                "primary_retriever": "hybrid",
                "secondary_retriever": None,
                "fusion_weight": 0.5,  # Balanced
                "fusion_method": "linear",
                "rerank_threshold": 0.5
            }


#############################################################################
# Performance Telemetry Components
#############################################################################

class PerformanceMetrics:
    """Track performance metrics for various pipeline operations."""
    
    def __init__(self):
        """Initialize performance metrics tracking system."""
        self.operations = {}
        self.process_data = {}
        self.errors = {}
    
    def start_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start tracking a new operation.
        
        Args:
            operation_name: Name of the operation
            metadata: Optional metadata about the operation
            
        Returns:
            Operation ID for tracking
        """
        op_id = str(uuid.uuid4()) if "process_id" not in (metadata or {}) else metadata["process_id"]
        
        self.operations[op_id] = {
            "name": operation_name,
            "start_time": time.time(),
            "end_time": None,
            "duration": None,
            "metadata": metadata or {}
        }
        
        return op_id
    
    def end_operation(self, op_id: str, status: str = "success") -> float:
        """End tracking for an operation and calculate duration.
        
        Args:
            op_id: Operation ID
            status: Operation status
            
        Returns:
            Operation duration in seconds
        """
        if op_id not in self.operations:
            logger.warning(f"Unknown operation ID: {op_id}")
            return 0.0
        
        end_time = time.time()
        self.operations[op_id]["end_time"] = end_time
        duration = end_time - self.operations[op_id]["start_time"]
        self.operations[op_id]["duration"] = duration
        self.operations[op_id]["status"] = status
        
        return duration
    
    def log_error(self, op_id: str, error_message: str) -> None:
        """Log an error for an operation.
        
        Args:
            op_id: Operation ID
            error_message: Error description
        """
        self.errors[op_id] = {
            "timestamp": time.time(),
            "error": error_message,
            "operation": self.operations.get(op_id, {}).get("name", "unknown")
        }
        
        # Mark operation as failed if it exists
        if op_id in self.operations:
            self.operations[op_id]["status"] = "error"
            if self.operations[op_id]["end_time"] is None:
                self.end_operation(op_id, "error")
    
    def get_metrics(self, process_id: Optional[str] = None) -> Dict[str, Any]:
        """Get metrics for all operations or a specific process.
        
        Args:
            process_id: Optional process ID to filter by
            
        Returns:
            Dictionary of performance metrics
        """
        if process_id and process_id in self.operations:
            # Get metrics for a specific process
            main_op = self.operations[process_id]
            
            # Find child operations
            child_ops = {
                op_id: op for op_id, op in self.operations.items()
                if op.get("metadata", {}).get("process_id") == process_id
            }
            
            # Calculate stage timings
            stage_timings = {
                op["name"]: op["duration"]
                for op_id, op in child_ops.items()
                if op["duration"] is not None
            }
            
            # Check for errors
            has_error = process_id in self.errors
            
            return {
                "total_time": main_op.get("duration"),
                "stage_timings": stage_timings,
                "error": self.errors.get(process_id, None),
                "success": not has_error and main_op.get("status") == "success"
            }
        else:
            # Get overall metrics
            total_ops = len(self.operations)
            completed_ops = sum(1 for op in self.operations.values() if op.get("end_time") is not None)
            error_count = len(self.errors)
            
            # Calculate average durations by operation type
            op_types = {}
            for op in self.operations.values():
                if op.get("duration") is not None:
                    op_name = op["name"]
                    if op_name not in op_types:
                        op_types[op_name] = {"count": 0, "total_time": 0}
                    
                    op_types[op_name]["count"] += 1
                    op_types[op_name]["total_time"] += op["duration"]
            
            avg_durations = {
                name: data["total_time"] / data["count"]
                for name, data in op_types.items()
            }
            
            return {
                "total_operations": total_ops,
                "completed_operations": completed_ops,
                "error_count": error_count,
                "avg_durations": avg_durations
            }


#############################################################################
# Enhanced Reranking Components
#############################################################################

class RerankerInterface:
    """Interface for reranking components that reorder results based on different criteria."""
    
    async def rerank(
        self,
        results: List[Any],
        query: str,
        query_vector: Optional[List[float]] = None,
        **kwargs
    ) -> List[Any]:
        """Rerank search results based on different criteria.
        
        Args:
            results: List of search results to rerank
            query: Original query text
            query_vector: Query embedding vector
            **kwargs: Additional parameters specific to reranking method
            
        Returns:
            Reordered list of results
        """
        raise NotImplementedError("Subclasses must implement rerank()")


class MMRReranker(RerankerInterface):
    """Maximal Marginal Relevance reranker for improving diversity."""
    
    async def rerank(
        self,
        results: List[Any],
        query: str,
        query_vector: Optional[List[float]] = None,
        **kwargs
    ) -> List[Any]:
        """Rerank results using MMR to balance relevance and diversity.
        
        Args:
            results: List of search results to rerank
            query: Original query text (not used)
            query_vector: Query embedding vector (not used)
            **kwargs: Additional parameters (mmr_lambda)
            
        Returns:
            Reordered list of results
        """
        if len(results) <= 1:
            return results
            
        mmr_lambda = kwargs.get('mmr_lambda', 0.5)
        
        try:
            from query_rag import _mmr_rerank
            return _mmr_rerank(results, mmr_lambda)
        except Exception as e:
            logger.error(f"MMR reranking failed: {e}")
            return results


class CrossEncoderReranker(RerankerInterface):
    """Cross-encoder based reranker for semantic relevance."""
    
    async def rerank(
        self,
        results: List[Any],
        query: str,
        query_vector: Optional[List[float]] = None,
        **kwargs
    ) -> List[Any]:
        """Rerank results using a cross-encoder model.
        
        Args:
            results: List of search results to rerank
            query: Original query text
            query_vector: Query embedding vector (not used)
            **kwargs: Additional parameters (rerank_top)
            
        Returns:
            Reordered list of results
        """
        if len(results) <= 1:
            return results
            
        rerank_top = min(kwargs.get('rerank_top', len(results)), len(results))
        model_name = kwargs.get('model_name', "cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        try:
            from sentence_transformers import CrossEncoder
            
            # Limit to requested number of results
            candidates = results[:rerank_top]
            
            # Prepare pairs for cross-encoder
            pairs = [
                (query, (getattr(p, 'payload', {}) or {}).get("chunk_text", ""))
                for p in candidates
            ]
            
            # Skip if no valid pairs
            if not pairs:
                logger.warning("No valid text pairs for cross-encoder reranking")
                return results
                
            # Load cross-encoder model
            ce = CrossEncoder(model_name)
            
            # Predict relevance scores
            rerank_scores = ce.predict(pairs)
            
            # Sort by new scores
            idxs = sorted(range(rerank_top), key=lambda i: rerank_scores[i], reverse=True)
            
            # Reorder results
            reranked = [candidates[i] for i in idxs]
            
            # Combine with remaining results
            return reranked + results[rerank_top:]
            
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            return results


class ContextAwareReranker(RerankerInterface):
    """Context-aware reranking that considers query semantics."""
    
    async def rerank(
        self,
        results: List[Any],
        query: str,
        query_vector: Optional[List[float]] = None,
        **kwargs
    ) -> List[Any]:
        """Perform context-aware reranking.
        
        Args:
            results: List of search results to rerank
            query: Original query text
            query_vector: Query embedding vector
            **kwargs: Additional parameters
            
        Returns:
            Reordered list of results
        """
        if len(results) <= 1:
            return results
        
        # Get query analysis (or create it)
        query_analysis = kwargs.get('query_analysis')
        if not query_analysis:
            analyzer = QueryAnalyzer()
            query_analysis = analyzer.analyze_query(query)
        
        # Determine if we need topic diversity
        query_type = query_analysis.get('query_type', 'balanced')
        complexity = query_analysis.get('complexity', {})
        
        # Configure diversity settings based on query characteristics
        need_diversity = complexity.get('is_complex', False) or query_type == 'conceptual'
        diversity_weight = 0.3 if need_diversity else 0.1
        
        # Get relevance factors
        entities = query_analysis.get('entities', [])
        has_entities = len(entities) > 0
        
        # Calculate relevance boosting scores for context-awareness
        boosted_scores = {}
        for i, result in enumerate(results):
            # Get original score and content
            original_score = getattr(result, 'score', 0.0)
            
            # Start with original score
            context_score = original_score
            
            # Get content text
            payload = getattr(result, 'payload', {}) or {}
            content = payload.get('chunk_text', '')
            
            # Boost if content contains query entities
            if has_entities:
                entity_matches = sum(1 for entity in entities if entity.lower() in content.lower())
                entity_boost = min(0.2, 0.05 * entity_matches)  # Cap at 0.2
                context_score += entity_boost
            
            # Consider document length - prefer medium-length docs for complex queries
            word_count = len(content.split())
            length_score = 0
            
            if complexity.get('is_complex', False):
                # For complex queries, prefer medium to longer content
                if 100 <= word_count <= 500:
                    length_score = 0.1
                elif word_count > 500:
                    length_score = 0.05
            else:
                # For simple queries, prefer shorter content
                if word_count < 100:
                    length_score = 0.1
            
            context_score += length_score
            
            # Store result with boosted score
            boosted_scores[i] = context_score
        
        # Apply diversity if needed
        if need_diversity and len(results) > 3:
            # Start with highest scoring result
            reranked_indices = [max(boosted_scores.items(), key=lambda x: x[1])[0]]
            remaining = set(range(len(results))) - set(reranked_indices)
            
            # Extract vectors for diversity calculation
            result_vectors = []
            for result in results:
                vector = getattr(result, 'vector', None)
                if vector is None:
                    # If no vector, use a different reranker
                    logger.warning("Context-aware reranking needs result vectors, falling back to MMR")
                    mmr_reranker = MMRReranker()
                    return await mmr_reranker.rerank(
                        results,
                        query,
                        query_vector,
                        mmr_lambda=1-diversity_weight
                    )
                result_vectors.append(vector)
            
            # Continue selecting based on diversity and relevance
            while remaining and len(reranked_indices) < len(results):
                best_score = -1
                best_idx = -1
                
                for idx in remaining:
                    # Calculate similarity to already selected documents
                    avg_sim = 0
                    if reranked_indices:
                        similarities = []
                        for selected_idx in reranked_indices:
                            sim = np.dot(result_vectors[idx], result_vectors[selected_idx])
                            similarities.append(sim)
                        avg_sim = sum(similarities) / len(similarities)
                    
                    # Combined score: relevance - diversity_weight * similarity
                    score = boosted_scores[idx] - diversity_weight * avg_sim
                    
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                
                if best_idx >= 0:
                    reranked_indices.append(best_idx)
                    remaining.remove(best_idx)
                else:
                    break
            
            # Create reranked list
            reranked = [results[i] for i in reranked_indices]
            return reranked
        else:
            # Just sort by boosted scores if no diversity needed
            sorted_indices = sorted(boosted_scores.keys(), key=lambda idx: boosted_scores[idx], reverse=True)
            return [results[i] for i in sorted_indices]


class DiversityAwareReranker(RerankerInterface):
    """Diversity-focused reranking to ensure varied results."""
    
    async def rerank(
        self,
        results: List[Any],
        query: str,
        query_vector: Optional[List[float]] = None,
        **kwargs
    ) -> List[Any]:
        """Rerank results to maximize diversity.
        
        Args:
            results: List of search results to rerank
            query: Original query text
            query_vector: Query embedding vector
            **kwargs: Additional parameters (diversity_weight, cluster_count)
            
        Returns:
            Reordered list of results
        """
        if len(results) <= 3:
            return results
            
        diversity_weight = kwargs.get('diversity_weight', 0.5)
        cluster_count = min(kwargs.get('cluster_count', 3), len(results) // 2)
        
        try:
            import numpy as np
            from sklearn.cluster import KMeans
            
            # Extract vectors from results
            vectors = []
            for result in results:
                vec = getattr(result, 'vector', None)
                if vec is None:
                    logger.warning("Diversity reranking requires vectors, falling back to MMR")
                    mmr_reranker = MMRReranker()
                    return await mmr_reranker.rerank(
                        results,
                        query,
                        query_vector,
                        mmr_lambda=1-diversity_weight
                    )
                vectors.append(vec)
            
            # Convert to numpy array
            vectors_array = np.array(vectors)
            
            # Create clusters
            kmeans = KMeans(n_clusters=cluster_count, random_state=42)
            clusters = kmeans.fit_predict(vectors_array)
            
            # Group results by cluster
            cluster_groups = {}
            for i, cluster_id in enumerate(clusters):
                if cluster_id not in cluster_groups:
                    cluster_groups[cluster_id] = []
                cluster_groups[cluster_id].append((i, getattr(results[i], 'score', 0.0)))
            
            # Sort within each cluster by score
            for cluster_id in cluster_groups:
                cluster_groups[cluster_id].sort(key=lambda x: x[1], reverse=True)
            
            # Round-robin selection from clusters
            reranked_indices = []
            cluster_counters = {cluster_id: 0 for cluster_id in cluster_groups}
            
            # Start with top result from each cluster
            for cluster_id in sorted(cluster_groups.keys()):
                if cluster_groups[cluster_id]:
                    reranked_indices.append(cluster_groups[cluster_id][0][0])
                    cluster_counters[cluster_id] += 1
            
            # Continue round-robin until all are selected
            while len(reranked_indices) < len(results):
                for cluster_id in sorted(cluster_groups.keys()):
                    if cluster_counters[cluster_id] < len(cluster_groups[cluster_id]):
                        idx = cluster_groups[cluster_id][cluster_counters[cluster_id]][0]
                        if idx not in reranked_indices:  # Safeguard against duplicates
                            reranked_indices.append(idx)
                            cluster_counters[cluster_id] += 1
                            
                            if len(reranked_indices) >= len(results):
                                break
            
            # Create reranked results
            reranked = [results[i] for i in reranked_indices]
            return reranked
            
        except ImportError:
            logger.error("Required packages not installed: scikit-learn, numpy")
            return results
        except Exception as e:
            logger.error(f"Diversity reranking failed: {e}")
            return results


class RerankerFactory:
    """Factory for creating reranker instances based on strategy."""
    
    @staticmethod
    def create_reranker(reranker_type: str) -> RerankerInterface:
        """Create reranker instance by type.
        
        Args:
            reranker_type: Type of reranker ('mmr', 'cross_encoder', etc.)
            
        Returns:
            Reranker instance
        """
        logger.info(f"Creating reranker of type: {reranker_type}")
        rerankers = {
            "mmr": MMRReranker(),
            "cross_encoder": CrossEncoderReranker(),
            "context_aware": ContextAwareReranker(),
            "diversity": DiversityAwareReranker(),
        }
        
        if reranker_type not in rerankers:
            logger.warning(f"Unknown reranker type: {reranker_type}, defaulting to MMR")
            return MMRReranker()
            
        return rerankers[reranker_type]


#############################################################################
# Main API Functions
#############################################################################

class SearchEngine:
    """High-level search engine that combines retrieval, fusion, and reranking."""
    
    def __init__(
        self,
        client: QdrantClient,
        openai_client: Any
    ):
        """Initialize search engine.
        
        Args:
            client: Qdrant client
            openai_client: OpenAI client
        """
        self.client = client
        self.openai_client = openai_client
        self.orchestrator = RetrievalOrchestrator(client, openai_client)
        logger.info("Initialized SearchEngine")
    
    async def search(
        self,
        query_text: str,
        collection: str,
        model: str = "text-embedding-3-large",
        k: int = 20,
        filter_obj: Optional[Filter] = None,
        alpha: float = 0.5,
        rrf_k: float = 60.0,
        fusion_method: str = "auto",
        bm25_index_path: str = None,
        use_mmr: bool = False,
        mmr_lambda: float = 0.5,
        rerank_top: int = 0
    ) -> List[Any]:
        """Main search API combining all improved retrieval components.
        
        Args:
            query_text: Query text
            collection: Collection name
            model: Embedding model name
            k: Number of results to return
            filter_obj: Optional filter
            alpha: Weight for vector scores in fusion
            rrf_k: RRF k parameter
            fusion_method: Fusion method (auto, rrf, linear, softmax)
            bm25_index_path: Path to BM25 index file
            use_mmr: Whether to apply MMR reranking
            mmr_lambda: MMR lambda parameter
            rerank_top: Number of top results to rerank (0 for no reranking)
            
        Returns:
            List of search results
        """
        logger.info(f"Search request: query='{query_text}', collection='{collection}'")
        
        # Get embedding for the query
        vector = await asyncio.to_thread(
            get_cached_embedding,
            query_text,
            model,
            self.openai_client
        )
        
        # Analyze query to determine optimal retrieval strategy
        query_analysis = query_analyzer.analyze_query(query_text, alpha)
        logger.info(f"Query analysis: type={query_analysis['query_type']}, alpha={query_analysis['alpha']:.2f}")
        
        # Select retrieval strategy based on query analysis
        strategy = query_analysis.get("strategy", {})
        retrieval_strategy = strategy.get("primary_retriever", "hybrid")
        
        # Get fusion method to use (either explicitly specified or from query analysis)
        actual_fusion_method = fusion_method
        if fusion_method == "auto":
            actual_fusion_method = query_analysis["fusion_method"]
            logger.info(f"Auto-selected fusion method: {actual_fusion_method}")
        
        # Create retrievers based on determined strategy
        retriever = RetrieverFactory.create_retriever(
            retrieval_strategy,
            self.client,
            self.openai_client,
            bm25_index_path
        )
        
        # Perform retrieval
        if retrieval_strategy == "hybrid":
            vector_results, bm25_results = await retriever.retrieve(
                query_text, vector, collection, k, filter_obj
            )
        else:
            # For non-hybrid strategies, we need to do the parallel retrieval ourselves
            vector_retriever = VectorRetriever(self.client)
            bm25_retriever = BM25Retriever(self.client, bm25_index_path)
            
            vector_results, bm25_results = await asyncio.gather(
                vector_retriever.retrieve(query_text, vector, collection, k, filter_obj),
                bm25_retriever.retrieve(query_text, vector, collection, k, filter_obj)
            )
        
        # Extract scores for fusion
        vector_scores = {getattr(r, 'id', None): getattr(r, 'score', 0.0) for r in vector_results}
        bm25_scores = {getattr(r, 'id', None): getattr(r, 'score', 0.0) for r in bm25_results}
        
        # Create appropriate fusion component
        fusion = FusionFactory.create_fusion(actual_fusion_method)
        
        # Perform fusion with query-dependent alpha
        query_alpha = query_analysis["alpha"]
        
        if actual_fusion_method == "rrf":
            fused_scores = fusion.fuse(vector_scores, bm25_scores, query_alpha, rrf_k=rrf_k)
        else:
            fused_scores = fusion.fuse(vector_scores, bm25_scores, query_alpha)
        
        # Create result map for lookups
        results_map = {}
        for r in vector_results + bm25_results:
            if not hasattr(r, 'id'):
                continue
            results_map[r.id] = r
        
        # Sort by fused score and create final results
        from types import SimpleNamespace
        fused_results = []
        
        for pid, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:k]:
            if pid in results_map:
                r = results_map[pid]
                result = SimpleNamespace(
                    id=pid,
                    score=score,
                    payload=getattr(r, 'payload', {}) or {},
                    vector=getattr(r, 'vector', None),
                )
                fused_results.append(result)
        
        logger.info(f"Fusion complete, got {len(fused_results)} results")
        
        # Apply reranking if requested
        if use_mmr:
            mmr_reranker = RerankerFactory.create_reranker("mmr")
            fused_results = await mmr_reranker.rerank(
                fused_results, query_text, vector, mmr_lambda=mmr_lambda
            )
            logger.info("Applied MMR reranking")
        
        if rerank_top > 0:
            cross_encoder_reranker = RerankerFactory.create_reranker("cross_encoder")
            fused_results = await cross_encoder_reranker.rerank(
                fused_results, query_text, vector, rerank_top=rerank_top
            )
            logger.info(f"Applied cross-encoder reranking to top {rerank_top} results")
        
        return fused_results


# For backward compatibility
async def search(
    query_text: str,
    collection: str,
    client: QdrantClient,
    openai_client: Any,
    model: str = "text-embedding-3-large",
    k: int = 20,
    filter_obj: Optional[Filter] = None,
    alpha: float = 0.5,
    rrf_k: float = 60.0,
    fusion_method: str = "auto",
    bm25_index_path: str = None,
    use_mmr: bool = False,
    mmr_lambda: float = 0.5,
    rerank_top: int = 0
) -> List[Any]:
    """Main search API combining all improved retrieval components (backward compatibility wrapper).
    
    Args:
        query_text: Query text
        collection: Collection name
        client: Qdrant client
        openai_client: OpenAI client
        model: Embedding model name
        k: Number of results to return
        filter_obj: Optional filter
        alpha: Weight for vector scores in fusion
        rrf_k: RRF k parameter
        fusion_method: Fusion method (auto, rrf, linear, softmax)
        bm25_index_path: Path to BM25 index file
        use_mmr: Whether to apply MMR reranking
        mmr_lambda: MMR lambda parameter
        rerank_top: Number of top results to rerank (0 for no reranking)
        
    Returns:
        List of search results
    """
    engine = SearchEngine(client, openai_client)
    return await engine.search(
        query_text,
        collection,
        model,
        k,
        filter_obj,
        alpha,
        rrf_k,
        fusion_method,
        bm25_index_path,
        use_mmr,
        mmr_lambda,
        rerank_top
    )