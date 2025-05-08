"""Query Service implementation.

This service is responsible for query analysis and orchestration of the retrieval process.
It serves as the entry point for RAG queries and coordinates the work of other services.
"""
import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple
import httpx
from pydantic import BaseModel

from microservices.service_interfaces import (
    ServiceRequest, 
    ServiceResponse,
    QueryServiceInterface
)

# Configure logging
logger = logging.getLogger(__name__)


class QueryRequest(ServiceRequest):
    """Query request model."""
    query_text: str
    collection: str
    k: int = 10
    filter_obj: Optional[Dict[str, Any]] = None
    strategy_params: Optional[Dict[str, Any]] = None
    use_cache: bool = True


class QueryResponse(ServiceResponse):
    """Query response model."""
    results: Optional[List[Dict[str, Any]]] = None
    strategy_used: Optional[Dict[str, Any]] = None
    execution_stats: Optional[Dict[str, Any]] = None


class QueryAnalysisResult(BaseModel):
    """Model for query analysis results."""
    query_type: str
    alpha: float
    entities: List[str]
    metric: str
    fusion_method: str
    strategy: Dict[str, Any]
    query_complexity: Dict[str, Any]


class QueryService(QueryServiceInterface):
    """Implementation of the Query Service."""
    
    def __init__(
        self,
        vector_service_url: str,
        bm25_service_url: str,
        fusion_service_url: str,
        cache_service_url: str,
        colbert_service_url: Optional[str] = None,
        splade_service_url: Optional[str] = None
    ):
        """Initialize the query service.
        
        Args:
            vector_service_url: URL for the Vector Search Service
            bm25_service_url: URL for the BM25 Service
            fusion_service_url: URL for the Fusion Service
            cache_service_url: URL for the Cache Service
            colbert_service_url: Optional URL for the ColBERT Service
            splade_service_url: Optional URL for the SPLADE Service
        """
        self.vector_service_url = vector_service_url
        self.bm25_service_url = bm25_service_url
        self.fusion_service_url = fusion_service_url
        self.cache_service_url = cache_service_url
        self.colbert_service_url = colbert_service_url
        self.splade_service_url = splade_service_url
        
        self.client = httpx.AsyncClient(timeout=30.0)
        self.is_running = True
        logger.info("Query Service initialized")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if the service is healthy and return status information."""
        services_status = {}
        
        # Check connectivity to all dependent services
        services = {
            "vector_service": self.vector_service_url,
            "bm25_service": self.bm25_service_url,
            "fusion_service": self.fusion_service_url,
            "cache_service": self.cache_service_url
        }
        
        if self.colbert_service_url:
            services["colbert_service"] = self.colbert_service_url
        
        if self.splade_service_url:
            services["splade_service"] = self.splade_service_url
        
        for name, url in services.items():
            try:
                response = await self.client.get(f"{url}/health")
                services_status[name] = "healthy" if response.status_code == 200 else "unhealthy"
            except Exception as e:
                services_status[name] = f"error: {str(e)}"
        
        return {
            "status": "healthy" if all(status == "healthy" for status in services_status.values()) else "degraded",
            "services": services_status,
            "version": "1.0.0"
        }
    
    async def process_request(self, request: ServiceRequest) -> ServiceResponse:
        """Process a service request and return a response."""
        if not isinstance(request, QueryRequest):
            return ServiceResponse(
                request_id=request.request_id,
                status="error",
                message="Invalid request type"
            )
        
        try:
            # Check cache first if enabled
            if request.use_cache:
                cache_key = f"query:{request.query_text}:{request.collection}:{request.k}:{str(request.filter_obj)}"
                cache_response = await self.client.get(
                    f"{self.cache_service_url}/get",
                    params={"key": cache_key}
                )
                
                if cache_response.status_code == 200 and cache_response.json().get("data"):
                    cached_result = cache_response.json()["data"]
                    logger.info(f"Cache hit for query: {request.query_text}")
                    return QueryResponse(
                        request_id=request.request_id,
                        status="success",
                        message="Results retrieved from cache",
                        data=cached_result
                    )
            
            # Analyze query to determine strategy
            query_analysis = await self.analyze_query(
                query_text=request.query_text,
                query_id=request.request_id
            )
            
            # Orchestrate retrieval based on analysis
            results = await self.orchestrate_retrieval(
                query_text=request.query_text,
                strategy=query_analysis["strategy"],
                query_id=request.request_id,
                collection=request.collection,
                k=request.k,
                filter_obj=request.filter_obj
            )
            
            # Cache results if caching is enabled
            if request.use_cache:
                cache_data = {
                    "results": results["results"],
                    "strategy_used": query_analysis["strategy"],
                    "execution_stats": results.get("execution_stats", {})
                }
                
                await self.client.post(
                    f"{self.cache_service_url}/set",
                    json={
                        "key": cache_key,
                        "value": cache_data,
                        "ttl": 300  # 5 minutes TTL
                    }
                )
            
            return QueryResponse(
                request_id=request.request_id,
                status="success",
                message="Query processed successfully",
                data={
                    "results": results["results"],
                    "strategy_used": query_analysis["strategy"],
                    "execution_stats": results.get("execution_stats", {})
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return QueryResponse(
                request_id=request.request_id,
                status="error",
                message=f"Error processing query: {str(e)}"
            )
    
    async def analyze_query(self, query_text: str, query_id: str) -> Dict[str, Any]:
        """Analyze a query to determine optimal retrieval strategy.
        
        This method analyzes the query text to determine the best retrieval strategy,
        including which retrieval methods to use, fusion methods, and parameters.
        
        Args:
            query_text: The text of the query
            query_id: Unique identifier for this query
            
        Returns:
            Dictionary containing query analysis and strategy
        """
        logger.info(f"Analyzing query: {query_text}")
        
        # First we examine the query to determine its characteristics
        # Check for factual indicators (exact entities, numbers, dates, etc.)
        factual_indicators = ['who', 'what', 'when', 'where', 'which', 'how many', 'list', 'name']
        factual_score = sum(1 for word in factual_indicators if word.lower() in query_text.lower().split())
        
        # Check for conceptual indicators (explanations, concepts, reasoning)
        conceptual_indicators = ['why', 'how', 'explain', 'describe', 'compare', 'analyze', 'evaluate']
        conceptual_score = sum(1 for word in conceptual_indicators if word.lower() in query_text.lower().split())
        
        # Extract entities (for potential boosting)
        import re
        entities = []
        
        # Extract quoted phrases as exact match entities
        quoted = re.findall(r'"([^"]*)"', query_text)
        entities.extend(quoted)
        
        # Look for capitalized phrases (potential named entities)
        capitalized = re.findall(r'\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\b', query_text)
        entities.extend(capitalized)
        
        # Analyze query complexity
        query_complexity = {
            "word_count": len(query_text.split()),
            "sentence_count": len(re.findall(r'[.!?]+', query_text)) + 1,
            "has_numbers": bool(re.search(r'\d+', query_text)),
            "is_complex": len(query_text.split()) > 10 or len(re.findall(r'[.!?]+', query_text)) > 1
        }
        
        # Determine query type and parameters
        if factual_score > conceptual_score:
            query_type = "factual"
            alpha = 0.3  # More weight to BM25 for factual queries
            fusion_method = "rrf"
            metric = "hybrid_bm25_heavy"
            strategy = {
                "primary_retriever": "bm25",
                "secondary_retriever": "vector",
                "fusion_weight": alpha,
                "fusion_method": fusion_method,
                "use_colbert": False,
                "use_splade": len(entities) > 0,  # Use SPLADE for entity-rich queries
                "rerank_method": "mmr" if len(entities) > 1 else None
            }
        elif conceptual_score > factual_score:
            query_type = "conceptual"
            alpha = 0.7  # More weight to vectors for conceptual queries
            fusion_method = "softmax"
            metric = "hybrid_vector_heavy"
            strategy = {
                "primary_retriever": "vector",
                "secondary_retriever": "bm25",
                "fusion_weight": alpha,
                "fusion_method": fusion_method,
                "use_colbert": True,  # ColBERT works well for conceptual queries
                "use_splade": False,
                "rerank_method": "context_aware"
            }
        else:
            query_type = "balanced"
            alpha = 0.5
            fusion_method = "linear"
            metric = "hybrid_balanced"
            strategy = {
                "primary_retriever": "hybrid",
                "secondary_retriever": None,
                "fusion_weight": alpha,
                "fusion_method": fusion_method,
                "use_colbert": query_complexity["is_complex"],
                "use_splade": True,
                "rerank_method": "diversity" if query_complexity["is_complex"] else "mmr"
            }
        
        analysis_result = {
            "query_type": query_type,
            "alpha": alpha,
            "entities": entities,
            "metric": metric,
            "fusion_method": fusion_method,
            "strategy": strategy,
            "query_complexity": query_complexity
        }
        
        logger.info(f"Query analysis completed: {query_type}, fusion: {fusion_method}")
        return analysis_result
    
    async def orchestrate_retrieval(
        self, 
        query_text: str, 
        strategy: Dict[str, Any],
        query_id: str,
        collection: str = "rag_data",
        k: int = 10,
        filter_obj: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Orchestrate the retrieval process based on query analysis.
        
        Args:
            query_text: The text of the query
            strategy: Retrieval strategy from query analysis
            query_id: Unique identifier for this query
            collection: Collection name to search
            k: Number of results to retrieve
            filter_obj: Optional filter for the search
            
        Returns:
            Dictionary containing search results and execution stats
        """
        logger.info(f"Orchestrating retrieval for query: {query_text}")
        start_time = asyncio.get_event_loop().time()
        execution_stats = {}
        
        # Step 1: Get query embedding for vector search
        vector_embedding_start = asyncio.get_event_loop().time()
        vector_response = await self.client.post(
            f"{self.vector_service_url}/embed",
            json={"text": query_text}
        )
        
        if vector_response.status_code != 200:
            raise Exception(f"Failed to get vector embedding: {vector_response.text}")
            
        query_vector = vector_response.json()["embedding"]
        execution_stats["vector_embedding_time"] = asyncio.get_event_loop().time() - vector_embedding_start
        
        # Step 2: Prepare tasks for all retrieval methods
        retrieval_tasks = []
        
        # Vector search
        if strategy["primary_retriever"] in ["vector", "hybrid"]:
            vector_task = self.client.post(
                f"{self.vector_service_url}/search",
                json={
                    "query_vector": query_vector,
                    "collection": collection,
                    "k": k,
                    "filter_obj": filter_obj
                }
            )
            retrieval_tasks.append(("vector", vector_task))
        
        # BM25 search
        if strategy["primary_retriever"] in ["bm25", "hybrid"]:
            bm25_task = self.client.post(
                f"{self.bm25_service_url}/search",
                json={
                    "query_text": query_text,
                    "collection": collection,
                    "k": k,
                    "filter_obj": filter_obj
                }
            )
            retrieval_tasks.append(("bm25", bm25_task))
        
        # ColBERT search if enabled
        if strategy.get("use_colbert", False) and self.colbert_service_url:
            colbert_task = self.client.post(
                f"{self.colbert_service_url}/encode_query",
                json={"query_text": query_text}
            )
            retrieval_tasks.append(("colbert", colbert_task))
        
        # SPLADE search if enabled
        if strategy.get("use_splade", False) and self.splade_service_url:
            splade_task = self.client.post(
                f"{self.splade_service_url}/encode_query",
                json={"query_text": query_text}
            )
            retrieval_tasks.append(("splade", splade_task))
        
        # Step 3: Execute all retrieval tasks in parallel
        retrieval_start = asyncio.get_event_loop().time()
        retrieval_results = {}
        
        for method, task in retrieval_tasks:
            try:
                response = await task
                if response.status_code == 200:
                    retrieval_results[method] = response.json()
                else:
                    logger.error(f"Error in {method} retrieval: {response.text}")
                    retrieval_results[method] = {"error": response.text}
            except Exception as e:
                logger.error(f"Exception in {method} retrieval: {str(e)}")
                retrieval_results[method] = {"error": str(e)}
        
        execution_stats["retrieval_time"] = asyncio.get_event_loop().time() - retrieval_start
        
        # Step 4: Process ColBERT results if available
        if "colbert" in retrieval_results and "vector" in retrieval_results and not "error" in retrieval_results["colbert"]:
            colbert_start = asyncio.get_event_loop().time()
            try:
                # Get document vectors from vector search
                vector_results = retrieval_results["vector"].get("results", [])
                doc_vectors = [{"id": doc["id"], "vector": doc.get("vector")} for doc in vector_results]
                
                # Score documents using ColBERT
                colbert_response = await self.client.post(
                    f"{self.colbert_service_url}/score",
                    json={
                        "query_embeddings": retrieval_results["colbert"],
                        "document_vectors": doc_vectors
                    }
                )
                
                if colbert_response.status_code == 200:
                    colbert_scores = colbert_response.json().get("scores", {})
                    
                    # Update vector results with ColBERT scores
                    for doc in vector_results:
                        if doc["id"] in colbert_scores:
                            doc["colbert_score"] = colbert_scores[doc["id"]]
                            # Blend original score with ColBERT score
                            doc["score"] = 0.7 * doc["score"] + 0.3 * colbert_scores[doc["id"]]
                    
                    # Update vector results in retrieval_results
                    retrieval_results["vector"]["results"] = vector_results
                
                execution_stats["colbert_time"] = asyncio.get_event_loop().time() - colbert_start
            except Exception as e:
                logger.error(f"Error in ColBERT processing: {str(e)}")
        
        # Step 5: Process SPLADE results if available
        if "splade" in retrieval_results and not "error" in retrieval_results["splade"]:
            splade_start = asyncio.get_event_loop().time()
            try:
                # Search using SPLADE sparse vectors
                splade_search_response = await self.client.post(
                    f"{self.splade_service_url}/search",
                    json={
                        "query_vector": retrieval_results["splade"].get("sparse_vector", {}),
                        "collection": collection,
                        "k": k
                    }
                )
                
                if splade_search_response.status_code == 200:
                    splade_results = splade_search_response.json().get("results", [])
                    retrieval_results["splade"]["results"] = splade_results
                
                execution_stats["splade_time"] = asyncio.get_event_loop().time() - splade_start
            except Exception as e:
                logger.error(f"Error in SPLADE processing: {str(e)}")
        
        # Step 6: Fusion of results
        fusion_start = asyncio.get_event_loop().time()
        
        # Prepare fusion inputs
        vector_results = retrieval_results.get("vector", {}).get("results", [])
        bm25_results = retrieval_results.get("bm25", {}).get("results", [])
        splade_results = retrieval_results.get("splade", {}).get("results", []) if "splade" in retrieval_results else []
        
        # Perform fusion
        fusion_params = {
            "vector_results": vector_results,
            "bm25_results": bm25_results,
            "strategy": {
                "alpha": strategy["fusion_weight"],
                "method": strategy["fusion_method"]
            }
        }
        
        # Add SPLADE results if available
        if splade_results:
            fusion_params["splade_results"] = splade_results
        
        fusion_response = await self.client.post(
            f"{self.fusion_service_url}/fuse",
            json=fusion_params
        )
        
        if fusion_response.status_code != 200:
            raise Exception(f"Failed to fuse results: {fusion_response.text}")
            
        fused_results = fusion_response.json()["results"]
        execution_stats["fusion_time"] = asyncio.get_event_loop().time() - fusion_start
        
        # Step 7: Reranking if specified
        if strategy.get("rerank_method") and fused_results:
            rerank_start = asyncio.get_event_loop().time()
            
            rerank_response = await self.client.post(
                f"{self.fusion_service_url}/rerank",
                json={
                    "results": fused_results,
                    "query_text": query_text,
                    "query_vector": query_vector,
                    "strategy": {
                        "method": strategy["rerank_method"]
                    }
                }
            )
            
            if rerank_response.status_code != 200:
                logger.error(f"Reranking failed: {rerank_response.text}")
            else:
                fused_results = rerank_response.json()["results"]
            
            execution_stats["rerank_time"] = asyncio.get_event_loop().time() - rerank_start
        
        # Record total execution time
        execution_stats["total_time"] = asyncio.get_event_loop().time() - start_time
        
        return {
            "results": fused_results,
            "execution_stats": execution_stats
        }
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the service."""
        logger.info("Shutting down Query Service")
        self.is_running = False
        await self.client.aclose()


# FastAPI specific code
def create_fastapi_app():
    """Create a FastAPI app for the Query Service."""
    from fastapi import FastAPI, HTTPException
    import os
    
    app = FastAPI(title="RAG Query Service", version="1.0.0")
    
    # Initialize service with environment variables or defaults
    service = QueryService(
        vector_service_url=os.getenv("VECTOR_SERVICE_URL", "http://localhost:8001"),
        bm25_service_url=os.getenv("BM25_SERVICE_URL", "http://localhost:8002"),
        fusion_service_url=os.getenv("FUSION_SERVICE_URL", "http://localhost:8003"),
        cache_service_url=os.getenv("CACHE_SERVICE_URL", "http://localhost:8004"),
        colbert_service_url=os.getenv("COLBERT_SERVICE_URL", "http://localhost:8005"),
        splade_service_url=os.getenv("SPLADE_SERVICE_URL", "http://localhost:8006")
    )
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return await service.health_check()
    
    @app.post("/query")
    async def query(request: QueryRequest):
        """Process a query request."""
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