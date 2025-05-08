#!/usr/bin/env python3
"""
Demo of Phase 2 improvements from the RAG Retrieval Pipeline Comprehensive Improvement Plan.

This script demonstrates:
1. Vector Index Optimization
2. Advanced Fusion Methods
3. Context-Aware Reranking
4. Retrieval Strategy Router
5. Asynchronous Processing Pipeline
6. Performance Telemetry

Usage:
    python demo_phase2_improvements.py --collection <collection_name> --query "your query here"
"""
import asyncio
import argparse
import time
import json
from typing import Dict, Any, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter

# Import OpenAI for embeddings
try:
    import openai
    from openai import OpenAI, AsyncOpenAI
except ImportError:
    print("OpenAI Python package not found. Please install it with 'pip install openai'")
    exit(1)

# Import our improved retrieval components
from improved_retrieval import (
    VectorIndexOptimizer,
    normalized_fusion,
    ContextAwareReranker,
    RerankerFactory,
    QueryStrategyRouter,
    AsyncRetrievalPipeline,
    PerformanceMetrics,
    preprocess_query,
    QueryAnalyzer
)

async def optimize_vector_index(client: QdrantClient, collection: str):
    """Demonstrate vector index optimization features"""
    print("\n=== Vector Index Optimization ===")
    
    # Create optimizer
    optimizer = VectorIndexOptimizer(client)
    
    # 1. Configure optimal HNSW parameters
    print("Configuring HNSW parameters...")
    hnsw_params = {
        "m": 16,              # Number of connections per node
        "ef_construct": 100,  # Construction parameter
        "full_scan_threshold": 10000,  # When to switch from HNSW to full scan
        "max_indexing_threads": 0,     # Number of threads (0 = auto)
    }
    success = await optimizer.configure_index_parameters(collection, hnsw_params)
    print(f"HNSW configuration {'successful' if success else 'failed'}")
    
    # 2. Apply vector quantization
    print("\nApplying vector quantization...")
    success = await optimizer.apply_vector_quantization(
        collection,
        scalar_quantization=True,
        quantization_config={
            "type": "scalar",
            "quantile": 0.99,
            "always_ram": True,
        }
    )
    print(f"Vector quantization {'successful' if success else 'failed'}")
    
    # 3. Demonstrate dimensionality reduction (with sample data)
    print("\nDemonstrating dimensionality reduction...")
    # Create sample vectors (normally these would come from your database)
    import numpy as np
    sample_vectors = np.random.randn(10, 1536).tolist()  # 10 vectors of dimension 1536
    
    # Reduce dimensions
    start_time = time.time()
    reduced_vectors = await optimizer.reduce_dimensions(
        sample_vectors,
        target_dim=256,
        method="pca"
    )
    end_time = time.time()
    
    if reduced_vectors and len(reduced_vectors) > 0:
        original_dim = len(sample_vectors[0])
        reduced_dim = len(reduced_vectors[0])
        compression = 100 * (1 - reduced_dim / original_dim)
        print(f"Reduced dimensions from {original_dim} to {reduced_dim} " 
              f"({compression:.1f}% compression) in {end_time - start_time:.3f}s")
    else:
        print("Dimensionality reduction failed")

async def demo_advanced_fusion_methods(query: str):
    """Demonstrate advanced fusion methods"""
    print("\n=== Advanced Fusion Methods ===")
    
    # Create mock scores for demonstration
    import numpy as np
    
    # Mock vector scores (higher is better)
    vec_scores = {
        f"doc_{i}": float(0.9 - 0.05 * i + 0.02 * np.random.randn())
        for i in range(10)
    }
    
    # Mock BM25 scores (higher is better)
    bm25_scores = {
        f"doc_{i}": float(20 - 1.5 * i + 0.5 * np.random.randn())
        for i in range(10)
    }
    
    # Add some unique docs to each set
    vec_scores["vec_only_1"] = 0.85
    vec_scores["vec_only_2"] = 0.82
    bm25_scores["bm25_only_1"] = 18.5
    bm25_scores["bm25_only_2"] = 17.8
    
    print(f"Query: '{query}'")
    print(f"Vector scores sample: {dict(list(vec_scores.items())[:3])}...")
    print(f"BM25 scores sample: {dict(list(bm25_scores.items())[:3])}...")
    
    # Demonstrate all fusion methods
    fusion_methods = ["softmax", "rrf", "linear"]
    alpha_values = [0.3, 0.5, 0.7]
    
    for method in fusion_methods:
        print(f"\nFusion method: {method}")
        for alpha in alpha_values:
            # Apply normalized fusion
            fused_scores = normalized_fusion(
                vec_scores=vec_scores,
                bm25_scores=bm25_scores,
                alpha=alpha,
                method=method
            )
            
            # Get top 3 results
            top_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            
            print(f"  Alpha={alpha:.1f}: Top 3 = {top_results}")

async def demo_context_aware_reranking(query: str, query_vector: List[float]):
    """Demonstrate context-aware reranking"""
    print("\n=== Context-Aware Reranking ===")
    
    # Create mock results for demonstration
    from types import SimpleNamespace
    
    # Create sample results
    results = []
    for i in range(10):
        # Create result with varied content
        if i % 3 == 0:
            # Technical document
            content = f"Technical specification for component {i}. " + \
                     "This includes detailed information about the architecture and implementation. " + \
                     "The system requires careful consideration of various factors."
        elif i % 3 == 1:
            # Short factual document
            content = f"Quick answer: Component {i} has 5 submodules and was created in 2024."
        else:
            # Conceptual document
            content = f"The theory behind component {i} involves several key concepts. " + \
                     "First, we need to understand how the underlying mechanisms work. " + \
                     "This requires a deep analysis of the interactions between various elements. " + \
                     "The implications of this design are far-reaching and complex."
        
        # Create result object
        result = SimpleNamespace(
            id=f"doc_{i}",
            score=0.9 - 0.05 * i,
            payload={"chunk_text": content},
            vector=[0.1] * 1536  # Mock embedding vector
        )
        results.append(result)
    
    print(f"Query: '{query}'")
    print(f"Original results: {[r.id for r in results[:5]]}...")
    
    # Demonstration of different reranking approaches
    rerankers = ["mmr", "context_aware", "diversity"]
    
    for reranker_type in rerankers:
        # Create reranker
        reranker = RerankerFactory.create_reranker(reranker_type)
        
        # Apply reranking
        reranked_results = await reranker.rerank(
            results=results,
            query=query,
            query_vector=query_vector
        )
        
        print(f"\n{reranker_type} reranking:")
        print(f"  New order: {[r.id for r in reranked_results[:5]]}...")
        
        # Show a sample of how content affects reranking
        if reranker_type == "context_aware":
            print("  Sample content relevance:")
            for r in reranked_results[:2]:
                content = r.payload.get("chunk_text", "")
                print(f"  - {r.id}: {content[:100]}...")

async def demo_retrieval_strategy_router(query: str):
    """Demonstrate the retrieval strategy router"""
    print("\n=== Retrieval Strategy Router ===")
    
    # Create query analyzer
    analyzer = QueryAnalyzer()
    
    # Create strategy router
    router = QueryStrategyRouter(analyzer)
    
    # Analyze the query
    query_analysis = analyzer.analyze_query(query)
    
    print(f"Query: '{query}'")
    print(f"Query analysis: {json.dumps(query_analysis, indent=2)}")
    
    # Route the query to determine strategy
    strategy = router.route_query(query, query_analysis)
    
    print(f"\nSelected retrieval strategy: {json.dumps(strategy, indent=2)}")
    
    # Try with different query types for comparison
    sample_queries = [
        "Who invented the light bulb?",  # Factual
        "How does quantum computing work?",  # Conceptual
        "Compare traditional CPUs with quantum processors",  # Complex
        "news from last week",  # Temporal
        "price of $50 in 2010",  # Numerical
    ]
    
    print("\nStrategy comparisons for different query types:")
    for sample_query in sample_queries:
        sample_analysis = analyzer.analyze_query(sample_query)
        sample_strategy = router.route_query(sample_query, sample_analysis)
        print(f"  '{sample_query}' → {sample_strategy['primary_retriever']} " +
              f"(fusion: {sample_strategy['fusion_method']}, weight: {sample_strategy['fusion_weight']})")

async def demo_async_pipeline(query: str, collection: str, client: QdrantClient, openai_client: Any):
    """Demonstrate the asynchronous processing pipeline"""
    print("\n=== Asynchronous Processing Pipeline ===")
    
    # Create the asynchronous pipeline
    pipeline = AsyncRetrievalPipeline(
        client=client,
        openai_client=openai_client,
        default_collection=collection,
        timeout=30.0  # Generous timeout for demo
    )
    
    print(f"Processing query: '{query}'")
    
    # Process the query with full pipeline
    start_time = time.time()
    result = await pipeline.process_query(
        query_text=query,
        collection=collection,
        k=5,  # Get 5 results for demo
        rerank_options={"method": "context_aware"}
    )
    end_time = time.time()
    
    # Display results
    print(f"\nQuery processed in {end_time - start_time:.3f} seconds")
    
    # Show performance metrics
    if "metrics" in result:
        print("\nPerformance metrics:")
        metrics = result["metrics"]
        if "stage_timings" in metrics:
            for stage, duration in metrics["stage_timings"].items():
                print(f"  {stage}: {duration:.3f}s")
        print(f"  Total: {metrics.get('total_time', 0):.3f}s")
    
    # Show strategy used
    if "strategy" in result:
        print(f"\nStrategy used: {result['strategy']['primary_retriever']}")
        print(f"Fusion method: {result['strategy']['fusion_method']}")
        print(f"Fusion weight: {result['strategy']['fusion_weight']}")
    
    # Show results
    if "results" in result and len(result["results"]) > 0:
        print("\nTop results:")
        for i, res in enumerate(result["results"][:3]):  # Show top 3
            content = (res.payload.get("chunk_text", ""))[:100] + "..."
            print(f"  {i+1}. {res.id} (score: {res.score:.4f})")
            print(f"     {content}")
    else:
        print("\nNo results found")

async def demo_performance_telemetry():
    """Demonstrate the performance telemetry capabilities"""
    print("\n=== Performance Telemetry ===")
    
    # Create metrics tracker
    metrics = PerformanceMetrics()
    
    # Track operations of different types
    operations = [
        ("embedding", 0.1),
        ("vector_search", 0.2),
        ("bm25_search", 0.15),
        ("fusion", 0.05),
        ("reranking", 0.1)
    ]
    
    # Run multiple operations
    process_id = metrics.start_operation("process_query", {"query": "sample query"})
    
    print("Executing sample operations...")
    for op_name, duration in operations:
        # Start operation
        op_id = metrics.start_operation(op_name, {"process_id": process_id})
        
        # Simulate work
        time.sleep(duration)
        
        # End operation
        metrics.end_operation(op_id)
        print(f"  {op_name}: {duration:.3f}s")
    
    # End main process
    metrics.end_operation(process_id)
    
    # Display metrics
    stats = metrics.get_metrics(process_id)
    print("\nCollected metrics:")
    print(json.dumps(stats, indent=2))
    
    # Display overall metrics
    overall_stats = metrics.get_metrics()
    print("\nOverall metrics:")
    print(json.dumps(overall_stats, indent=2))

async def main():
    parser = argparse.ArgumentParser(description="Demo of Phase 2 RAG improvements")
    parser.add_argument("--collection", type=str, default="rag_data", 
                        help="Qdrant collection name")
    parser.add_argument("--query", type=str, default="How does quantum computing work?",
                        help="Query to use for demonstrations")
    parser.add_argument("--qdrant-url", type=str, default="http://localhost:6333",
                        help="Qdrant server URL")
    parser.add_argument("--openai-api-key", type=str, 
                        help="OpenAI API key (if not set in environment)")
    parser.add_argument("--openai-model", type=str, default="text-embedding-3-large",
                        help="OpenAI embedding model")
    args = parser.parse_args()
    
    # Create Qdrant client
    client = QdrantClient(url=args.qdrant_url)
    
    # Create OpenAI client
    openai_api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("Warning: No OpenAI API key provided. Some features may not work.")
        openai_client = None
    else:
        openai_client = OpenAI(api_key=openai_api_key)
    
    print(f"Demo using collection: {args.collection}")
    print(f"Demo query: '{args.query}'")
    
    # Create a mock embedding for the query
    # In a real application, we would use the OpenAI API to get the embedding
    import numpy as np
    query_vector = np.random.randn(1536).tolist()  # Mock embedding vector
    
    try:
        # Demo Vector Index Optimization
        await optimize_vector_index(client, args.collection)
        
        # Demo Advanced Fusion Methods
        await demo_advanced_fusion_methods(args.query)
        
        # Demo Context-Aware Reranking
        await demo_context_aware_reranking(args.query, query_vector)
        
        # Demo Retrieval Strategy Router
        await demo_retrieval_strategy_router(args.query)
        
        # Demo Performance Telemetry
        await demo_performance_telemetry()
        
        # Only run the full pipeline if we have an OpenAI client
        if openai_client is not None:
            # Demo Asynchronous Processing Pipeline
            await demo_async_pipeline(args.query, args.collection, client, openai_client)
        else:
            print("\n=== Asynchronous Processing Pipeline ===")
            print("Skipping full pipeline demo (no OpenAI API key provided)")
            
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import os
    asyncio.run(main())