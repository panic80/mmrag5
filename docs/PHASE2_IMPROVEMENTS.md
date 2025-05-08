# Phase 2: Advanced Retrieval & Optimization

This document describes the implementation of Phase 2 improvements from the RAG Retrieval Pipeline Comprehensive Improvement Plan.

## Overview

Phase 2 focuses on "Advanced Retrieval & Optimization" with medium impact and medium complexity. These improvements build upon the foundation laid in Phase 1, adding more sophisticated retrieval techniques, better optimization, and advanced processing capabilities.

The following components have been implemented:

1. **Vector Index Optimization**
2. **Advanced Fusion Methods**
3. **Context-Aware Reranking**
4. **Retrieval Strategy Router**
5. **Asynchronous Processing Pipeline**
6. **Performance Telemetry**

## Vector Index Optimization

The `VectorIndexOptimizer` class provides tools to optimize vector indexes for better performance and efficiency:

- **HNSW Parameter Configuration**: Fine-tune the parameters of the HNSW (Hierarchical Navigable Small World) algorithm used in Qdrant for approximate nearest neighbor search. This allows balancing between search speed and recall accuracy.

- **Vector Quantization**: Implement scalar quantization to reduce the memory footprint of vector embeddings while maintaining search quality. This is especially important for large collections where memory usage can become a bottleneck.

- **Dimensionality Reduction**: Provide techniques to reduce the dimensionality of embedding vectors (e.g., from 1536 to 256 dimensions) using PCA or random projection. This improves both search speed and memory efficiency.

### Usage Example

```python
# Create optimizer
optimizer = VectorIndexOptimizer(client)

# Configure HNSW parameters
await optimizer.configure_index_parameters(
    collection="my_collection",
    hnsw_params={
        "m": 16,              # Connections per node
        "ef_construct": 100,  # Construction parameter
        "full_scan_threshold": 10000,
    }
)

# Apply vector quantization
await optimizer.apply_vector_quantization(
    collection="my_collection",
    scalar_quantization=True
)

# Reduce dimensions of vectors
reduced_vectors = await optimizer.reduce_dimensions(
    vectors=original_vectors,
    target_dim=256,
    method="pca"
)
```

## Advanced Fusion Methods

New fusion methods have been implemented to better combine scores from different retrieval approaches:

- **Normalized Fusion**: A flexible fusion approach that properly normalizes scores from different retrievers before combining them, ensuring fair comparison across different scoring distributions.

- **Multiple Fusion Strategies**: Support for various fusion techniques:
  - **RRF (Reciprocal Rank Fusion)**: Works well for factual queries
  - **Softmax Fusion**: Better for conceptual queries, with temperature control
  - **Linear Fusion**: Simple weighted combination with min-max normalization
  - **Logistic Fusion**: Uses logistic normalization for smoother score distributions
  - **Ensemble Fusion**: Combines multiple fusion methods

### Usage Example

```python
# Use normalized fusion with softmax method
fused_scores = normalized_fusion(
    vec_scores=vector_scores,
    bm25_scores=bm25_scores,
    alpha=0.6,    # Weighting (0.6 = 60% vector, 40% BM25)
    method="softmax"
)

# Use the fusion factory for class-based approach
fusion = FusionFactory.create_fusion("softmax")
fused_scores = fusion.fuse(vector_scores, bm25_scores, alpha=0.6)
```

## Context-Aware Reranking

The reranking system has been enhanced with context-aware capabilities:

- **Query-Context Interaction**: Reranking considers the relationship between query context and document content, not just similarity scores

- **Diversity-Aware Reranking**: Option to boost diversity in results, particularly valuable for complex or ambiguous queries

- **Multiple Reranking Strategies**:
  - **MMR (Maximal Marginal Relevance)**: Balances relevance with diversity
  - **Cross-Encoder Reranking**: Uses cross-encoder models for deeper relevance assessment
  - **Context-Aware Reranking**: Considers query characteristics and document properties
  - **Diversity-Aware Reranking**: Uses clustering to ensure diverse results

### Usage Example

```python
# Create a context-aware reranker
reranker = RerankerFactory.create_reranker("context_aware")

# Apply reranking
reranked_results = await reranker.rerank(
    results=initial_results,
    query=query_text,
    query_vector=query_embedding,
    query_analysis=query_analysis
)
```

## Retrieval Strategy Router

The `QueryStrategyRouter` intelligently selects the optimal retrieval approach based on query analysis:

- **Query Type Detection**: Identifies factual, conceptual, temporal, or complex queries

- **Dynamic Parameter Selection**: Chooses appropriate fusion weights, methods, and reranking thresholds

- **Strategy Mapping**: Maps query characteristics to retrieval strategies

### Usage Example

```python
# Create the router
router = QueryStrategyRouter(query_analyzer)

# Get the optimal strategy for a query
strategy = router.route_query(query_text, query_analysis)

# Example strategy output
{
  "primary_retriever": "vector",
  "secondary_retriever": "bm25",
  "fusion_weight": 0.7,
  "fusion_method": "softmax",
  "rerank_threshold": 0.4
}
```

## Asynchronous Processing Pipeline

The `AsyncRetrievalPipeline` provides a fully asynchronous pipeline for the entire retrieval process:

- **End-to-End Async Execution**: Handles the entire query processing flow asynchronously

- **Integrated Error Handling**: Includes retries, timeouts, and graceful error management

- **Caching Integration**: Seamlessly integrates with the caching system

### Usage Example

```python
# Create pipeline
pipeline = AsyncRetrievalPipeline(
    client=qdrant_client,
    openai_client=openai_client,
    default_collection="my_collection",
    timeout=10.0
)

# Process a query through the entire pipeline
result = await pipeline.process_query(
    query_text="How does quantum computing work?",
    k=10,
    rerank_options={"method": "context_aware"}
)
```

## Performance Telemetry

The `PerformanceMetrics` class provides comprehensive monitoring of retrieval performance:

- **Detailed Stage Timing**: Track performance of each pipeline stage

- **Operation Tracking**: Log performance data for all operations

- **Error Monitoring**: Capture and analyze error patterns

### Usage Example

```python
# Create metrics tracker
metrics = PerformanceMetrics()

# Track an operation
process_id = metrics.start_operation("process_query", {"query": query_text})

# Track sub-operations
embed_id = metrics.start_operation("embedding", {"process_id": process_id})
# ... perform embedding ...
metrics.end_operation(embed_id)

# ... more operations ...

# End main operation
metrics.end_operation(process_id)

# Get performance data
stats = metrics.get_metrics(process_id)
```

## Demo

A demonstration script (`demo_phase2_improvements.py`) is provided to showcase all Phase 2 improvements. Run it with:

```bash
python demo_phase2_improvements.py --collection your_collection --query "your query here"
```

The demo illustrates:
- Vector index optimization capabilities
- Advanced fusion methods with different parameters
- Context-aware reranking with different strategies
- Strategy routing based on query analysis
- Full asynchronous pipeline execution
- Performance telemetry data collection and reporting

## Conclusion

The Phase 2 improvements significantly enhance the RAG retrieval pipeline with:

- Better performance through vector optimization techniques
- More sophisticated fusion methods for improved relevance
- Context-aware reranking for better result ordering
- Intelligent strategy selection based on query analysis
- Fully asynchronous operation for better user experience
- Comprehensive performance tracking

These improvements build upon the Phase 1 foundation and set the stage for the more advanced Phase 3 enhancements.