# Advanced RAG Features Integration Guide

This guide explains how to integrate the advanced RAG features into your existing RAG pipeline.

## Overview

We've implemented three advanced features to improve your RAG system:

1. **Semantic Document Chunking**: Divides documents based on semantic boundaries rather than arbitrary character counts
2. **Query Expansion**: Expands user queries into multiple variations to improve retrieval
3. **RAG Self-Evaluation**: Evaluates the quality of RAG responses for continuous improvement

## Installation

### Dependencies

These advanced features require additional dependencies:

```bash
pip install transformers torch sentence-transformers openai tqdm
```

## Integration Points

### 1. Semantic Document Chunking

Replace your existing chunking method with the semantic chunking function in your ingestion pipeline.

#### In `ingest_rag.py`:

```python
from advanced_rag import semantic_chunk_text

# Replace the existing chunking method
def _smart_chunk_text(text: str, max_chars: int, overlap: int = 0) -> list[str]:
    """
    Chunk text on semantic boundaries up to max_chars.
    """
    return semantic_chunk_text(text, max_chars=max_chars)
```

### 2. Query Expansion

Add query expansion to your search flow to improve retrieval results.

#### In `query_rag.py`:

```python
from advanced_rag import expand_query

# In your query handling function
def search_with_expansion(query: str, client, collection: str, limit: int = 10):
    """
    Search with query expansion for better retrieval.
    """
    # Initialize OpenAI client
    openai_client = get_openai_client()
    
    # Expand the query into multiple variations
    expanded_queries = expand_query(query, openai_client)
    
    # Search with all query variations
    all_results = []
    for expanded_query in expanded_queries:
        # Get embeddings for this query variation
        embedding = get_embedding(expanded_query, openai_client)
        
        # Search vector database
        results = client.search(
            collection_name=collection,
            query_vector=embedding,
            limit=limit
        )
        all_results.extend(results)
    
    # Deduplicate results
    seen_ids = set()
    unique_results = []
    for result in all_results:
        if result.id not in seen_ids:
            seen_ids.add(result.id)
            unique_results.append(result)
    
    # Return top results
    return sorted(unique_results, key=lambda x: x.score, reverse=True)[:limit]
```

### 3. RAG Self-Evaluation

Add evaluation to measure and improve your RAG system quality.

#### In your application code:

```python
from advanced_rag import evaluate_rag_quality

# After generating a response
def generate_and_evaluate(query: str, retrieved_chunks: list[str]):
    """
    Generate a response and evaluate its quality.
    """
    # Generate answer from retrieved chunks
    answer = generate_answer(query, retrieved_chunks)
    
    # Initialize OpenAI client
    openai_client = get_openai_client()
    
    # Evaluate the RAG response
    evaluation = evaluate_rag_quality(
        query=query,
        retrieved_chunks=retrieved_chunks,
        generated_answer=answer,
        openai_client=openai_client
    )
    
    # Optional: Log evaluation for improvement
    log_evaluation(query, retrieved_chunks, answer, evaluation)
    
    # Return both the answer and evaluation
    return {
        "answer": answer,
        "evaluation": evaluation
    }
```

## Complete Integration Example

Here's how to integrate all three features together:

```python
from advanced_rag import (
    semantic_chunk_text,
    expand_query,
    evaluate_rag_quality
)

# 1. Enhanced Ingestion Process
def enhanced_ingest(document_text, collection, openai_client):
    # Use semantic chunking
    chunks = semantic_chunk_text(document_text, max_chars=1000)
    
    # Process and store chunks as usual
    for chunk in chunks:
        # Get embedding
        embedding = get_embedding(chunk, openai_client)
        
        # Store in vector database
        store_chunk(chunk, embedding, collection)
    
    return len(chunks)

# 2. Enhanced Query Process
def enhanced_query(user_query, collection, openai_client):
    # Expand query to improve retrieval
    expanded_queries = expand_query(user_query, openai_client)
    
    # Search with all expanded queries
    all_results = []
    for query in expanded_queries:
        embedding = get_embedding(query, openai_client)
        results = search_collection(embedding, collection)
        all_results.extend(results)
    
    # Deduplicate and sort results
    deduplicated_results = deduplicate_results(all_results)
    
    # Extract text chunks from results
    chunks = [result.payload.get("chunk_text", "") for result in deduplicated_results]
    
    # Generate answer
    answer = generate_answer(user_query, chunks, openai_client)
    
    # Evaluate RAG quality
    evaluation = evaluate_rag_quality(
        user_query, 
        chunks, 
        answer, 
        openai_client
    )
    
    return {
        "query": user_query,
        "expanded_queries": expanded_queries,
        "chunks": chunks,
        "answer": answer,
        "evaluation": evaluation
    }
```

## Tips for Effective Integration

1. **Gradually adopt features**: Start with one feature at a time to measure its impact
2. **Benchmark performance**: Compare results before and after implementing each feature
3. **Adjust parameters**: Fine-tune chunking sizes, query expansion limits, etc. based on your specific use case
4. **Log evaluations**: Store evaluation results to identify recurring issues and improvement opportunities
5. **Optimize for production**: Add caching for embeddings and query expansions in high-volume applications

## Example Command-Line Integration

```python
# In query_rag.py

parser.add_argument("--use-expansion", action="store_true", help="Use query expansion")
parser.add_argument("--evaluate", action="store_true", help="Evaluate RAG quality")

# In the main function
if args.use_expansion:
    expanded_queries = expand_query(args.query, openai_client)
    print(f"Expanded queries: {expanded_queries}")
    # Use expanded queries for search

if args.evaluate:
    evaluation = evaluate_rag_quality(args.query, chunks, answer, openai_client)
    print(f"RAG Evaluation: {json.dumps(evaluation, indent=2)}")
```

## Next Steps

After integrating these features, consider these additional enhancements:

1. Implement a feedback loop using evaluation results to continuously improve your system
2. Add specialized document processors for different content types (PDFs, URLs, etc.)
3. Explore local embedding models to reduce API costs and latency
4. Implement cross-encoder reranking for higher precision retrieval

For more advanced RAG techniques, refer to the `RAGIMPROVE.md` document.