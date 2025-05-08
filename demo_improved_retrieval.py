#!/usr/bin/env python3
"""
demo_improved_retrieval.py

Demonstration script for the improved RAG retrieval system.
"""
import os
import sys
import json
import asyncio
import argparse
from typing import Any, List, Dict, Optional

import click
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from ingest_rag import get_openai_client
import improved_retrieval


async def format_results(results: List[Any], include_snippets: bool = True) -> List[Dict[str, Any]]:
    """Format search results for display."""
    formatted = []
    
    for idx, result in enumerate(results, start=1):
        item = {
            "rank": idx,
            "id": getattr(result, "id", "unknown"),
            "score": getattr(result, "score", 0.0),
        }
        
        if include_snippets:
            payload = getattr(result, "payload", {}) or {}
            text = payload.get("chunk_text", "")
            if text:
                snippet = text[:200] + "..." if len(text) > 200 else text
                item["snippet"] = snippet
            
            # Add metadata if available
            metadata = payload.get("metadata", {})
            if metadata:
                item["metadata"] = metadata
        
        formatted.append(item)
    
    return formatted


async def demo_search(args):
    """Run demonstration search with provided arguments."""
    # Initialize clients
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    openai_client = get_openai_client(openai_api_key)
    
    qdrant_url = args.qdrant_url or f"http://{args.qdrant_host}:{args.qdrant_port}"
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    
    # Build filter if provided
    filter_obj = None
    if args.filter:
        conditions = []
        for f in args.filter:
            if "=" not in f:
                print(f"ERROR: Invalid filter '{f}'; must be key=value")
                sys.exit(1)
            key, val = f.split("=", 1)
            conditions.append(FieldCondition(key=key, match=MatchValue(value=val)))
        filter_obj = Filter(must=conditions)
    
    # Preprocess query
    processed_query = improved_retrieval.preprocess_query(args.query)
    print(f"Query: {args.query}")
    print(f"Processed query: {processed_query}")
    
    # Analyze query
    analysis = improved_retrieval.analyze_query(processed_query, args.alpha)
    print("\nQuery Analysis:")
    print(f"  Type: {analysis['query_type']}")
    print(f"  Optimal alpha: {analysis['alpha']:.2f}")
    print(f"  Fusion method: {analysis['fusion_method']}")
    if analysis['entities']:
        print(f"  Entities: {', '.join(analysis['entities'])}")
    
    # Get embedding
    print("\nGenerating embedding...")
    embedding = improved_retrieval.get_cached_embedding(
        processed_query, args.model, openai_client
    )
    print(f"  Embedding dimensions: {len(embedding)}")
    
    # Perform search
    print("\nPerforming retrieval...")
    results = await improved_retrieval.retrieve_and_fuse(
        query_text=processed_query,
        collection=args.collection,
        client=client,
        openai_client=openai_client,
        model=args.model,
        k=args.k,
        filter_obj=filter_obj,
        alpha=args.alpha,
        rrf_k=args.rrf_k,
        fusion_method="auto",
        bm25_index=None,
        index_path=args.bm25_index
    )
    
    # Format and display results
    formatted_results = await format_results(results, include_snippets=True)
    
    print(f"\nFound {len(results)} results:")
    for item in formatted_results:
        print(f"\n[{item['rank']}] ID: {item['id']}, Score: {item['score']:.4f}")
        if "snippet" in item:
            print(f"  Snippet: {item['snippet']}")
        if "metadata" in item:
            print(f"  Metadata: {json.dumps(item['metadata'], indent=2)}")
    
    # Additional information
    cache_stats = improved_retrieval.results_cache.get_stats()
    print("\nCache Stats:")
    print(f"  Size: {cache_stats['size']} entries")
    print(f"  Hit ratio: {cache_stats['hit_ratio']:.2%}")
    
    # Save results to file if requested
    if args.output:
        output = {
            "query": args.query,
            "processed_query": processed_query,
            "analysis": analysis,
            "results": formatted_results,
            "cache_stats": cache_stats
        }
        
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to {args.output}")


def main():
    """Main function to parse arguments and run demo."""
    parser = argparse.ArgumentParser(description="Demo of improved RAG retrieval")
    
    parser.add_argument("query", help="Query text")
    parser.add_argument("--collection", default="rag_data", help="Qdrant collection name")
    parser.add_argument("--k", type=int, default=5, help="Number of results to retrieve")
    parser.add_argument("--model", default="text-embedding-3-large", help="OpenAI embedding model")
    parser.add_argument("--alpha", type=float, default=0.5, help="Fusion weight for vector scores")
    parser.add_argument("--rrf-k", type=float, default=60.0, help="RRF k parameter")
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--qdrant-url", help="Full Qdrant URL (overrides host/port)")
    parser.add_argument("--bm25-index", help="Path to BM25 index file")
    parser.add_argument("--filter", action="append", help="Filter by payload key=value")
    parser.add_argument("--output", help="Path to save results JSON")
    
    args = parser.parse_args()
    
    # Run the demo
    asyncio.run(demo_search(args))


if __name__ == "__main__":
    main()