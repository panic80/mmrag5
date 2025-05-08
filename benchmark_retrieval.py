#!/usr/bin/env python3
"""
benchmark_retrieval.py

Benchmark script to compare performance between original and improved RAG retrieval.
"""
import os
import sys
import time
import json
import argparse
import asyncio
from typing import Dict, List, Any
from statistics import mean, stdev

import click
from qdrant_client import QdrantClient

# Import components from our modules
from ingest_rag import get_openai_client
import improved_retrieval
from query_rag import build_incremental_bm25_index as original_build_index
from query_rag import _mmr_rerank, preprocess_query as original_preprocess_query


class BenchmarkTimer:
    """Simple timer for benchmarking code execution."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time


async def run_improved_retrieval(
    query: str,
    collection: str,
    client: QdrantClient,
    openai_client: Any,
    model: str,
    k: int,
    filter_obj: Any = None,
    alpha: float = 0.5,
    rrf_k: float = 60.0,
    fusion_method: str = "auto",
    bm25_index_path: str = None
) -> List[Any]:
    """Run improved retrieval with parallel execution and caching."""
    return await improved_retrieval.retrieve_and_fuse(
        query,
        collection,
        client,
        openai_client,
        model,
        k,
        filter_obj,
        alpha,
        rrf_k,
        fusion_method,
        None,
        bm25_index_path
    )


def run_original_retrieval(
    query: str,
    collection: str,
    client: QdrantClient,
    openai_client: Any,
    model: str,
    k: int,
    filter_obj: Any = None,
    alpha: float = 0.5,
    rrf_k: float = 60.0,
    bm25_index_path: str = None
) -> List[Any]:
    """Simulate original retrieval flow without running full CLI."""
    # This is a simplified version that follows the original query_rag.py flow
    
    # Preprocess query
    processed_query = original_preprocess_query(query)
    
    # Get embedding
    if hasattr(openai_client, "embeddings"):  # openai>=1.0 style
        resp = openai_client.embeddings.create(model=model, input=[processed_query])
        vector = resp.data[0].embedding
    else:  # legacy style
        resp = openai_client.Embedding.create(model=model, input=[processed_query])
        vector = resp["data"][0]["embedding"]
    
    # Vector search
    if hasattr(client, "query_points"):
        resp = client.query_points(
            collection_name=collection,
            query=vector,
            limit=k,
            with_payload=True,
            with_vectors=True,
            query_filter=filter_obj,
        )
        vector_results = getattr(resp, 'points', [])
    else:
        vector_results = client.search(
            collection_name=collection,
            query_vector=vector,
            limit=k,
            with_payload=True,
            with_vectors=True,
            query_filter=filter_obj,
        )
    
    # Create ranking from vector results
    vec_rank = {result.id: rank for rank, result in enumerate(vector_results, start=1)}
    
    # BM25 search
    from rank_bm25 import BM25Okapi
    
    # Build or load BM25 index
    id2text = original_build_index(
        collection, 
        bm25_index_path or f"{collection}_bm25_index.json", 
        client
    )
    
    # Basic BM25 search
    ids = list(id2text.keys())
    tokenized = [id2text[_id].split() for _id in ids]
    if tokenized and not all(len(tokens) == 0 for tokens in tokenized):
        bm25 = BM25Okapi(tokenized)
        query_tokens = processed_query.split()
        bm25_scores = bm25.get_scores(query_tokens)
        
        top_n = k
        bm25_sorted = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:top_n]
        bm25_rank = {ids[idx]: rank + 1 for rank, (idx, _) in enumerate(bm25_sorted)}
    else:
        bm25_rank = {}
    
    # RRF fusion
    fused_scores = {}
    all_ids = set(vec_rank) | set(bm25_rank)
    
    for pid in all_ids:
        vec_score = 0.0
        bm25_score = 0.0
        
        if pid in vec_rank:
            vec_score = alpha * (1.0 / (rrf_k + vec_rank[pid]))
        if pid in bm25_rank:
            bm25_score = (1.0 - alpha) * (1.0 / (rrf_k + bm25_rank[pid]))
        
        fused_scores[pid] = vec_score + bm25_score
    
    # Lookup payloads
    from qdrant_client.http.models import Filter, HasIdCondition
    from types import SimpleNamespace
    
    fused_ids = [pid for pid, _ in sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)][:k]
    
    if fused_ids:
        # Get payloads for these IDs
        fobj = Filter(must=[HasIdCondition(has_id=fused_ids)])
        records, _ = client.scroll(
            collection_name=collection, 
            scroll_filter=fobj, 
            with_payload=True,
            limit=len(fused_ids)
        )
        
        # Create result objects
        records_by_id = {rec.id: rec for rec in records}
        results = []
        
        for pid in fused_ids:
            if pid in records_by_id:
                record = records_by_id[pid]
                result = SimpleNamespace(
                    id=pid,
                    score=fused_scores[pid],
                    payload=getattr(record, 'payload', {}) or {},
                    vector=getattr(record, 'vector', None),
                )
                results.append(result)
        
        return results
    else:
        return []


async def benchmark_single_query(
    query: str,
    collection: str,
    client: QdrantClient,
    openai_client: Any,
    runs: int = 3,
    warmup: int = 1,
    **kwargs
) -> Dict[str, Any]:
    """Benchmark a single query with both original and improved retrieval."""
    results = {
        "query": query,
        "original": {"times": [], "results_count": 0},
        "improved": {"times": [], "results_count": 0},
    }
    
    # Ensure BM25 index exists (affects both methods)
    index_path = kwargs.get("bm25_index_path", f"{collection}_bm25_index.json")
    
    # Optional warmup runs
    for _ in range(warmup):
        # Warmup original
        with BenchmarkTimer("warmup_original"):
            run_original_retrieval(query, collection, client, openai_client, **kwargs)
        
        # Warmup improved
        with BenchmarkTimer("warmup_improved"):
            await run_improved_retrieval(query, collection, client, openai_client, **kwargs)
        
        # Clear improved cache between runs
        improved_retrieval.results_cache.invalidate()
    
    # Benchmark runs
    for i in range(runs):
        # Run original retrieval
        with BenchmarkTimer("original") as timer:
            original_results = run_original_retrieval(
                query, collection, client, openai_client, **kwargs
            )
        results["original"]["times"].append(timer.elapsed)
        
        if i == 0:  # Only count results once
            results["original"]["results_count"] = len(original_results)
        
        # Run improved retrieval
        with BenchmarkTimer("improved") as timer:
            improved_results = await run_improved_retrieval(
                query, collection, client, openai_client, **kwargs
            )
        results["improved"]["times"].append(timer.elapsed)
        
        if i == 0:  # Only count results once
            results["improved"]["results_count"] = len(improved_results)
        
        # Clear improved cache between runs for fair comparison
        improved_retrieval.results_cache.invalidate()
    
    # Calculate statistics
    for method in ["original", "improved"]:
        times = results[method]["times"]
        results[method]["min"] = min(times)
        results[method]["max"] = max(times)
        results[method]["mean"] = mean(times)
        results[method]["stdev"] = stdev(times) if len(times) > 1 else 0
    
    # Calculate improvement percentage
    if results["original"]["mean"] > 0:
        speedup = (results["original"]["mean"] - results["improved"]["mean"]) / results["original"]["mean"] * 100
        results["speedup_percent"] = speedup
    else:
        results["speedup_percent"] = 0
    
    return results


async def run_benchmark(args):
    """Run benchmark with provided arguments."""
    # Initialize clients
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    openai_client = get_openai_client(openai_api_key)
    
    qdrant_url = args.qdrant_url or f"http://{args.qdrant_host}:{args.qdrant_port}"
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    
    # Load queries
    if args.queries_file:
        with open(args.queries_file, 'r') as f:
            queries = json.load(f)
    else:
        queries = args.queries or ["What is a RAG retrieval pipeline?"]
    
    # Common parameters for retrieval
    common_params = {
        "collection": args.collection,
        "model": args.model,
        "k": args.k,
        "alpha": args.alpha,
        "rrf_k": args.rrf_k,
        "fusion_method": "auto",
        "bm25_index_path": args.bm25_index
    }
    
    print(f"Starting benchmark with {len(queries)} queries...")
    print(f"Collection: {args.collection}")
    print(f"Parameters: k={args.k}, alpha={args.alpha}, runs={args.runs}")
    
    all_results = []
    
    for i, query in enumerate(queries, 1):
        print(f"\nBenchmarking query {i}/{len(queries)}: '{query}'")
        result = await benchmark_single_query(
            query, args.collection, client, openai_client, 
            runs=args.runs, warmup=args.warmup, **common_params
        )
        
        # Print results for this query
        print(f"  Original: {result['original']['mean']:.4f}s ± {result['original']['stdev']:.4f}s")
        print(f"  Improved: {result['improved']['mean']:.4f}s ± {result['improved']['stdev']:.4f}s")
        print(f"  Speedup: {result['speedup_percent']:.1f}%")
        
        all_results.append(result)
    
    # Calculate aggregate statistics
    if all_results:
        original_means = [r["original"]["mean"] for r in all_results]
        improved_means = [r["improved"]["mean"] for r in all_results]
        
        avg_original = mean(original_means)
        avg_improved = mean(improved_means)
        
        overall_speedup = (avg_original - avg_improved) / avg_original * 100
        
        print("\nOverall Results:")
        print(f"  Average Original Time: {avg_original:.4f}s")
        print(f"  Average Improved Time: {avg_improved:.4f}s")
        print(f"  Overall Speedup: {overall_speedup:.1f}%")
    
    # Save results to file if requested
    if args.output:
        output = {
            "params": vars(args),
            "queries": queries,
            "results": all_results,
            "summary": {
                "avg_original": avg_original,
                "avg_improved": avg_improved,
                "overall_speedup": overall_speedup
            }
        }
        
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Benchmark results saved to {args.output}")


def main():
    """Main function to parse arguments and run benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark RAG retrieval performance")
    
    parser.add_argument("--collection", default="rag_data", help="Qdrant collection name")
    parser.add_argument("--k", type=int, default=20, help="Number of results to retrieve")
    parser.add_argument("--model", default="text-embedding-3-large", help="OpenAI embedding model")
    parser.add_argument("--alpha", type=float, default=0.5, help="Fusion weight for vector scores")
    parser.add_argument("--rrf-k", type=float, default=60.0, help="RRF k parameter")
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--qdrant-url", help="Full Qdrant URL (overrides host/port)")
    parser.add_argument("--bm25-index", help="Path to BM25 index file")
    
    parser.add_argument("--queries", nargs="+", help="Queries to benchmark")
    parser.add_argument("--queries-file", help="Path to JSON file containing queries")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs per query")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup runs")
    parser.add_argument("--output", help="Path to save benchmark results JSON")
    
    args = parser.parse_args()
    
    # Ensure we have either queries or queries-file
    if not args.queries and not args.queries_file:
        parser.error("Either --queries or --queries-file must be provided")
    
    # Run the benchmark
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()