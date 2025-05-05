#!/usr/bin/env python3

"""
Post-Ingest Validation and Testing for RAG Systems

This module provides functionality to validate and test the quality of RAG system
ingestion. It includes checks for embedding quality, retrieval accuracy, and
system performance. These validations help ensure that the embeddings will work
effectively in retrieval scenarios.

Features:
- Embedding consistency checks
- Retrieval validation with test queries
- Semantic drift detection
- Content coverage analysis
- Embedding quality metrics

Usage:
    from ingest_validation import validate_ingestion, run_test_queries
    
    # Validate ingestion results for a Qdrant collection
    validation_results = validate_ingestion(qdrant_client, "my_collection")
    
    # Run test queries to verify retrieval quality
    query_results = run_test_queries(qdrant_client, "my_collection", openai_client)
"""

import logging
import time
import json
import re
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import random
import math
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Results from validation testing."""
    test_name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    message: str

@dataclass
class ValidationSummary:
    """Summary of all validation results."""
    total_tests: int
    passed_tests: int
    failed_tests: int
    average_score: float
    results: List[ValidationResult]
    overall_status: str
    critical_issues: List[str]

def generate_test_queries(documents: List[Dict[str, Any]], num_queries: int = 5) -> List[Dict[str, Any]]:
    """
    Generate test queries based on document content for validation.
    
    Args:
        documents: List of documents with content and metadata
        num_queries: Number of test queries to generate
        
    Returns:
        List of test query objects
    """
    if not documents:
        return []
    
    # Ensure we don't try to generate more queries than we have documents
    num_queries = min(num_queries, len(documents))
    
    # Randomly select documents to generate queries from
    selected_docs = random.sample(documents, num_queries)
    
    queries = []
    
    for doc in selected_docs:
        content = doc.get("content", "")
        if not content:
            continue
            
        # Extract a key sentence or phrase from the document
        sentences = re.split(r'(?<=[.!?])\s+', content)
        if not sentences:
            continue
            
        # Try to find an informative sentence (longer than 30 chars, contains a verb)
        informative_sentences = [s for s in sentences if len(s) > 30 and re.search(r'\b(?:is|are|was|were|has|have|do|does|did|can|could|may|might|shall|should|will|would)\b', s.lower())]
        
        if informative_sentences:
            base_sentence = random.choice(informative_sentences)
        else:
            # Fall back to longest sentence
            base_sentence = max(sentences, key=len)
        
        # Transform the sentence into a question
        # First try to identify key information to query
        key_terms = []
        for term in re.findall(r'\b[A-Z][a-zA-Z0-9_]+\b', base_sentence):  # Proper nouns
            if len(term) > 2 and term.lower() not in ["the", "and", "for", "this", "that"]:
                key_terms.append(term)
                
        if not key_terms and len(base_sentence.split()) > 5:
            # Extract common nouns as fallback
            for term in re.findall(r'\b[a-z]+\b', base_sentence.lower()):
                if len(term) > 4 and term not in ["about", "these", "those", "their", "there"]:
                    key_terms.append(term)
        
        if key_terms:
            # Create a query about a key term
            key_term = random.choice(key_terms)
            query_text = f"Tell me about {key_term} in context of this document."
        else:
            # Create a generic summary query
            query_text = f"Summarize the main points from this text: '{base_sentence}'"
        
        # Create query object with expected document ID
        metadata = doc.get("metadata", {})
        query = {
            "text": query_text,
            "expected_doc_id": metadata.get("id", ""),
            "source": metadata.get("source", ""),
            "context": base_sentence
        }
        
        queries.append(query)
    
    return queries

def validate_embedding_consistency(embeddings: List[List[float]]) -> ValidationResult:
    """
    Check embedding vectors for consistency and quality.
    
    Args:
        embeddings: List of embedding vectors
        
    Returns:
        ValidationResult with consistency metrics
    """
    if not embeddings or len(embeddings) < 2:
        return ValidationResult(
            test_name="embedding_consistency",
            passed=False,
            score=0.0,
            details={"error": "Not enough embeddings to validate"},
            message="Insufficient embeddings for validation"
        )
    
    # Check embedding dimensions
    dimensions = [len(emb) for emb in embeddings]
    consistent_dimensions = len(set(dimensions)) == 1
    
    # Check for zero vectors
    zero_vectors = sum(1 for emb in embeddings if all(abs(val) < 1e-6 for val in emb))
    zero_ratio = zero_vectors / len(embeddings) if embeddings else 0
    
    # Check for null values
    null_values = sum(1 for emb in embeddings for val in emb if math.isnan(val) or math.isinf(val))
    
    # Calculate embedding statistics (mean, std, etc)
    magnitudes = [math.sqrt(sum(val**2 for val in emb)) for emb in embeddings]
    mean_magnitude = statistics.mean(magnitudes) if magnitudes else 0
    std_magnitude = statistics.stdev(magnitudes) if len(magnitudes) > 1 else 0
    
    # Compute cosine similarities between random pairs
    similarities = []
    sample_count = min(100, len(embeddings))
    for _ in range(sample_count):
        i, j = random.sample(range(len(embeddings)), 2)
        emb1, emb2 = embeddings[i], embeddings[j]
        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        mag1 = math.sqrt(sum(a**2 for a in emb1))
        mag2 = math.sqrt(sum(b**2 for b in emb2))
        similarity = dot_product / (mag1 * mag2) if mag1 * mag2 > 0 else 0
        similarities.append(similarity)
    
    avg_similarity = statistics.mean(similarities) if similarities else 0
    
    # Calculate overall score
    score = 1.0
    
    if not consistent_dimensions:
        score -= 0.5
    
    if zero_ratio > 0:
        score -= min(0.5, zero_ratio * 5)  # Penalize up to 0.5 for zero vectors
    
    if null_values > 0:
        score -= min(0.5, null_values / (len(embeddings) * dimensions[0]) * 10)
    
    # Similarity should be in a reasonable range (not too high, not too low)
    if avg_similarity > 0.95:  # Too similar (might be duplicates)
        score -= min(0.3, (avg_similarity - 0.95) * 6)
    elif avg_similarity < 0.1:  # Too dissimilar (might be random)
        score -= min(0.3, (0.1 - avg_similarity) * 3)
    
    # Check if passed
    passed = score >= 0.7 and consistent_dimensions and zero_ratio < 0.01 and null_values == 0
    
    return ValidationResult(
        test_name="embedding_consistency",
        passed=passed,
        score=score,
        details={
            "consistent_dimensions": consistent_dimensions,
            "dimensions": dimensions[0] if consistent_dimensions else dimensions,
            "zero_vectors": zero_vectors,
            "zero_ratio": zero_ratio,
            "null_values": null_values,
            "mean_magnitude": mean_magnitude,
            "std_magnitude": std_magnitude,
            "avg_similarity": avg_similarity
        },
        message="Embedding consistency check " + ("passed" if passed else "failed")
    )

def validate_embedding_coverage(collection_info: Dict[str, Any]) -> ValidationResult:
    """
    Validate that embeddings cover the expected content range.
    
    Args:
        collection_info: Information about the Qdrant collection
        
    Returns:
        ValidationResult with coverage metrics
    """
    vector_count = collection_info.get("vectors_count", 0)
    
    if vector_count == 0:
        return ValidationResult(
            test_name="embedding_coverage",
            passed=False,
            score=0.0,
            details={"error": "No vectors found in collection"},
            message="Collection has no vectors"
        )
    
    # Check if vector count seems reasonable (at least 10)
    reasonable_count = vector_count >= 10
    
    # Calculate score based on vector count
    if vector_count < 10:
        score = 0.3
    elif vector_count < 50:
        score = 0.6
    elif vector_count < 100:
        score = 0.8
    else:
        score = 1.0
    
    return ValidationResult(
        test_name="embedding_coverage",
        passed=reasonable_count,
        score=score,
        details={
            "vector_count": vector_count,
            "reasonable_count": reasonable_count
        },
        message=f"Found {vector_count} vectors in collection"
    )

def run_test_query(
    client: Any, 
    collection: str, 
    query: Dict[str, Any],
    openai_client: Any,
    model: str = "text-embedding-3-large",
    limit: int = 5
) -> Dict[str, Any]:
    """
    Run a test query against the Qdrant collection.
    
    Args:
        client: Qdrant client
        collection: Collection name
        query: Query object with text and expected result
        openai_client: OpenAI client for embeddings
        model: Embedding model name
        limit: Number of results to retrieve
        
    Returns:
        Query test results
    """
    query_text = query.get("text", "")
    expected_doc_id = query.get("expected_doc_id", "")
    
    # Create embedding for query
    if hasattr(openai_client, "embeddings"):  # openai>=1.0 style
        resp = openai_client.embeddings.create(model=model, input=[query_text])
        query_vector = resp.data[0].embedding
    else:  # legacy style
        resp = openai_client.Embedding.create(model=model, input=[query_text])
        query_vector = resp["data"][0]["embedding"]
    
    # Search Qdrant
    start_time = time.time()
    
    try:
        if hasattr(client, "query_points"):
            # New API
            search_results = client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=limit,
                with_payload=True
            )
            results = search_results.points
        else:
            # Deprecated API
            results = client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=limit,
                with_payload=True
            )
    except Exception as e:
        return {
            "query": query_text,
            "error": str(e),
            "success": False,
            "latency": time.time() - start_time
        }
    
    latency = time.time() - start_time
    
    # Extract result information
    hits = []
    hit_position = None
    
    for i, hit in enumerate(results):
        hit_id = getattr(hit, 'id', '')
        score = getattr(hit, 'score', 0.0)
        payload = getattr(hit, 'payload', {}) or {}
        
        # Check if this is the expected document
        if hit_id == expected_doc_id:
            hit_position = i
        
        # Add to hits list
        hits.append({
            "id": hit_id,
            "score": score,
            "payload": payload
        })
    
    # Calculate success metrics
    expected_found = hit_position is not None
    
    return {
        "query": query_text,
        "expected_doc_id": expected_doc_id,
        "hits": hits,
        "hit_position": hit_position,
        "expected_found": expected_found,
        "latency": latency,
        "success": True
    }

def validate_retrieval_quality(query_results: List[Dict[str, Any]]) -> ValidationResult:
    """
    Validate retrieval quality based on test query results.
    
    Args:
        query_results: Results from test queries
        
    Returns:
        ValidationResult with retrieval quality metrics
    """
    if not query_results:
        return ValidationResult(
            test_name="retrieval_quality",
            passed=False,
            score=0.0,
            details={"error": "No query results to evaluate"},
            message="No test queries were run"
        )
    
    # Count successful queries
    successful_queries = sum(1 for result in query_results if result.get("success", False))
    
    # Count queries that found the expected document
    found_expected = sum(1 for result in query_results if result.get("expected_found", False))
    
    # Calculate retrieval metrics
    total_queries = len(query_results)
    success_rate = successful_queries / total_queries if total_queries > 0 else 0
    hit_rate = found_expected / total_queries if total_queries > 0 else 0
    
    # Calculate average position of expected document (when found)
    positions = [result.get("hit_position", None) for result in query_results]
    valid_positions = [pos for pos in positions if pos is not None]
    avg_position = sum(valid_positions) / len(valid_positions) if valid_positions else None
    
    # Calculate average latency
    latencies = [result.get("latency", 0) for result in query_results]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    
    # Calculate score based on hit rate and positions
    score = hit_rate  # Base score on hit rate
    
    # Bonus for finding in top positions
    if valid_positions:
        position_score = sum((5 - pos) / 5 if pos < 5 else 0 for pos in valid_positions) / len(valid_positions)
        score = 0.7 * hit_rate + 0.3 * position_score
    
    # Penalty for failed queries
    if successful_queries < total_queries:
        score *= successful_queries / total_queries
    
    # Determine if test passed
    passed = successful_queries == total_queries and hit_rate >= 0.5
    
    return ValidationResult(
        test_name="retrieval_quality",
        passed=passed,
        score=score,
        details={
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "found_expected": found_expected,
            "hit_rate": hit_rate,
            "avg_position": avg_position,
            "avg_latency": avg_latency
        },
        message=f"Retrieval test: found {found_expected}/{total_queries} expected documents"
    )

def validate_ingestion(client: Any, collection: str) -> ValidationSummary:
    """
    Validate the quality of ingested embeddings in a Qdrant collection.
    
    Args:
        client: Qdrant client
        collection: Collection name
        
    Returns:
        ValidationSummary with test results
    """
    logger.info(f"Validating ingestion for collection '{collection}'")
    
    # Initialize results
    results = []
    
    try:
        # Get collection info
        if hasattr(client, "get_collection"):
            collection_info = client.get_collection(collection_name=collection)
            # Convert from protobuf-like object to dict if needed
            collection_info = {
                "vectors_count": getattr(collection_info, "vectors_count", 0),
                "points_count": getattr(collection_info, "points_count", 0),
                "dimension": getattr(collection_info, "config", {}).get("params", {}).get("dimension", 0),
                "distance": getattr(collection_info, "config", {}).get("params", {}).get("distance", "")
            }
        else:
            # Fallback API
            collections = client.get_collections()
            collection_info = next((c for c in collections.collections if c.name == collection), {})
            collection_info = {
                "vectors_count": getattr(collection_info, "vectors_count", 0),
                "points_count": getattr(collection_info, "points_count", 0),
            }
        
        # Validate embedding coverage
        coverage_result = validate_embedding_coverage(collection_info)
        results.append(coverage_result)
        
        # Get sample of vectors for consistency check
        sample_size = min(1000, collection_info.get("vectors_count", 0))
        
        if sample_size > 0:
            embeddings = []
            
            # Fetch sample vectors
            try:
                if hasattr(client, "scroll"):
                    # Use scroll API to get vectors
                    records, _ = client.scroll(
                        collection_name=collection,
                        limit=sample_size,
                        with_vectors=True
                    )
                    
                    for rec in records:
                        vector = getattr(rec, "vector", None)
                        if vector:
                            embeddings.append(vector)
                else:
                    # No good way to get vectors with older API
                    pass
                
                # Validate embedding consistency if we got vectors
                if embeddings:
                    consistency_result = validate_embedding_consistency(embeddings)
                    results.append(consistency_result)
            except Exception as e:
                logger.error(f"Error fetching vectors: {e}")
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        results.append(ValidationResult(
            test_name="collection_validation",
            passed=False,
            score=0.0,
            details={"error": str(e)},
            message=f"Error validating collection: {e}"
        ))
    
    # Compile summary
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.passed)
    average_score = sum(r.score for r in results) / total_tests if total_tests > 0 else 0
    
    # Determine overall status
    if passed_tests == total_tests:
        status = "PASSED"
    elif passed_tests >= total_tests * 0.7:
        status = "PARTIAL"
    else:
        status = "FAILED"
    
    # Identify critical issues
    critical_issues = [
        r.message for r in results 
        if not r.passed and r.score < 0.5
    ]
    
    return ValidationSummary(
        total_tests=total_tests,
        passed_tests=passed_tests,
        failed_tests=total_tests - passed_tests,
        average_score=average_score,
        results=results,
        overall_status=status,
        critical_issues=critical_issues
    )

def run_test_queries(
    client: Any, 
    collection: str, 
    openai_client: Any,
    model: str = "text-embedding-3-large",
    num_queries: int = 5
) -> ValidationResult:
    """
    Run test queries to validate retrieval quality.
    
    Args:
        client: Qdrant client
        collection: Collection name
        openai_client: OpenAI client for embeddings
        model: Embedding model name
        num_queries: Number of test queries to run
        
    Returns:
        ValidationResult with retrieval quality metrics
    """
    logger.info(f"Running {num_queries} test queries on collection '{collection}'")
    
    try:
        # Fetch some documents to generate queries from
        if hasattr(client, "scroll"):
            # Use scroll API to get documents with payload
            records, _ = client.scroll(
                collection_name=collection,
                limit=100,  # Get a sample to choose from
                with_payload=True
            )
            
            # Convert to document dictionaries
            documents = []
            for rec in records:
                payload = getattr(rec, "payload", {}) or {}
                content = payload.get("chunk_text", "")
                if not content and "text" in payload:
                    content = payload.get("text", "")
                
                if content:
                    documents.append({
                        "content": content,
                        "metadata": {
                            "id": getattr(rec, "id", ""),
                            "source": payload.get("source", "")
                        }
                    })
        else:
            # No good way to get documents with older API
            documents = []
        
        if not documents:
            return ValidationResult(
                test_name="retrieval_quality",
                passed=False,
                score=0.0,
                details={"error": "No documents found to generate test queries"},
                message="No documents available for testing"
            )
        
        # Generate test queries
        test_queries = generate_test_queries(documents, num_queries)
        
        if not test_queries:
            return ValidationResult(
                test_name="retrieval_quality",
                passed=False,
                score=0.0,
                details={"error": "Failed to generate test queries"},
                message="Could not generate test queries"
            )
        
        # Run the queries
        query_results = []
        
        for query in test_queries:
            result = run_test_query(
                client=client,
                collection=collection,
                query=query,
                openai_client=openai_client,
                model=model
            )
            query_results.append(result)
        
        # Validate retrieval quality
        return validate_retrieval_quality(query_results)
        
    except Exception as e:
        logger.error(f"Test query error: {e}")
        return ValidationResult(
            test_name="retrieval_quality",
            passed=False,
            score=0.0,
            details={"error": str(e)},
            message=f"Error running test queries: {e}"
        )

if __name__ == "__main__":
    """Command line interface for testing ingestion validation."""
    import argparse
    import sys
    import os
    
    parser = argparse.ArgumentParser(description="Validate RAG ingestion quality")
    parser.add_argument("--collection", help="Qdrant collection to validate", required=True)
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--openai-api-key", help="OpenAI API key (for test queries)")
    parser.add_argument("--test-queries", action="store_true", help="Run test queries")
    
    args = parser.parse_args()
    
    try:
        # Import Qdrant client
        from qdrant_client import QdrantClient
        
        # Connect to Qdrant
        client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)
        
        # Run validation
        validation = validate_ingestion(client, args.collection)
        
        # Print validation results
        print(f"\nValidation Results for Collection: {args.collection}")
        print(f"Status: {validation.overall_status}")
        print(f"Tests Passed: {validation.passed_tests}/{validation.total_tests}")
        print(f"Overall Score: {validation.average_score:.2f}/1.00")
        
        # Print individual test results
        print("\nTest Details:")
        for result in validation.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"{status} [{result.test_name}] Score: {result.score:.2f} - {result.message}")
        
        # Print critical issues
        if validation.critical_issues:
            print("\nCritical Issues:")
            for issue in validation.critical_issues:
                print(f"- {issue}")
        
        # Run test queries if requested
        if args.test_queries:
            if not args.openai_api_key and "OPENAI_API_KEY" not in os.environ:
                print("\nError: OpenAI API key required for test queries")
                print("Provide with --openai-api-key or set OPENAI_API_KEY environment variable")
                sys.exit(1)
            
            api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
            
            # Initialize OpenAI client
            try:
                import openai
                
                if hasattr(openai, "OpenAI"):
                    # OpenAI v1.x
                    openai_client = openai.OpenAI(api_key=api_key)
                else:
                    # OpenAI v0.x
                    openai.api_key = api_key
                    openai_client = openai
                
                # Run test queries
                print("\nRunning Test Queries...")
                query_result = run_test_queries(client, args.collection, openai_client)
                
                # Print query test results
                status = "✅ PASS" if query_result.passed else "❌ FAIL"
                print(f"\nQuery Test: {status} - Score: {query_result.score:.2f}")
                print(f"Message: {query_result.message}")
                
                # Print detailed metrics
                details = query_result.details
                print(f"Queries Run: {details.get('total_queries', 0)}")
                print(f"Hit Rate: {details.get('hit_rate', 0):.2f}")
                if details.get('avg_position') is not None:
                    print(f"Average Position: {details.get('avg_position', 0):.1f}")
                print(f"Average Latency: {details.get('avg_latency', 0):.3f} seconds")
                
            except ImportError:
                print("\nError: OpenAI Python package not installed")
                print("Install with: pip install openai")
                sys.exit(1)
            except Exception as e:
                print(f"\nError running test queries: {e}")
                sys.exit(1)
        
    except ImportError:
        print("Error: qdrant_client not installed")
        print("Install with: pip install qdrant-client")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)