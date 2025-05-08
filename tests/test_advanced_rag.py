#!/usr/bin/env python3

"""
Test script for advanced RAG features.

This script demonstrates how to use the advanced RAG features:
1. Semantic Document Chunking
2. Query Expansion
3. RAG Self-Evaluation

Example usage:
    python test_advanced_rag.py
"""

import os
import sys
import argparse
import json
from typing import List, Dict, Any, Optional

from advanced_rag import (
    semantic_chunk_text,
    expand_query,
    evaluate_rag_quality
)

# Check for OpenAI API key
if "OPENAI_API_KEY" not in os.environ:
    print("Warning: OPENAI_API_KEY not found in environment variables.")
    print("Some features may not work unless you set this environment variable.")
    print("You can set it temporarily with: export OPENAI_API_KEY=your_key_here")


def get_openai_client():
    """Get OpenAI client using either v0 or v1 API based on what's installed."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("Error: OPENAI_API_KEY not set. Cannot initialize client.")
        return None

    try:
        # Try OpenAI v1 first
        import openai
        client = openai.OpenAI(api_key=api_key)
        print("Using OpenAI Python SDK v1")
        return client
    except (ImportError, AttributeError):
        try:
            # Fall back to v0
            import openai
            openai.api_key = api_key
            print("Using OpenAI Python SDK v0")
            return openai
        except ImportError:
            print("Error: OpenAI SDK not installed. Please run: pip install openai")
            return None


def test_semantic_chunking(text=None, max_chars=1000):
    """Test semantic document chunking."""
    print("\n=== Testing Semantic Document Chunking ===\n")
    
    if text is None:
        text = """# Introduction to RAG Systems

Retrieval-Augmented Generation (RAG) is a technique that enhances large language models by incorporating external knowledge. 
Instead of relying solely on the model's parameters, RAG systems retrieve relevant information from a knowledge base to inform the generation process.

## How RAG Works

RAG systems work in three main steps:
1. The user query is processed
2. Relevant documents are retrieved from a knowledge base
3. The retrieved documents and the query are fed to a language model to generate a response

This approach helps address limitations of traditional LLMs, such as:
- Outdated knowledge from training data
- Hallucinations (generating incorrect information)
- Limited context windows

## Key Components

Effective RAG systems require several components:
- Document processing and chunking
- Vector embeddings for semantic search
- Efficient vector databases
- Advanced retrieval strategies

### Document Processing

Document processing involves converting various document formats into chunks that can be embedded and retrieved. 
This includes techniques for splitting documents, handling tables and images, and preserving document structure.

### Vector Embeddings

Vector embeddings are numerical representations of text that capture semantic meaning.
They allow for similarity search based on meaning rather than just keywords.

## Advanced Techniques

Recent advances in RAG systems include:
- Hybrid search combining semantic and keyword approaches
- Re-ranking retrieved documents to improve relevance
- Query expansion to capture different aspects of the user's intent
- Multi-vector retrieval for better representation of documents
"""
    
    if not text:
        print("Error: Empty text provided")
        return
    
    print(f"Original text length: {len(text)} characters")
    
    # Get semantic chunks
    chunks = semantic_chunk_text(text, max_chars=max_chars)
    
    print(f"Generated {len(chunks)} semantic chunks")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} ({len(chunk)} chars):")
        # Print first 100 chars of each chunk
        print(chunk[:100] + "..." if len(chunk) > 100 else chunk)
    
    return chunks


def test_query_expansion(query=None):
    """Test query expansion."""
    print("\n=== Testing Query Expansion ===\n")
    
    if query is None:
        query = "How does RAG help with hallucination?"
    
    print(f"Original query: '{query}'")
    
    # Initialize OpenAI client
    client = get_openai_client()
    if not client:
        print("Error: OpenAI client initialization failed")
        return
    
    # Expand query
    expanded_queries = expand_query(query, client)
    
    print(f"Generated {len(expanded_queries)} expanded queries:")
    for i, expanded in enumerate(expanded_queries):
        print(f"{i+1}. {expanded}")
    
    return expanded_queries


def test_rag_evaluation():
    """Test RAG self-evaluation."""
    print("\n=== Testing RAG Self-Evaluation ===\n")
    
    # Sample query
    query = "What are the benefits of using RAG systems?"
    
    # Sample retrieved chunks
    retrieved_chunks = [
        """Effective RAG systems require several components:
- Document processing and chunking
- Vector embeddings for semantic search
- Efficient vector databases
- Advanced retrieval strategies""",
        
        """This approach helps address limitations of traditional LLMs, such as:
- Outdated knowledge from training data
- Hallucinations (generating incorrect information)
- Limited context windows"""
    ]
    
    # Sample generated answer
    generated_answer = """RAG (Retrieval-Augmented Generation) systems offer several key benefits:

1. They help address the problem of outdated knowledge in LLMs by retrieving current information from external sources.
2. They significantly reduce hallucinations by grounding responses in retrieved documents.
3. They effectively extend the context window by bringing in relevant information from external sources.
4. They allow for more transparent responses since the retrieved documents can be cited as sources.

These benefits make RAG an important technique for creating more reliable and trustworthy AI systems."""
    
    print(f"Query: {query}")
    print("\nRetrieved chunks:")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"\nChunk {i+1}:")
        print(chunk)
    
    print("\nGenerated answer:")
    print(generated_answer)
    
    # Initialize OpenAI client
    client = get_openai_client()
    if not client:
        print("Error: OpenAI client initialization failed")
        return
    
    # Evaluate RAG quality
    evaluation = evaluate_rag_quality(query, retrieved_chunks, generated_answer, client)
    
    print("\nEvaluation results:")
    print(json.dumps(evaluation, indent=2))
    
    return evaluation


def main():
    """Main function to run tests."""
    parser = argparse.ArgumentParser(description="Test advanced RAG features")
    parser.add_argument("--feature", choices=["chunking", "expansion", "evaluation", "all"], 
                      default="all", help="Feature to test")
    args = parser.parse_args()
    
    if args.feature in ["chunking", "all"]:
        test_semantic_chunking()
    
    if args.feature in ["expansion", "all"]:
        test_query_expansion()
    
    if args.feature in ["evaluation", "all"]:
        test_rag_evaluation()


if __name__ == "__main__":
    main()