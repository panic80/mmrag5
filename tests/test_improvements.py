#!/usr/bin/env python3

"""
Test script to verify our semantic chunking improvements.
This script tests both functionalities directly and ensures they are properly integrated.
"""

import os
import sys
import time

# Import the improved functions
from ingest_rag import semantic_chunk_text, Document

def test_semantic_chunking():
    """Test the improved semantic chunking functionality."""
    print("Testing semantic chunking...")
    
    # Sample text with mixed content
    sample_text = """
    # Introduction
    
    This is a sample document with multiple paragraphs and different types of content.
    
    ## Section 1
    
    This section contains regular text that should be chunked semantically.
    Sentences within the same paragraph should stay together when possible.
    
    * Bullet point 1
    * Bullet point 2
    
    ## Section 2
    
    ```python
    def example_function():
        # This demonstrates code content
        print("Hello world")
        return True
    ```
    
    | Header 1 | Header 2 |
    |----------|----------|
    | Value 1  | Value 2  |
    """
    
    # Test with different configurations
    print("\nStandard semantic chunking:")
    chunks = semantic_chunk_text(sample_text, max_chars=300, fast_mode=True)
    print(f"Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} ({len(chunk)} chars):")
        print("-" * 50)
        print(chunk[:150] + "..." if len(chunk) > 150 else chunk)
    
    print("\nPrecise semantic chunking:")
    chunks = semantic_chunk_text(sample_text, max_chars=300, fast_mode=False)
    print(f"Created {len(chunks)} chunks")
    
    print("\nAdaptive chunking (if available):")
    try:
        chunks = semantic_chunk_text(sample_text, max_chars=300, use_adaptive=True)
        print(f"Created {len(chunks)} chunks")
    except Exception as e:
        print(f"Adaptive chunking not available: {e}")

def test_document_creation():
    """Test basic Document functionality."""
    print("\nTesting Document creation...")
    
    # Sample text
    sample_text = "This is a sample document for testing."
    
    # Metadata
    metadata = {"source": "test_document", "type": "sample"}
    
    # Create a Document directly
    document = Document(content=sample_text, metadata=metadata)
    print(f"Content: {document.content}")
    print(f"Metadata: {document.metadata}")

def run_tests():
    """Run all improvement tests."""
    print("Starting improvement tests...")
    start_time = time.time()
    
    test_semantic_chunking()
    test_document_creation()
    
    elapsed = time.time() - start_time
    print(f"\nAll tests completed in {elapsed:.2f} seconds")
    print("To run the full test suite, use: pytest tests/test_semantic_chunking.py -v")

if __name__ == "__main__":
    run_tests()