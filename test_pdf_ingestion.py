#!/usr/bin/env python3

"""
Test script for PDF ingestion flow
This script tests the PDF ingestion and embedding pipeline without requiring all dependencies
"""

import os
import sys
import json
from datetime import datetime

# Test document path
TEST_DOC = "test_document.txt"

def log(message):
    """Print a log message with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def test_document_loading():
    """Test document loading functionality"""
    log("Testing document loading...")
    
    try:
        with open(TEST_DOC, "r") as f:
            content = f.read()
        
        log(f"Successfully loaded document: {len(content)} characters")
        return content
    except Exception as e:
        log(f"Error loading document: {e}")
        return None

def test_chunking(text):
    """Test text chunking functionality"""
    log("Testing text chunking...")
    
    try:
        # Simple paragraph-based chunking
        paragraphs = text.split("\n\n")
        chunks = []
        
        for i, para in enumerate(paragraphs):
            if para.strip():
                chunks.append({
                    "content": para.strip(),
                    "metadata": {
                        "chunk_index": i,
                        "source": TEST_DOC
                    }
                })
        
        log(f"Successfully chunked text into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        log(f"Error chunking text: {e}")
        return []

def test_metadata_extraction(chunks):
    """Test metadata extraction functionality"""
    log("Testing metadata extraction...")
    
    try:
        import re
        
        enriched_chunks = []
        
        for chunk in chunks:
            metadata = chunk["metadata"].copy()
            content = chunk["content"]
            
            # Extract basic metadata
            # 1. Title - first line if it starts with # 
            if content.startswith("#"):
                title_match = re.match(r"^#+\s+(.*?)$", content.split("\n")[0])
                if title_match:
                    metadata["title"] = title_match.group(1)
            
            # 2. Key topics - extract bullet points
            bullet_points = re.findall(r"^-\s+(.*?)$", content, re.MULTILINE)
            if bullet_points:
                metadata["bullet_points"] = bullet_points
            
            # 3. Section headers
            headers = re.findall(r"^#{2,}\s+(.*?)$", content, re.MULTILINE)
            if headers:
                metadata["section_headers"] = headers
            
            # 4. Text statistics
            words = content.split()
            metadata["statistics"] = {
                "word_count": len(words),
                "char_count": len(content),
                "line_count": content.count('\n') + 1
            }
            
            # Simple sentiment analysis
            positive_words = ["good", "great", "successfully", "correct", "properly", "effectively"]
            negative_words = ["error", "fail", "issue", "problem"]
            
            pos_count = sum(1 for word in words if word.lower() in positive_words)
            neg_count = sum(1 for word in words if word.lower() in negative_words)
            
            if pos_count > neg_count:
                metadata["sentiment"] = "positive"
            elif neg_count > pos_count:
                metadata["sentiment"] = "negative"
            else:
                metadata["sentiment"] = "neutral"
            
            # Add to enriched chunks
            enriched_chunks.append({
                "content": content,
                "metadata": metadata
            })
        
        log(f"Successfully extracted metadata for {len(enriched_chunks)} chunks")
        return enriched_chunks
    except Exception as e:
        log(f"Error extracting metadata: {e}")
        return chunks  # Return original chunks if metadata extraction fails

def simulate_embedding(chunks):
    """Simulate embedding generation (without actually calling OpenAI)"""
    log("Simulating embedding generation...")
    
    try:
        embedded_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Create a simulated embedding vector (just for testing)
            # In real use, this would call the OpenAI API
            simulated_vector = [0.1] * 20  # 20-dim dummy vector
            
            embedded_chunks.append({
                "id": f"chunk_{i}",
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "vector": simulated_vector
            })
        
        log(f"Successfully simulated embeddings for {len(embedded_chunks)} chunks")
        return embedded_chunks
    except Exception as e:
        log(f"Error simulating embeddings: {e}")
        return []

def simulate_vector_storage(embedded_chunks):
    """Simulate vector storage in Qdrant"""
    log("Simulating vector storage...")
    
    try:
        # Just write to a JSON file to simulate storage
        output_file = "simulated_vector_db.json"
        
        with open(output_file, "w") as f:
            json.dump(embedded_chunks, f, indent=2)
        
        log(f"Successfully simulated storage of {len(embedded_chunks)} vectors to {output_file}")
        return True
    except Exception as e:
        log(f"Error simulating vector storage: {e}")
        return False

def run_pdf_ingestion_test():
    """Run the complete PDF ingestion test"""
    log("Starting PDF ingestion test...")
    
    # Step 1: Document loading
    content = test_document_loading()
    if not content:
        log("Document loading failed. Exiting test.")
        return False
    
    # Step 2: Text chunking
    chunks = test_chunking(content)
    if not chunks:
        log("Text chunking failed. Exiting test.")
        return False
    
    # Step 3: Metadata extraction
    enriched_chunks = test_metadata_extraction(chunks)
    
    # Step 4: Embedding generation (simulated)
    embedded_chunks = simulate_embedding(enriched_chunks)
    if not embedded_chunks:
        log("Embedding simulation failed. Exiting test.")
        return False
    
    # Step 5: Vector storage (simulated)
    storage_success = simulate_vector_storage(embedded_chunks)
    
    # Test summary
    log("PDF ingestion test completed.")
    log(f"Document loaded: {bool(content)}")
    log(f"Chunks created: {len(chunks)}")
    log(f"Metadata extracted: {len(enriched_chunks)}")
    log(f"Embeddings generated: {len(embedded_chunks)}")
    log(f"Storage simulated: {storage_success}")
    
    return all([content, chunks, enriched_chunks, embedded_chunks, storage_success])

if __name__ == "__main__":
    success = run_pdf_ingestion_test()
    sys.exit(0 if success else 1)