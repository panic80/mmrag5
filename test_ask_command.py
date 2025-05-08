#!/usr/bin/env python3
"""Test script for the /ask command functionality"""

import os
import subprocess
import sys

def test_ask_command(question):
    """Test the /ask command functionality with a given question"""
    print(f"Testing question: '{question}'")
    
    # Build command similar to how server.py does it
    cmd = ["python3", "-m", "query_rag"]
    
    # Add the --use-expansion flag by default
    cmd.append("--use-expansion")
    
    # Add Qdrant URL if available
    qdrant_url = os.environ.get("QDRANT_URL")
    if qdrant_url:
        cmd += ["--qdrant-url", qdrant_url]

    # Add default collection from environment if available
    default_coll = os.environ.get("QDRANT_COLLECTION_NAME")
    if default_coll:
        cmd += ["--collection", default_coll]

    # Add BM25 index JSON if it exists for the collection
    coll_name = default_coll or "rag_data"
    bm25_index_file = f"{coll_name}_bm25_index.json"
    if os.path.exists(bm25_index_file):
        cmd += ["--bm25-index", bm25_index_file]
    
    # Add the question at the end
    cmd.append(question)
    
    print(f"Executing command: {' '.join(cmd)}")
    
    # Execute the command
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("\nOutput:")
        print(result.stdout)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print("\nError:")
        print(f"Return code: {e.returncode}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False, e.stderr

if __name__ == "__main__":
    question = "what is lunch rate in yukon?"
    if len(sys.argv) > 1:
        question = sys.argv[1]
    
    success, output = test_ask_command(question)
    
    if success:
        print("Test passed: The /ask command executed successfully.")
    else:
        print("Test failed: The /ask command encountered an error.")
        
    sys.exit(0 if success else 1)