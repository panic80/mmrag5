#!/usr/bin/env python3

"""Test script to verify flag handling in both server.py and ingest_rag.py"""

import subprocess
import os
import time
import sys
import json
from pathlib import Path

# If no OPENAI_API_KEY in environment, use a dummy one for testing
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "test-dummy-key-for-cmd-line-arg-testing-only"

def test_ingest_rag_flags():
    """Test direct flag handling in ingest_rag.py"""
    print("\n=== Testing ingest_rag.py flag handling ===\n")
    
    # Create a test file
    test_content = (
        "# Test Document\n\n"
        "This is a test document to verify flag handling.\n\n"
        "## Section 1\n\n"
        "Testing adaptive chunking and other features.\n\n"
        "## Section 2\n\n"
        "More test content to ensure proper chunking and processing.\n\n"
        "```python\n"
        "def test_function():\n"
        "    print('This is code that should be chunked differently')\n"
        "```\n\n"
        "## Section 3\n\n"
        "Final test section with some data in a table format:\n\n"
        "| Header 1 | Header 2 | Header 3 |\n"
        "| -------- | -------- | -------- |\n"
        "| Value 1  | Value 2  | Value 3  |\n"
        "| Value 4  | Value 5  | Value 6  |\n"
    )
    
    test_file = Path("test_document.md")
    test_file.write_text(test_content)
    
    try:
        # Test adaptive chunking flag
        print("Testing --adaptive-chunking flag...")
        cmd = [
            sys.executable, "-u", "-m", "ingest_rag",
            "--source", str(test_file),
            "--collection", "test_collection",
            "--adaptive-chunking",
            "--purge"  # Start with a clean collection
        ]
        
        # Run the command with a timeout
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=30
        )
        
        print(f"Exit code: {result.returncode}")
        
        # Check output for indicators that adaptive chunking was used
        if "Using content-aware adaptive chunking" in result.stdout:
            print("✅ Success: Adaptive chunking flag was properly recognized and used")
        else:
            print("❌ Failure: Adaptive chunking flag doesn't appear to be working")
            print("\nOutput:")
            print(result.stdout)
            print("\nErrors:")
            print(result.stderr)
        
        # Test multiple flags together
        print("\nTesting multiple flags together...")
        cmd = [
            sys.executable, "-u", "-m", "ingest_rag",
            "--source", str(test_file),
            "--collection", "test_collection",
            "--adaptive-chunking",
            "--entity-extraction",
            "--similarity-threshold", "0.85",
            "--chunk-size", "300"
        ]
        
        # Run the command with a timeout
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=30
        )
        
        print(f"Exit code: {result.returncode}")
        
        # Check output for various flags
        success = True
        
        if "Using content-aware adaptive chunking" in result.stdout:
            print("✅ --adaptive-chunking recognized")
        else:
            print("❌ --adaptive-chunking not recognized")
            success = False
            
        if "--entity-extraction" in ' '.join(cmd):
            print("✅ --entity-extraction included in command")
        else:
            print("❌ --entity-extraction missing from command")
            success = False
            
        if "--similarity-threshold 0.85" in ' '.join(cmd):
            print("✅ --similarity-threshold parameter passed correctly")
        else:
            print("❌ --similarity-threshold parameter issue")
            success = False
            
        if "--chunk-size 300" in ' '.join(cmd):
            print("✅ --chunk-size parameter passed correctly")
        else:
            print("❌ --chunk-size parameter issue")
            success = False
            
        if success:
            print("\n✅ Success: All flags were properly handled")
        else:
            print("\n❌ Failure: Some flags were not properly handled")
            print("\nOutput:")
            print(result.stdout)
            print("\nErrors:")
            print(result.stderr)
            
    finally:
        # Clean up test file
        if test_file.exists():
            test_file.unlink()
            print(f"\nCleaned up test file: {test_file}")
            
    print("\n=== Completed ingest_rag.py flag tests ===\n")

if __name__ == "__main__":
    test_ingest_rag_flags()