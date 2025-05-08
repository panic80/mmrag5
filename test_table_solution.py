#!/usr/bin/env python3
"""
Test script for the improved HTML table ingestion solution

This script demonstrates how our new table processing system handles HTML tables
of varying complexity, comparing the result with the original problematic behavior.
"""

import os
import sys
import json
import logging
from pprint import pprint
from pathlib import Path

# Import our custom modules
from table_graph_model import Table
from table_parser import html_file_to_table
from table_formatter import format_table_for_rag
from table_ingestion import process_html_file, Document

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("table_test")

def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)

def compare_with_original_behavior(html_file):
    """
    Compare our new table processing with the original problematic behavior.
    
    Args:
        html_file: Path to the HTML file to process
    """
    print_section(f"PROCESSING: {os.path.basename(html_file)}")
    
    # Import original diagnosis code to simulate old behavior
    sys.path.append('.')
    try:
        import diagnose_table_ingestion
        
        # Simulate original processing
        print("Simulating original processing method...")
        simulated_docs = diagnose_table_ingestion.simulate_ingest_process(html_file)
        orig_table_fragments = sum(1 for doc in simulated_docs if doc.metadata.get('is_table', False))
        
        print(f"  Original method created {len(simulated_docs)} documents total")
        print(f"  Original method created {orig_table_fragments} table fragments")
        
        # Show an example of a table fragment
        for doc in simulated_docs:
            if doc.metadata.get('is_table', False):
                print("\nOriginal Method - Example Table Fragment:")
                print("-" * 60)
                print(doc.content)
                print("-" * 60)
                print(f"Metadata: {doc.metadata}")
                break
                
    except ImportError:
        print("Warning: diagnose_table_ingestion.py not found, skipping original behavior simulation")
    
    # Process with our new solution
    print("\nProcessing with new table ingestion solution...")
    new_docs = process_html_file(html_file)
    new_table_docs = [doc for doc in new_docs if doc.metadata.get('is_table', False)]
    
    print(f"  New method created {len(new_docs)} documents total")
    print(f"  New method created {len(new_table_docs)} table documents")
    
    # Show an example of the improved table representation
    if new_table_docs:
        print("\nNew Method - Example Table Document:")
        print("-" * 60)
        
        # Show a snippet if it's very long
        content = new_table_docs[0].content
        if len(content) > 500:
            print(content[:500] + "...\n[Content truncated]")
        else:
            print(content)
            
        print("-" * 60)
        
        # Print key metadata
        metadata = new_table_docs[0].metadata
        print("Key Metadata:")
        important_keys = ['is_table', 'table_type', 'table_format', 'row_count', 
                         'column_count', 'has_merged_cells', 'has_nested_tables',
                         'context_preserved']
        
        for key in important_keys:
            if key in metadata:
                print(f"  {key}: {metadata[key]}")
    
    # Show detailed analysis for complex tables
    if new_table_docs and any(doc.metadata.get('has_merged_cells', False) for doc in new_table_docs):
        print("\nDetailed Analysis of Complex Table Structure:")
        
        # Get the graph representation
        tables = html_file_to_table(html_file)
        if tables:
            complex_table = next((t for t in tables if any(
                cell.rowspan > 1 or cell.colspan > 1 for cell in t.cells.values()
            )), tables[0])
            
            # Show merged cell information
            merged_cells = [cell for cell in complex_table.cells.values() 
                          if cell.rowspan > 1 or cell.colspan > 1]
            
            if merged_cells:
                print(f"Found {len(merged_cells)} merged cells:")
                for i, cell in enumerate(merged_cells[:3]):  # Show first 3 for brevity
                    print(f"  Cell {i+1}: content='{cell.content}', rowspan={cell.rowspan}, "
                         f"colspan={cell.colspan}, position=({cell.row_index},{cell.col_index})")
                
                if len(merged_cells) > 3:
                    print(f"  ...and {len(merged_cells) - 3} more merged cells")
    
    return new_docs

def test_html_table_files():
    """
    Test our solution on all HTML files in the test_html_tables directory.
    """
    test_dir = "test_html_tables"
    
    if not os.path.exists(test_dir):
        logger.error(f"Test directory {test_dir} not found")
        return
    
    html_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) 
                  if f.endswith('.html')]
    
    if not html_files:
        logger.error(f"No HTML files found in {test_dir}")
        return
    
    results = []
    
    for html_file in html_files:
        docs = compare_with_original_behavior(html_file)
        results.append({
            'file': os.path.basename(html_file),
            'doc_count': len(docs),
            'table_docs': sum(1 for doc in docs if doc.metadata.get('is_table', False))
        })
    
    # Summary
    print_section("SUMMARY")
    for result in results:
        print(f"{result['file']}: {result['doc_count']} documents ({result['table_docs']} table docs)")

def explore_formats(html_file):
    """
    Demonstrate the different formatting options available.
    
    Args:
        html_file: Path to HTML file to format
    """
    print_section(f"FORMAT OPTIONS FOR: {os.path.basename(html_file)}")
    
    # Parse the HTML file
    tables = html_file_to_table(html_file)
    
    if not tables:
        print("No tables found in the HTML file.")
        return
    
    # Get the first table for demonstration
    table = tables[0]
    
    # Show available formats
    formats = ["preserving", "enhanced_markdown", "row_based", "interactive"]
    
    for format_type in formats:
        print(f"\n--- {format_type.upper()} FORMAT ---")
        
        result = format_table_for_rag(table, format_type=format_type)
        
        if format_type == "row_based":
            # Multiple documents
            print(f"Created {len(result)} row-based documents")
            
            # Show first row document
            if result:
                content, metadata = result[0]
                print("\nExample Row Document:")
                print(content[:300] + "..." if len(content) > 300 else content)
                print(f"\nMetadata Keys: {', '.join(metadata.keys())}")
        else:
            # Single document
            content, metadata = result
            print("\nFormatted Content:")
            print(content[:300] + "..." if len(content) > 300 else content)
            print(f"\nMetadata Keys: {', '.join(metadata.keys())}")

def main():
    """Main entry point for the test script."""
    if len(sys.argv) > 1:
        # Test a specific file
        html_file = sys.argv[1]
        if not os.path.isfile(html_file):
            print(f"Error: File not found: {html_file}")
            return
            
        compare_with_original_behavior(html_file)
        explore_formats(html_file)
    else:
        # Test all files in the test directory
        test_html_table_files()

if __name__ == "__main__":
    main()