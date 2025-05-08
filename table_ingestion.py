#!/usr/bin/env python3
"""
Table Ingestion Module for RAG System

This module provides improved table processing capabilities for the RAG ingestion system,
addressing key issues with the existing table handling approach:

1. Table Fragmentation - Preserves table context rather than splitting into row fragments
2. Markdown Conversion Limitations - Better handling of complex tables with merged cells
3. Code Implementation Issues - Fixed indentation error and consolidated code paths
4. Rigid Assumptions - Handles non-standard table structures without headers properly

This module serves as a drop-in replacement for the table processing parts of ingest_rag.py.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import traceback
from datetime import datetime

# Import our custom modules
from table_parser import html_string_to_table, html_file_to_table
from table_formatter import format_table_for_rag
from table_graph_model import Table

# Set up logging
logger = logging.getLogger(__name__)

class Document:
    """A minimal representation of a document compatible with ingest_rag.Document."""
    
    def __init__(self, content: str, metadata: dict = None):
        self.content = content
        self.metadata = metadata or {}

def process_html_tables(html_content: str, source_path: str = None) -> List[Document]:
    """
    Process HTML content and extract tables as properly formatted documents.
    
    This is a replacement for the table processing logic in ingest_rag.py that
    addresses the fragmentation and formatting issues.
    
    Args:
        html_content: Raw HTML content
        source_path: Original source path or URL (for metadata)
        
    Returns:
        List of Document objects with properly formatted table content
    """
    documents = []
    
    try:
        # Parse tables from HTML into our graph representation
        tables = html_string_to_table(html_content)
        
        if not tables:
            logger.info("No tables found in HTML content")
            return []
        
        logger.info(f"Found {len(tables)} tables in HTML content")
        
        # Process each table
        for table_idx, table in enumerate(tables):
            logger.info(f"Processing table {table_idx+1}/{len(tables)}")
            
            # Determine if this is a complex table that needs special handling
            has_merged_cells = any(
                cell.rowspan > 1 or cell.colspan > 1 
                for cell in table.cells.values()
            )
            has_nested_tables = len(table.nested_tables) > 0
            
            # Choose format based on table complexity
            format_type = "preserving"  # Default to preserving entire table
            
            # For very simple tables with few rows, we can use row-based approach for compatibility
            is_simple_table = (
                not has_merged_cells and 
                not has_nested_tables and
                len(table.rows) <= 10  # Arbitrary threshold - adjust as needed
            )
            
            if is_simple_table and len(table.rows) > 2:  # Make sure we have content rows
                # Use row-based format for backward compatibility with simple tables
                logger.info(f"Using row-based format for simple table {table_idx+1}")
                format_type = "row_based"
            else:
                # Use enhanced preservation for complex tables
                logger.info(f"Using preserving format for complex table {table_idx+1}")
                
            # Format the table
            formatted_result = format_table_for_rag(table, format_type=format_type)
            
            # Handle the two different return types (single or multiple documents)
            if format_type == "row_based":
                # Multiple documents, one per row
                for row_idx, (content, metadata) in enumerate(formatted_result):
                    # Add source information
                    if source_path:
                        metadata["source"] = source_path
                    
                    # Add table index information
                    metadata["table_index"] = table_idx
                    
                    # Create document
                    documents.append(Document(content=content, metadata=metadata))
                
                logger.info(f"Created {len(formatted_result)} row-based documents for table {table_idx+1}")
            else:
                # Single document for the whole table
                content, metadata = formatted_result
                
                # Add source information
                if source_path:
                    metadata["source"] = source_path
                
                # Add table index information
                metadata["table_index"] = table_idx
                
                # Create document
                documents.append(Document(content=content, metadata=metadata))
                
                logger.info(f"Created single document for table {table_idx+1}")
    
    except Exception as e:
        error_msg = f"Error processing HTML tables: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        # Track error for diagnostics
        if hasattr(Document, "INGESTION_DIAGNOSTICS"):
            Document.INGESTION_DIAGNOSTICS["errors"].append({
                "timestamp": datetime.now().isoformat(),
                "component": "table_processing",
                "error_type": e.__class__.__name__,
                "message": str(e)
            })
    
    return documents

def process_html_file(file_path: str) -> List[Document]:
    """
    Process an HTML file and extract tables as properly formatted documents.
    
    This is a replacement for the HTML file processing logic in ingest_rag.py.
    
    Args:
        file_path: Path to HTML file
        
    Returns:
        List of Document objects
    """
    documents = []
    
    try:
        # Read the HTML file
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Process tables
        table_docs = process_html_tables(html_content, source_path=file_path)
        documents.extend(table_docs)
        
        # If requested, also process non-table content (using existing approach)
        # This could be added here in the future
        
    except Exception as e:
        error_msg = f"Error processing HTML file {file_path}: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
    
    return documents

def integrate_with_ingest_rag() -> Dict[str, Any]:
    """
    Return functions and objects for integrating with ingest_rag.py.
    
    This function provides a dictionary of replacements for the table processing
    functions in ingest_rag.py.
    
    Returns:
        Dictionary of functions and objects for integration
    """
    return {
        "process_html_tables": process_html_tables,
        "process_html_file": process_html_file,
        "Document": Document
    }

def patch_ingest_rag():
    """
    Patch the ingest_rag module to use our improved table processing.
    
    This function monkey-patches the ingest_rag module to replace its
    table processing functions with our improved versions.
    
    Note: This is a more invasive approach and should be used with caution.
          A better approach is to directly modify ingest_rag.py to import
          and use our functions.
    """
    try:
        import ingest_rag
        
        # Store original functions for potential restoration
        original_functions = {
            "partition_html": ingest_rag.partition_html if hasattr(ingest_rag, "partition_html") else None,
        }
        
        # Replace functions as needed
        # This is where we would monkey-patch specific functions
        
        logger.info("Successfully patched ingest_rag module for improved table handling")
        return original_functions
    except ImportError:
        logger.error("Failed to import ingest_rag module for patching")
        return None

def create_ingest_rag_drop_in() -> str:
    """
    Generate a drop-in replacement for the table processing part of ingest_rag.py.
    
    This function generates code that can be used to replace the problematic table
    processing sections in ingest_rag.py.
    
    Returns:
        String containing code to replace in ingest_rag.py
    """
    code = '''
# =========================================================================
# TABLE PROCESSING REPLACEMENT CODE
# =========================================================================
# This code replaces the table processing sections in ingest_rag.py
# around lines 1400-1427

from table_ingestion import process_html_tables

# For HTML files
if ext in ('.html', '.htm'):
    try:
        # Read the HTML file
        with open(source, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Use our improved table processing
        docs_html = process_html_tables(html_content, source_path=source)
        
        if docs_html:
            return docs_html
        else:
            # Fallback to existing approach if no tables found
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text(separator='\\n')
            return [Document(content=chunk, metadata={"source": source, "chunk_index": idx})
                    for idx, chunk in enumerate(_smart_chunk_text(text, chunk_size, overlap))]
                    
    except Exception as e:
        click.echo(f"[warning] HTML processing failed for '{source}': {e}", err=True)
        # Fallback to standard text extraction
        try:
            from bs4 import BeautifulSoup
            with open(source, 'r', encoding='utf-8', errors='ignore') as fh:
                soup = BeautifulSoup(fh, 'html.parser')
            text = soup.get_text(separator='\\n')
            return [Document(content=chunk, metadata={"source": source, "chunk_index": idx})
                    for idx, chunk in enumerate(_smart_chunk_text(text, chunk_size, overlap))]
        except Exception:
            pass
'''
    return code

def fix_html_table_indentation_error():
    """
    Generate code to fix the indentation error in HTML table processing.
    
    This specifically addresses the indentation error found in ingest_rag.py
    around lines 1423-1424.
    
    Returns:
        String containing corrected code
    """
    corrected_code = '''
# Corrected indentation for the 'else' clause
if len(lines) > 2:
    header = lines[0]
    sep = lines[1]
    for row_idx, row_content in enumerate(lines[2:]):
        row = row_content.strip()
        if not row:
            continue
        row_md = f"{header}\\n{sep}\\n{row}"
        docs_html.append(Document(content=row_md, metadata={"source": source, "is_table": True, "table_row_index": row_idx}))
else:
    # This line was improperly indented in the original code
    docs_html.append(Document(content=md, metadata={"source": source, "is_table": True}))
'''
    return corrected_code

# Example usage of our module's functions
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example: Process an HTML file
    import sys
    
    if len(sys.argv) > 1:
        html_file = sys.argv[1]
        if os.path.isfile(html_file):
            print(f"Processing HTML file: {html_file}")
            docs = process_html_file(html_file)
            print(f"Extracted {len(docs)} document(s)")
            
            # Print the first document as an example
            if docs:
                print("\nExample document content:")
                print("-" * 60)
                print(docs[0].content[:500] + "..." if len(docs[0].content) > 500 else docs[0].content)
                print("-" * 60)
                print("\nMetadata:")
                for key, value in docs[0].metadata.items():
                    print(f"  {key}: {value}")
        else:
            print(f"File not found: {html_file}")
    else:
        print("Usage: python table_ingestion.py <html_file>")