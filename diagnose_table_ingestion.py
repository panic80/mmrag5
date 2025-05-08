#!/usr/bin/env python3
"""
Diagnostic script for HTML table ingestion issues.
This script adds detailed logging to the HTML table processing parts of ingest_rag.py.
"""

import os
import sys
import logging
import json
from datetime import datetime
from bs4 import BeautifulSoup
import traceback

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("table_ingestion_diagnosis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("table_diagnosis")

# Import the ingest_rag module to use its functions
sys.path.append('.')
try:
    import ingest_rag
    logger.info("Successfully imported ingest_rag module")
except ImportError as e:
    logger.error(f"Failed to import ingest_rag: {e}")
    sys.exit(1)

def diagnose_table_processing(html_file):
    """Process an HTML file and diagnose any table processing issues."""
    logger.info(f"Processing HTML file: {html_file}")
    
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Log raw HTML content size
        logger.info(f"HTML content size: {len(html_content)} bytes")
        
        # Use BeautifulSoup to parse tables - this is how we'd expect tables to be processed
        soup = BeautifulSoup(html_content, 'html.parser')
        tables = soup.find_all('table')
        logger.info(f"Found {len(tables)} tables in the HTML")
        
        # Examine each table's structure
        for i, table in enumerate(tables):
            logger.info(f"Table {i+1} structure:")
            rows = table.find_all('tr')
            logger.info(f"  - Contains {len(rows)} rows")
            
            # Check for thead, tbody, tfoot sections
            thead = table.find('thead')
            tbody = table.find('tbody')
            tfoot = table.find('tfoot')
            logger.info(f"  - Has thead: {thead is not None}")
            logger.info(f"  - Has tbody: {tbody is not None}")
            logger.info(f"  - Has tfoot: {tfoot is not None}")
            
            # Check for merged cells (rowspan/colspan)
            merged_cells = table.find_all(['td', 'th'], attrs={'rowspan': True}) + \
                          table.find_all(['td', 'th'], attrs={'colspan': True})
            logger.info(f"  - Contains {len(merged_cells)} merged cells")
            
            # Check for nested tables
            nested_tables = table.find_all('table')
            logger.info(f"  - Contains {len(nested_tables)} nested tables")
            
            # Examine first row to check headers
            if rows:
                first_row = rows[0]
                headers = first_row.find_all('th')
                logger.info(f"  - First row has {len(headers)} header cells")
            
            # Generate markdown representation (like the system does)
            try:
                markdown = table_to_markdown(table)
                logger.info(f"  - Generated markdown representation ({len(markdown)} chars)")
                logger.debug(f"  - Markdown:\n{markdown}")
            except Exception as e:
                logger.error(f"  - Failed to generate markdown: {e}")
        
        # Try to simulate how ingest_rag processes this file
        try:
            # This approximates what happens in ingest_rag.load_documents when loading HTML
            simulated_docs = simulate_ingest_process(html_file)
            logger.info(f"Simulated ingestion produced {len(simulated_docs)} documents")
            
            # Check if tables were preserved in the simulated documents
            table_fragments_found = 0
            for doc in simulated_docs:
                # Look for typical markdown table patterns
                if '|' in doc.content and '---' in doc.content:
                    table_fragments_found += 1
                    logger.info(f"Found table fragment in document: {doc.content[:100]}...")
            
            logger.info(f"Identified {table_fragments_found} table fragments in documents")
            
        except Exception as e:
            logger.error(f"Error in simulated ingestion: {e}")
            logger.error(traceback.format_exc())
    
    except Exception as e:
        logger.error(f"Error processing HTML file: {e}")
        logger.error(traceback.format_exc())

def table_to_markdown(table):
    """Convert an HTML table to markdown format (similar to how the system would)."""
    markdown_lines = []
    
    # Process rows
    rows = table.find_all('tr')
    for i, row in enumerate(rows):
        cells = row.find_all(['th', 'td'])
        
        # Build row content
        row_content = []
        for cell in cells:
            # Handle merged cells (for diagnosis - the actual processing may not handle these well)
            rowspan = int(cell.get('rowspan', 1))
            colspan = int(cell.get('colspan', 1))
            
            # Log merged cell for diagnosis
            if rowspan > 1 or colspan > 1:
                logger.debug(f"Found merged cell: rowspan={rowspan}, colspan={colspan}")
            
            # Get text content
            cell_content = cell.get_text(strip=True)
            row_content.append(cell_content)
        
        # Add row to markdown
        if row_content:
            markdown_lines.append('| ' + ' | '.join(row_content) + ' |')
        
        # Add separator row after headers
        if i == 0 and row.find('th'):
            separator = '|'
            for _ in range(len(row_content)):
                separator += ' --- |'
            markdown_lines.append(separator)
    
    return '\n'.join(markdown_lines)

def simulate_ingest_process(html_file):
    """Simulate how ingest_rag would process this HTML file."""
    from ingest_rag import Document
    
    logger.info(f"Simulating ingest process for {html_file}")
    
    # First, read the file
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # This is a simulation of what happens in ingest_rag.py lines 1403-1427
        try:
            from unstructured.partition.html import partition_html
            from unstructured.documents.elements import Table
            
            logger.info("Using unstructured.partition.html to process file")
            elements = partition_html(html_file)
            
            docs = []
            for elem in elements:
                if isinstance(elem, Table):
                    logger.info(f"Found table element: {type(elem)}")
                    try:
                        md = elem.to_markdown()
                        logger.info(f"Converted table to markdown: {len(md)} chars")
                        logger.debug(f"Table markdown:\n{md}")
                    except Exception as table_error:
                        logger.error(f"Error converting table to markdown: {table_error}")
                        md = elem.get_text()
                        logger.info(f"Falling back to plain text: {len(md)} chars")
                    
                    # Check for critical issue: Table row splitting
                    lines = md.splitlines()
                    if len(lines) > 2:
                        logger.info(f"Table has {len(lines)} lines")
                        header, sep = lines[0], lines[1]
                        
                        # This is the problematic section in ingest_rag.py that splits tables
                        for row_idx, row_content in enumerate(lines[2:]):
                            row = row_content.strip()
                            if not row:
                                continue
                            # The system recreates a mini-table for each row, losing table context
                            row_md = f"{header}\n{sep}\n{row}"
                            docs.append(Document(content=row_md, metadata={"source": html_file, "is_table": True, "table_row_index": row_idx}))
                            logger.info(f"Created separate document for row {row_idx+1}")
                    else:
                        # Keep table as single document if it's small
                        docs.append(Document(content=md, metadata={"source": html_file, "is_table": True}))
                        logger.info("Kept table as a single document")
                        
                elif hasattr(elem, 'text') and isinstance(elem.text, str):
                    logger.info(f"Found text element: {type(elem)}")
                    docs.append(Document(content=elem.text, metadata={"source": html_file}))
            
            return docs
            
        except ImportError as e:
            logger.error(f"unstructured.partition.html not available: {e}")
            
            # Fall back to BeautifulSoup (simulating the fallback in ingest_rag.py)
            logger.info("Falling back to BeautifulSoup")
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text(separator='\n')
            
            # Simulate chunking
            chunks = text.split('\n\n')
            return [Document(content=chunk, metadata={"source": html_file, "chunk_index": idx}) 
                   for idx, chunk in enumerate(chunks) if chunk.strip()]
    
    except Exception as e:
        logger.error(f"Error in simulation: {e}")
        logger.error(traceback.format_exc())
        return []

def main():
    """Run diagnostics on test HTML files."""
    test_dir = "test_html_tables"
    
    if not os.path.exists(test_dir):
        logger.error(f"Test directory {test_dir} not found")
        return
    
    html_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) 
                 if f.endswith('.html')]
    
    if not html_files:
        logger.error(f"No HTML files found in {test_dir}")
        return
    
    logger.info(f"Found {len(html_files)} HTML test files")
    for html_file in html_files:
        logger.info(f"=" * 80)
        logger.info(f"PROCESSING: {html_file}")
        logger.info(f"=" * 80)
        diagnose_table_processing(html_file)
        logger.info("\n")
    
    logger.info("Diagnosis complete. Check table_ingestion_diagnosis.log for details.")

if __name__ == "__main__":
    main()