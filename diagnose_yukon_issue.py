#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diagnostic script to investigate the Yukon lunch rate issue in the RAG system.
This script analyzes how table data is processed during ingestion, particularly 
focusing on how column relationships are maintained (or lost) during the process.
"""

import os
import sys
import logging
import re
import json
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('yukon_rate_diagnosis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Try to import table_ingestion, table_parser, and table_formatter
# These might be in the current directory or in the project's import path
try:
    from table_ingestion import process_html_file
    from table_parser import TableParser
    from table_formatter import PreservingMarkdownFormatter
    logger.info("Successfully imported table modules")
except ImportError:
    logger.error("Failed to import table modules. Make sure they're in the current directory or PYTHONPATH")
    # Add the current directory to the path as a fallback
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from table_ingestion import process_html_file
        from table_parser import TableParser
        from table_formatter import PreservingMarkdownFormatter
        logger.info("Successfully imported table modules after path adjustment")
    except ImportError as e:
        logger.error(f"Still couldn't import required modules: {e}")
        raise

def download_html_table(url):
    """Download HTML from the given URL."""
    logger.info(f"Downloading HTML from {url}")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.error(f"Failed to download HTML: {e}")
        raise

def save_html_to_file(html_content, filename="meal_rates_table.html"):
    """Save the HTML content to a file for reference."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"Saved HTML content to {filename}")
    except Exception as e:
        logger.error(f"Failed to save HTML: {e}")

def extract_meal_rates_table(html_content):
    """Extract the meal rates table from the HTML content."""
    soup = BeautifulSoup(html_content, 'html.parser')
    logger.info("Searching for the meal rates table...")
    
    # First, look for tables with meal rate related content
    tables = soup.find_all('table')
    target_table = None
    
    for table in tables:
        # Check for text content related to meal rates
        if table.text and ('meal' in table.text.lower() or 'lunch' in table.text.lower() or 'yukon' in table.text.lower()):
            # Check specifically for the pattern of columns we're interested in
            headers = table.find_all('th') or table.find_all('td')
            header_text = ' '.join([h.text.strip() for h in headers])
            if 'yukon' in header_text.lower() and 'alaska' in header_text.lower():
                target_table = table
                logger.info("Found the meal rates table with Yukon & Alaska column")
                break
    
    if not target_table:
        tables_with_meals = [t for t in tables if 'meal' in t.text.lower()]
        if tables_with_meals:
            target_table = tables_with_meals[0]
            logger.info("Using first table with 'meal' in content as fallback")
        else:
            logger.error("Could not find the meal rates table in the HTML content")
            return None
    
    return target_table

def analyze_table_structure(table):
    """Analyze the structure of the table, particularly the header rows and columns."""
    logger.info("Analyzing table structure...")
    
    # Find header rows (rows in thead or tr with th elements)
    thead = table.find('thead')
    
    if thead:
        header_rows = thead.find_all('tr')
        logger.info(f"Found {len(header_rows)} header rows in thead")
    else:
        # If no thead, look for rows with th elements
        header_rows = [row for row in table.find_all('tr') if row.find('th')]
        logger.info(f"Found {len(header_rows)} header rows with th elements")
    
    # Analyze header structure
    header_analysis = []
    
    for i, row in enumerate(header_rows):
        cells = row.find_all(['th', 'td'])
        cell_info = []
        
        for j, cell in enumerate(cells):
            colspan = int(cell.get('colspan', 1))
            rowspan = int(cell.get('rowspan', 1))
            
            cell_info.append({
                'position': (i, j),
                'text': cell.text.strip(),
                'colspan': colspan,
                'rowspan': rowspan
            })
            
            # Specifically log Yukon-related header cells
            if 'yukon' in cell.text.lower():
                logger.info(f"Found 'Yukon' in header at position ({i},{j}): '{cell.text.strip()}', "
                           f"colspan={colspan}, rowspan={rowspan}")
        
        header_analysis.append(cell_info)
    
    # Extract data rows (tbody rows or tr not in thead)
    tbody = table.find('tbody')
    
    if tbody:
        data_rows = tbody.find_all('tr')
        logger.info(f"Found {len(data_rows)} data rows in tbody")
    else:
        # If no tbody, all rows except header rows are data rows
        all_rows = table.find_all('tr')
        data_rows = [row for row in all_rows if row not in header_rows]
        logger.info(f"Found {len(data_rows)} data rows outside of thead")
    
    # Log information about lunch rate rows
    for i, row in enumerate(data_rows):
        cells = row.find_all(['td', 'th'])
        for j, cell in enumerate(cells):
            if 'lunch' in cell.text.lower():
                logger.info(f"Found 'lunch' in data row {i}, cell {j}: '{cell.text.strip()}'")
                
                # Check the values in this row across different columns
                # We're especially interested in the Yukon & Alaska column
                if len(cells) > 2:  # Ensure there are enough cells
                    values = [c.text.strip() for c in cells]
                    logger.info(f"Lunch row values: {values}")
                    
                    # Log specific columns where the values should be
                    col_indices = {
                        "row_label": 0,
                        "canada_usa": 1,
                        "yukon_alaska": 2,
                        "nwt": 3,
                        "nunavut": 4
                    }
                    
                    for name, idx in col_indices.items():
                        if idx < len(cells):
                            logger.info(f"Column '{name}' value: '{cells[idx].text.strip()}'")
    
    return {
        'header_rows': header_analysis,
        'data_rows_count': len(data_rows)
    }

def test_table_extraction_with_parser(table_html):
    """Test how the system's table parser handles the meal rates table."""
    logger.info("Testing table extraction with the system's parser...")
    
    try:
        # Parse the table using the system's parser
        parser = TableParser()
        parsed_table = parser.parse_html_table(table_html)
        
        if parsed_table:
            # Log the parsed structure
            logger.info(f"Parser returned a table with {len(parsed_table.rows)} rows and "
                       f"{len(parsed_table.columns) if hasattr(parsed_table, 'columns') else 'unknown'} columns")
            
            # Check if the parser correctly identified header rows
            header_count = sum(1 for row in parsed_table.rows if getattr(row, 'is_header', False))
            logger.info(f"Parser identified {header_count} header rows")
            
            # Check for column merging/span detection
            merged_cells = []
            for row_idx, row in enumerate(parsed_table.rows):
                for cell_idx, cell in enumerate(getattr(row, 'cells', [])):
                    colspan = getattr(cell, 'colspan', 1)
                    rowspan = getattr(cell, 'rowspan', 1)
                    
                    if colspan > 1 or rowspan > 1:
                        merged_cells.append({
                            'position': (row_idx, cell_idx),
                            'text': getattr(cell, 'text', ''),
                            'colspan': colspan,
                            'rowspan': rowspan
                        })
            
            logger.info(f"Parser found {len(merged_cells)} merged cells")
            for cell in merged_cells:
                logger.info(f"Merged cell at {cell['position']}: '{cell['text']}', "
                           f"colspan={cell['colspan']}, rowspan={cell['rowspan']}")
            
            # Look for Yukon-related columns
            for row_idx, row in enumerate(parsed_table.rows):
                for cell_idx, cell in enumerate(getattr(row, 'cells', [])):
                    cell_text = getattr(cell, 'text', '')
                    if isinstance(cell_text, str) and 'yukon' in cell_text.lower():
                        logger.info(f"Parser found 'Yukon' in cell at ({row_idx},{cell_idx}): '{cell_text}'")
            
            # Look for lunch rate rows
            for row_idx, row in enumerate(parsed_table.rows):
                row_text = ' '.join(getattr(cell, 'text', '') for cell in getattr(row, 'cells', []))
                if 'lunch' in row_text.lower():
                    logger.info(f"Parser found 'lunch' in row {row_idx}: '{row_text}'")
                    
                    # Log the cells in this row to see column values
                    for cell_idx, cell in enumerate(getattr(row, 'cells', [])):
                        cell_text = getattr(cell, 'text', '')
                        logger.info(f"  Cell {cell_idx}: '{cell_text}'")
            
            return parsed_table
        else:
            logger.warning("Parser returned None for the table")
            return None
    
    except Exception as e:
        logger.error(f"Error in table parser: {e}")
        logger.error(traceback.format_exc())
        return None

def test_markdown_conversion(parsed_table):
    """Test how the system converts the parsed table to markdown."""
    logger.info("Testing markdown conversion...")
    
    try:
        # Convert the parsed table to markdown
        formatter = PreservingMarkdownFormatter()
        markdown = formatter.format(parsed_table)
        
        if markdown:
            logger.info("Successfully converted table to markdown")
            
            # Save the markdown for inspection
            with open("meal_rates_table.md", "w", encoding="utf-8") as f:
                f.write(markdown)
            logger.info("Saved markdown to meal_rates_table.md")
            
            # Analyze the markdown structure
            lines = markdown.strip().split('\n')
            logger.info(f"Markdown has {len(lines)} lines")
            
            # Check if header separator is present (|---|---|...)
            header_separator_lines = [i for i, line in enumerate(lines) if re.match(r'\|[\s\-]+\|[\s\-]+\|', line)]
            if header_separator_lines:
                logger.info(f"Found header separator at line {header_separator_lines[0] + 1}")
            else:
                logger.warning("No header separator found in markdown")
            
            # Look for Yukon in the headers
            for i, line in enumerate(lines[:5]):  # Check first few lines for headers
                if 'yukon' in line.lower():
                    logger.info(f"Found 'Yukon' in markdown line {i + 1}: '{line}'")
            
            # Look for lunch rates in the markdown
            for i, line in enumerate(lines):
                if 'lunch' in line.lower():
                    logger.info(f"Found 'lunch' in markdown line {i + 1}: '{line}'")
                    
                    # Try to extract the values for different regions
                    parts = line.split('|')
                    if len(parts) >= 5:
                        # Remove leading/trailing whitespace from each part
                        parts = [p.strip() for p in parts]
                        logger.info(f"Lunch rate row values in markdown: {parts}")
            
            return markdown
        else:
            logger.warning("Formatter returned None for the markdown")
            return None
    
    except Exception as e:
        logger.error(f"Error in markdown formatter: {e}")
        logger.error(traceback.format_exc())
        return None

def test_row_based_chunking(markdown):
    """Test how row-based chunking affects column relationships."""
    logger.info("Testing row-based chunking impact...")
    
    if not markdown:
        logger.warning("No markdown provided to test chunking")
        return
    
    # Simulate row-based chunking as done in ingest_rag.py
    lines = markdown.strip().split('\n')
    
    # Find header rows (before the separator)
    header_separator_idx = next((i for i, line in enumerate(lines) 
                                if re.match(r'\|[\s\-]+\|[\s\-]+\|', line)), -1)
    
    if header_separator_idx == -1:
        logger.warning("Could not find header separator in markdown")
        return
    
    # Header is everything before the separator
    header_rows = lines[:header_separator_idx]
    logger.info(f"Found {len(header_rows)} header row(s) in markdown")
    
    # Data rows are everything after the separator
    data_rows = lines[header_separator_idx + 1:]
    logger.info(f"Found {len(data_rows)} data row(s) in markdown")
    
    # Simulate creating row-level chunks as in ingest_rag.py
    chunks = []
    yukon_lunch_chunks = []
    
    for row in data_rows:
        if not row.strip():
            continue
        
        # Create a chunk with the header and this data row
        chunk = '\n'.join(header_rows + [lines[header_separator_idx], row])
        
        # Check if this is a lunch rate row for Yukon
        if 'lunch' in row.lower():
            logger.info(f"Creating chunk for lunch row: '{row}'")
            
            # Check if the data actually includes the Yukon values correctly
            parts = row.split('|')
            parts = [p.strip() for p in parts if p.strip()]
            
            if len(parts) >= 4:  # Label + Canada/USA + Yukon/Alaska + NWT + Nunavut
                logger.info(f"Chunked lunch row has these columns: {parts}")
                
                # The label should be in parts[0]
                # Canada & USA should be in parts[1]
                # Yukon & Alaska should be in parts[2]
                # NWT should be in parts[3]
                label = parts[0] if len(parts) > 0 else "MISSING"
                canada_usa = parts[1] if len(parts) > 1 else "MISSING"
                yukon_alaska = parts[2] if len(parts) > 2 else "MISSING"
                nwt = parts[3] if len(parts) > 3 else "MISSING"
                
                logger.info(f"In chunked lunch row: Label='{label}', Canada/USA='{canada_usa}', "
                          f"Yukon/Alaska='{yukon_alaska}', NWT='{nwt}'")
                
                # Store this chunk for later reference
                yukon_lunch_chunks.append({
                    'chunk': chunk,
                    'row': row,
                    'label': label,
                    'canada_usa': canada_usa,
                    'yukon_alaska': yukon_alaska,
                    'nwt': nwt
                })
        
        chunks.append(chunk)
    
    logger.info(f"Created {len(chunks)} row-based chunks in total")
    logger.info(f"Found {len(yukon_lunch_chunks)} chunks related to lunch rates for Yukon")
    
    # Save a sample chunk for inspection
    if yukon_lunch_chunks:
        with open("sample_lunch_chunk.md", "w", encoding="utf-8") as f:
            f.write(yukon_lunch_chunks[0]['chunk'])
        logger.info("Saved a sample lunch rate chunk to sample_lunch_chunk.md")
    
    return yukon_lunch_chunks

def simulate_query_process(yukon_lunch_chunks):
    """Simulate how a query for 'lunch rate in Yukon' would be processed."""
    logger.info("Simulating query processing for 'lunch rate in Yukon'...")
    
    if not yukon_lunch_chunks:
        logger.warning("No Yukon lunch chunks available to simulate query")
        return
    
    # In a vector search, the system would identify relevant chunks
    # Here we're simulating with the chunks we already identified
    
    # The query is about lunch rates in Yukon, so let's examine each chunk
    for i, chunk_info in enumerate(yukon_lunch_chunks):
        logger.info(f"\nAnalyzing lunch rate chunk {i+1}:")
        
        # Check if the chunk explicitly mentions "Yukon" 
        chunk = chunk_info['chunk']
        has_yukon_term = 'yukon' in chunk.lower()
        logger.info(f"Chunk {i+1} explicitly mentions 'Yukon': {has_yukon_term}")
        
        # Check which column is actually for Yukon
        logger.info(f"Label: {chunk_info['label']}")
        logger.info(f"Canada & USA value: {chunk_info['canada_usa']}")
        logger.info(f"Yukon & Alaska value: {chunk_info['yukon_alaska']}")
        logger.info(f"NWT value: {chunk_info['nwt']}")
        
        # Check if column header is correctly preserved in the chunk
        headers = chunk.split('\n')[0] if '\n' in chunk else ''
        logger.info(f"Headers in chunk: '{headers}'")
        
        # Check if this chunk contains all three rate types (100%, 75%, 50%)
        is_100pct = '100%' in chunk_info['label']
        is_75pct = '75%' in chunk_info['label']
        is_50pct = '50%' in chunk_info['label']
        
        logger.info(f"Rate type - 100%: {is_100pct}, 75%: {is_75pct}, 50%: {is_50pct}")
        
        # Compare to the values mentioned in the problem
        # Problem mentioned:
        # - Returned: $27.75 (N.W.T. 75% rate)
        # - Should be: $25.65 (Yukon & Alaska 100% rate)
        if '27.75' in chunk:
            logger.info("Chunk contains the incorrect value $27.75")
        if '25.65' in chunk:
            logger.info("Chunk contains the correct value $25.65")
        
        # Check for other values mentioned
        if '20.55' in chunk:
            logger.info("Chunk contains the incorrect value $20.55")
        if '19.25' in chunk:
            logger.info("Chunk contains the correct value $19.25")
            
        if '13.70' in chunk:
            logger.info("Chunk contains the incorrect value $13.70")
        if '12.85' in chunk:
            logger.info("Chunk contains the correct value $12.85")
        
        # Check for potential column alignment issues
        # The problem suggests the system is confusing columns
        col_values = [
            chunk_info['canada_usa'],
            chunk_info['yukon_alaska'],
            chunk_info['nwt']
        ]
        
        if '27.75' in col_values:
            idx = col_values.index('27.75')
            logger.info(f"Value $27.75 is in column {idx+1} (should be in NWT column)")
        
        if '25.65' in col_values:
            idx = col_values.index('25.65')
            logger.info(f"Value $25.65 is in column {idx+1} (should be in Yukon & Alaska column)")

def main():
    """Main function to run the diagnostic process."""
    logger.info("Starting diagnosis of Yukon lunch rate issue")
    
    # Option 1: Download and process the HTML from the web
    url = "https://www.njc-cnm.gc.ca/directive/d10/v238/en?print"
    try:
        html_content = download_html_table(url)
        save_html_to_file(html_content)
        logger.info("Successfully downloaded and saved HTML content")
    except Exception as e:
        logger.error(f"Failed to download HTML: {e}")
        logger.info("Will try to load from a local file instead")
        try:
            with open("meal_rates_table.html", "r", encoding="utf-8") as f:
                html_content = f.read()
            logger.info("Loaded HTML from local file")
        except Exception as e:
            logger.error(f"Failed to load local HTML file: {e}")
            return
    
    # Extract the meal rates table
    table_html = extract_meal_rates_table(html_content)
    if table_html:
        logger.info("Successfully extracted the meal rates table")
        
        # Save just the table HTML for reference
        with open("meal_rates_table_only.html", "w", encoding="utf-8") as f:
            f.write(str(table_html))
        logger.info("Saved the extracted table HTML to meal_rates_table_only.html")
        
        # Analyze the table structure
        analyze_table_structure(table_html)
        
        # Test the system's table parser
        parsed_table = test_table_extraction_with_parser(table_html)
        
        # Test markdown conversion
        if parsed_table:
            markdown = test_markdown_conversion(parsed_table)
            
            # Test row-based chunking
            if markdown:
                yukon_lunch_chunks = test_row_based_chunking(markdown)
                
                # Simulate query processing
                simulate_query_process(yukon_lunch_chunks)
    else:
        logger.error("Failed to extract the meal rates table")
    
    logger.info("Completed diagnosis of Yukon lunch rate issue")

if __name__ == "__main__":
    main()