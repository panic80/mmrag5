#!/usr/bin/env python3
"""
HTML Table Parser - Converts HTML tables to graph-based representation

This module provides the parsing capability to convert HTML tables into our
graph-based table representation, preserving structure and relationships.
"""

import re
import json
import logging
from typing import List, Dict, Optional, Any, Union, Tuple, Set
from bs4 import BeautifulSoup, Tag
from dataclasses import dataclass

from table_graph_model import Table, Row, Column, Cell

# Set up logger
logger = logging.getLogger(__name__)

class TableParser:
    """
    Parser to convert HTML tables to graph-based representation.
    Implements the three-tier parsing strategy:
    1. Structural Tier: Parse HTML tables using BeautifulSoup
    2. Graph Tier: Convert parsed structure to relationship graph
    3. Output Tier: Generate different output formats
    """
    
    def __init__(self):
        """Initialize the table parser."""
        # Track occupied cells in the table for merged cell handling
        self.occupied_cells: Set[Tuple[int, int]] = set()
    
    def parse_html_table(self, html_table: Tag) -> Table:
        """
        Parse an HTML table element (from BeautifulSoup) into our graph-based table model.
        
        Args:
            html_table: BeautifulSoup Tag representing an HTML table
            
        Returns:
            Table: Our graph-based representation of the table
        """
        # Reset occupied cells for this table
        self.occupied_cells = set()
        
        # Create a new table
        table = Table()
        
        # Extract caption if present
        caption_tag = html_table.find('caption')
        if caption_tag and hasattr(caption_tag, 'text'):
            table.caption = caption_tag.text.strip()
        
        # Extract table structure information
        thead = html_table.find('thead')
        tbody = html_table.find('tbody')
        tfoot = html_table.find('tfoot')
        
        # Store structure information in metadata
        table.metadata['has_thead'] = thead is not None
        table.metadata['has_tbody'] = tbody is not None
        table.metadata['has_tfoot'] = tfoot is not None
        
        # Process table rows in order: thead, tbody, tfoot
        row_index = 0
        
        # Process header rows (thead)
        if thead:
            header_rows = thead.find_all('tr')
            for tr in header_rows:
                row = self._process_row(tr, row_index, is_header=True)
                table.add_row(row)
                row_index += 1
        
        # Process body rows (tbody)
        if tbody:
            body_rows = tbody.find_all('tr')
            for tr in body_rows:
                # Check if this is a header row (has th cells)
                is_header = len(tr.find_all('th')) > 0
                row = self._process_row(tr, row_index, is_header=is_header)
                table.add_row(row)
                row_index += 1
        
        # If no thead/tbody, process rows directly from table
        if not thead and not tbody:
            rows = html_table.find_all('tr')
            for tr in rows:
                # Detect if this is a header row (first row or has th cells)
                is_header = (row_index == 0) or (len(tr.find_all('th')) > 0)
                row = self._process_row(tr, row_index, is_header=is_header)
                table.add_row(row)
                row_index += 1
        
        # Process footer rows (tfoot)
        if tfoot:
            footer_rows = tfoot.find_all('tr')
            for tr in footer_rows:
                row = self._process_row(tr, row_index, is_header=False)
                table.add_row(row)
                row_index += 1
        
        # Build columns based on the cells we've processed
        self._build_columns(table)
        
        # Process nested tables
        self._process_nested_tables(table, html_table)
        
        # Process merged cells - establish relationships between spanned cells
        self._process_merged_cells(table)
        
        return table
    
    def _process_row(self, tr: Tag, row_index: int, is_header: bool = False) -> Row:
        """
        Process a table row and its cells.
        
        Args:
            tr: BeautifulSoup row Tag (tr)
            row_index: The index of this row in the table
            is_header: Whether this row contains header cells
            
        Returns:
            Row: Row object containing processed cells
        """
        row = Row(is_header=is_header)
        
        # Find all cells in this row (th and td)
        cells = tr.find_all(['th', 'td'])
        
        # Track current column position in the row
        col_index = 0
        
        # Skip positions that are already occupied by cells spanning from previous rows
        while (row_index, col_index) in self.occupied_cells:
            col_index += 1
        
        # Process each cell
        for cell_tag in cells:
            # Skip this position if it's already occupied by a rowspan/colspan
            while (row_index, col_index) in self.occupied_cells:
                col_index += 1
            
            # Get cell attributes
            is_cell_header = cell_tag.name == 'th'
            rowspan = int(cell_tag.get('rowspan', 1))
            colspan = int(cell_tag.get('colspan', 1))
            content = cell_tag.get_text(strip=True)
            
            # Check for nested tables
            nested_table = cell_tag.find('table')
            has_nested_table = nested_table is not None
            
            # Create cell
            cell = Cell(
                content=content,
                row_index=row_index,
                col_index=col_index,
                rowspan=rowspan,
                colspan=colspan,
                is_header=is_header or is_cell_header
            )
            
            # Flag cells with nested tables in metadata
            if has_nested_table:
                cell.metadata['has_nested_table'] = True
            
            # Add cell to row
            row.add_cell(cell)
            
            # Mark positions occupied by this cell due to rowspan/colspan
            for r in range(row_index, row_index + rowspan):
                for c in range(col_index, col_index + colspan):
                    if r != row_index or c != col_index:  # Don't mark the current cell position
                        self.occupied_cells.add((r, c))
            
            # Move to next column position
            col_index += colspan
        
        return row
    
    def _build_columns(self, table: Table):
        """
        Build column objects based on cells in the table.
        
        Args:
            table: The table to build columns for
        """
        # Find the maximum column index
        max_col = 0
        for row in table.rows:
            for cell in row.cells:
                max_col = max(max_col, cell.col_index + cell.colspan - 1)
        
        # Create columns and assign cells
        for col_idx in range(max_col + 1):
            column = Column()
            
            # Find cells in this column
            for row in table.rows:
                for cell in row.cells:
                    # Check if cell spans this column
                    if cell.col_index <= col_idx < (cell.col_index + cell.colspan):
                        column.add_cell(cell)
                        # Mark column as header if it contains header cells
                        if cell.is_header:
                            column.is_header = True
            
            table.add_column(column)
    
    def _process_nested_tables(self, table: Table, html_table: Tag):
        """
        Find and process nested tables within the current table.
        
        Args:
            table: Parent table object
            html_table: BeautifulSoup table Tag
        """
        # Find cells containing nested tables
        for row_idx, row in enumerate(table.rows):
            for cell in row.cells:
                # Find the corresponding HTML cell
                html_row = html_table.find_all('tr')[row_idx]
                html_cells = html_row.find_all(['th', 'td'])
                
                # Match the cell in the HTML based on position
                col_counter = 0
                matched_html_cell = None
                
                for html_cell in html_cells:
                    if col_counter == cell.col_index:
                        matched_html_cell = html_cell
                        break
                    col_counter += int(html_cell.get('colspan', 1))
                
                if matched_html_cell:
                    # Check for nested table
                    nested_html_table = matched_html_cell.find('table')
                    if nested_html_table:
                        # Parse the nested table
                        nested_table = self.parse_html_table(nested_html_table)
                        # Add to parent table
                        table.add_nested_table(nested_table, cell)
                        # Update cell metadata
                        cell.metadata['nested_table_id'] = nested_table.id
    
    def _process_merged_cells(self, table: Table):
        """
        Establish relationships between merged cells.
        
        Args:
            table: The table to process
        """
        # Find cells with rowspan/colspan > 1
        merged_cells = []
        for cell_id, cell in table.cells.items():
            if cell.rowspan > 1 or cell.colspan > 1:
                merged_cells.append(cell)
        
        # For each merged cell, find the cells it spans
        for merged_cell in merged_cells:
            # Calculate the range of positions this cell spans
            for r in range(merged_cell.row_index, merged_cell.row_index + merged_cell.rowspan):
                for c in range(merged_cell.col_index, merged_cell.col_index + merged_cell.colspan):
                    # Skip the primary cell position
                    if r == merged_cell.row_index and c == merged_cell.col_index:
                        continue
                    
                    # Find cells spanned by this merged cell
                    # The position may be empty due to rowspan/colspan
                    spanned_cell = table.get_cell(r, c)
                    if spanned_cell:
                        merged_cell.add_merged_cell(spanned_cell)
                        # Also add bidirectional reference
                        spanned_cell.add_merged_cell(merged_cell)

def html_string_to_table(html: str) -> List[Table]:
    """
    Parse HTML content and extract all tables as graph-based table models.
    
    Args:
        html: HTML content string
        
    Returns:
        List of Table objects representing all tables in the HTML
    """
    soup = BeautifulSoup(html, 'html.parser')
    tables = []
    
    # Find all tables in the HTML
    html_tables = soup.find_all('table')
    
    if not html_tables:
        logger.info("No tables found in HTML content")
        return []
    
    # Parse each table
    parser = TableParser()
    for html_table in html_tables:
        table = parser.parse_html_table(html_table)
        tables.append(table)
    
    return tables

def html_file_to_table(file_path: str) -> List[Table]:
    """
    Parse an HTML file and extract all tables as graph-based table models.
    
    Args:
        file_path: Path to HTML file
        
    Returns:
        List of Table objects representing all tables in the HTML file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return html_string_to_table(html_content)
    except Exception as e:
        logger.error(f"Error reading or parsing HTML file {file_path}: {e}")
        return []

def markdown_to_table(markdown: str) -> Optional[Table]:
    """
    Convert a markdown table representation to our graph-based table model.
    This is useful for handling tables already converted to markdown.
    
    Args:
        markdown: Markdown table string
        
    Returns:
        Table object or None if parsing fails
    """
    lines = markdown.strip().split('\n')
    if len(lines) < 3:
        logger.warning("Markdown table too short to be valid")
        return None
    
    # Check for our enhanced markdown format
    metadata = {}
    if lines[0].startswith('{table_metadata:'):
        try:
            metadata_line = lines[0].strip()
            # Extract JSON between the first { and the last }
            metadata_json = metadata_line[metadata_line.find('{'): metadata_line.rfind('}')+1]
            metadata = json.loads(metadata_json)
            # Remove metadata line
            lines = lines[1:]
        except Exception as e:
            logger.warning(f"Failed to parse table metadata: {e}")
    
    # Create table
    table = Table()
    if metadata:
        table.id = metadata.get('table_id', table.id)
        table.caption = metadata.get('caption', '')
        table.metadata.update(metadata)
    
    row_index = 0
    header_processed = False
    
    # Process each line
    for line_idx, line in enumerate(lines):
        # Skip separator line
        if '---' in line and all(c == '-' or c == '|' or c == ' ' for c in line):
            continue
        
        # Process as row
        cells = [cell.strip() for cell in line.split('|')[1:-1]]
        
        # Check if this is a header (first row or explicitly marked)
        is_header = (line_idx == 0 and line_idx < len(lines) - 2 
                     and '---' in lines[line_idx + 1])
        
        row = Row(is_header=is_header)
        
        # Process cells
        for col_idx, content in enumerate(cells):
            # Check for span annotations in enhanced markdown
            rowspan = 1
            colspan = 1
            
            # Extract span metadata from *(...)*
            span_match = re.search(r'\*\((.*?)\)\*$', content)
            if span_match:
                span_text = span_match.group(1)
                # Remove span annotation from content
                content = content[:content.rfind(' *(')].strip()
                
                # Parse span values
                for span in span_text.split(','):
                    span = span.strip()
                    if span.startswith('rowspan='):
                        try:
                            rowspan = int(span[8:])
                        except ValueError:
                            pass
                    elif span.startswith('colspan='):
                        try:
                            colspan = int(span[8:])
                        except ValueError:
                            pass
            
            # Create cell
            cell = Cell(
                content=content,
                row_index=row_index,
                col_index=col_idx,
                rowspan=rowspan,
                colspan=colspan,
                is_header=is_header
            )
            
            row.add_cell(cell)
        
        # Add row to table
        table.add_row(row)
        row_index += 1
    
    # Build columns
    for col_idx in range(max(len(row.cells) for row in table.rows)):
        column = Column()
        for row in table.rows:
            if col_idx < len(row.cells):
                column.add_cell(row.cells[col_idx])
                if row.cells[col_idx].is_header:
                    column.is_header = True
        table.add_column(column)
    
    return table