#!/usr/bin/env python3
"""
Table Formatter Module for converting graph-based table representations to various output formats.

This module provides formatters for converting our rich graph-based table representation
into formats suitable for embedding and retrieval in the RAG system.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field

from table_graph_model import Table, Row, Cell, Column

# Configure logging
logger = logging.getLogger(__name__)

class TableFormatterBase:
    """Base class for all table formatters."""
    
    def format(self, table: Table) -> str:
        """Convert a Table to the target format."""
        raise NotImplementedError("Subclasses must implement format()")
    
    def format_with_metadata(self, table: Table) -> Tuple[str, Dict[str, Any]]:
        """
        Convert a Table to the target format and return both the formatted content
        and metadata that should be preserved with the document.
        
        Returns:
            Tuple of (formatted_content, metadata_dict)
        """
        content = self.format(table)
        metadata = self._extract_metadata(table)
        return content, metadata
    
    def _extract_metadata(self, table: Table) -> Dict[str, Any]:
        """Extract metadata from the table that should be preserved."""
        metadata = {
            "is_table": True,
            "table_id": table.id,
            "table_type": "html_table",
            "row_count": len(table.rows),
            "column_count": len(table.columns),
            "has_header": any(row.is_header for row in table.rows),
            "has_merged_cells": any(
                cell.rowspan > 1 or cell.colspan > 1
                for cell in table.cells.values()
            ),
            "has_nested_tables": len(table.nested_tables) > 0,
            "original_metadata": table.metadata
        }
        
        if table.caption:
            metadata["caption"] = table.caption
            
        # Add metadata for table context preservation
        if metadata.get("has_nested_tables"):
            metadata["nested_table_count"] = len(table.nested_tables)
            
        return metadata

class EnhancedMarkdownFormatter(TableFormatterBase):
    """
    Formatter that produces enhanced markdown with structural annotations.
    
    This formatter generates markdown with special annotations to preserve
    information about merged cells and other complex structures.
    """
    
    def format(self, table: Table) -> str:
        """
        Convert a Table to enhanced markdown format with structural annotations.
        
        Args:
            table: The graph-based table representation to format
            
        Returns:
            Enhanced markdown string with structural annotations
        """
        return table.to_enhanced_markdown()
    
    def _extract_metadata(self, table: Table) -> Dict[str, Any]:
        """Extract metadata with enhanced information for markdown format."""
        metadata = super()._extract_metadata(table)
        metadata["table_format"] = "enhanced_markdown"
        
        # Add more specific metadata for enhanced markdown
        header_rows = [row for row in table.rows if row.is_header]
        if header_rows:
            # Extract header text to help with retrieval
            headers = []
            for row in header_rows:
                for cell in row.cells:
                    if cell.is_header:
                        headers.append(cell.content)
            if headers:
                metadata["headers"] = headers
        
        return metadata

class PreservingMarkdownFormatter(TableFormatterBase):
    """
    Formatter that produces markdown that preserves the entire table structure.
    
    Unlike the default behavior where tables are split row-by-row, this formatter
    keeps the entire table intact in a single document, which is more suitable
    for preserving context in complex tables.
    """
    
    def format(self, table: Table) -> str:
        """
        Convert a Table to context-preserving markdown format.
        
        Args:
            table: The graph-based table representation to format
            
        Returns:
            Markdown string with the entire table represented
        """
        markdown_lines = []
        
        # Add caption if present
        if table.caption:
            markdown_lines.append(f"**{table.caption}**\n")
        
        # Build header row(s)
        header_rows = [row for row in table.rows if row.is_header]
        content_rows = [row for row in table.rows if not row.is_header]
        
        # Process header rows
        for row in header_rows:
            line = []
            for cell in sorted(row.cells, key=lambda c: c.col_index):
                cell_text = cell.content
                line.append(cell_text)
            
            markdown_lines.append("| " + " | ".join(line) + " |")
        
        # Add separator row
        if header_rows:
            # Count total columns by finding the row with the most cells
            max_cols = max(
                sum(cell.colspan for cell in row.cells)
                for row in header_rows + content_rows
            )
            separator = "| " + " | ".join(["---"] * max_cols) + " |"
            markdown_lines.append(separator)
        
        # Process content rows
        for row in content_rows:
            line = []
            for cell in sorted(row.cells, key=lambda c: c.col_index):
                cell_text = cell.content
                line.append(cell_text)
            
            markdown_lines.append("| " + " | ".join(line) + " |")
        
        # Process nested tables if present
        if table.nested_tables:
            markdown_lines.append("\n**Nested Tables:**")
            
            for i, nested_table in enumerate(table.nested_tables):
                parent_cell_id = nested_table.parent_cell.id if nested_table.parent_cell else "unknown"
                markdown_lines.append(f"\n**Nested Table {i+1}** (in cell {parent_cell_id}):")
                nested_markdown = self.format(nested_table)
                # Indent nested table content
                indented_lines = ["  " + line for line in nested_markdown.splitlines()]
                markdown_lines.append("\n".join(indented_lines))
        
        return "\n".join(markdown_lines)
    
    def _extract_metadata(self, table: Table) -> Dict[str, Any]:
        """Extract metadata with preservation context."""
        metadata = super()._extract_metadata(table)
        metadata["table_format"] = "preserving_markdown"
        metadata["context_preserved"] = True
        metadata["is_chunked"] = False  # Indicate this is a complete table, not a fragment
        
        return metadata

class InteractiveTableFormatter(TableFormatterBase):
    """
    Creates a JSON representation of the table suitable for interactive applications.
    
    This representation preserves all structural information and is suitable for
    reconstructing an interactive table view in a UI application.
    """
    
    def format(self, table: Table) -> str:
        """
        Convert a Table to a JSON string representation for interactive use.
        
        Args:
            table: The graph-based table representation to format
            
        Returns:
            JSON string representation of the table
        """
        return table.to_json(pretty=True)
    
    def _extract_metadata(self, table: Table) -> Dict[str, Any]:
        """Extract metadata with interactive formatting details."""
        metadata = super()._extract_metadata(table)
        metadata["table_format"] = "interactive_json"
        metadata["is_json"] = True
        
        return metadata

class TableRowFormatter(TableFormatterBase):
    """
    Formatter that produces individual document per table row but with improved context.
    
    This maintains compatibility with the existing row-based approach in the RAG system
    but adds contextual annotations to preserve table structure information.
    """
    
    def format_rows(self, table: Table) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Convert a Table to a list of row-based documents with enhanced context.
        
        This method creates multiple documents (one per row) like the original system
        but with improved context preservation.
        
        Args:
            table: The graph-based table representation to format
            
        Returns:
            List of (content, metadata) tuples, one per row
        """
        result = []
        
        # Find header rows
        header_rows = [row for row in table.rows if row.is_header]
        content_rows = [row for row in table.rows if not row.is_header]
        
        # If no explicit headers, use first row as header if it has cells
        if not header_rows and content_rows:
            first_row = content_rows[0]
            if first_row.cells:
                header_rows = [first_row]
                content_rows = content_rows[1:]
        
        # Construct header string
        header_str = ""
        if header_rows:
            header_lines = []
            for row in header_rows:
                header_cells = []
                for cell in sorted(row.cells, key=lambda c: c.col_index):
                    header_cells.append(cell.content)
                header_lines.append("| " + " | ".join(header_cells) + " |")
            
            # Add separator
            max_cols = max(len(row.cells) for row in header_rows)
            separator = "| " + " | ".join(["---"] * max_cols) + " |"
            
            header_str = "\n".join(header_lines) + "\n" + separator
        
        # Caption if available
        caption = f"**{table.caption}**\n\n" if table.caption else ""
        
        # Process each content row
        for row_idx, row in enumerate(content_rows):
            cells = sorted(row.cells, key=lambda c: c.col_index)
            if not cells:
                continue
                
            # Format row
            row_content = []
            for cell in cells:
                row_content.append(cell.content)
            
            row_md = "| " + " | ".join(row_content) + " |"
            
            # Combine with header for final markdown
            if header_str:
                formatted_content = f"{caption}{header_str}\n{row_md}"
            else:
                formatted_content = f"{caption}{row_md}"
            
            # Create metadata
            metadata = self._extract_row_metadata(table, row, row_idx)
            
            result.append((formatted_content, metadata))
        
        return result
    
    def format(self, table: Table) -> str:
        """
        Format the first row as example (used when a single output is expected).
        This is mainly for compatibility with the formatter interface.
        """
        formatted_rows = self.format_rows(table)
        if formatted_rows:
            return formatted_rows[0][0]
        return ""
    
    def _extract_row_metadata(self, table: Table, row: Row, row_idx: int) -> Dict[str, Any]:
        """Extract metadata for a specific row."""
        base_metadata = self._extract_metadata(table)
        
        # Add row-specific metadata
        row_metadata = {
            **base_metadata,
            "table_format": "row_markdown",
            "is_row_fragment": True,
            "table_row_index": row_idx,
            "total_rows": len([r for r in table.rows if not r.is_header]),
            "row_id": row.id
        }
        
        return row_metadata

class HtmlPreservingFormatter(TableFormatterBase):
    """
    Formatter that preserves the original HTML structure.
    
    This is useful for cases where markdown conversion loses too much information
    and we want to keep the original HTML for more accurate representation.
    """
    
    def __init__(self, html_source: str):
        """
        Initialize with the original HTML source.
        
        Args:
            html_source: Original HTML content containing the table
        """
        self.html_source = html_source
    
    def format(self, table: Table) -> str:
        """
        Format as HTML with structural annotations.
        
        Args:
            table: The graph-based table representation
            
        Returns:
            HTML string representation
        """
        # In a real implementation, we might reconstruct HTML from the graph model
        # But for now, we'll just return the original HTML with annotations
        html_lines = []
        html_lines.append("```html")
        html_lines.append(self.html_source)
        html_lines.append("```")
        
        return "\n".join(html_lines)
    
    def _extract_metadata(self, table: Table) -> Dict[str, Any]:
        """Extract metadata with HTML context."""
        metadata = super()._extract_metadata(table)
        metadata["table_format"] = "preserved_html"
        metadata["contains_html"] = True
        
        return metadata

def format_table_for_rag(
    table: Table, 
    format_type: str = "preserving", 
    html_source: Optional[str] = None
) -> Union[
    Tuple[str, Dict[str, Any]], 
    List[Tuple[str, Dict[str, Any]]]
]:
    """
    Format a table for use in the RAG system.
    
    Args:
        table: The graph-based table representation
        format_type: The formatting approach to use:
            - "preserving": Preserve the entire table (default)
            - "enhanced_markdown": Enhanced markdown with annotations
            - "row_based": Split into row-based chunks (compatible with existing system)
            - "interactive": JSON format for interactive applications
            - "html": Preserve original HTML
        html_source: Original HTML source (required for html format)
        
    Returns:
        Either a single (content, metadata) tuple or a list of such tuples for row_based format
    """
    if format_type == "row_based":
        formatter = TableRowFormatter()
        return formatter.format_rows(table)
    elif format_type == "enhanced_markdown":
        formatter = EnhancedMarkdownFormatter()
        return formatter.format_with_metadata(table)
    elif format_type == "interactive":
        formatter = InteractiveTableFormatter()
        return formatter.format_with_metadata(table)
    elif format_type == "html":
        if html_source is None:
            raise ValueError("html_source is required for html format type")
        formatter = HtmlPreservingFormatter(html_source)
        return formatter.format_with_metadata(table)
    else:
        # Default to preserving formatter
        formatter = PreservingMarkdownFormatter()
        return formatter.format_with_metadata(table)