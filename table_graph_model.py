#!/usr/bin/env python3
"""
Graph-based Table Representation Model for HTML Table Ingestion

This module provides a comprehensive model for representing HTML tables as graph structures,
preserving all structural elements including merged cells and nested tables.
"""

import uuid
import json
from typing import List, Dict, Optional, Any, Union, Tuple
from bs4 import BeautifulSoup, Tag
from dataclasses import dataclass, field, asdict

class TableObject:
    """Base class for all table objects."""
    
    def __init__(self, id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Initialize a table object with optional ID and metadata."""
        self.id = id or str(uuid.uuid4())
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert the object to a dictionary representation."""
        raise NotImplementedError("Subclasses must implement to_dict()")

@dataclass
class Cell(TableObject):
    """Represents a cell in a table."""
    
    content: str = ""
    row_index: int = 0
    col_index: int = 0
    rowspan: int = 1
    colspan: int = 1
    is_header: bool = False
    merged_with: List['Cell'] = field(default_factory=list)
    parent_row: Optional['Row'] = None
    parent_table: Optional['Table'] = None
    
    def __post_init__(self):
        """Initialize the TableObject base class after dataclass fields."""
        super().__init__()
        # Remove circular references for serialization
        if hasattr(self, 'parent_row'):
            self.parent_row = None
        if hasattr(self, 'parent_table'):
            self.parent_table = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the cell to a dictionary without circular references."""
        result = {
            'id': self.id,
            'content': self.content,
            'row_index': self.row_index,
            'col_index': self.col_index,
            'rowspan': self.rowspan,
            'colspan': self.colspan,
            'is_header': self.is_header,
            'metadata': self.metadata
        }
        
        # Handle merged cells without circular references
        if self.merged_with:
            result['merged_with'] = [
                {'id': cell.id, 'row': cell.row_index, 'col': cell.col_index}
                for cell in self.merged_with
            ]
        
        return result
    
    def add_merged_cell(self, cell: 'Cell'):
        """Add a cell to the merged_with list."""
        if cell not in self.merged_with:
            self.merged_with.append(cell)

@dataclass
class Row(TableObject):
    """Represents a row in a table."""
    
    cells: List[Cell] = field(default_factory=list)
    is_header: bool = False
    
    def __post_init__(self):
        """Initialize the TableObject base class after dataclass fields."""
        super().__init__()
    
    def add_cell(self, cell: Cell):
        """Add a cell to the row and set parent_row reference."""
        cell.parent_row = self
        self.cells.append(cell)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the row to a dictionary."""
        return {
            'id': self.id,
            'is_header': self.is_header,
            'cells': [cell.id for cell in self.cells],
            'metadata': self.metadata
        }

@dataclass
class Column(TableObject):
    """Represents a column in a table."""
    
    cells: List[Cell] = field(default_factory=list)
    is_header: bool = False
    
    def __post_init__(self):
        """Initialize the TableObject base class after dataclass fields."""
        super().__init__()
    
    def add_cell(self, cell: Cell):
        """Add a cell to the column."""
        self.cells.append(cell)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the column to a dictionary."""
        return {
            'id': self.id,
            'is_header': self.is_header,
            'cells': [cell.id for cell in self.cells],
            'metadata': self.metadata
        }

@dataclass
class Table(TableObject):
    """Represents a table with its rows, columns, and cell relationships."""
    
    rows: List[Row] = field(default_factory=list)
    columns: List[Column] = field(default_factory=list)
    caption: str = ""
    nested_tables: List['Table'] = field(default_factory=list)
    parent_table: Optional['Table'] = None
    parent_cell: Optional[Cell] = None
    cells: Dict[str, Cell] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize the TableObject base class after dataclass fields."""
        super().__init__()
        # Remove circular references for serialization
        if hasattr(self, 'parent_table'):
            self.parent_table = None
        if hasattr(self, 'parent_cell'):
            self.parent_cell = None
    
    def add_row(self, row: Row):
        """Add a row to the table and set cell parent_table references."""
        for cell in row.cells:
            cell.parent_table = self
            self.cells[cell.id] = cell
        self.rows.append(row)
    
    def add_column(self, column: Column):
        """Add a column to the table."""
        self.columns.append(column)
    
    def add_nested_table(self, table: 'Table', parent_cell: Cell):
        """Add a nested table with reference to its parent cell."""
        table.parent_table = self
        table.parent_cell = parent_cell
        self.nested_tables.append(table)
    
    def get_cell(self, row_idx: int, col_idx: int) -> Optional[Cell]:
        """Get a cell by its row and column indices."""
        for cell in self.cells.values():
            if cell.row_index == row_idx and cell.col_index == col_idx:
                return cell
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the table to a dictionary."""
        result = {
            'id': self.id,
            'caption': self.caption,
            'rows': [row.to_dict() for row in self.rows],
            'columns': [col.to_dict() for col in self.columns],
            'cells': {cell_id: cell.to_dict() for cell_id, cell in self.cells.items()},
            'metadata': self.metadata,
        }
        
        if self.nested_tables:
            result['nested_tables'] = [
                {
                    'id': table.id,
                    'parent_cell_id': table.parent_cell.id if table.parent_cell else None
                }
                for table in self.nested_tables
            ]
        
        return result
    
    def to_json(self, pretty: bool = False) -> str:
        """Convert the table to a JSON string."""
        indent = 2 if pretty else None
        return json.dumps(self.to_dict(), indent=indent)
    
    def to_enhanced_markdown(self) -> str:
        """Generate an enhanced markdown representation of the table."""
        markdown_lines = []
        
        # Add table metadata as a comment
        metadata = {
            'table_id': self.id,
            'caption': self.caption,
            'merged_cells': any(
                cell.rowspan > 1 or cell.colspan > 1
                for cell in self.cells.values()
            ),
            'nested_tables': len(self.nested_tables) > 0
        }
        
        markdown_lines.append(f"{{table_metadata: {json.dumps(metadata)}}}")
        
        # Build header row(s)
        header_rows = [row for row in self.rows if row.is_header]
        content_rows = [row for row in self.rows if not row.is_header]
        
        # Process header rows
        for row in header_rows:
            line = []
            for cell in sorted(row.cells, key=lambda c: c.col_index):
                cell_text = cell.content
                
                # Add span annotations
                if cell.rowspan > 1 or cell.colspan > 1:
                    spans = []
                    if cell.rowspan > 1:
                        spans.append(f"rowspan={cell.rowspan}")
                    if cell.colspan > 1:
                        spans.append(f"colspan={cell.colspan}")
                    
                    cell_text = f"{cell_text} *({', '.join(spans)})*"
                
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
                
                # Add span annotations
                if cell.rowspan > 1 or cell.colspan > 1:
                    spans = []
                    if cell.rowspan > 1:
                        spans.append(f"rowspan={cell.rowspan}")
                    if cell.colspan > 1:
                        spans.append(f"colspan={cell.colspan}")
                    
                    cell_text = f"{cell_text} *({', '.join(spans)})*"
                
                line.append(cell_text)
            
            markdown_lines.append("| " + " | ".join(line) + " |")
        
        return "\n".join(markdown_lines)

def table_to_graph(html_table: BeautifulSoup) -> Table:
    """
    Convert an HTML table from BeautifulSoup to our graph-based representation.
    
    Args:
        html_table: BeautifulSoup table element
        
    Returns:
        Table object representing the graph structure of the table
    """
    pass  # We'll implement this in table_parser.py

def markdown_to_table(markdown: str) -> Table:
    """
    Convert a markdown table to our graph-based representation.
    
    Args:
        markdown: A string containing a markdown table
        
    Returns:
        Table object representing the graph structure of the table
    """
    pass  # We'll implement this in table_parser.py