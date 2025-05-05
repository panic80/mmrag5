#!/usr/bin/env python3

"""
Adaptive Chunking for RAG Systems

This module provides content-aware chunking strategies that adapt to different document types
and content density. By using different chunking approaches for different content types,
we can create more semantically coherent chunks that better preserve context and meaning.

Features:
- Content type detection and classification
- Density-based chunk size adaptation
- Special handling for different content types (code, tables, lists)
- Content boundary preservation
- Markdown-aware chunking

Usage:
    from adaptive_chunking import adaptive_chunk_text, detect_content_type
    
    # Adaptively chunk a document based on its content type
    chunks = adaptive_chunk_text(text, max_chars=500)
"""

import re
import logging
import json
from typing import Dict, Any, List, Optional, Tuple, Callable
from collections import defaultdict, Counter
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Content type definitions
CONTENT_TYPES = [
    "CODE", "TABLE", "LIST", "PROSE", "HEADING", 
    "DIALOGUE", "TECHNICAL", "MATH", "CITATION", 
    "STRUCTURED", "DENSE", "SPARSE"
]

# Regular expressions for content type detection
CODE_BLOCK_PATTERN = re.compile(r'```(?:\w+)?\n[\s\S]*?\n```|`[^`]+`')
TABLE_PATTERN = re.compile(r'\|[^|]+\|[^|]+\|[^|]*\||\+[-+]+\+')
LIST_PATTERN = re.compile(r'^\s*[-*+]\s+.+$|^\s*\d+\.\s+.+$', re.MULTILINE)
HEADING_PATTERN = re.compile(r'^#{1,6}\s+.+$|^[^\n]+\n[=-]{2,}$', re.MULTILINE)
MATH_PATTERN = re.compile(r'\$\$.+?\$\$|\$.+?\$', re.DOTALL)
DIALOGUE_PATTERN = re.compile(r'^\s*[A-Za-z]+:\s+.+$', re.MULTILINE)
CITATION_PATTERN = re.compile(r'^\s*\[\d+\].*|\(\w+,\s+\d{4}\)|^\s*\d+\.\s+.+\(\d{4}\)', re.MULTILINE)

def detect_density(text: str) -> float:
    """
    Measure content density by estimating information per character.
    
    Args:
        text: Input text
        
    Returns:
        Density score (0-1 scale, where higher values mean denser content)
    """
    if not text or len(text) < 10:
        return 0.5  # Default for very short text
    
    # Get character count without whitespace
    char_count = sum(1 for c in text if not c.isspace())
    total_chars = len(text)
    
    # Calculate character density
    char_density = char_count / total_chars if total_chars > 0 else 0
    
    # Measure line length statistics
    lines = text.splitlines()
    if not lines:
        return 0.5
    
    # Average line length (excluding very short lines)
    content_lines = [line for line in lines if len(line.strip()) > 5]
    if not content_lines:
        return 0.5
    
    avg_line_length = sum(len(line) for line in content_lines) / len(content_lines)
    
    # Normalize avg line length (longer lines often indicate denser content)
    # Consider values between 30-100 as the normal range
    line_length_score = min(1.0, avg_line_length / 100)
    
    # Vocabulary diversity (approximate)
    words = re.findall(r'\b\w+\b', text.lower())
    unique_words = len(set(words))
    word_diversity = unique_words / len(words) if words else 0.5
    
    # Calculate final density score (weighted combination)
    density = 0.4 * char_density + 0.3 * line_length_score + 0.3 * word_diversity
    
    return density

def detect_content_type(text: str) -> Dict[str, float]:
    """
    Detect document content types with confidence scores.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary of content types and confidence scores (0-1)
    """
    if not text:
        return {"SPARSE": 1.0}
    
    # Initialize scores
    scores = defaultdict(float)
    
    # Detect code patterns
    code_matches = len(CODE_BLOCK_PATTERN.findall(text))
    if code_matches:
        code_ratio = min(1.0, code_matches / (len(text) / 500))
        scores["CODE"] = code_ratio
    
    # Detect table patterns
    table_matches = len(TABLE_PATTERN.findall(text))
    if table_matches:
        table_ratio = min(1.0, table_matches / (len(text) / 1000))
        scores["TABLE"] = table_ratio
    
    # Detect list patterns
    list_matches = len(LIST_PATTERN.findall(text))
    if list_matches:
        list_ratio = min(1.0, list_matches / (len(text.splitlines()) / 10))
        scores["LIST"] = list_ratio
    
    # Detect heading patterns (section-rich document)
    heading_matches = len(HEADING_PATTERN.findall(text))
    if heading_matches:
        heading_ratio = min(1.0, heading_matches / (len(text.splitlines()) / 20))
        scores["HEADING"] = heading_ratio
    
    # Detect mathematical content
    math_matches = len(MATH_PATTERN.findall(text))
    if math_matches:
        math_ratio = min(1.0, math_matches / (len(text) / 500))
        scores["MATH"] = math_ratio
    
    # Detect dialogue patterns
    dialogue_matches = len(DIALOGUE_PATTERN.findall(text))
    if dialogue_matches:
        dialogue_ratio = min(1.0, dialogue_matches / (len(text.splitlines()) / 5))
        scores["DIALOGUE"] = dialogue_ratio
    
    # Detect citation patterns (academic/reference text)
    citation_matches = len(CITATION_PATTERN.findall(text))
    if citation_matches:
        citation_ratio = min(1.0, citation_matches / (len(text.splitlines()) / 10))
        scores["CITATION"] = citation_ratio
    
    # Measure content density
    density = detect_density(text)
    if density > 0.7:
        scores["DENSE"] = (density - 0.7) * 3.33  # Scale 0.7-1.0 to 0-1.0
    elif density < 0.4:
        scores["SPARSE"] = (0.4 - density) * 2.5  # Scale 0-0.4 to 0-1.0
    
    # Add prose as the default content type if nothing else is strongly detected
    if not any(score > 0.3 for score in scores.values()):
        scores["PROSE"] = 0.8
    
    # If scores suggest multiple strong content types, identify as STRUCTURED
    if len([s for s in scores.values() if s > 0.5]) > 1:
        scores["STRUCTURED"] = 0.9
    
    # Check for technical content based on domain-specific words
    technical_terms = [
        "algorithm", "function", "method", "procedure", "implementation",
        "parameter", "variable", "constant", "return", "interface", "class",
        "object", "instance", "module", "package", "library", "framework",
        "api", "dependency", "configuration", "environment", "deployment",
        "runtime", "compile", "debug", "exception", "error", "query", "database"
    ]
    
    technical_count = sum(1 for term in technical_terms if term in text.lower())
    if technical_count > 5:
        technical_score = min(1.0, technical_count / 15)
        scores["TECHNICAL"] = technical_score
    
    # Normalize scores
    return dict(scores)

def get_optimal_chunk_size(content_type: Dict[str, float], base_size: int = 500) -> int:
    """
    Determine optimal chunk size based on content type.
    
    Args:
        content_type: Dictionary of content types and confidence scores
        base_size: Base chunk size to adjust from
        
    Returns:
        Adjusted chunk size for the content
    """
    # Set adjustment factors for different content types
    type_adjustments = {
        "CODE": 0.5,         # Smaller chunks for code (preserve function boundaries)
        "TABLE": 0.4,        # Very small chunks for tables
        "LIST": 0.6,         # Slightly smaller chunks for lists
        "PROSE": 1.0,        # Normal chunks for prose
        "HEADING": 1.2,      # Larger chunks for sectioned documents
        "DIALOGUE": 0.8,     # Slightly smaller for dialogue
        "TECHNICAL": 0.7,    # Smaller for technical content
        "MATH": 0.5,         # Smaller for math-heavy content
        "CITATION": 0.9,     # Slightly smaller for citations
        "STRUCTURED": 0.7,   # Smaller for highly structured content
        "DENSE": 0.6,        # Smaller chunks for dense content
        "SPARSE": 1.5        # Larger chunks for sparse content
    }
    
    # Start with the base size
    adjusted_size = base_size
    
    # Get dominant content types (scores > 0.3)
    dominant_types = {k: v for k, v in content_type.items() if v > 0.3}
    
    if not dominant_types:
        return base_size
    
    # Calculate weighted adjustment based on content type scores
    total_weight = sum(dominant_types.values())
    weighted_adjustment = 0
    
    for ctype, score in dominant_types.items():
        if ctype in type_adjustments:
            weighted_adjustment += type_adjustments[ctype] * (score / total_weight)
    
    # Apply adjustment
    adjusted_size = int(base_size * weighted_adjustment)
    
    # Ensure size is within reasonable bounds
    return max(200, min(adjusted_size, 1500))

def chunk_code(text: str, max_chars: int) -> List[str]:
    """
    Specialized chunking for code that preserves function/class boundaries.
    
    Args:
        text: Code text to chunk
        max_chars: Maximum character length for chunks
        
    Returns:
        List of code chunks
    """
    # Detect programming language based on common patterns
    lang = "generic"
    if re.search(r'function\s+\w+\s*\(|class\s+\w+', text):
        lang = "javascript/typescript"
    elif re.search(r'def\s+\w+\s*\(|class\s+\w+', text):
        lang = "python"
    elif re.search(r'public\s+(?:static\s+)?(?:void|class|int|String)', text):
        lang = "java"
    
    chunks = []
    
    # Define boundaries based on language
    if lang == "python":
        # Python functions and classes
        boundaries = re.split(r'(?=^(?:async\s+)?def\s+|^class\s+)', text, flags=re.MULTILINE)
    elif lang == "javascript/typescript":
        # JavaScript/TypeScript functions and classes
        boundaries = re.split(r'(?=^(?:async\s+)?function\s+|^class\s+|^\w+\s*=\s*(?:async\s+)?\()', text, flags=re.MULTILINE)
    else:
        # Generic code - split on common structural patterns
        boundaries = re.split(r'(?=^[a-zA-Z_]\w*\s+\w+\s*\(|^class\s+|^interface\s+)', text, flags=re.MULTILINE)
    
    current_chunk = []
    current_length = 0
    
    for boundary in boundaries:
        if not boundary.strip():
            continue
            
        # If this boundary is small enough to fit
        if len(boundary) <= max_chars:
            if current_length + len(boundary) <= max_chars:
                current_chunk.append(boundary)
                current_length += len(boundary)
            else:
                # Start a new chunk
                chunks.append("".join(current_chunk))
                current_chunk = [boundary]
                current_length = len(boundary)
        else:
            # This boundary is too big - split it further by lines
            if current_chunk:
                chunks.append("".join(current_chunk))
                current_chunk = []
                current_length = 0
                
            # Split large boundary by lines
            lines = boundary.splitlines()
            sub_chunk = []
            sub_length = 0
            
            for line in lines:
                line_length = len(line) + 1  # +1 for newline
                
                if sub_length + line_length <= max_chars:
                    sub_chunk.append(line)
                    sub_length += line_length
                else:
                    if sub_chunk:
                        chunks.append("\n".join(sub_chunk))
                        sub_chunk = [line]
                        sub_length = line_length
                    else:
                        # Line is too long, split within the line
                        chunks.append(line[:max_chars])
                        remaining = line[max_chars:]
                        while remaining:
                            chunks.append(remaining[:max_chars])
                            remaining = remaining[max_chars:]
            
            if sub_chunk:
                chunks.append("\n".join(sub_chunk))
    
    # Add final chunk
    if current_chunk:
        chunks.append("".join(current_chunk))
    
    # Ensure we have at least one chunk
    if not chunks and text.strip():
        chunks = [text]
    
    return chunks

def chunk_table(text: str, max_chars: int) -> List[str]:
    """
    Specialized chunking for tables that preserves row structure.
    
    Args:
        text: Table text to chunk
        max_chars: Maximum character length for chunks
        
    Returns:
        List of table chunks
    """
    chunks = []
    
    # Extract table header pattern (first few rows)
    lines = text.splitlines()
    if not lines:
        return [text] if text else []
    
    # Identify markdown tables
    if lines and '|' in lines[0]:
        # Find header row(s) - typically first row and separator row
        header_rows = []
        for i, line in enumerate(lines[:3]):  # Look at first 3 rows max
            if '|' in line:
                header_rows.append(line)
                # If we see a separator row with dashes, we've found our header section
                if re.match(r'\s*\|[\s\-+:]*\|', line):
                    break
        
        # Process table in chunks, keeping header with each chunk
        header = "\n".join(header_rows)
        current_rows = []
        current_length = len(header)
        
        for line in lines[len(header_rows):]:
            line_length = len(line) + 1  # +1 for newline
            
            if current_length + line_length <= max_chars:
                current_rows.append(line)
                current_length += line_length
            else:
                if current_rows:
                    # Create a chunk with header + rows
                    chunks.append(header + "\n" + "\n".join(current_rows))
                    current_rows = [line]
                    current_length = len(header) + line_length
                else:
                    # Handle rare case where a single row exceeds max_chars
                    chunks.append(header + "\n" + line[:max_chars-len(header)-1])
        
        # Add final chunk
        if current_rows:
            chunks.append(header + "\n" + "\n".join(current_rows))
    else:
        # Just use simple line-based chunking for other structured content
        current_chunk = []
        current_length = 0
        
        for line in lines:
            line_length = len(line) + 1  # +1 for newline
            
            if current_length + line_length <= max_chars:
                current_chunk.append(line)
                current_length += line_length
            else:
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = [line]
                    current_length = line_length
                else:
                    # Line is too long, split within the line
                    chunks.append(line[:max_chars])
                    
        # Add final chunk
        if current_chunk:
            chunks.append("\n".join(current_chunk))
    
    # Ensure we have at least one chunk
    if not chunks and text.strip():
        chunks = [text]
    
    return chunks

def chunk_list(text: str, max_chars: int) -> List[str]:
    """
    Specialized chunking for lists that maintains list item grouping.
    
    Args:
        text: List text to chunk
        max_chars: Maximum character length for chunks
        
    Returns:
        List of list chunks
    """
    chunks = []
    lines = text.splitlines()
    
    # Identify list items and their continuation lines
    list_items = []
    current_item = []
    in_list_item = False
    
    for line in lines:
        # Check if line starts a new list item
        is_list_start = re.match(r'^\s*[-*+]\s+|^\s*\d+\.\s+', line)
        
        if is_list_start:
            # Save previous item if exists
            if current_item:
                list_items.append("\n".join(current_item))
                current_item = []
            
            # Start new item
            current_item.append(line)
            in_list_item = True
        elif in_list_item:
            # If this is an empty line, it might end a list item
            if not line.strip():
                if current_item:
                    list_items.append("\n".join(current_item))
                    current_item = []
                in_list_item = False
            # Otherwise it's a continuation of the current item
            else:
                current_item.append(line)
        else:
            # Not in a list item, treat as normal text
            if current_item:
                list_items.append("\n".join(current_item))
                current_item = []
            
            list_items.append(line)
    
    # Add final item if exists
    if current_item:
        list_items.append("\n".join(current_item))
    
    # Group list items into chunks
    current_chunk = []
    current_length = 0
    
    for item in list_items:
        item_length = len(item) + 1  # +1 for newline
        
        if current_length + item_length <= max_chars:
            current_chunk.append(item)
            current_length += item_length
        else:
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = [item]
                current_length = item_length
            else:
                # Item is too long, split within the item
                # Try to preserve list marker
                match = re.match(r'^(\s*[-*+]\s+|\s*\d+\.\s+)(.*)$', item, re.DOTALL)
                if match:
                    marker, content = match.groups()
                    
                    # Add first chunk with marker
                    first_chunk_content = content[:max_chars-len(marker)]
                    chunks.append(marker + first_chunk_content)
                    
                    # Process remaining content
                    remaining = content[max_chars-len(marker):]
                    indent = " " * len(marker)
                    while remaining:
                        chunks.append(indent + remaining[:max_chars-len(indent)])
                        remaining = remaining[max_chars-len(indent):]
                else:
                    # Not a list item or couldn't parse marker
                    chunks.append(item[:max_chars])
                    remaining = item[max_chars:]
                    while remaining:
                        chunks.append(remaining[:max_chars])
                        remaining = remaining[max_chars:]
    
    # Add final chunk
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    
    # Ensure we have at least one chunk
    if not chunks and text.strip():
        chunks = [text]
    
    return chunks

def chunk_prose(text: str, max_chars: int) -> List[str]:
    """
    Specialized chunking for prose text that preserves paragraph structure.
    
    Args:
        text: Prose text to chunk
        max_chars: Maximum character length for chunks
        
    Returns:
        List of prose chunks
    """
    # Split into paragraphs (text blocks separated by blank lines)
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para_length = len(para) + 2  # +2 for paragraph separators
        
        # If single paragraph is larger than max, split into sentences
        if para_length > max_chars:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Split by sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            sentence_chunk = []
            sentence_length = 0
            
            for sentence in sentences:
                if sentence_length + len(sentence) + 1 <= max_chars:
                    sentence_chunk.append(sentence)
                    sentence_length += len(sentence) + 1
                else:
                    if sentence_chunk:
                        chunks.append(" ".join(sentence_chunk))
                        sentence_chunk = [sentence]
                        sentence_length = len(sentence)
                    else:
                        # Sentence is too long, split within the sentence
                        chunks.append(sentence[:max_chars])
                        remaining = sentence[max_chars:]
                        while remaining:
                            chunks.append(remaining[:max_chars])
                            remaining = remaining[max_chars:]
            
            if sentence_chunk:
                chunks.append(" ".join(sentence_chunk))
        else:
            # Normal paragraph, check if it fits in current chunk
            if current_length + para_length <= max_chars:
                current_chunk.append(para)
                current_length += para_length
            else:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = [para]
                    current_length = para_length
                else:
                    # Should never happen but handle just in case
                    chunks.append(para)
    
    # Add final chunk
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    
    # Ensure we have at least one chunk
    if not chunks and text.strip():
        chunks = [text]
    
    return chunks

def adaptive_chunk_text(text: str, max_chars: int = 500) -> List[str]:
    """
    Adaptively chunk text based on content type detection.
    
    Args:
        text: Text to chunk
        max_chars: Base maximum character length (will be adjusted)
        
    Returns:
        List of chunks adapted to content type
    """
    logger.info(f"Starting adaptive chunking with base size {max_chars}")
    
    if not text or not text.strip():
        return []
    
    # Detect content types
    content_types = detect_content_type(text)
    logger.info(f"Detected content types: {content_types}")
    
    # Determine optimal chunk size for this content
    adapted_size = get_optimal_chunk_size(content_types, max_chars)
    logger.info(f"Adapted chunk size: {adapted_size} (base: {max_chars})")
    
    # Identify primary content type for specialized chunking
    primary_type = max(content_types.items(), key=lambda x: x[1])[0] if content_types else "PROSE"
    logger.info(f"Primary content type: {primary_type}")
    
    # Apply specialized chunking based on content type
    if primary_type == "CODE" and content_types.get("CODE", 0) > 0.6:
        logger.info("Using specialized code chunking")
        return chunk_code(text, adapted_size)
    elif primary_type == "TABLE" and content_types.get("TABLE", 0) > 0.6:
        logger.info("Using specialized table chunking")
        return chunk_table(text, adapted_size)
    elif primary_type == "LIST" and content_types.get("LIST", 0) > 0.6:
        logger.info("Using specialized list chunking")
        return chunk_list(text, adapted_size)
    else:
        # For all other types, use prose chunking with adapted size
        logger.info("Using prose chunking with adapted size")
        return chunk_prose(text, adapted_size)

if __name__ == "__main__":
    """Command line interface for testing adaptive chunking."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Adaptively chunk text based on content type")
    parser.add_argument("--input", "-i", help="Input file path", required=False)
    parser.add_argument("--output", "-o", help="Output file path for chunked results", required=False)
    parser.add_argument("--text", "-t", help="Text to process", required=False)
    parser.add_argument("--size", "-s", type=int, default=500, help="Base maximum chunk size (chars)")
    
    args = parser.parse_args()
    
    # Get input text
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        print("Either --input or --text must be provided")
        parser.print_help()
        sys.exit(1)
    
    # Detect content type
    content_types = detect_content_type(text)
    print(f"Detected content types: {json.dumps(content_types, indent=2)}")
    
    # Determine optimal chunk size
    adapted_size = get_optimal_chunk_size(content_types, args.size)
    print(f"Adapted chunk size: {adapted_size} (base: {args.size})")
    
    # Chunk the text
    chunks = adaptive_chunk_text(text, args.size)
    print(f"Created {len(chunks)} chunks")
    
    # Output results
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                f.write(f"--- CHUNK {i+1} ---\n")
                f.write(chunk)
                f.write("\n\n")
        print(f"Chunks written to {args.output}")
    else:
        for i, chunk in enumerate(chunks):
            print(f"\n--- CHUNK {i+1} (length: {len(chunk)}) ---")
            print(chunk[:100] + "..." if len(chunk) > 100 else chunk)