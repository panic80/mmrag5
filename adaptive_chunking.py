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

# Configure logging with enhanced format for easier debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# Dictionary to track errors and fallbacks for diagnostic purposes
# This helps track the hierarchy of fallbacks and error reasons
ERROR_CONTEXT = {
    "error_count": 0,
    "fallback_path": [],
    "detected_errors": {},
    "last_error": None,
}

# Content type definitions
CONTENT_TYPES = [
    "CODE", "TABLE", "LIST", "PROSE", "HEADING",
    "DIALOGUE", "TECHNICAL", "MATH", "CITATION",
    "STRUCTURED", "DENSE", "SPARSE"
]

# Supported code languages for detection and chunking
SUPPORTED_CODE_LANGUAGES = [
    "python", "javascript", "java", "sql", "html", "css", "json", "xml", "shell", "markdown",
    "go", "c", "cpp", "ruby", "php", "rust", "typescript", "swift", "kotlin", "scala", "r", "perl"
]

# Regular expressions for content type detection and boundary preservation
# Code patterns - including improved language-specific syntax detection and boundaries
CODE_BLOCK_PATTERN = re.compile(r'```(?P<lang>\w+)?\n(?P<code>[\s\S]*?)\n```|`(?P<inline_code>[^`]+)`')
CODE_PATTERNS = {
    "python": re.compile(r'(?:^|\n)(?:async\s+)?(?:def|class)\s+\w+.*?:|(?:^|\n)(?:import|from\s+\w+\s+import)\s+.*?|(?:^|\n)(?:if|for|while|try|except|finally|with)\s+.*?:|@\w+.*?:', re.MULTILINE),
    "javascript": re.compile(r'(?:^|\n)(?:async\s+)?(?:function\s+\w*|class\s+\w*)\s*\(.*?\)|(?:^|\n)(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?function\s*\(.*?\)|(?:^|\n)(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?\(.*?\)\s*=>\s*[\{\w]+|import\s+.*?from\s+.*?;|export\s+(?:default\s+)?(?:class|function|\w+);', re.MULTILINE),
    "java": re.compile(r'(?:^|\n)(?:public|private|protected|static|final|abstract|\s*)+\s+(?:class|interface|enum)\s+\w+.*?\{|(?:^|\n)(?:public|private|protected|static|final|abstract|\s*)+\s+\w+.*?\(.*?\)\s*\{|@\w+', re.MULTILINE),
    "sql": re.compile(r'(?i)(?:^|\n)(?:SELECT|INSERT INTO|UPDATE|DELETE FROM|CREATE TABLE|ALTER TABLE|DROP TABLE|CREATE INDEX|DROP INDEX|GRANT|REVOKE|BEGIN|COMMIT|ROLLBACK|DECLARE)\s+.*?;?', re.MULTILINE),
    "html": re.compile(r'<\/?(?:html|head|body|div|span|p|a|h\d|ul|ol|li|table|thead|tbody|tr|td|th|form|input|button|select|textarea|img|script|style)[^>]*>', re.IGNORECASE),
    "css": re.compile(r'(?:^|\})[^\{]+\{\s*[^}]+\s*\}', re.MULTILINE),
    "json": re.compile(r'\{\s*\"[^"]+\"\s*:\s*[^,\]\}]+\s*(?:,\s*\"[^"]+\"\s*:\s*[^,\]\}]+\s*)*\}|\[\s*(?:[^,\]\}]+\s*(?:,\s*[^,\]\}]+\s*)*)?\]', re.DOTALL),
    "xml": re.compile(r'<\w+[^>]*>.*?<\/\w+>|<[^!\?]+\/>', re.DOTALL),
    "shell": re.compile(r'^#!\/(?:bin\/bash|bin\/sh)|\$\s+.+|^[^#\s].*?=', re.MULTILINE),
    "markdown": re.compile(r'^#{1,6}\s+.+$|^[-*+]\s+.+$|^\d+\.\s+.+$|^>\s+.+|^```.*?```', re.MULTILINE | re.DOTALL),
    "go": re.compile(r'(?:^|\n)(?:func|type|var|const|package|import)\s+\w+.*\{?', re.MULTILINE),
    "c": re.compile(r'(?:^|\n)(?:struct|union|enum|typedef)\s+\w+.*\{?|(?:^|\n)\w+\s+\w+\s*\(.*?\);?|(?:^|\n)\w+\s+\*?\w+\s*\(.*?\)\s*\{', re.MULTILINE),
    "cpp": re.compile(r'(?:^|\n)(?:class|struct|union|enum|typedef)\s+\w+.*\{?|(?:^|\n)(?:template\s*<.*?>\s*)?\w+::\w+\s*\(.*?\).*\{?|(?:^|\n)\w+\s+\*?\w+\s*\(.*?\)\s*\{', re.MULTILINE),
    "ruby": re.compile(r'(?:^|\n)(?:def|class|module)\s+\w+|require\s+.*', re.MULTILINE),
    "php": re.compile(r'<\?php.*?|\?>|(?:^|\n)(?:class|function|namespace|use)\s+\w+.*\{?', re.MULTILINE | re.DOTALL),
}
CODE_BRACKETS_PATTERN = re.compile(r'[\{\}\[\]\(\)\<\>]{5,}')

# Pattern to detect nested code structures and scope boundaries
CODE_SCOPE_START_PATTERN = re.compile(r'[\{\[\(]')
CODE_SCOPE_END_PATTERN = re.compile(r'[\}\]\)]')
CODE_INDENTATION_PATTERN = re.compile(r'^(\s+)\S', re.MULTILINE)

# Table patterns - enhanced to detect different table formats and boundaries
TABLE_MARKDOWN_PATTERN = re.compile(r'(\|[^|]+\|[^|]+\|[^|]*\|(?:\n\|[\s\-:]+\|[\s\-:]+\|[\s\-:]*\|)?(?:\n\|[^|]+\|[^|]+\|[^|]*\|)*)', re.MULTILINE)
TABLE_ASCII_PATTERN = re.compile(r'(\+[-+]+\+|\+[=+]+\+)(?:\n(?:\||\+)[^\n]*)+(?:\n\+[-+]+\+|\+[=+]+\+)', re.MULTILINE)
TABLE_GRID_PATTERN = re.compile(r'(┌[─┬]+┐|┏[━┳]+┓|╔[═╦]+╗)(?:\n(?:│|┃|║)[^\n]*)+(?:\n└[─┴]+┘|┗[━┻]+┛|╚[═╩]+╝)', re.MULTILINE)
TABLE_CSV_PATTERN = re.compile(r'((?:^|\n)(?:[^,\n]*,){2,}[^,\n]*(?:$|\n)(?:[^,\n]*,){2,}[^,\n]*(?:$|\n))*', re.MULTILINE) # Adjusted CSV to not require trailing newline
# Enhanced pattern to detect table separators in various formats
TABLE_SEPARATOR_PATTERN = re.compile(
    r'(\|[\s\-:=+]+\||\+[\-=+]+\+|┼[─━═┼┴┬]+┼|┌[─┬]+┐|┏[━┳]+┓|╔[═╦]+╗|└[─┴]+┘|┗[━┻]+┛|╚[═╩]+╝)',
    re.MULTILINE
) # Pattern to detect table separators

# List patterns - enhanced for nested lists, different formats, and boundaries
# Enhanced patterns for list detection with better support for nested and hierarchical structures
LIST_BULLET_PATTERN = re.compile(r'^(\s*[-*+•◦▪▫]\s+.+(?:\n\s{2,}[^\n]*)*)', re.MULTILINE)
LIST_NUMBER_PATTERN = re.compile(r'^(\s*(?:\d+\.|\d+\)|\(\d+\)|[a-zA-Z]\.|\([a-zA-Z]\)))\s+.+(?:\n\s*(?:\d+\.|\d+\)|\(\d+\)|[a-zA-Z]\.|\([a-zA-Z]\)|\s{2,})[^\n]*)*', re.MULTILINE)
LIST_DEFINITION_PATTERN = re.compile(r'^(\s*[^:\n]+:\s*.+(?:\n\s{2,}[^\n]*)*)', re.MULTILINE)
LIST_INDENT_PATTERN = re.compile(r'^((\s{2,}|\t+)[-*+•◦▪▫]\s+.+(?:\n\s{2,}[^\n]*)*)', re.MULTILINE)
LIST_NESTED_PATTERN = re.compile(r'^(\s+)[-*+•◦▪▫]\s+|\s+(?:\d+\.|\d+\)|\(\d+\)|[a-zA-Z]\.|\([a-zA-Z]\))', re.MULTILINE)

# List patterns - enhanced for nested lists and different formats
# These are primarily for detection, the boundary patterns above are for splitting
LIST_BULLET_DETECTION_PATTERN = re.compile(r'^\s*[-*+•◦▪▫]\s+.+$', re.MULTILINE)
LIST_NUMBER_DETECTION_PATTERN = re.compile(r'^\s*(?:\d+\.|\d+\)|\(\d+\)|[a-zA-Z]\.|\([a-zA-Z]\))\s+.+$', re.MULTILINE)
LIST_NESTED_DETECTION_PATTERN = re.compile(r'^\s{2,}[-*+•◦▪▫]\s+.+$', re.MULTILINE)
LIST_DEFINITION_DETECTION_PATTERN = re.compile(r'^\s*[^:\n]+:\s*.+$', re.MULTILINE)

# Structural patterns
HEADING_PATTERN = re.compile(r'^#{1,6}\s+.+$|^[^\n]+\n[=-]{2,}$', re.MULTILINE)
MATH_PATTERN = re.compile(r'\$\$.+?\$\$|\$.+?\$|\\\[\s*(?:\\begin{(?:align|equation|matrix)}.*?\\end{(?:align|equation|matrix)}|\S.*?)\s*\\\]', re.DOTALL)
DIALOGUE_PATTERN = re.compile(r'^\s*(?:[A-Za-z][A-Za-z\s]*?|"[^"]+"):\s+.+$', re.MULTILINE)
CITATION_PATTERN = re.compile(r'^\s*\[\d+\].*|\(\w+(?:(?:\s+and|\s*[,&])\s+\w+)*,\s+\d{4}(?:[a-z]?)\)|^\s*\d+\.\s+.+\(\d{4}\)', re.MULTILINE)

def reset_error_context():
    """Reset the error tracking context to clean state"""
    global ERROR_CONTEXT
    ERROR_CONTEXT = {
        "error_count": 0,
        "fallback_path": [],
        "detected_errors": {},
        "last_error": None,
    }

def log_error(error_type, message, exception=None):
    """
    Log an error with detailed context and update error tracking
    
    Args:
        error_type: Type/category of the error
        message: Descriptive message about what happened
        exception: Optional exception object
    """
    global ERROR_CONTEXT
    
    ERROR_CONTEXT["error_count"] += 1
    ERROR_CONTEXT["last_error"] = {
        "type": error_type,
        "message": message,
        "exception": str(exception) if exception else None,
        "timestamp": logging.Formatter.formatTime(logging.Formatter(), logging.LogRecord("", 0, "", 0, "", (), None))
    }
    
    if error_type not in ERROR_CONTEXT["detected_errors"]:
        ERROR_CONTEXT["detected_errors"][error_type] = []
    
    ERROR_CONTEXT["detected_errors"][error_type].append(ERROR_CONTEXT["last_error"])
    
    if exception:
        logger.error(f"{error_type}: {message} - Exception: {exception.__class__.__name__}: {exception}")
    else:
        logger.error(f"{error_type}: {message}")

def log_fallback(from_method, to_method, reason):
    """
    Log a fallback event with context and update fallback tracking
    
    Args:
        from_method: Original method that failed
        to_method: Fallback method being used
        reason: Reason for the fallback
    """
    global ERROR_CONTEXT
    
    fallback_info = {
        "from": from_method,
        "to": to_method,
        "reason": reason,
        "timestamp": logging.Formatter.formatTime(logging.Formatter(), logging.LogRecord("", 0, "", 0, "", (), None))
    }
    
    ERROR_CONTEXT["fallback_path"].append(fallback_info)
    logger.warning(f"Fallback from {from_method} to {to_method}: {reason}")

def detect_density(text: str) -> float:
    """
    Measure content density by estimating information per character.
    
    Args:
        text: Input text
        
    Returns:
        Density score (0-1 scale, where higher values mean denser content)
    """
    logger.debug(f"Calculating content density for text of length {len(text)}")
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
    Detect document content types with confidence scores using sophisticated pattern matching.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary of content types and confidence scores (0-1)
    """
    if not text:
        logger.info("Empty text provided to detect_content_type, returning SPARSE classification")
        return {"SPARSE": 1.0}
    
    logger.info(f"Detecting content types for text of length {len(text)}")
    
    # Initialize scores with weighted feature components
    scores = defaultdict(lambda: defaultdict(float))
    final_scores = defaultdict(float)
    
    # Calculate basic metrics for normalization
    line_count = len(text.splitlines()) or 1
    char_count = len(text)
    word_count = len(re.findall(r'\b\w+\b', text)) or 1
    
    # Detect code patterns with language-specific patterns
    code_block_matches = len(CODE_BLOCK_PATTERN.findall(text))
    
    lang_matches = {}
    for lang, pattern in CODE_PATTERNS.items():
        lang_matches[lang] = len(pattern.findall(text))
    
    bracket_density = len(CODE_BRACKETS_PATTERN.findall(text))

    # Weight different code features
    scores["CODE"]["blocks"] = min(1.0, code_block_matches * 2 / math.sqrt(line_count))
    for lang, count in lang_matches.items():
        scores["CODE"][lang] = min(1.0, count * 3 / math.sqrt(line_count)) # Increased weight for specific language patterns
    scores["CODE"]["brackets"] = min(1.0, bracket_density / math.sqrt(line_count))
    
    # Calculate indentation patterns (common in code)
    indent_pattern = re.compile(r'^(?:    |\t)(?:    |\t)*\S', re.MULTILINE)
    indent_matches = len(indent_pattern.findall(text))
    scores["CODE"]["indentation"] = min(1.0, indent_matches / (line_count / 5))
    
    # Weighted code score aggregation
    code_weights = {
        "blocks": 0.2, "python": 0.1, "javascript": 0.1, "java": 0.1,
        "sql": 0.05, "html": 0.05, "css": 0.05, "json": 0.05, "xml": 0.05,
        "shell": 0.05, "markdown": 0.05, "brackets": 0.05, "indentation": 0.1
    }
    for feature, weight in code_weights.items():
        if feature in scores["CODE"]:
            final_scores["CODE"] += scores["CODE"][feature] * weight
    
    # Detect table patterns with enhanced recognition
    markdown_table = len(TABLE_MARKDOWN_PATTERN.findall(text))
    table_separators = len(TABLE_SEPARATOR_PATTERN.findall(text))
    ascii_tables = len(TABLE_ASCII_PATTERN.findall(text))
    grid_tables = len(TABLE_GRID_PATTERN.findall(text))
    csv_tables = len(TABLE_CSV_PATTERN.findall(text))
    
    # Calculate data-to-text ratio for potential tables
    data_chars = sum(len(m) for m in re.findall(r'\b[0-9.,%$]+\b', text))
    data_ratio = data_chars / char_count if char_count > 0 else 0
    
    # Weight different table features
    scores["TABLE"]["markdown"] = min(1.0, markdown_table * 2 / math.sqrt(line_count))
    scores["TABLE"]["separators"] = min(1.0, table_separators * 3 / math.sqrt(line_count))
    scores["TABLE"]["ascii"] = min(1.0, ascii_tables * 2 / math.sqrt(line_count))
    scores["TABLE"]["grid"] = min(1.0, grid_tables * 3 / math.sqrt(line_count))
    scores["TABLE"]["csv"] = min(1.0, csv_tables / math.sqrt(line_count))
    scores["TABLE"]["data_ratio"] = min(1.0, data_ratio * 5)  # High data ratio often indicates tables
    
    # Weighted table score aggregation
    table_weights = {
        "markdown": 0.25, "separators": 0.25, "ascii": 0.15,
        "grid": 0.15, "csv": 0.1, "data_ratio": 0.1
    }
    for feature, weight in table_weights.items():
        final_scores["TABLE"] += scores["TABLE"][feature] * weight
    
    # Detect list patterns with improved recognition
    bullet_lists = len(LIST_BULLET_DETECTION_PATTERN.findall(text))
    number_lists = len(LIST_NUMBER_DETECTION_PATTERN.findall(text))
    nested_lists = len(LIST_NESTED_DETECTION_PATTERN.findall(text))
    definition_lists = len(LIST_DEFINITION_DETECTION_PATTERN.findall(text))
    
    # Analyze line patterns that might indicate lists (consistent leading patterns)
    line_starts = Counter()
    for line in text.splitlines():
        if line.strip():
            leading_chars = re.match(r'^(\s*\W+\s*)', line)
            if leading_chars:
                line_starts[leading_chars.group(1)] += 1
    
    repetitive_starts = sum(count for pattern, count in line_starts.items() if count > 2)
    
    # Weight different list features
    scores["LIST"]["bullet"] = min(1.0, bullet_lists / (line_count / 8))
    scores["LIST"]["numbered"] = min(1.0, number_lists / (line_count / 8))
    scores["LIST"]["nested"] = min(1.0, nested_lists / (line_count / 15))
    scores["LIST"]["definition"] = min(1.0, definition_lists / (line_count / 10))
    scores["LIST"]["repetitive"] = min(1.0, repetitive_starts / (line_count / 3))
    
    # Weighted list score aggregation
    list_weights = {
        "bullet": 0.3, "numbered": 0.3, "nested": 0.2,
        "definition": 0.1, "repetitive": 0.1
    }
    for feature, weight in list_weights.items():
        final_scores["LIST"] += scores["LIST"][feature] * weight
    
    # Detect structural patterns (headings, math, dialogue, citations)
    heading_matches = len(HEADING_PATTERN.findall(text))
    math_matches = len(MATH_PATTERN.findall(text))
    dialogue_matches = len(DIALOGUE_PATTERN.findall(text))
    citation_matches = len(CITATION_PATTERN.findall(text))
    
    # Calculate significance based on document size
    heading_significance = heading_matches / (line_count / 25)
    final_scores["HEADING"] = min(1.0, heading_significance)
    
    math_significance = math_matches / (char_count / 1000)
    final_scores["MATH"] = min(1.0, math_significance * 2)
    
    dialogue_significance = dialogue_matches / (line_count / 8)
    final_scores["DIALOGUE"] = min(1.0, dialogue_significance)
    
    citation_significance = citation_matches / (line_count / 15)
    final_scores["CITATION"] = min(1.0, citation_significance * 1.5)
    
    # Measure content density with improved metrics
    density = detect_density(text)
    if density > 0.65:  # Lower threshold for dense content
        final_scores["DENSE"] = min(1.0, (density - 0.65) * 2.85)  # Scale 0.65-1.0 to 0-1.0
    elif density < 0.45:  # Higher threshold for sparse content
        final_scores["SPARSE"] = min(1.0, (0.45 - density) * 2.2)  # Scale 0-0.45 to 0-1.0
    
    # Technical content detection with expanded vocabulary
    technical_domains = {
        "programming": [
            "algorithm", "function", "method", "procedure", "implementation",
            "parameter", "variable", "constant", "return", "interface", "class",
            "object", "instance", "module", "package", "library", "framework",
            "api", "dependency", "configuration", "environment", "deployment",
            "runtime", "compile", "debug", "exception", "error", "query", "database",
            "async", "synchronous", "thread", "process", "memory", "cache", "buffer"
        ],
        "data_science": [
            "dataset", "feature", "label", "training", "model", "accuracy", "precision",
            "recall", "f1", "classification", "regression", "cluster", "neural",
            "vector", "matrix", "tensor", "gradient", "epoch", "batch", "validation",
            "test", "prediction", "inference", "overfitting", "underfitting"
        ],
        "network": [
            "protocol", "tcp", "ip", "http", "request", "response", "server",
            "client", "packet", "router", "firewall", "gateway", "dns", "domain",
            "url", "endpoint", "latency", "bandwidth", "throughput"
        ],
        "system": [
            "kernel", "driver", "firmware", "hardware", "cpu", "gpu", "memory",
            "disk", "storage", "filesystem", "mount", "boot", "process", "thread",
            "scheduler", "interrupt", "signal", "pipe", "socket"
        ]
    }
    
    # Process each technical domain
    tech_word_count = 0
    tech_domain_scores = defaultdict(float)
    
    words_lower = set(w.lower() for w in re.findall(r'\b\w+\b', text))
    
    for domain, terms in technical_domains.items():
        domain_matches = sum(1 for term in terms if term in words_lower)
        domain_ratio = domain_matches / len(terms)
        tech_word_count += domain_matches
        tech_domain_scores[domain] = domain_ratio
    
    # Calculate technical score based on density and diversity of technical terms
    if tech_word_count > 3:
        tech_term_density = tech_word_count / word_count
        tech_term_diversity = sum(1 for score in tech_domain_scores.values() if score > 0.1)
        
        # Weighted technical score
        technical_base = min(1.0, tech_term_density * 15)
        technical_diversity = min(1.0, tech_term_diversity / len(technical_domains))
        final_scores["TECHNICAL"] = 0.7 * technical_base + 0.3 * technical_diversity
    
    # Look for mixed content sections with sliding window analysis
    if len(text) > 1000:
        window_size = min(len(text) // 3, 2000)
        section_count = max(3, len(text) // window_size)
        
        section_types = []
        for i in range(section_count):
            start = i * len(text) // section_count
            end = (i + 1) * len(text) // section_count
            section = text[start:end]
            
            # Simplified content type detection for the section
            section_type = None
            section_score = 0
            
            # Check for code in section
            code_in_section = any(pattern.search(section) for pattern in CODE_PATTERNS.values())

            # Check for table in section
            table_in_section = bool(TABLE_MARKDOWN_PATTERN.search(section) or
                                   TABLE_ASCII_PATTERN.search(section) or
                                   TABLE_GRID_PATTERN.search(section) or
                                   TABLE_CSV_PATTERN.search(section))

            # Check for list in section
            list_in_section = bool(LIST_BULLET_DETECTION_PATTERN.search(section) or
                                  LIST_NUMBER_DETECTION_PATTERN.search(section) or
                                  LIST_DEFINITION_DETECTION_PATTERN.search(section) or
                                  LIST_INDENT_PATTERN.search(section))

            if code_in_section:
                section_type = "CODE"
            elif table_in_section:
                section_type = "TABLE"
            elif list_in_section:
                section_type = "LIST"
            else:
                section_type = "PROSE"
            
            section_types.append(section_type)
        
        # Detect content type transitions
        transitions = sum(1 for i in range(1, len(section_types))
                          if section_types[i] != section_types[i-1])
        
        # If we have multiple different section types, mark as structured with mixed content
        if transitions > 0:
            structured_score = min(1.0, transitions / (section_count - 1) * 1.5)
            final_scores["STRUCTURED"] = max(final_scores.get("STRUCTURED", 0), structured_score)
    
    # Add prose as the fallback content type if nothing else is strongly detected
    prose_indicators = [
        len(re.findall(r'[.!?]\s+[A-Z]', text)) / max(1, line_count / 5),  # Sentences
        sum(1 for line in text.splitlines() if len(line.strip()) > 40) / max(1, line_count),  # Long lines
        1 - sum(s for s in final_scores.values()) * 0.5  # Inverse of other type strengths
    ]
    
    if not any(score > 0.4 for score in final_scores.values()):
        prose_score = min(1.0, sum(prose_indicators) / len(prose_indicators))
        final_scores["PROSE"] = max(0.6, prose_score)
    elif sum(s for s in final_scores.values() if s > 0.4) <= 1:
        # Add some prose score if only one other type is strong
        prose_score = min(0.5, sum(prose_indicators) / len(prose_indicators))
        final_scores["PROSE"] = prose_score
    
    # Return normalized final scores
    return dict(final_scores)

def get_optimal_chunk_size(content_type: Dict[str, float], base_size: int = 500) -> int:
    """
    Determine optimal chunk size based on content type.
    
    Args:
        content_type: Dictionary of content types and confidence scores
        base_size: Base chunk size to adjust from
        
    Returns:
        Adjusted chunk size for the content
    """
    logger.info(f"Determining optimal chunk size from base size {base_size}")
    logger.debug(f"Content type scores: {json.dumps(content_type)}")
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
    adjusted_size = max(200, min(adjusted_size, 1500))
    logger.info(f"Adjusted chunk size: {adjusted_size} (from base: {base_size})")
    return adjusted_size

def chunk_code(text: str, max_chars: int) -> List[Dict[str, Any]]:
    """
    Specialized chunking for code that preserves function/class boundaries and includes metadata.
    
    Args:
        text: Code text to chunk
        max_chars: Maximum character length for chunks
        
    Returns:
        List of code chunks with metadata.
    """
    logger.info(f"Chunking code of length {len(text)} with max_chars={max_chars}")
    # Detect programming language based on common patterns with improved detection
    detected_lang = "generic"
    lang_scores = {}
    for lang, pattern in CODE_PATTERNS.items():
        lang_scores[lang] = len(pattern.findall(text))

    # Determine the language with the most matches, with more specific threshold
    sorted_langs = sorted(lang_scores.items(), key=lambda item: item[1], reverse=True)
    if sorted_langs and sorted_langs[0][1] > 1:  # Threshold of 2 matches
        detected_lang = sorted_langs[0][0]
    
    logger.info(f"Detected code language: {detected_lang} (scores: {sorted_langs[:3] if sorted_langs else 'none'})")

    # Analyze code structure for better boundary preservation
    # Analyze indentation patterns to understand code block structure
    indentation_levels = {}
    for match in CODE_INDENTATION_PATTERN.finditer(text):
        indent = match.group(1)
        indentation_levels[len(indent)] = indentation_levels.get(len(indent), 0) + 1
    
    # Sort indentation levels to find the most common ones
    common_indents = sorted(indentation_levels.items(), key=lambda x: x[1], reverse=True)
    primary_indent_level = common_indents[0][0] if common_indents else 0
    
    # Detect scope nesting depth to better handle nested structures
    scope_stack = []
    scope_boundaries = []
    
    # Enhanced boundary patterns with more programming languages and better boundaries
    boundary_patterns = {
       "python": re.compile(r'(?=^(?:async\s+)?def\s+\w+|^class\s+\w+|^(?:if|for|while|try|except|finally|with)\s+.*?:)', re.MULTILINE),
       "javascript": re.compile(r'(?=^(?:async\s+)?(?:function\s+\w*|class\s+\w*)\s*\(.*?\)|^(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?function\s*\(.*?\)|^(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?\(.*?\)\s*=>\s*[\{\w]+)', re.MULTILINE),
       "java": re.compile(r'(?=^(?:public|private|protected|static|final|abstract|\s*)+\s+(?:class|interface|enum)\s+\w+.*?\{|^(?:public|private|protected|static|final|abstract|\s*)+\s+\w+.*?\(.*?\)\s*\{)', re.MULTILINE),
       "sql": re.compile(r'(?i)(?=^(?:SELECT|INSERT INTO|UPDATE|DELETE FROM|CREATE TABLE|ALTER TABLE|DROP TABLE|CREATE INDEX|DROP INDEX|GRANT|REVOKE|BEGIN|COMMIT|ROLLBACK|DECLARE)\s+)', re.MULTILINE),
       "html": re.compile(r'(?=<\w+[^>]*>)', re.DOTALL),
       "css": re.compile(r'(?=^\s*\S[^\{]*\{)', re.MULTILINE),
       "json": re.compile(r'(?=[\{\[])', re.DOTALL), # Split at top-level objects/arrays
       "xml": re.compile(r'(?=<\w+[^>]*>|<[^!\?]+\/>)', re.DOTALL),
       "shell": re.compile(r'(?=^\$\s+|^#!\/.*)', re.MULTILINE),
       "markdown": re.compile(r'(?=^#{1,6}\s+|^```.*?```|\n\s*[-*+•◦▪▫]\s+)', re.MULTILINE | re.DOTALL),
       "go": re.compile(r'(?=^(?:func|type|var|const|package)\s+\w+.*\{?)', re.MULTILINE),
       "c": re.compile(r'(?=^(?:struct|union|enum|typedef)\s+\w+.*\{?|^(?:static\s+)?\w+\s+\*?\w+\s*\(.*?\)\s*\{)', re.MULTILINE),
       "cpp": re.compile(r'(?=^(?:class|struct|union|enum|typedef)\s+\w+.*\{?|^(?:template\s*<.*?>\s*)?\w+::\w+\s*\(.*?\).*\{?|^(?:static\s+)?\w+\s+\*?\w+\s*\(.*?\)\s*\{)', re.MULTILINE),
       "ruby": re.compile(r'(?=^(?:def|class|module)\s+\w+)', re.MULTILINE),
       "php": re.compile(r'(?=<\?php.*?|\?>|^(?:class|function|namespace)\s+\w+.*\{?)', re.MULTILINE | re.DOTALL),
       "rust": re.compile(r'(?=^(?:pub\s+)?(?:fn|struct|enum|trait|impl|use|mod)\s+\w+)', re.MULTILINE),
       "typescript": re.compile(r'(?=^(?:export\s+)?(?:class|interface|enum|type|function|const|let|var)\s+\w+)', re.MULTILINE),
       "swift": re.compile(r'(?=^(?:func|class|struct|enum|protocol|extension)\s+\w+)', re.MULTILINE),
       "kotlin": re.compile(r'(?=^(?:fun|class|interface|object|data\s+class|enum\s+class)\s+\w+)', re.MULTILINE),
       "scala": re.compile(r'(?=^(?:def|class|object|trait|val|var)\s+\w+)', re.MULTILINE),
       "generic": re.compile(r'(?=^(?:(?:public|private|protected|static|\s*)+)?(?:function|def|class|interface|enum|import)\s+\w+)', re.MULTILINE) # Fallback generic
    }

    # Specialized language detection for languages not in the standard set
    if detected_lang == "generic" and primary_indent_level > 0:
        # Check for specific language indicators that might have been missed
        if re.search(r'func\s+\w+\s*\(', text):
            detected_lang = "go"
        elif re.search(r'fn\s+\w+\s*\(', text):
            detected_lang = "rust"
        elif re.search(r'val\s+\w+\s*[:=]|var\s+\w+\s*[:=]', text):
            detected_lang = "kotlin"

    chunks = []
    
    # Initial metadata for code chunks
    chunk_metadata = {
        "content_type": "CODE",
        "language": detected_lang,
        "split_reason": "boundary_split",
        "scope_level": 0,  # Track nesting level for better context
    }
    
    # Get boundaries based on language detection
    boundaries = []
    if detected_lang in boundary_patterns:
        boundaries = boundary_patterns[detected_lang].split(text)
    else:
        # Fallback for languages with less specific patterns or if detection was uncertain
        boundaries = boundary_patterns["generic"].split(text)

    # Handle JSON specifically to ensure valid JSON chunks
    if detected_lang == "json":
        try:
            parsed_json = json.loads(text)
            if isinstance(parsed_json, list):
                # For arrays, keep each item as a separate boundary
                boundaries = [json.dumps(item, indent=2) for item in parsed_json]
            elif isinstance(parsed_json, dict):
                # For objects, keep each property as a separate boundary when possible
                if len(parsed_json) > 1:
                    boundaries = [f'"{k}": {json.dumps(v, indent=2)}' for k, v in parsed_json.items()]
                else:
                    # Small objects should be kept together
                    boundaries = [text]
            else:
                boundaries = [text]  # Not a list or dict, keep as one
        except json.JSONDecodeError:
            logger.warning("Invalid JSON detected, applying heuristic-based chunking")
            log_fallback("json_parse", "heuristic_json_chunking", f"JSON parse error: {str(e)}")
            # Use a more sophisticated approach to find JSON-like boundaries
            json_obj_pattern = re.compile(r'(\{(?:[^{}]|(?1))*\})', re.DOTALL)
            json_arr_pattern = re.compile(r'(\[(?:[^\[\]]|(?1))*\])', re.DOTALL)
            
            # Try to find complete JSON objects/arrays
            obj_matches = json_obj_pattern.findall(text)
            arr_matches = json_arr_pattern.findall(text)
            
            if obj_matches or arr_matches:
                boundaries = obj_matches + arr_matches
            else:
                # Fallback to lines for invalid JSON with no clear structure
                boundaries = text.splitlines(keepends=True)

    # Filter out empty boundaries and add newline for splitting if not already present
    boundaries = [b for b in boundaries if b.strip()]
    if not boundaries:
        logger.warning("No specific code boundaries found, falling back to intelligent line grouping")
        log_fallback("boundary_detection", "line_grouping", "No clear code boundaries detected")
        # Group lines by indentation to preserve code blocks
        lines = text.splitlines(keepends=True)
        current_group = []
        current_indent = -1
        
        for line in lines:
            if not line.strip():
                # Keep empty lines with their surrounding context
                if current_group:
                    current_group.append(line)
                continue
            
            # Calculate indentation level
            indent = len(line) - len(line.lstrip())
            
            # Start a new group on zero indentation or significant changes
            if current_indent == -1 or (indent == 0 and current_indent > 0) or current_indent > indent:
                if current_group:
                    boundaries.append("".join(current_group))
                current_group = [line]
                current_indent = indent
            else:
                # Continue current group
                current_group.append(line)
                # Update indent level to track nested structures
                if indent > current_indent:
                    current_indent = indent
        
        # Add the last group
        if current_group:
            boundaries.append("".join(current_group))

    # Process boundaries into chunks with scope awareness
    current_chunk_content = []
    current_length = 0
    current_scope_level = 0
    
    for i, boundary in enumerate(boundaries):
        # Analyze scope changes in this boundary for metadata
        scope_starts = len(CODE_SCOPE_START_PATTERN.findall(boundary))
        scope_ends = len(CODE_SCOPE_END_PATTERN.findall(boundary))
        scope_change = scope_starts - scope_ends
        
        boundary_length = len(boundary)
        
        # Track scope level for code context
        boundary_scope_level = current_scope_level
        current_scope_level = max(0, current_scope_level + scope_change)

        # If this boundary is small enough to fit in the current chunk
        if current_length + boundary_length <= max_chars:
            current_chunk_content.append(boundary)
            current_length += boundary_length
        else:
            # Start a new chunk with appropriate context metadata
            if current_chunk_content:
                chunks.append({
                    "content": "".join(current_chunk_content),
                    "metadata": {
                        **chunk_metadata.copy(),
                        "scope_level": boundary_scope_level,
                        "has_continuation": True if i < len(boundaries) - 1 else False
                    }
                })
            
            # Reset for next chunk
            current_chunk_content = [boundary]
            current_length = boundary_length
            
            # Handle case where a single boundary is too large
            if boundary_length > max_chars:
                logger.debug(f"Code boundary too large ({boundary_length}), using structure-aware splitting.")
                
                # If current boundary is already in the chunks list, remove it
                if current_chunk_content and chunks and chunks[-1]["content"] == "".join(current_chunk_content):
                    chunks.pop()
                
                # Try to split at meaningful points like line breaks first
                lines = boundary.splitlines(keepends=True)
                sub_chunk_content = []
                sub_length = 0
                sub_scope_level = boundary_scope_level
                
                for line_num, line in enumerate(lines):
                    line_length = len(line)
                    
                    # Track scope changes within lines for better context
                    line_scope_starts = len(CODE_SCOPE_START_PATTERN.findall(line))
                    line_scope_ends = len(CODE_SCOPE_END_PATTERN.findall(line))
                    sub_scope_level += line_scope_starts - line_scope_ends
                    
                    if sub_length + line_length <= max_chars:
                        sub_chunk_content.append(line)
                        sub_length += line_length
                    else:
                        # Add completed sub-chunk with scope context
                        if sub_chunk_content:
                            chunks.append({
                                "content": "".join(sub_chunk_content),
                                "metadata": {
                                    **chunk_metadata.copy(),
                                    "split_reason": "line_split",
                                    "scope_level": boundary_scope_level,
                                    "has_continuation": True
                                }
                            })
                            
                            # Update context for next chunk
                            boundary_scope_level = sub_scope_level
                            sub_chunk_content = [line]
                            sub_length = line_length
                        else:
                            # Handle case where a single line is too long
                            logger.warning(f"Code line too large ({line_length}), preserving indent and syntax")
                            log_error("OVERSIZED_CODE_LINE", f"Code line exceeds max_chars ({line_length} > {max_chars})")
                            
                            # For long lines, try to preserve syntax structure
                            if line.lstrip().startswith(("def ", "class ", "function", "interface", "struct")):
                                # For function/class definitions, keep the signature intact if possible
                                signature_end = line.find(":")
                                if signature_end == -1:
                                    signature_end = line.find("{")
                                
                                if signature_end != -1 and signature_end < max_chars:
                                    chunks.append({
                                        "content": line[:signature_end+1],
                                        "metadata": {
                                            **chunk_metadata.copy(),
                                            "split_reason": "signature_split",
                                            "scope_level": boundary_scope_level,
                                            "has_continuation": True
                                        }
                                    })
                                    
                                    remaining = line[signature_end+1:]
                                    while remaining:
                                        split_size = min(len(remaining), max_chars)
                                        chunks.append({
                                            "content": remaining[:split_size],
                                            "metadata": {
                                                **chunk_metadata.copy(),
                                                "split_reason": "char_split",
                                                "scope_level": boundary_scope_level + 1,
                                                "has_continuation": len(remaining) > split_size
                                            }
                                        })
                                        remaining = remaining[split_size:]
                                else:
                                    # Fallback to char split with indentation preservation
                                    indent = re.match(r'^\s*', line).group(0)
                                    content = line.lstrip()
                                    
                                    first_chunk = True
                                    while content:
                                        chunk_content = indent + content[:max_chars - len(indent)] if first_chunk else indent + content[:max_chars - len(indent)]
                                        chunks.append({
                                            "content": chunk_content,
                                            "metadata": {
                                                **chunk_metadata.copy(),
                                                "split_reason": "char_split",
                                                "scope_level": boundary_scope_level,
                                                "has_continuation": len(content) > max_chars - len(indent)
                                            }
                                        })
                                        content = content[max_chars - len(indent):]
                                        first_chunk = False
                            else:
                                # For other long lines, preserve indentation
                                indent = re.match(r'^\s*', line).group(0)
                                content = line.lstrip()
                                
                                first_chunk = True
                                while content:
                                    chunk_content = indent + content[:max_chars - len(indent)] if first_chunk else indent + content[:max_chars - len(indent)]
                                    chunks.append({
                                        "content": chunk_content,
                                        "metadata": {
                                            **chunk_metadata.copy(),
                                            "split_reason": "char_split",
                                            "scope_level": boundary_scope_level,
                                            "has_continuation": len(content) > max_chars - len(indent)
                                        }
                                    })
                                    content = content[max_chars - len(indent):]
                                    first_chunk = False
                
                # Add final sub-chunk
                if sub_chunk_content:
                    chunks.append({
                        "content": "".join(sub_chunk_content),
                        "metadata": {
                            **chunk_metadata.copy(),
                            "split_reason": "line_split",
                            "scope_level": sub_scope_level,
                            "has_continuation": False
                        }
                    })
                
                # Reset for next boundary
                current_chunk_content = []
                current_length = 0
            else:
                # Just a regular boundary that didn't fit in previous chunk
                if chunks:
                    # Update previous chunk's metadata to indicate continuation
                    chunks[-1]["metadata"]["has_continuation"] = True

    # Add final chunk
    if current_chunk_content:
        chunks.append({
            "content": "".join(current_chunk_content),
            "metadata": {
                **chunk_metadata.copy(),
                "scope_level": current_scope_level,
                "has_continuation": False
            }
        })

    # Ensure we have at least one chunk if the original text was not empty
    if not chunks and text.strip():
        chunks = [{
            "content": text,
            "metadata": {
                "content_type": "CODE",
                "language": detected_lang,
                "split_reason": "no_split",
                "scope_level": 0,
                "has_continuation": False
            }
        }]
        logger.warning("Original text was not empty but chunking resulted in no chunks. Adding original text as single chunk.")
        log_fallback("code_chunking", "single_chunk_fallback", "Chunking produced no valid chunks")

    return chunks

def chunk_table(text: str, max_chars: int) -> List[Dict[str, Any]]:
    """
    Specialized chunking for tables with improved header preservation and complex structure handling.
    
    Args:
        text: Table content to chunk
        max_chars: Maximum character length for chunks
        
    Returns:
        List of table chunks with metadata.
    """
    logger.info(f"Chunking table of length {len(text)} with max_chars={max_chars}")
    # Enhanced table format detection
    table_format = "unknown"
    header_lines_content = [] # Stores the string content of header lines
    data_rows_content = [] # Stores the string content of data rows
    lines = text.strip().split('\n') # Process lines without keepends initially for easier logic
    
    # Regex for common table separators
    markdown_separator_pattern = re.compile(r'^\s*\|?(:?-+:?\|)+:?-+:?\s*$')
    # ascii_header_pattern is used for detecting bordered lines, not primary format detection
    
    # Markdown Table Detection
    if any("|" in line and "-" in line for line in lines): # Basic check for markdown
        table_format = "markdown"
        separator_indices = [i for i, line_content in enumerate(lines) if markdown_separator_pattern.match(line_content)]
        if separator_indices:
            header_end_index = separator_indices[0]
            header_lines_content = lines[:header_end_index + 1] # Include separator line
            data_rows_content = lines[header_end_index + 1:]
        elif lines and "|" in lines[0]: # No clear separator, assume first line is header if it has pipes
            header_lines_content = [lines[0]]
            data_rows_content = lines[1:]
        else: # Fallback: treat all as data rows if no header structure
            data_rows_content = lines
                
    # ASCII/Unicode Table Detection (using TABLE_SEPARATOR_PATTERN for more robust detection)
    elif TABLE_SEPARATOR_PATTERN.search(text): # Search in original text with newlines
        table_format = "ascii"
        header_end_index = -1
        # Iterate through original lines with keepends for TABLE_SEPARATOR_PATTERN
        original_lines_with_keepends = text.splitlines(keepends=True)

        temp_header_lines_indices = []
        potential_header_end = -1
        for i, line_content_with_keepends in enumerate(original_lines_with_keepends):
            line_stripped = line_content_with_keepends.strip()
            if TABLE_SEPARATOR_PATTERN.fullmatch(line_stripped):
                # This is a separator line. If content above it is not a separator, it's a header.
                if i > 0 and not TABLE_SEPARATOR_PATTERN.fullmatch(original_lines_with_keepends[i-1].strip()):
                    potential_header_end = i
                    break
                elif i == 0 and len(original_lines_with_keepends) > 1 and not TABLE_SEPARATOR_PATTERN.fullmatch(original_lines_with_keepends[i+1].strip()):
                    # Separator at top, next line is data. Header is just this separator.
                    header_lines_content = [lines[i]] # Use non-keepends version for content storage
                    data_rows_content = lines[i+1:]
                    potential_header_end = i # Mark that header processing is done
                    break
        
        if potential_header_end != -1 and not data_rows_content: # If data_rows not assigned by early exit
             header_lines_content = lines[:potential_header_end + 1]
             data_rows_content = lines[potential_header_end + 1:]
        elif not header_lines_content and not data_rows_content: # Fallback if above logic didn't set anything
            # Try simpler ASCII border detection if complex separator pattern fails
            ascii_border_pattern = re.compile(r'^\s*[\+\|].*[\+\|]\s*$')
            if lines and ascii_border_pattern.match(lines[0]):
                header_lines_content.append(lines[0])
                if len(lines) > 1 and not ascii_border_pattern.match(lines[1]): # Next line is content
                    header_lines_content.append(lines[1])
                    if len(lines) > 2 and ascii_border_pattern.match(lines[2]): # Separator below content
                        header_lines_content.append(lines[2])
                        data_rows_content = lines[3:]
                    else:
                        data_rows_content = lines[2:]
                else: data_rows_content = lines[1:] # No clear content or separator
            else: data_rows_content = lines # No clear header structure

    # CSV-like format (simple heuristic)
    elif any("," in line for line in lines) and len(lines) > 0 : # Check if any line has a comma
        table_format = "csv"
        if lines:
            first_line_commas = lines[0].count(',')
            # A simple heuristic: if first line has commas and is not purely numeric, assume header
            if first_line_commas > 0 and not lines[0].replace(',', '').replace('.', '').strip().isdigit():
                header_lines_content = [lines[0]]
                data_rows_content = lines[1:]
            else:
                data_rows_content = lines
    else: # Fallback for unknown formats
        data_rows_content = lines

    chunks = []
    current_chunk_assembled_content = [] # Stores lines to be joined for the current chunk
    current_length = 0
    
    # Initial metadata for table chunks
    base_metadata = {
        "content_type": "TABLE",
        "table_format": table_format,
        "has_header": bool(header_lines_content),
        "split_reason": "row_split" # Default, can be overridden
    }

    # Convert header lines to string with newlines for length calculation and prepending
    header_block_text_with_newlines = "\n".join(header_lines_content) + ("\n" if header_lines_content else "")
    header_block_len = len(header_block_text_with_newlines)

    # Add header lines to the first chunk if they exist and fit
    if header_lines_content:
        if header_block_len <= max_chars:
            current_chunk_assembled_content.extend(header_lines_content)
            current_length += header_block_len
        else:
            # Header is too large, split it. Each part of header is a chunk.
            logger.warning(f"Table header too large ({header_block_len} chars), splitting header")
            log_error("OVERSIZED_TABLE_HEADER", f"Table header exceeds max_chars ({header_block_len} > {max_chars})")
            temp_header_part_lines = []
            temp_header_part_len = 0
            for h_line_content in header_lines_content:
                h_line_len_with_newline = len(h_line_content) + 1 # +1 for newline
                if temp_header_part_len + h_line_len_with_newline <= max_chars:
                    temp_header_part_lines.append(h_line_content)
                    temp_header_part_len += h_line_len_with_newline
                else:
                    if temp_header_part_lines:
                        chunks.append({
                            "content": "\n".join(temp_header_part_lines),
                            "metadata": {**base_metadata, "is_header_chunk": True, "split_reason": "header_too_large"}
                        })
                    temp_header_part_lines = [h_line_content]
                    temp_header_part_len = h_line_len_with_newline
            if temp_header_part_lines: # Add remaining part of header
                 chunks.append({
                    "content": "\n".join(temp_header_part_lines),
                    "metadata": {**base_metadata, "is_header_chunk": True, "split_reason": "header_too_large"}
                })
            # After splitting header, next chunk will be data, so reset current_chunk_assembled_content
            current_chunk_assembled_content = []
            current_length = 0 # Header already processed

    # Process data rows
    for i, row_text_content in enumerate(data_rows_content):
        row_len_with_newline = len(row_text_content) + 1  # +1 for newline
        
        # Check if a new chunk needs to start with the header
        # This happens if current_chunk_assembled_content is empty (meaning we are starting a new chunk)
        # AND header_lines_content exist AND this is not the very first set of rows being processed after initial header.
        if not current_chunk_assembled_content and header_lines_content and (chunks or current_length == 0 and not header_lines_content): # Second part of condition ensures this is not the initial header placement
            # Try to add header if it fits with the current row
            if header_block_len + row_len_with_newline <= max_chars:
                current_chunk_assembled_content.extend(header_lines_content)
                current_length += header_block_len
            # If header + row doesn't fit, the row starts the chunk, header is omitted for this specific chunk.

        if current_length + row_len_with_newline <= max_chars:
            current_chunk_assembled_content.append(row_text_content)
            current_length += row_len_with_newline
        else:
            # Finalize current chunk
            if current_chunk_assembled_content:
                chunk_meta = base_metadata.copy()
                # Mark continuation if there are more data rows OR if this chunk doesn't end with the last data row processed so far
                chunk_meta["has_continuation"] = True # Assume true, can be set to false for the very last chunk
                chunks.append({
                    "content": "\n".join(current_chunk_assembled_content),
                    "metadata": chunk_meta
                })
            
            # Start new chunk
            current_chunk_assembled_content = []
            current_length = 0
            
            # Try to prepend header to the new chunk if it fits with the current row
            if header_lines_content:
                if header_block_len + row_len_with_newline <= max_chars:
                    current_chunk_assembled_content.extend(header_lines_content)
                    current_length += header_block_len
            
            # Add current row to the new chunk (or what's left of it if header took space)
            current_chunk_assembled_content.append(row_text_content)
            current_length += row_len_with_newline # This might make current_length > max_chars if row_text_content is huge
            
            # Handle rows larger than max_chars (even after attempting to prepend header)
            if current_length > max_chars and len(current_chunk_assembled_content) == (len(header_lines_content) + 1 if header_lines_content and current_chunk_assembled_content[:len(header_lines_content)] == header_lines_content else 1) :
                # This means the row itself (with or without header) is too large.
                # Pop the row and its header (if added), then split the row.
                
                large_row_content = current_chunk_assembled_content.pop() # Removes the data row
                current_length -= (len(large_row_content) +1)
                
                header_for_split_row_parts = []
                if current_chunk_assembled_content and current_chunk_assembled_content == header_lines_content: # Header was indeed prepended
                    header_for_split_row_parts = current_chunk_assembled_content # Keep header for parts
                    current_chunk_assembled_content = [] # Clear it from being a standalone chunk
                    current_length = 0
                elif not current_chunk_assembled_content and header_lines_content: # No, header was not prepended because it didn't fit with row
                     pass # header_for_split_row_parts remains empty

                logger.warning(f"Table row too large ({len(large_row_content)} chars), splitting row")
                log_error("OVERSIZED_TABLE_ROW", f"Table row exceeds max_chars ({len(large_row_content)} > {max_chars})")
                
                remaining_row_part_text = large_row_content
                is_first_part_of_row = True
                while remaining_row_part_text:
                    # Determine available space for this part of the row
                    # If it's the first part, and we have a header for split parts, account for its length
                    effective_max_chars_for_part = max_chars
                    prefix_for_this_part = ""

                    if is_first_part_of_row and header_for_split_row_parts:
                        header_text_for_part = "\n".join(header_for_split_row_parts) + "\n"
                        if len(header_text_for_part) < max_chars:
                             effective_max_chars_for_part -= len(header_text_for_part)
                             prefix_for_this_part = header_text_for_part
                    
                    split_point = min(len(remaining_row_part_text), effective_max_chars_for_part)
                    if split_point <= 0: # No space left, probably very small max_chars or huge header
                        split_point = min(len(remaining_row_part_text), max_chars) # Try to take at least something
                        prefix_for_this_part = "" # Cannot afford prefix

                    chunk_part_final_content = prefix_for_this_part + remaining_row_part_text[:split_point]
                    
                    split_row_meta = base_metadata.copy()
                    split_row_meta["split_reason"] = "row_too_large"
                    split_row_meta["is_partial_row"] = True
                    split_row_meta["has_continuation"] = len(remaining_row_part_text[split_point:]) > 0
                    if is_first_part_of_row and header_for_split_row_parts:
                         split_row_meta["has_header_context"] = True

                    chunks.append({
                        "content": chunk_part_final_content,
                        "metadata": split_row_meta
                    })
                    remaining_row_part_text = remaining_row_part_text[split_point:]
                    is_first_part_of_row = False
                
                # After splitting a large row, current_chunk_assembled_content should be empty
                current_chunk_assembled_content = []
                current_length = 0

    # Add final chunk
    if current_chunk_assembled_content:
        chunk_meta = base_metadata.copy()
        chunk_meta["has_continuation"] = False # Last chunk
        
        # Avoid adding a chunk that is ONLY the header if the previous chunk already contained data rows and thus the header.
        is_only_header = False
        if header_lines_content:
            if len(current_chunk_assembled_content) == len(header_lines_content):
                is_only_header = all(current_chunk_assembled_content[k] == header_lines_content[k] for k in range(len(header_lines_content)))
        
        if not (is_only_header and chunks and chunks[-1]["metadata"].get("has_header")):
            chunks.append({
                "content": "\n".join(current_chunk_assembled_content),
                "metadata": chunk_meta
            })
    
    # Fallback if no chunks were created for non-empty text
    if not chunks and text.strip():
        fallback_meta = base_metadata.copy()
        fallback_meta["split_reason"] = "no_split_fallback"
        # Add the whole text as one chunk, trying to preserve original newlines for fallback
        chunks = [{"content": text.strip(), "metadata": fallback_meta}]
        logger.warning("Original table text was not empty but chunking resulted in no chunks. Adding original text as single chunk.")
        log_fallback("table_chunking", "single_chunk_fallback", "Table chunking produced no valid chunks")

    # Add context metadata for split boundaries (linking chunks)
    for idx, chunk_dict in enumerate(chunks):
        if idx > 0 and chunks[idx-1]["content"]: # Not the first chunk and previous chunk has content
            chunk_dict["metadata"]["previous_chunk_ends_with"] = chunks[idx-1]["content"].split('\n')[-1][:30] # Preview of previous
        if idx < len(chunks) - 1 and chunks[idx+1]["content"]: # Not the last chunk and next chunk has content
            chunk_dict["metadata"]["next_chunk_starts_with"] = chunks[idx+1]["content"].split('\n')[0][:30] # Preview of next

    return chunks

def chunk_list(text: str, max_chars: int) -> List[Dict[str, Any]]:
    """
    Specialized chunking for lists that maintains list item grouping,
    preserves hierarchical structures, and includes metadata. Enhanced for
    nested lists, definition lists, and context preservation.

    Args:
        text: List text to chunk
        max_chars: Maximum character length for chunks

    Returns:
        List of list chunks with metadata.
    """
    logger.info(f"Chunking list of length {len(text)} with max_chars={max_chars}")
    chunks = []
    base_metadata = {"content_type": "LIST"}

    lines = text.splitlines(keepends=True) # Keep newlines for accurate length and reconstruction
    if not lines:
        if text.strip(): # Handle case where text is non-empty but has no newlines
            return [{"content": text, "metadata": {**base_metadata, "split_reason": "no_split"}}]
        return []

    # Hierarchical parsing of list items
    # Each item in parsed_items will be a dictionary:
    # {"content": str, "indent": int, "type": "bullet/number/definition", "level": int, "original_lines": list_of_strings}
    parsed_items = []
    current_item_lines = []
    current_item_indent = -1
    current_item_level = 0 # For tracking nesting
    indent_stack = [] # To manage changes in indentation for nesting levels

    for line_idx, line_content_with_nl in enumerate(lines):
        line_content = line_content_with_nl.rstrip('\n') # Work with content without NL for regex, add NL back for length
        stripped_line = line_content.lstrip()
        leading_whitespace_len = len(line_content) - len(stripped_line)

        # Regex for list item starts (more specific)
        bullet_match = re.match(r'^[-*+•◦▪▫]\s+', stripped_line)
        number_match = re.match(r'^(?:\d+\.|\d+\)|\(\d+\)|[a-zA-Z]\.|\([a-zA-Z]\))\s+', stripped_line)
        definition_term_match = re.match(r'^([^:\n]+):\s+', stripped_line) # Term part of definition list

        is_new_item_start = False
        item_type = None

        if bullet_match:
            is_new_item_start = True
            item_type = "bullet"
        elif number_match:
            is_new_item_start = True
            item_type = "number"
        elif definition_term_match:
            is_new_item_start = True
            item_type = "definition_term"
        
        # Logic to finalize previous item and start a new one
        if is_new_item_start:
            if current_item_lines: # Finalize the previous item
                # Update the content of the item being finalized before adding it
                if parsed_items: # Ensure parsed_items is not empty
                    parsed_items[-1]["content"] = "".join(current_item_lines)
                    # Type was already set when item started, or updated if it was definition_description
                # This direct append is removed as items are added when they start
                # and their content is built up.
            
            # Start new item
            current_item_lines = [line_content_with_nl]
            current_item_indent = leading_whitespace_len
            
            # Update nesting level based on indent_stack
            if not indent_stack or leading_whitespace_len > indent_stack[-1]:
                indent_stack.append(leading_whitespace_len)
                current_item_level = len(indent_stack) -1
            else:
                while indent_stack and leading_whitespace_len < indent_stack[-1]:
                    indent_stack.pop()
                # After popping, if stack is not empty and current indent matches top, it's same level
                if indent_stack and leading_whitespace_len == indent_stack[-1]:
                     current_item_level = len(indent_stack) -1
                # If stack is empty, or current indent is less than new top (should not happen if popped correctly)
                # or current indent is greater (new sublevel, handled by first condition)
                # This means we are back to a previous level or base.
                elif not indent_stack : # Back to base level
                    indent_stack = [leading_whitespace_len] # Reset stack with current indent
                    current_item_level = 0
                else: # indent_stack not empty, but leading_whitespace_len != indent_stack[-1]
                      # This case implies an inconsistent indentation not matching previous levels.
                      # Default to current stack depth or treat as new base if very different.
                      # For simplicity, treat as a new level at current stack depth or new base.
                      if leading_whitespace_len > indent_stack[-1]: # Should be caught by first if
                           indent_stack.append(leading_whitespace_len)
                           current_item_level = len(indent_stack) - 1
                      else: # Indentation is less than stack top but not matching any previous after pop
                            # This is tricky, could be misformatted. Treat as a new base for this item.
                            indent_stack = [leading_whitespace_len]
                            current_item_level = 0


            # Add the new item being started
            parsed_items.append({
                "content": line_content_with_nl, # Start with the first line
                "indent": current_item_indent,
                "level": current_item_level,
                "type": item_type,
                "original_lines": [line_content_with_nl]
            })

        elif current_item_lines: # Continuation of an existing item
            # Add line if it's blank (part of item), or indented further or same (part of item body or nested)
            # For definition lists, the description part might not have a marker but belongs to the term.
            is_definition_description = parsed_items and parsed_items[-1]["type"] == "definition_term" and \
                                       leading_whitespace_len >= parsed_items[-1]["indent"] +1 # Description typically indented

            if not stripped_line or leading_whitespace_len >= current_item_indent or is_definition_description:
                current_item_lines.append(line_content_with_nl)
                if parsed_items: # Add to the content of the last started item
                    parsed_items[-1]["original_lines"].append(line_content_with_nl)
                    # Content will be fully joined later, or upon finalization of this item.
                    # For now, parsed_items[-1]["content"] is accumulating.
                    # No, better to update content directly:
                    parsed_items[-1]["content"] += line_content_with_nl
                    if is_definition_description and parsed_items[-1]["type"] == "definition_term":
                        parsed_items[-1]["type"] = "definition_item" # Mark as complete definition item

            else: # Line is less indented and not a new item start, signifies end of current list block
                if current_item_lines and parsed_items: # Finalize the current item
                     parsed_items[-1]["content"] = "".join(current_item_lines) # Ensure its content is set
                current_item_lines = [] # Reset for non-list content
                current_item_indent = -1
                indent_stack = []
                current_item_level = 0
        
        elif not is_new_item_start and not current_item_lines and stripped_line:
            # Non-list line outside of any active list item processing.
            # This specialized chunker should ideally only receive list content.
            # If it gets mixed, these lines are effectively ignored by this function.
            pass


    # Finalize the very last item if it was being built up
    if current_item_lines and parsed_items: # Ensure parsed_items is not empty
         # The content of the last item is already being accumulated in parsed_items[-1]["content"]
         # No explicit update needed here unless current_item_lines holds something not yet added.
         # However, the logic above tries to add to parsed_items[-1]["content"] directly.
         # For safety, ensure the collected lines are fully represented:
         parsed_items[-1]["content"] = "".join(parsed_items[-1]["original_lines"])


    # Filter out any empty items that might have been added as placeholders or had no content
    parsed_items = [item for item in parsed_items if item["content"].strip()]

    if not parsed_items and text.strip():
        logger.warning("Enhanced list item parsing failed to identify items, using fallback line grouping")
        log_fallback("list_item_detection", "line_grouping", "Failed to identify list items using enhanced parsing")
        temp_items_content = []
        current_fallback_item_lines = []
        for line_text_with_nl in lines:
            line_text_stripped = line_text_with_nl.strip()
            # Use broader detection for fallback
            if LIST_BULLET_DETECTION_PATTERN.match(line_text_stripped) or \
               LIST_NUMBER_DETECTION_PATTERN.match(line_text_stripped) or \
               LIST_DEFINITION_DETECTION_PATTERN.match(line_text_stripped): # Check for definition start
                if current_fallback_item_lines:
                    temp_items_content.append("".join(current_fallback_item_lines))
                current_fallback_item_lines = [line_text_with_nl]
            elif current_fallback_item_lines: # Continuation of a fallback item
                current_fallback_item_lines.append(line_text_with_nl)
            elif line_text_stripped: # A non-list, non-empty line starting a new "item"
                if current_fallback_item_lines: temp_items_content.append("".join(current_fallback_item_lines)) # Finalize previous
                current_fallback_item_lines = [line_text_with_nl]
        if current_fallback_item_lines: # Add last collected item
            temp_items_content.append("".join(current_fallback_item_lines))
        
        parsed_items = [{"content": content, "indent": 0, "level": 0, "type": "unknown_fallback", "original_lines": content.splitlines(keepends=True)} for content in temp_items_content if content.strip()]


    # Group parsed items into chunks
    current_chunk_assembled_content_strings = []
    current_chunk_length = 0
    current_chunk_base_metadata = {} # To store metadata for the current chunk being built

    for item_idx, item_data in enumerate(parsed_items):
        item_content_str = item_data["content"]
        item_length = len(item_content_str)

        item_specific_metadata = {
            "list_item_type": item_data["type"],
            "indent_level": item_data["indent"],
            "nesting_level": item_data["level"]
        }

        if not current_chunk_assembled_content_strings: # Starting a new chunk
            current_chunk_base_metadata = {**base_metadata, **item_specific_metadata, "split_reason": "list_item_group"}
        
        if current_chunk_length + item_length <= max_chars:
            current_chunk_assembled_content_strings.append(item_content_str)
            current_chunk_length += item_length
            # Update chunk metadata, e.g., max nesting level seen in this chunk
            current_chunk_base_metadata["nesting_level"] = max(current_chunk_base_metadata.get("nesting_level",0), item_data["level"])
            current_chunk_base_metadata["contains_mixed_item_types"] = current_chunk_base_metadata.get("contains_mixed_item_types", False) or \
                (len(current_chunk_assembled_content_strings) > 1 and item_data["type"] != current_chunk_base_metadata.get("list_item_type"))


        else: # Item does not fit, finalize previous chunk and start new one
            if current_chunk_assembled_content_strings:
                current_chunk_base_metadata["has_continuation"] = True # This chunk is not the last one if there's more items
                chunks.append({
                    "content": "".join(current_chunk_assembled_content_strings),
                    "metadata": current_chunk_base_metadata
                })
            
            # Start new chunk with the current item
            current_chunk_assembled_content_strings = [item_content_str]
            current_chunk_length = item_length
            current_chunk_base_metadata = {**base_metadata, **item_specific_metadata, "split_reason": "list_item_group"}

            # If the item itself is too large, split it by its original lines
            if item_length > max_chars:
                logger.warning(f"List item too large ({item_length}), splitting by its internal lines")
                log_error("OVERSIZED_LIST_ITEM", f"List item exceeds max_chars ({item_length} > {max_chars})")
                # The large item chunk was started above, pop it if it was added to current_chunk_assembled_content_strings
                if current_chunk_assembled_content_strings and current_chunk_assembled_content_strings[0] == item_content_str:
                    current_chunk_assembled_content_strings = []
                    current_chunk_length = 0
                
                # If it was already added to chunks (e.g. if it was the *only* item in the previous chunk attempt)
                # this is harder to detect. Assume it wasn't finalized yet.

                item_original_lines = item_data["original_lines"]
                sub_item_lines_for_chunk = []
                sub_item_length_for_chunk = 0

                for line_in_item_str_idx, line_in_item_str in enumerate(item_original_lines):
                    line_in_item_len = len(line_in_item_str)
                    if sub_item_length_for_chunk + line_in_item_len <= max_chars:
                        sub_item_lines_for_chunk.append(line_in_item_str)
                        sub_item_length_for_chunk += line_in_item_len
                    else:
                        if sub_item_lines_for_chunk: # Finalize this part of the large item
                            split_item_meta = {**base_metadata, **item_specific_metadata,
                                               "split_reason": "line_split_in_list_item",
                                               "is_partial_item": True,
                                               "has_continuation": True} # More lines to come for this item
                            chunks.append({
                                "content": "".join(sub_item_lines_for_chunk),
                                "metadata": split_item_meta
                            })
                        sub_item_lines_for_chunk = [line_in_item_str] # Start new part with current line
                        sub_item_length_for_chunk = line_in_item_len
                        
                        # If a single line of a list item is too long, split it by characters
                        if line_in_item_len > max_chars:
                            sub_item_lines_for_chunk.pop() # Remove the too-long line
                            sub_item_length_for_chunk = 0
                            
                            char_split_remaining = line_in_item_str
                            while char_split_remaining:
                                char_part = char_split_remaining[:max_chars]
                                char_split_meta = {**base_metadata, **item_specific_metadata,
                                                   "split_reason": "char_split_in_list_item_line",
                                                   "is_partial_item": True, "is_partial_line": True}
                                char_split_meta["has_continuation"] = len(char_split_remaining[max_chars:]) > 0 or \
                                                                     line_in_item_str_idx < len(item_original_lines) -1
                                chunks.append({"content": char_part, "metadata": char_split_meta})
                                char_split_remaining = char_split_remaining[max_chars:]
                
                if sub_item_lines_for_chunk: # Add the last part of the split large item
                    split_item_meta = {**base_metadata, **item_specific_metadata,
                                       "split_reason": "line_split_in_list_item",
                                       "is_partial_item": True,
                                       "has_continuation": False} # End of this specific item
                    chunks.append({
                        "content": "".join(sub_item_lines_for_chunk),
                        "metadata": split_item_meta
                    })
                current_chunk_base_metadata = {} # Reset, new chunk will start with next item or be empty
                current_chunk_assembled_content_strings = [] # Ensure it's reset
                current_chunk_length = 0


    # Add the final assembled chunk
    if current_chunk_assembled_content_strings:
        current_chunk_base_metadata["has_continuation"] = False # This is the last chunk from the list
        chunks.append({
            "content": "".join(current_chunk_assembled_content_strings),
            "metadata": current_chunk_base_metadata
        })

    if not chunks and text.strip():
        logger.warning("List chunking resulted in no chunks despite input. Adding original text as single fallback chunk.")
        log_fallback("list_chunking", "single_chunk_fallback", "List chunking produced no valid chunks")
        chunks = [{"content": text, "metadata": {**base_metadata, "split_reason": "no_split_final_fallback"}}]
    
    # Add context metadata for split boundaries (linking chunks)
    for idx, chunk_dict in enumerate(chunks):
        # Ensure metadata exists
        if "metadata" not in chunk_dict: chunk_dict["metadata"] = {}
        
        if idx > 0 and chunks[idx-1]["content"]:
            prev_meta = chunks[idx-1].get("metadata", {})
            chunk_dict["metadata"]["previous_chunk_item_type"] = prev_meta.get("list_item_type", prev_meta.get("type", "unknown"))
            chunk_dict["metadata"]["previous_chunk_nesting"] = prev_meta.get("nesting_level", 0)
        if idx < len(chunks) - 1 and chunks[idx+1]["content"]:
            next_meta = chunks[idx+1].get("metadata", {})
            chunk_dict["metadata"]["next_chunk_item_type"] = next_meta.get("list_item_type", next_meta.get("type", "unknown"))
            chunk_dict["metadata"]["next_chunk_nesting"] = next_meta.get("nesting_level", 0)

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

def adaptive_chunk_text(text: str, max_chars: int = 500) -> List[Dict[str, Any]]:
    """
    Adaptively chunk text based on content type detection, adding metadata to preserve context.
    
    Args:
        text: Text to chunk
        max_chars: Base maximum character length (will be adjusted)
        
    Returns:
        List of chunks with metadata, adapted to content type
    """
    # Reset error tracking for fresh start
    reset_error_context()
    
    logger.info(f"Starting adaptive chunking with base size {max_chars}")
    
    if not text or not text.strip():
        logger.info("Empty text provided to adaptive_chunk_text, returning empty list")
        return []
    
    try:
        # Detect content types
        content_types = detect_content_type(text)
        logger.info(f"Detected content types: {content_types}")
        
        # Determine optimal chunk size for this content
        adapted_size = get_optimal_chunk_size(content_types, max_chars)
        logger.info(f"Adapted chunk size: {adapted_size} (base: {max_chars})")
    except Exception as e:
        logger.error(f"Error in content type detection or size adaptation: {e}")
        log_error("CONTENT_DETECTION_FAILED", "Content detection or size adaptation failed", e)
        # Use base size as fallback
        adapted_size = max_chars
        content_types = {"PROSE": 1.0}  # Default to prose
        log_fallback("content_detection", "default_prose", "Exception in content detection")
    
    try:
        # Check for mixed/structured content
        is_mixed_content = content_types.get("STRUCTURED", 0) > 0.5
        
        # If we have mixed content, consider splitting by content sections first
        if is_mixed_content and len(text) > 1000:
            logger.info("Detected mixed content structure, using section-aware chunking")
            try:
                return chunk_mixed_content(text, adapted_size, content_types)
            except Exception as e:
                logger.error(f"Error in mixed content chunking: {e}")
                log_error("MIXED_CONTENT_CHUNKING_FAILED", "Failed to chunk mixed content", e)
                log_fallback("mixed_content", "primary_type_detection", "Exception in mixed content chunking")
                # Continue to primary type detection
        
        # Identify primary content type for specialized chunking
        primary_type = max(content_types.items(), key=lambda x: x[1])[0] if content_types else "PROSE"
        logger.info(f"Primary content type: {primary_type} (confidence: {content_types.get(primary_type, 0):.2f})")
        
        chunks = None
        chunking_method = None
        
        # Hierarchical chunking strategy with clear fallbacks
        # Lower thresholds for specialized chunking to make better use of our improved detection
        if primary_type == "CODE" and content_types.get("CODE", 0) > 0.4:
            chunking_method = "code"
            logger.info("Attempting specialized code chunking")
            try:
                chunks = chunk_code(text, adapted_size)
                if chunks:
                    logger.info(f"Successfully chunked code into {len(chunks)} chunks")
                    return chunks
            except Exception as e:
                logger.error(f"Error in code chunking: {e}")
                log_error("CODE_CHUNKING_FAILED", "Failed to chunk code content", e)
                log_fallback("code_chunking", "next_specialized_chunker", "Exception in code chunking")
        
        if (primary_type == "TABLE" and content_types.get("TABLE", 0) > 0.4) or (chunks is None and chunking_method == "code"):
            chunking_method = "table"
            logger.info("Attempting specialized table chunking")
            try:
                chunks = chunk_table(text, adapted_size)
                if chunks:
                    logger.info(f"Successfully chunked table into {len(chunks)} chunks")
                    return chunks
            except Exception as e:
                logger.error(f"Error in table chunking: {e}")
                log_error("TABLE_CHUNKING_FAILED", "Failed to chunk table content", e)
                log_fallback("table_chunking", "next_specialized_chunker", "Exception in table chunking")
        
        if (primary_type == "LIST" and content_types.get("LIST", 0) > 0.4) or (chunks is None and chunking_method in ["code", "table"]):
            chunking_method = "list"
            logger.info("Attempting specialized list chunking")
            try:
                chunks = chunk_list(text, adapted_size)
                if chunks:
                    logger.info(f"Successfully chunked list into {len(chunks)} chunks")
                    return chunks
            except Exception as e:
                logger.error(f"Error in list chunking: {e}")
                log_error("LIST_CHUNKING_FAILED", "Failed to chunk list content", e)
                log_fallback("list_chunking", "technical_or_prose", "Exception in list chunking")
        
        # If we get here, either specialized chunking was not applicable or it failed
        
        # For technical content with high confidence, use smaller chunks
        if primary_type == "TECHNICAL" and content_types.get("TECHNICAL", 0) > 0.5:
            logger.info("Using technical content chunking (smaller prose chunks)")
            try:
                tech_size = min(adapted_size, 400)  # Further reduce size for technical content
                prose_chunks = chunk_prose(text, tech_size)
                # Convert prose chunks to metadata format
                result = [{"content": c, "metadata": {"content_type": "PROSE", "technical": True}} for c in prose_chunks]
                logger.info(f"Successfully chunked technical content into {len(result)} chunks")
                return result
            except Exception as e:
                logger.error(f"Error in technical content chunking: {e}")
                log_error("TECHNICAL_CHUNKING_FAILED", "Failed to chunk technical content", e)
                log_fallback("technical_chunking", "standard_prose", "Exception in technical content chunking")
                # Continue to prose chunking
        
        # For all other types or as fallback, use prose chunking with adapted size
        logger.info("Using prose chunking with adapted size")
        try:
            prose_chunks = chunk_prose(text, adapted_size)
            # Convert prose chunks to metadata format
            result = [{"content": c, "metadata": {
                "content_type": "PROSE",
                "fallback_info": ERROR_CONTEXT["fallback_path"] if ERROR_CONTEXT["fallback_path"] else None
            }} for c in prose_chunks]
            logger.info(f"Successfully chunked prose into {len(result)} chunks")
            return result
        except Exception as e:
            logger.error(f"Error in prose chunking: {e}")
            log_error("PROSE_CHUNKING_FAILED", "Failed to chunk prose content", e)
            log_fallback("prose_chunking", "character_splitting", "Exception in prose chunking")
            
            # Ultimate fallback: simple character splitting
            logger.warning("All specialized chunking methods failed, using simple character splitting")
            try:
                # Very basic character splitting as last resort
                simple_chunks = []
                for i in range(0, len(text), adapted_size):
                    chunk = text[i:i + adapted_size]
                    if chunk.strip():
                        simple_chunks.append({
                            "content": chunk,
                            "metadata": {
                                "content_type": "UNKNOWN",
                                "chunking_method": "character_fallback",
                                "error_context": ERROR_CONTEXT
                            }
                        })
                logger.info(f"Created {len(simple_chunks)} chunks using character fallback")
                return simple_chunks
            except Exception as final_e:
                logger.error(f"Even character splitting failed: {final_e}")
                log_error("CHARACTER_SPLITTING_FAILED", "Failed to perform character splitting", final_e)
                # Return a single chunk with the whole text as absolute last resort
                return [{"content": text, "metadata": {
                    "content_type": "UNKNOWN",
                    "chunking_method": "full_text_fallback",
                    "error_context": ERROR_CONTEXT
                }}]
    
    except Exception as outer_e:
        # Catch-all for any unhandled exceptions in the chunking process
        logger.error(f"Unhandled exception in adaptive chunking: {outer_e}")
        log_error("UNHANDLED_CHUNKING_ERROR", "Unhandled exception in adaptive chunking", outer_e)
        
        # Return a single chunk with the whole text as absolute last resort
        return [{"content": text, "metadata": {
            "content_type": "UNKNOWN",
            "chunking_method": "exception_fallback",
            "error": str(outer_e),
            "error_context": ERROR_CONTEXT
        }}]


def chunk_mixed_content(text: str, max_chars: int, content_types: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Specialized chunking for mixed content that tries to identify and separate different content sections,
    adding metadata to chunks.

    Args:
        text: Text to chunk
        max_chars: Maximum character length for chunks
        content_types: Detected content types

    Returns:
        List of chunks with metadata.
    """
    logger.info(f"Chunking mixed content of length {len(text)} with max_chars={max_chars}")
    chunks: List[Dict[str, Any]] = []

    # Use regex patterns to identify content type boundaries
    code_sections = []
    table_sections = []
    list_sections = []

    # Find code blocks
    for match in CODE_BLOCK_PATTERN.finditer(text):
        code_sections.append((match.start(), match.end(), "CODE"))

    # Find potential code regions with high density of code markers
    # Iterate through all code patterns
    for lang, pattern in CODE_PATTERNS.items():
         for match in pattern.finditer(text):
             # Look for clusters of code patterns around the match
             section_start = max(0, match.start() - 100) # Look back 100 characters
             section_end = min(len(text), match.end() + 100) # Look forward 100 characters

             # Check if this section has high code pattern density
             section = text[section_start:section_end]
             # Count matches of any code pattern within this small section
             code_pattern_count = sum(len(p.findall(section)) for p in CODE_PATTERNS.values()) + len(CODE_BRACKETS_PATTERN.findall(section))

             if code_pattern_count >= 2: # Threshold of 2 code pattern matches in proximity
                 code_sections.append((section_start, section_end, "CODE"))

    # Find table sections
    for pattern in [TABLE_MARKDOWN_PATTERN, TABLE_ASCII_PATTERN, TABLE_GRID_PATTERN, TABLE_CSV_PATTERN]:
        for match in pattern.finditer(text):
            # For structural patterns, the match usually captures the whole unit
            table_sections.append((match.start(), match.end(), "TABLE"))

    # Find list sections
    for pattern in [LIST_BULLET_PATTERN, LIST_NUMBER_PATTERN, LIST_DEFINITION_PATTERN, LIST_INDENT_PATTERN]:
        for match in pattern.finditer(text):
            # For structural patterns, the match usually captures the whole unit
            list_sections.append((match.start(), match.end(), "LIST"))

    # Combine and sort all content sections
    all_sections = code_sections + table_sections + list_sections

    # Merge overlapping sections
    if all_sections:
        all_sections.sort(key=lambda x: x[0])
        merged_sections = [all_sections[0]]

        for current in all_sections[1:]:
            prev = merged_sections[-1]

            # If current section overlaps with previous one
            if current[0] <= prev[1]:
                # Merge by taking the union and keeping the type with higher confidence score in the document
                if content_types.get(current[2], 0) > content_types.get(prev[2], 0):
                     merged_sections[-1] = (prev[0], max(prev[1], current[1]), current[2])
                else:
                     merged_sections[-1] = (prev[0], max(prev[1], current[1]), prev[2])
            else:
                merged_sections.append(current)

        all_sections = merged_sections

    # Process document by sections
    last_end = 0
    for start, end, section_type in all_sections:
        # Process text before this section as prose
        if start > last_end:
            prose_text = text[last_end:start]
            if prose_text.strip():
                # chunk_prose returns list of strings, need to add metadata
                prose_chunks = chunk_prose(prose_text, max_chars)
                for chunk_content in prose_chunks:
                    chunks.append({"content": chunk_content, "metadata": {"content_type": "PROSE"}})

        # Process this section with specialized chunker
        section_text = text[start:end]
        if section_text.strip():
            # Specialized chunkers now return list of dicts with metadata
            if section_type == "CODE":
                chunks.extend(chunk_code(section_text, max_chars))
            elif section_type == "TABLE":
                chunks.extend(chunk_table(section_text, max_chars))
            elif section_type == "LIST":
                chunks.extend(chunk_list(section_text, max_chars))
        
        last_end = end

    # Process remaining text as prose
    if last_end < len(text):
        prose_text = text[last_end:]
        if prose_text.strip():
            # chunk_prose returns list of strings, need to add metadata
            prose_chunks = chunk_prose(prose_text, max_chars)
            for chunk_content in prose_chunks:
                chunks.append({"content": chunk_content, "metadata": {"content_type": "PROSE"}})

    # If no sections were detected or processing failed, fall back to prose chunking
    if not chunks and text.strip():
        logger.warning("Mixed content section detection yielded no chunks, falling back to prose chunking")
        log_fallback("mixed_content_detection", "prose_chunking", "No chunks produced by mixed content detection")
        # chunk_prose returns list of strings, need to add metadata
        prose_chunks = chunk_prose(text, max_chars)
        chunks = [{"content": chunk_content, "metadata": {"content_type": "PROSE"}} for chunk_content in prose_chunks]

    return chunks

if __name__ == "__main__":
    """Command line interface for testing adaptive chunking."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Adaptively chunk text based on content type")
    parser.add_argument("--input", "-i", help="Input file path", required=False)
    parser.add_argument("--output", "-o", help="Output file path for chunked results", required=False)
    parser.add_argument("--text", "-t", help="Text to process", required=False)
    parser.add_argument("--size", "-s", type=int, default=500, help="Base maximum chunk size (chars)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
         logger.setLevel(logging.DEBUG)

    # Get input text
    if args.input:
        try:
            with open(args.input, "r", encoding="utf-8") as f:
                text = f.read()
        except FileNotFoundError:
            print(f"Error: Input file not found at {args.input}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error reading input file: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.text:
        text = args.text
    else:
        print("Either --input or --text must be provided", file=sys.stderr)
        parser.print_help(sys.stderr)
        sys.exit(1)

    if not text.strip():
         print("Warning: Input text is empty or contains only whitespace.")
         sys.exit(0)

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
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                for i, chunk_data in enumerate(chunks):
                    f.write(f"--- CHUNK {i+1} (length: {len(chunk_data['content'])}) ---\n")
                    f.write(f"Metadata: {json.dumps(chunk_data.get('metadata', {}), indent=2)}\n")
                    f.write(chunk_data['content'])
                    f.write("\n\n")
            print(f"Chunks written to {args.output}")
        except Exception as e:
            print(f"Error writing output file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        for i, chunk_data in enumerate(chunks):
            print(f"\n--- CHUNK {i+1} (length: {len(chunk_data['content'])}) ---")
            print(f"Metadata: {json.dumps(chunk_data.get('metadata', {}), indent=2)}")
            print(chunk_data['content'][:200] + "..." if len(chunk_data['content']) > 200 else chunk_data['content'])