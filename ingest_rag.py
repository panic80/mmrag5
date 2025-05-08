#!/usr/bin/env python3
# -*- coding: cp1252 -*-

"""ingest_rag.py

CLI utility for building a Retrieval-Augmented Generation (RAG) vector
database with Qdrant.

The tool is intentionally *source-agnostic* and *schema-agnostic* � it relies
on the `docling` project to ingest documents from *any* kind of data source
(local files, URLs, databases, etc.). After the documents are loaded, their
content is embedded with OpenAI's `text-embedding-3-large` model and stored in
a Qdrant collection.

Example
-------
    $ OPENAI_API_KEY=... python ingest_rag.py --source ./my_corpus \
          --collection my_rag_collection

Requirements
------------
    pip install qdrant-client docling openai tqdm

Environment variables
---------------------
OPENAI_API_KEY
    Your OpenAI API key. It can also be passed explicitly with the
    ``--openai-api-key`` CLI option (the environment variable takes
    precedence).
QDRANT_API_KEY
    If you are using a managed Qdrant Cloud instance, set your API key here or
    use the ``--qdrant-api-key`` option.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
import traceback
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Dict, Any, Optional
import logging
from datetime import datetime

# Core dependencies
import click
from tqdm.auto import tqdm
import re
from dateutil.parser import parse as _parse_date

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize diagnostic tracking
INGESTION_DIAGNOSTICS = {
    "started_at": datetime.now().isoformat(),
    "errors": [],
    "fallbacks": [],
    "warnings": [],
    "chunks_created": 0,
    "chunking_methods_used": {}
}

# Regex to detect ISO dates (YYYY-MM-DD) in text
DATE_REGEX = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
# deterministic UUID generation does not require hashlib

# Global flags for chunking configuration
# ---------------------------------------------------------------------------
# Global configuration flags (toggled by CLI)
# ---------------------------------------------------------------------------

_use_fast_chunking: bool = True
_adaptive_chunking: bool = False

# Default set of ISO language codes that will be accepted by the language
# filter. The list is modified by the CLI option ``--languages``.
_allowed_languages: set[str] = {"en"}

# Minimum probability reported by the language detector for us to trust the
# detection result.
_lang_prob_threshold: float = 0.9

# Optional dependencies � import lazily so that the error message is clearer.


def _lazy_import(name: str):
    try:
        return __import__(name)
    except ImportError as exc:  # pragma: no cover � dev convenience
        click.echo(
            f"[fatal] The Python package '{name}' is required but not installed.\n"
            f"Install it with: pip install {name}",
            err=True,
        )
        raise SystemExit(1) from exc


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Document:
    """A minimal representation of a document to be embedded."""

    content: str
    metadata: dict[str, object] = field(default_factory=dict)

# ---------------------------------------------------------------------------
# API Error classes
# ---------------------------------------------------------------------------

class RateLimitError(Exception):
    """Rate limit error from API"""
    pass

class AuthenticationError(Exception):
    """Authentication error from API"""
    pass

class ServiceUnavailableError(Exception):
    """Service temporarily unavailable"""
    pass


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def extract_column_mapping(lines: list, separator_idx: int) -> dict:
    """
    Extract column-region mapping from multi-row headers in tables.
    
    This function analyzes table headers to create a mapping between columns and regions,
    which is essential for tables with multi-row headers (like geographical regions
    specified in the second header row).
    
    Args:
        lines: List of all table lines as strings
        separator_idx: Index of the separator line (with dashes/pipes)
        
    Returns:
        Dictionary mapping column indices to their regional context
    """
    # If no headers or only one header row, return empty mapping
    if separator_idx <= 0:
        return {}
    
    # Get header rows (excluding the separator line)
    header_rows = [lines[i] for i in range(separator_idx)]
    
    # Parse column positions from separator line
    separator = lines[separator_idx]
    column_positions = []
    in_column = False
    
    for i, char in enumerate(separator):
        if char == '|':
            if in_column:
                column_positions.append((start_pos, i))
                in_column = False
            else:
                start_pos = i
                in_column = True
    
    # Handle last column if needed
    if in_column:
        column_positions.append((start_pos, len(separator)))
    
    # Initialize mapping
    column_mapping = {}
    
    # If we have multiple header rows, extract region information
    if len(header_rows) >= 2:
        # Extract region labels from the second row (index 1)
        if len(header_rows) > 1:
            region_row = header_rows[1]
            # Extract labels from each column position
            for idx, (start, end) in enumerate(column_positions):
                if start < len(region_row) and end <= len(region_row):
                    # Extract region label, handling limited-width columns
                    label = region_row[start:end].strip(' |').strip()
                    if label:
                        column_mapping[idx] = label
    
    # If no regions found in second row but we have more header rows, try the first row
    if not column_mapping and header_rows:
        first_row = header_rows[0]
        for idx, (start, end) in enumerate(column_positions):
            if start < len(first_row) and end <= len(first_row):
                label = first_row[start:end].strip(' |').strip()
                if label:
                    column_mapping[idx] = label
    
    return column_mapping

def iter_batches(seq: Sequence[Document], batch_size: int) -> Iterable[List[Document]]:
    """Yield items *seq* in lists of length *batch_size* (the last one may be shorter)."""

    batch: list[Document] = []
    for item in seq:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split *text* into chunks of up to *max_chars* characters, breaking on whitespace."""
    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        # If not at end, try to break at last whitespace before end
        if end < length:
            split_at = text.rfind(' ', start, end)
            if split_at > start:
                end = split_at
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks

def _smart_chunk_text(text: str, max_chars: int, overlap: int = 0) -> list[str]:
    """
    # For now, use paragraph-based chunking
    return _smart_chunk_text(text, max_chars, overlap)
    Chunk text on paragraph and sentence boundaries up to max_chars,

    This is the original chunking method, kept as fallback if semantic chunking fails.
    and apply character-level overlap between chunks.

    This is the original chunking method, kept as fallback if semantic chunking fails.
    """
    # Split into paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    chunks: list[str] = []
    current_paras: list[str] = []
    current_len = 0
    for para in paragraphs:
        p = para.strip()
        if not p:
            continue
        # If paragraph itself is too long, split by sentences
        if len(p) > max_chars:
            # flush existing
            if current_paras:
                chunk = "\n\n".join(current_paras).strip()
                if chunk:
                    chunks.append(chunk)
                current_paras = []
                current_len = 0
            # split paragraph into sentences
            sentences = re.split(r'(?<=[\.!?])\s+', p)
            curr = ""
            for sent in sentences:
                s = sent.strip()
                if not s:
                    continue
                if len(curr) + len(s) <= max_chars:
                    curr = f"{curr} {s}".strip() if curr else s
                else:
                    if curr:
                        chunks.append(curr)
                    curr = s
            if curr:
                chunks.append(curr)
        else:
            # Add paragraph to current group
            if current_len + len(p) + 2 <= max_chars:
                current_paras.append(p)
                current_len += len(p) + 2
            else:
                # flush existing
                chunk = "\n\n".join(current_paras).strip()
                if chunk:
                    chunks.append(chunk)
                current_paras = [p]
                current_len = len(p) + 2
    # flush remainder
    if current_paras:
        chunk = "\n\n".join(current_paras).strip()
        if chunk:
            chunks.append(chunk)
    # ------------------------------------------------------------------
    # Apply overlap � prefer *token*-level if *tiktoken* is available, fall
    # back to character-level otherwise so behaviour degrades gracefully.
    # ------------------------------------------------------------------

    if overlap > 0 and len(chunks) > 1:
        try:
            import tiktoken  # type: ignore

            enc = tiktoken.get_encoding("cl100k_base")

            overlapped: list[str] = []
            overlapped.append(chunks[0])  # first chunk unchanged

            for idx in range(1, len(chunks)):
                prev_chunk = chunks[idx - 1]
                curr_chunk = chunks[idx]

                prev_tokens = enc.encode(prev_chunk)
                # guard for short previous chunk
                if not prev_tokens:
                    overlap_text = ""
                else:
                    overlap_tokens = prev_tokens[-overlap:]
                    overlap_text = enc.decode(overlap_tokens)

                overlapped.append(f"{overlap_text}{curr_chunk}")

            return overlapped
        except ImportError:
            # Fallback to simple character-level overlap if tiktoken unavailable
            pass

        overlapped: list[str] = []
        for idx, chunk in enumerate(chunks):
            if idx == 0:
                overlapped.append(chunk)
            else:
                prev = overlapped[idx - 1]
                ov = prev[-overlap:] if len(prev) >= overlap else prev
                overlapped.append(f"{ov} {chunk}")
        return overlapped

    return chunks


def semantic_chunk_text(text: str, max_chars: int, overlap: int = 0, fast_mode: bool = True, use_adaptive: bool = False) -> list[str]:
    """
    Chunk text based on semantic topic boundaries.

    Args:
        text: The text to chunk
        max_chars: Maximum character length per chunk
        overlap: Overlap between chunks (used for token-level overlap if applied)
        fast_mode: Use faster semantic chunking method (default: True)
        use_adaptive: Use adaptive content-aware chunking (default: False)

    Returns:
        List of semantically chunked text segments
    """
    # Set up structured chunking flow with proper fallbacks and diagnostic tracking
    # This tracks what chunking method we're using
    chunking_method = "adaptive" if (use_adaptive or _adaptive_chunking) else "semantic"
    chunks = None
    log_prefix = "[adaptive]" if chunking_method == "adaptive" else "[semantic]"
    
    # For tracking fallbacks and hierarchy
    fallback_path = []
    
    def log_chunking_attempt(method, status, details=None, exc=None):
        """Log chunking attempt to both console and diagnostics"""
        timestamp = datetime.now().isoformat()
        
        if status == "error":
            error_info = {
                "timestamp": timestamp,
                "method": method,
                "message": details,
                "exception": f"{exc.__class__.__name__}: {exc}" if exc else None,
                "traceback": traceback.format_exc() if exc else None
            }
            INGESTION_DIAGNOSTICS["errors"].append(error_info)
            logger.error(f"{log_prefix} {details}")
            click.echo(f"[error] {log_prefix} {details}", err=True)
            
        elif status == "fallback":
            from_method, to_method = method, details
            reason = str(exc) if exc else "Method failed to produce valid chunks"
            
            fallback_info = {
                "timestamp": timestamp,
                "from_method": from_method,
                "to_method": to_method,
                "reason": reason
            }
            
            fallback_path.append(fallback_info)
            INGESTION_DIAGNOSTICS["fallbacks"].append(fallback_info)
            
            logger.warning(f"{log_prefix} Falling back from {from_method} to {to_method}: {reason}")
            click.echo(f"[warning] {log_prefix} Falling back from {from_method} to {to_method}: {reason}", err=True)
            
        elif status == "success":
            # Track successful chunking method
            if method not in INGESTION_DIAGNOSTICS["chunking_methods_used"]:
                INGESTION_DIAGNOSTICS["chunking_methods_used"][method] = 0
            
            chunk_count = len(details) if isinstance(details, list) else 0
            INGESTION_DIAGNOSTICS["chunking_methods_used"][method] += 1
            INGESTION_DIAGNOSTICS["chunks_created"] += chunk_count
            
            logger.info(f"{log_prefix} Successfully used {method} chunking: created {chunk_count} chunks")
            click.echo(f"[success] {log_prefix} Successfully used {method} chunking: created {chunk_count} chunks")
    
    try:
        import time
        start_time = time.time()
        
        # 1. Try adaptive chunking if enabled
        if chunking_method == "adaptive":
            logger.info(f"{log_prefix} Attempting content-aware adaptive chunking")
            click.echo(f"[info] {log_prefix} Attempting content-aware adaptive chunking")
            
            try:
                # Import adaptive chunking module
                try:
                    from adaptive_chunking import adaptive_chunk_text, reset_error_context
                    # Reset error context to ensure clean state
                    if 'reset_error_context' in locals():
                        reset_error_context()
                    logger.info(f"{log_prefix} Successfully imported adaptive_chunking module")
                    click.echo(f"[info] {log_prefix} Successfully imported adaptive_chunking module")
                except ImportError as ie:
                    error_msg = f"Failed to import adaptive_chunking module: {ie}"
                    log_chunking_attempt("adaptive_chunking_import", "error", error_msg, ie)
                    log_chunking_attempt("adaptive_chunking", "fallback", "semantic_chunking", ie)
                    chunking_method = "semantic"
                    # Don't re-raise, just change the method and continue to semantic chunking
                
                # Only execute adaptive chunking if we haven't fallen back to semantic yet
                if chunking_method == "adaptive":
                    # Execute adaptive chunking
                    logger.info(f"{log_prefix} Executing adaptive chunking with max_chars={max_chars}")
                    chunks = adaptive_chunk_text(text, max_chars=max_chars)
                    
                    # Validate chunks
                    if not chunks:
                        error_msg = "Adaptive chunking returned empty result"
                        log_chunking_attempt("adaptive_chunking", "error", error_msg)
                        log_chunking_attempt("adaptive_chunking", "fallback", "semantic_chunking")
                        chunking_method = "semantic"
                    elif not all(isinstance(c, (str, dict)) for c in chunks):
                        invalid_types = set(type(c).__name__ for c in chunks if not isinstance(c, (str, dict)))
                        error_msg = f"Adaptive chunking returned invalid chunk types: {invalid_types}"
                        log_chunking_attempt("adaptive_chunking", "error", error_msg)
                        log_chunking_attempt("adaptive_chunking", "fallback", "semantic_chunking")
                        chunking_method = "semantic"
                    else:
                        # Success path for adaptive chunking
                        elapsed_time = time.time() - start_time
                        logger.info(f"{log_prefix} Adaptive chunking completed in {elapsed_time:.2f} seconds")
                        
                        # Convert dict chunks to strings if needed
                        if all(isinstance(c, dict) for c in chunks):
                            # Extract content from dictionary chunks
                            processed_chunks = [c["content"] for c in chunks if "content" in c and c["content"]]
                            
                            # Store metadata for diagnostics
                            for chunk in chunks:
                                if isinstance(chunk, dict) and "metadata" in chunk:
                                    # Store metadata for error diagnostics if present
                                    if "error_context" in chunk["metadata"]:
                                        error_ctx = chunk["metadata"]["error_context"]
                                        if error_ctx and "detected_errors" in error_ctx:
                                            for err_type, errors in error_ctx["detected_errors"].items():
                                                for err in errors:
                                                    INGESTION_DIAGNOSTICS["errors"].append({
                                                        "source": "adaptive_chunking",
                                                        "type": err_type,
                                                        "message": err.get("message", "Unknown error"),
                                                        "timestamp": err.get("timestamp")
                                                    })
                                    
                                    # Store fallback paths if present
                                    if "fallback_info" in chunk["metadata"]:
                                        fallback_info = chunk["metadata"]["fallback_info"]
                                        if fallback_info:
                                            INGESTION_DIAGNOSTICS["fallbacks"].extend(fallback_info)
                            
                            chunks = processed_chunks
                        
                        log_chunking_attempt("adaptive_chunking", "success", chunks)
                        return chunks
            except ValueError as ve:
                error_msg = f"Value error in adaptive chunking: {ve}"
                log_chunking_attempt("adaptive_chunking", "error", error_msg, ve)
                log_chunking_attempt("adaptive_chunking", "fallback", "semantic_chunking", ve)
                chunking_method = "semantic"
            except TypeError as te:
                error_msg = f"Type error in adaptive chunking: {te}"
                log_chunking_attempt("adaptive_chunking", "error", error_msg, te)
                log_chunking_attempt("adaptive_chunking", "fallback", "semantic_chunking", te)
                chunking_method = "semantic"
            except Exception as e:
                error_msg = f"Unexpected error in adaptive chunking: {e.__class__.__name__}: {e}"
                log_chunking_attempt("adaptive_chunking", "error", error_msg, e)
                log_chunking_attempt("adaptive_chunking", "fallback", "semantic_chunking", e)
                chunking_method = "semantic"
    
        # 2. Semantic chunking (either as primary method or fallback from adaptive)
        # Update log prefix in case we transitioned from adaptive to semantic
        log_prefix = "[adaptive]" if chunking_method == "adaptive" else "[semantic]"
        
        if chunking_method == "semantic":
            # Reset start_time for semantic chunking timing measurement
            start_time = time.time()
            
            semantic_approach = "transformer" if not fast_mode else "heuristic"
            logger.info(f"{log_prefix} Using {semantic_approach} approach for semantic chunking")
            click.echo(f"[info] {log_prefix} Using {semantic_approach} approach for semantic chunking")
            
            # 2a. Attempt to use transformer-based semantic chunking if fast_mode is False
            if semantic_approach == "transformer":
                transformer_start = time.time()
                try:
                    # Try to import langchain's text splitter that can do semantic splits
                    try:
                        from langchain.text_splitter import RecursiveCharacterTextSplitter
                        logger.info(f"{log_prefix} Successfully imported RecursiveCharacterTextSplitter")
                        click.echo(f"[info] {log_prefix} Successfully imported RecursiveCharacterTextSplitter")
                    except ImportError as ie:
                        error_msg = f"Failed to import RecursiveCharacterTextSplitter: {ie}"
                        log_chunking_attempt("transformer_import", "error", error_msg, ie)
                        log_chunking_attempt("transformer_chunking", "fallback", "heuristic_chunking", ie)
                        semantic_approach = "heuristic"
                        # Don't re-raise, just change the approach and continue
                
                    # Only execute transformer-based chunking if we haven't fallen back to heuristic
                    if semantic_approach == "transformer":
                        # Execute transformer-based semantic chunking
                        logger.info(f"{log_prefix} Executing transformer-based chunking with chunk_size={max_chars*4}, overlap={overlap*4}")
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=max_chars * 4,  # approximate char length
                            chunk_overlap=overlap * 4,
                            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                            length_function=len,
                        )
                        chunks = splitter.split_text(text)
                        
                        # Validate chunks
                        if not chunks:
                            error_msg = "Transformer-based chunking returned empty result"
                            log_chunking_attempt("transformer_chunking", "error", error_msg)
                            log_chunking_attempt("transformer_chunking", "fallback", "heuristic_chunking")
                            semantic_approach = "heuristic"
                        elif not all(isinstance(c, str) for c in chunks):
                            invalid_types = set(type(c).__name__ for c in chunks if not isinstance(c, str))
                            error_msg = f"Transformer-based chunking returned invalid types: {invalid_types}"
                            log_chunking_attempt("transformer_chunking", "error", error_msg)
                            log_chunking_attempt("transformer_chunking", "fallback", "heuristic_chunking")
                            semantic_approach = "heuristic"
                        else:
                            # Success path for transformer-based chunking
                            elapsed_time = time.time() - transformer_start
                            logger.info(f"{log_prefix} Transformer-based chunking completed in {elapsed_time:.2f} seconds")
                            log_chunking_attempt("transformer_chunking", "success", chunks)
                            
                            # Add diagnostic information to INGESTION_DIAGNOSTICS
                            INGESTION_DIAGNOSTICS["chunking_details"] = {
                                "method": "transformer_based",
                                "execution_time": elapsed_time,
                                "chunk_count": len(chunks),
                                "avg_chunk_size": sum(len(c) for c in chunks) / len(chunks) if chunks else 0,
                                "max_chunk_size": max(len(c) for c in chunks) if chunks else 0,
                                "min_chunk_size": min(len(c) for c in chunks) if chunks else 0
                            }
                            
                            return chunks
                except ValueError as ve:
                    error_msg = f"Value error in transformer-based chunking: {ve}"
                    log_chunking_attempt("transformer_chunking", "error", error_msg, ve)
                    log_chunking_attempt("transformer_chunking", "fallback", "heuristic_chunking", ve)
                    semantic_approach = "heuristic"
                except TypeError as te:
                    error_msg = f"Type error in transformer-based chunking: {te}"
                    log_chunking_attempt("transformer_chunking", "error", error_msg, te)
                    log_chunking_attempt("transformer_chunking", "fallback", "heuristic_chunking", te)
                    semantic_approach = "heuristic"
                except Exception as e:
                    error_msg = f"Unexpected error in transformer-based chunking: {e.__class__.__name__}: {e}"
                    log_chunking_attempt("transformer_chunking", "error", error_msg, e)
                    log_chunking_attempt("transformer_chunking", "fallback", "heuristic_chunking", e)
                    semantic_approach = "heuristic"
            
            # 2b. Heuristic-based semantic chunking with better boundary detection
            if semantic_approach == "heuristic":
                logger.info(f"{log_prefix} Using heuristic-based semantic chunking with boundary preservation")
                click.echo(f"[info] {log_prefix} Using heuristic-based semantic chunking with boundary preservation")
                heuristic_start = time.time()
            
                try:
                    # Split into potential semantic units (paragraphs, sections, etc.)
                    paragraphs = re.split(r'\n\s*\n', text)
                    chunks = []
                    current_chunk = []
                    current_len = 0
                
                    for para in paragraphs:
                        para = para.strip()
                        if not para:
                            continue
                            
                        para_len = len(para)
                        
                        # If this paragraph would exceed max_chars, process it separately
                        if current_len + para_len > max_chars and current_chunk:
                            chunks.append("\n\n".join(current_chunk))
                            current_chunk = []
                            current_len = 0
                        
                        # If the paragraph itself is too long for a single chunk
                        if para_len > max_chars:
                            # If we have accumulated text, add it as a chunk first
                            if current_chunk:
                                chunks.append("\n\n".join(current_chunk))
                                current_chunk = []
                                current_len = 0
                            
                            # Split the long paragraph into sentences
                            sentences = re.split(r'(?<=[.!?])\s+', para)
                            sent_chunk = []
                            sent_len = 0
                            
                            for sentence in sentences:
                                sentence = sentence.strip()
                                if not sentence:
                                    continue
                                    
                                sent_chunk_len = len(sentence)
                                
                                # If adding this sentence would exceed max_chars, flush the buffer
                                if sent_len + sent_chunk_len > max_chars and sent_chunk:
                                    chunks.append(" ".join(sent_chunk))
                                    sent_chunk = []
                                    sent_len = 0
                                
                                # If the sentence itself is too long, split it further
                                if sent_chunk_len > max_chars:
                                    # Flush any accumulated sentences first
                                    if sent_chunk:
                                        chunks.append(" ".join(sent_chunk))
                                        sent_chunk = []
                                        sent_len = 0
                                    
                                    # Split long sentence by clause boundaries or ultimately by words
                                    parts = re.split(r',\s+', sentence)
                                    clause_chunk = []
                                    clause_len = 0
                                    
                                    for part in parts:
                                        part = part.strip()
                                        if not part:
                                            continue
                                            
                                        part_len = len(part)
                                        
                                        if clause_len + part_len <= max_chars:
                                            clause_chunk.append(part)
                                            clause_len += part_len + 2  # account for delimiter
                                        else:
                                            if clause_chunk:
                                                chunks.append(", ".join(clause_chunk))
                                                clause_chunk = []
                                                clause_len = 0
                                            
                                            # If the clause itself is too long, split by words
                                            if part_len > max_chars:
                                                words = part.split()
                                                word_chunk = []
                                                word_len = 0
                                                
                                                for word in words:
                                                    word_len_with_space = len(word) + 1
                                                    if word_len + word_len_with_space <= max_chars:
                                                        word_chunk.append(word)
                                                        word_len += word_len_with_space
                                                    else:
                                                        if word_chunk:
                                                            chunks.append(" ".join(word_chunk))
                                                            word_chunk = []
                                                            word_len = 0
                                                        word_chunk.append(word)
                                                        word_len = len(word) + 1
                                                
                                                if word_chunk:
                                                    chunks.append(" ".join(word_chunk))
                                            else:
                                                clause_chunk.append(part)
                                                clause_len = part_len + 2
                                    
                                    if clause_chunk:
                                        chunks.append(", ".join(clause_chunk))
                                else:
                                    sent_chunk.append(sentence)
                                    sent_len += sent_chunk_len + 1  # +1 for space
                            
                            if sent_chunk:
                                chunks.append(" ".join(sent_chunk))
                        else:
                            # Normal paragraph that fits within limits
                            current_chunk.append(para)
                            current_len += para_len + 2  # +2 for paragraph breaks
                    
                    # Don't forget remaining text
                    if current_chunk:
                        chunks.append("\n\n".join(current_chunk))
                    
                    # Apply token-level overlap if specified and possible
                    if overlap > 0 and len(chunks) > 1:
                        chunks = _apply_overlap(chunks, overlap)
                        
                    if chunks:
                        elapsed_time = time.time() - heuristic_start
                        logger.info(f"{log_prefix} Heuristic-based chunking completed in {elapsed_time:.2f} seconds")
                        click.echo(f"[success] {log_prefix} Heuristic-based chunking produced {len(chunks)} chunks in {elapsed_time:.2f} seconds")
                        
                        # Add detailed diagnostics
                        INGESTION_DIAGNOSTICS["chunking_details"] = {
                            "method": "heuristic_based",
                            "execution_time": elapsed_time,
                            "chunk_count": len(chunks),
                            "avg_chunk_size": sum(len(c) for c in chunks) / len(chunks) if chunks else 0
                        }
                        
                        log_chunking_attempt("heuristic_chunking", "success", chunks)
                        return chunks
                    else:
                        logger.error(f"{log_prefix} Heuristic-based chunking returned no chunks")
                        error_msg = "Heuristic-based chunking returned no chunks"
                        log_chunking_attempt("heuristic_chunking", "error", error_msg)
                        log_chunking_attempt("heuristic_chunking", "fallback", "basic_chunking")
                        chunks = None
                        
                except Exception as heuristic_e:
                    error_msg = f"Error in heuristic chunking: {heuristic_e.__class__.__name__}: {heuristic_e}"
                    log_chunking_attempt("heuristic_chunking", "error", error_msg, heuristic_e)
                    log_chunking_attempt("heuristic_chunking", "fallback", "basic_chunking", heuristic_e)
                    chunks = None
            
    except Exception as outer_e:
        # Catch any unexpected errors at the top level
        error_msg = f"Critical error in chunking pipeline: {outer_e.__class__.__name__}: {outer_e}"
        log_chunking_attempt("chunking_pipeline", "error", error_msg, outer_e)
        log_chunking_attempt("all_chunking_methods", "fallback", "basic_chunking", outer_e)
        
        # Track detailed diagnostic info about the failure
        INGESTION_DIAGNOSTICS["critical_error"] = {
            "message": str(outer_e),
            "type": outer_e.__class__.__name__,
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat(),
            "fallback_path": fallback_path
        }
        
        # Last resort - basic chunking
        logger.warning("Falling back to basic chunking as last resort after critical error")
        click.echo("[warning] Falling back to basic chunking as last resort", err=True)
        
        try:
            basic_chunks = _smart_chunk_text(text, max_chars, overlap)
            log_chunking_attempt("basic_chunking", "success", basic_chunks)
            return basic_chunks
        except Exception as basic_e:
            # If even basic chunking fails, log the error and try character chunking
            error_msg = f"Basic chunking failed: {basic_e.__class__.__name__}: {basic_e}"
            log_chunking_attempt("basic_chunking", "error", error_msg, basic_e)
            log_chunking_attempt("basic_chunking", "fallback", "character_chunking", basic_e)
            
            # Very simple character chunking as absolute last resort
            try:
                char_chunks = []
                for i in range(0, len(text), max_chars):
                    chunk = text[i:i+max_chars].strip()
                    if chunk:
                        char_chunks.append(chunk)
                log_chunking_attempt("character_chunking", "success", char_chunks)
                return char_chunks
            except Exception as char_e:
                # If even character chunking fails, return the original text as one chunk
                error_msg = f"Character chunking failed: {char_e.__class__.__name__}: {char_e}"
                log_chunking_attempt("character_chunking", "error", error_msg, char_e)
                # Return text as single chunk to avoid completely failing
                return [text]
        
    # If we reach this point, all methods failed or weren't even attempted
    if chunks is None:
        logger.error("All chunking methods failed, returning basic chunking as last resort")
        click.echo("[error] All chunking methods failed, returning basic chunking as last resort", err=True)
        
        # Track the failure chain
        INGESTION_DIAGNOSTICS["complete_failure"] = {
            "timestamp": datetime.now().isoformat(),
            "fallback_path": fallback_path,
            "final_fallback": "basic_chunking"
        }
        
        try:
            basic_chunks = _smart_chunk_text(text, max_chars, overlap)
            log_chunking_attempt("basic_chunking_final", "success", basic_chunks)
            return basic_chunks
        except Exception as final_e:
            # Last defense - return original text as one chunk if all else fails
            error_msg = f"Even final basic chunking failed: {final_e.__class__.__name__}: {final_e}"
            log_chunking_attempt("basic_chunking_final", "error", error_msg, final_e)
            logger.error("ALL chunking methods failed! Returning text as single chunk to avoid complete failure")
            return [text]
    
    # Return chunks if we have them
    log_chunking_attempt("final_chunks", "success", chunks)
    return chunks

def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Apply token-level overlap between chunks."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        
        logger.info(f"Applying token-level overlap of {overlap} tokens between {len(chunks)} chunks")
        
        overlapped = [chunks[0]]  # First chunk unchanged
        
        for idx in range(1, len(chunks)):
            prev_chunk = chunks[idx - 1]
            curr_chunk = chunks[idx]
            
            prev_tokens = enc.encode(prev_chunk)
            # Guard for short previous chunk
            if not prev_tokens:
                overlap_text = ""
                logger.warning(f"Previous chunk {idx-1} has no tokens for overlap")
            else:
                overlap_tokens = prev_tokens[-min(overlap, len(prev_tokens)):]
                overlap_text = enc.decode(overlap_tokens)
                logger.debug(f"Added {len(overlap_tokens)} tokens of overlap between chunks {idx-1} and {idx}")
            
            overlapped.append(f"{overlap_text}{curr_chunk}")
        
        logger.info(f"Successfully applied token-level overlap to {len(chunks)} chunks")
        return overlapped
    except ImportError as ie:
        # Fallback to character-level overlap if tiktoken unavailable
        logger.warning(f"tiktoken not available ({str(ie)}), using character-level overlap")
        click.echo("[warning] tiktoken not available, using character-level overlap", err=True)
        
        # Track this fallback
        INGESTION_DIAGNOSTICS["fallbacks"].append({
            "timestamp": datetime.now().isoformat(),
            "from_method": "token_overlap",
            "to_method": "character_overlap",
            "reason": f"ImportError: {str(ie)}"
        })
        
        try:
            overlapped = [chunks[0]]  # First chunk unchanged
            for idx in range(1, len(chunks)):
                prev = chunks[idx - 1]
                ov = prev[-min(overlap*4, len(prev)):] if len(prev) >= overlap else prev
                overlapped.append(f"{ov} {chunks[idx]}")
            
            logger.info(f"Successfully applied character-level overlap to {len(chunks)} chunks")
            return overlapped
        except Exception as e:
            # If even character overlap fails, return the original chunks
            logger.error(f"Character-level overlap failed: {e}")
            logger.warning("Returning chunks without overlap due to error")
            
            # Track this error
            INGESTION_DIAGNOSTICS["errors"].append({
                "timestamp": datetime.now().isoformat(),
                "method": "character_overlap",
                "message": f"Character-level overlap failed: {str(e)}",
                "exception": f"{e.__class__.__name__}: {e}",
                "traceback": traceback.format_exc()
            })
            
            return chunks
# ---------------------------------------------------------------------------
# Language detection helper (optional � requires the *cld3* package)
# ---------------------------------------------------------------------------


def _detect_language(text: str) -> tuple[str | None, float]:
    """Return `(lang, probability)` for *text* using *cld3* if available.

    If *cld3* is missing or language cannot be determined, returns ``(None,
    0.0)`` so that callers can treat the result as *unknown*.
    """

    if len(text) < 20:  # too little signal for detection � treat as unknown
        return None, 0.0

    try:
        import cld3  # type: ignore

        res = cld3.get_language(text)
        if res is None:
            return None, 0.0

        return res.language, res.probability  # type: ignore[attr-defined]
    except ImportError:
        # cld3 not installed � silently degrade (handled by caller)
        return None, 0.0
    except Exception:  # pragma: no cover � unexpected detector failure
        return None, 0.0


def get_openai_client(api_key: str):
    """
    Get an OpenAI client instance with improved version detection.

    This function attempts to handle different versions of the OpenAI Python client:
    - v0.x (openai<1.0): Global module with openai.api_key and openai.Embedding.create
    - v1.x (openai>=1.0): Client instance with client.embeddings.create

    Args:
        api_key: OpenAI API key

    Returns:
        OpenAI client (either module or instance depending on version)
    """
    openai = _lazy_import("openai")

    # First, check if we're using v1.x by attempting to detect the OpenAI class
    # This is the most reliable method for detecting v1.x
    if hasattr(openai, "OpenAI"):
        try:
            click.echo("[info] Detected OpenAI Python SDK v1.x")
            client = openai.OpenAI(api_key=api_key)

            # Verify this is really a v1 client by checking for crucial methods
            if hasattr(client, "embeddings") and hasattr(client.embeddings, "create"):
                return client
        except Exception as e:
            click.echo(f"[warning] Failed to initialize OpenAI v1 client: {e}", err=True)

    # If we get here, either we're using v0.x or the v1 client creation failed
    # Try to set the API key on the module (v0.x style)
    try:
        click.echo("[info] Attempting to use OpenAI Python SDK v0.x")
        openai.api_key = api_key

        # Verify this is really a v0 client by checking for crucial methods
        if hasattr(openai, "Embedding") and hasattr(openai.Embedding, "create"):
            return openai
        else:
            click.echo("[warning] OpenAI client doesn't have expected v0.x methods", err=True)
    except AttributeError:
        click.echo("[warning] Unable to set api_key on OpenAI module", err=True)

    # As a last resort, try again with v1 but with base_url and organization=None
    try:
        click.echo("[info] Attempting alternate OpenAI SDK v1.x initialization")
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.openai.com/v1",
            organization=None,
        )
        return client
    except Exception as e:
        click.echo(f"[error] All methods of OpenAI client initialization failed: {e}", err=True)
        click.echo(f"[info] Returning potentially incomplete OpenAI client", err=True)

    # Return whatever we have, even if it might not work correctly
    return openai


# ---------------------------------------------------------------------------
# Main ingestion routine
# ---------------------------------------------------------------------------


def load_documents(source: str, chunk_size: int = 500, overlap: int = 50, crawl_depth: int = 0) -> List[Document]:
    """
    Use *docling* to load and chunk documents from *source*.

    The function tries to stay *schema-agnostic*. For every item docling
    yields, we keep its raw representation as the ``payload`` (metadata), and
    attempt to locate a reasonable textual representation for embedding.
    """
    # Early HTTP(S) URL fallback using BeautifulSoup
    if source.lower().startswith(("http://", "https://")):
        click.echo(f"[info] Fetching and chunking URL content: {source}")
        try:
            import requests
            from bs4 import BeautifulSoup

            resp = requests.get(source, timeout=120)
            resp.raise_for_status()
            html = resp.text
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(separator="\n")
            chunks = _smart_chunk_text(text, chunk_size, overlap)
            return [Document(content=chunk, metadata={"source": source, "chunk_index": idx}) for idx, chunk in enumerate(chunks)]
        except Exception as e_url:
            click.echo(f"[warning] URL extraction with BeautifulSoup failed: {e_url}", err=True)
    # ---------------------------------------------------------------------
    # Helper: chunk arbitrary raw *text* into smaller passages. We keep the
    # helper nested so it can implicitly capture the *chunk_size* / *overlap*
    # parameters from the enclosing ``load_documents`` call, but we define it
    # right at the beginning of the function so that every subsequent code
    # path can freely reference it.
    # ---------------------------------------------------------------------

    def _chunk_text_tokenwise(text: str, metadata: dict[str, object]) -> List[Document]:
        """Split *text* into token-aware chunks, falling back to character
        splitting if the preferred tokenisers are unavailable. A small helper
        that always returns a ``List[Document]`` with a ``chunk_index``
        injected into the copied *metadata* for downstream processing."""

        docs_out: list[Document] = []

        # Ensure global variables are properly initialized
        global _use_fast_chunking, _adaptive_chunking
        if not hasattr(sys.modules[__name__], '_use_fast_chunking'):
            _use_fast_chunking = True
        if not hasattr(sys.modules[__name__], '_adaptive_chunking'):
            _adaptive_chunking = False

        # 1. Try semantic chunking first (from advanced_rag)
        try:
            # Use semantic chunking with token-based size; fallback chunkers
            # may interpret the value as characters, but semantic chunkers
            # (sechunk, adaptive, etc.) honor token counts correctly.
            click.echo(f"[info] Attempting semantic chunking (adaptive={_adaptive_chunking}, fast_mode={_use_fast_chunking})")
            
            chunks = semantic_chunk_text(
                text,
                max_chars=chunk_size,
                overlap=overlap,
                fast_mode=_use_fast_chunking,
                use_adaptive=_adaptive_chunking,
            )
            
            # Verify we got valid chunks
            if chunks and all(isinstance(c, str) for c in chunks):
                method_str = "adaptive" if _adaptive_chunking else "semantic"
                mode_str = "fast" if _use_fast_chunking else "precise"
                click.echo(f"[info] Successfully used {mode_str} {method_str} chunking: {len(chunks)} chunks created")
                
                # Create documents from chunks
                for idx, chunk in enumerate(chunks):
                    if not chunk.strip():  # Skip empty chunks
                        continue
                    new_meta = dict(metadata)
                    new_meta["chunk_index"] = idx
                    new_meta["chunking_method"] = f"{method_str}_{mode_str}"
                    docs_out.append(Document(content=chunk, metadata=new_meta))
                
                return docs_out
            else:
                click.echo("[warning] Semantic chunking returned invalid chunks, trying fallbacks", err=True)
        except Exception as e:
            click.echo(f"[warning] Semantic chunking failed, trying fallbacks: {e}", err=True)
        
        # 2. Try docling's GPT-aware splitter
        try:
            from docling.text import TextSplitter
            click.echo("[info] Using docling TextSplitter for chunking")
            
            splitter = TextSplitter.from_model(  # type: ignore[attr-defined]
                model="gpt-4.1-mini",
                chunk_size=chunk_size,
                chunk_overlap=overlap,
            )
            chunks = splitter.split(text)  # type: ignore[attr-defined]
            
            if chunks and all(isinstance(c, str) for c in chunks):
                click.echo(f"[info] Successfully used docling GPT-aware splitter: {len(chunks)} chunks created")
                
                # Create documents from chunks
                for idx, chunk in enumerate(chunks):
                    if not chunk.strip():  # Skip empty chunks
                        continue
                    new_meta = dict(metadata)
                    new_meta["chunk_index"] = idx
                    new_meta["chunking_method"] = "docling_gpt"
                    docs_out.append(Document(content=chunk, metadata=new_meta))
                
                return docs_out
            else:
                click.echo("[warning] Docling splitter returned invalid chunks, trying fallbacks", err=True)
        except Exception as docling_error:
            click.echo(f"[warning] Docling splitter failed: {docling_error}", err=True)

        # 3. Try LangChain's recursive character splitter
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
            click.echo("[info] Using LangChain RecursiveCharacterTextSplitter")
            
            lc_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size * 4,  # approximate char length
                chunk_overlap=overlap * 4,
                separators=["\n\n", "\n", ".", "!", "?", " "],
            )
            chunks = lc_splitter.split_text(text)
            
            if chunks and all(isinstance(c, str) for c in chunks):
                click.echo(f"[info] Successfully used LangChain recursive splitter: {len(chunks)} chunks created")
                
                # Create documents from chunks
                for idx, chunk in enumerate(chunks):
                    if not chunk.strip():  # Skip empty chunks
                        continue
                    new_meta = dict(metadata)
                    new_meta["chunk_index"] = idx
                    new_meta["chunking_method"] = "langchain_recursive"
                    docs_out.append(Document(content=chunk, metadata=new_meta))
                
                return docs_out
            else:
                click.echo("[warning] LangChain splitter returned invalid chunks, trying fallbacks", err=True)
        except Exception as langchain_error:
            click.echo(f"[warning] LangChain splitter failed: {langchain_error}", err=True)

        # 4. Smarter fallback - _smart_chunk_text
        try:
            click.echo("[info] Using smart_chunk_text as fallback")
            chunks = _smart_chunk_text(text, chunk_size * 4, overlap * 4)
            
            if chunks and all(isinstance(c, str) for c in chunks):
                click.echo(f"[info] Successfully used smart_chunk_text fallback: {len(chunks)} chunks created")
                
                # Create documents from chunks
                for idx, chunk in enumerate(chunks):
                    if not chunk.strip():  # Skip empty chunks
                        continue
                    new_meta = dict(metadata)
                    new_meta["chunk_index"] = idx
                    new_meta["chunking_method"] = "smart_fallback"
                    docs_out.append(Document(content=chunk, metadata=new_meta))
                
                return docs_out
            else:
                click.echo("[warning] smart_chunk_text returned invalid chunks, trying basic chunking", err=True)
        except Exception as smart_error:
            click.echo(f"[warning] smart_chunk_text failed: {smart_error}", err=True)
        
        # 5. Absolute last-ditch fallback - basic character splitter
        click.echo("[warning] All sophisticated chunking methods failed, using basic character chunking", err=True)
        try:
            chunks = chunk_text(text, chunk_size * 4)
            
            for idx, chunk in enumerate(chunks):
                if not chunk.strip():  # Skip empty chunks
                    continue
                new_meta = dict(metadata)
                new_meta["chunk_index"] = idx
                new_meta["chunking_method"] = "basic_character"
                docs_out.append(Document(content=chunk, metadata=new_meta))
            
            click.echo(f"[info] Created {len(docs_out)} chunks with basic character chunking")
            return docs_out
        except Exception as basic_error:
            click.echo(f"[error] Even basic chunking failed: {basic_error}", err=True)
        
        # No valid chunks found, return empty list as last resort
        if not docs_out:
            click.echo("[error] All chunking methods failed, unable to chunk text", err=True)
        
        return docs_out

    # ------------------------------------------------------------------
    # If *source* is a URL, prefer the unstructured/LangChain loader.
    # ------------------------------------------------------------------

    if source.lower().startswith(("http://", "https://")):
        click.echo(f"[info] Processing URL: {source}")
        all_url_docs: list[Document] = []
        url_load_success = False

        # Attempt LangChain UnstructuredURLLoader
        try:
            # ------------------------------------------------------------------
            # LangChain split up into *langchain* (core) and *langchain-community*.
            # The URL loader we need was moved to the latter. We therefore try
            # the new location first, then fall back to the pre-split paths so
            # that older installations remain compatible.
            # ------------------------------------------------------------------
            click.echo("[info] Trying LangChain URL loader...")

            # Track which imports we actually have
            have_langchain_community = False
            have_langchain = False
            have_unstructured = False

            # Try importing the necessary modules
            try:
                import langchain_community
                have_langchain_community = True
                click.echo("[info] Found langchain-community package")
            except ImportError:
                pass

            try:
                import langchain
                have_langchain = True
                click.echo("[info] Found langchain package")
            except ImportError:
                pass

            try:
                import unstructured
                have_unstructured = True
                click.echo("[info] Found unstructured package")
            except ImportError:
                pass

            if not (have_langchain or have_langchain_community):
                click.echo("[warning] LangChain packages not found. Please install with: pip install langchain langchain-community", err=True)
                raise ImportError("LangChain not available")

            if not have_unstructured:
                click.echo("[warning] Unstructured package not found. Please install with: pip install unstructured", err=True)
                raise ImportError("Unstructured not available")

            # Now attempt to import the specific loader
            UnstructuredURLLoader = None

            if have_langchain_community:
                try:  # New �0.1.0 structure
                    from langchain_community.document_loaders import UnstructuredURLLoader
                    click.echo("[info] Using langchain-community.document_loaders.UnstructuredURLLoader")
                except ImportError:
                    pass

            if UnstructuredURLLoader is None and have_langchain:
                try:  # Legacy structure (�0.0.x)
                    from langchain.document_loaders import UnstructuredURLLoader
                    click.echo("[info] Using langchain.document_loaders.UnstructuredURLLoader")
                except ImportError:
                    try:
                        from langchain.document_loaders.unstructured_url import UnstructuredURLLoader
                        click.echo("[info] Using langchain.document_loaders.unstructured_url.UnstructuredURLLoader")
                    except ImportError:
                        UnstructuredURLLoader = None

            if UnstructuredURLLoader is None:
                raise ImportError("Could not find UnstructuredURLLoader in any package")

            # Create and use the loader
            click.echo(f"[info] Loading URL with UnstructuredURLLoader: {source}")
            loader = UnstructuredURLLoader(urls=[source])
            raw_docs = loader.load()

            if not raw_docs:
                click.echo(f"[warning] UnstructuredURLLoader returned no documents for '{source}'", err=True)
            else:
                click.echo(f"[info] UnstructuredURLLoader loaded {len(raw_docs)} documents")

                for raw in raw_docs:
                    text = raw.page_content
                    meta = raw.metadata or {}
                    chunked_docs = _chunk_text_tokenwise(text, meta)
                    all_url_docs.extend(chunked_docs)

                if all_url_docs:
                    url_load_success = True
                    click.echo(f"[info] Successfully processed {len(all_url_docs)} chunks from URL using LangChain")

        except ImportError as ie:
            click.echo(
                f"[warning] LangChain or UnstructuredURLLoader not available: {ie}. "
                "Please install with: pip install langchain langchain-community unstructured",
                err=True,
            )
        except Exception as e:
            click.echo(f"[warning] LangChain URL load failed for '{source}': {e}", err=True)
        # Secondary fallback: use BeautifulSoup for URL content
        if not url_load_success:
            click.echo("[info] Trying secondary URL loader with BeautifulSoup...")
            try:
                import requests
                from bs4 import BeautifulSoup

                click.echo(f"[info] Fetching URL content: {source}")
                resp = requests.get(source, timeout=120)
                resp.raise_for_status()

                html = resp.text
                soup = BeautifulSoup(html, "html.parser")
                text = soup.get_text(separator="\n")
                chunks = _smart_chunk_text(text, chunk_size, overlap)
                docs_bs: list[Document] = []
                for idx, chunk in enumerate(chunks):
                    docs_bs.append(Document(content=chunk, metadata={"source": source, "chunk_index": idx}))
                if docs_bs:
                    click.echo(f"[info] Successfully processed {len(docs_bs)} chunks from URL using BeautifulSoup")
                    return docs_bs
            except Exception as e_bs:
                click.echo(f"[warning] BeautifulSoup URL loader failed: {e_bs}", err=True)

        # Fallback: Unstructured.io partition of remote HTML
        if not url_load_success:
            click.echo("[info] Trying fallback URL loader with BeautifulSoup...")
            try:
                import requests
                import tempfile
                import os as _os
                from unstructured.partition.html import partition_html
                from unstructured.documents.elements import Table

                # Fetch remote HTML with a longer timeout
                click.echo(f"[info] Fetching URL content: {source}")
                resp = requests.get(source, timeout=120)
                resp.raise_for_status()

                # Write to temp file
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
                tmp_path = tmp.name
                tmp.write(resp.content)
                tmp.close()

                click.echo(f"[info] Saved URL content to temporary file: {tmp_path}")

                # Partition HTML content and chunk elements
                click.echo("[info] Partitioning HTML content...")
                try:
                    elements = partition_html(tmp_path)
                except Exception as e_html:
                    click.echo(f"[warning] Partitioning HTML failed: {e_html}", err=True)
                    elements = []

                docs_url: list[Document] = []
                if elements:
                    click.echo(f"[info] Extracted {len(elements)} elements from HTML")
                    for elem in elements:
                        if isinstance(elem, Table):
                            try:
                                md = elem.to_markdown()
                            except Exception:
                                md = elem.get_text()
                            lines = md.splitlines()
                            if len(lines) > 2:
                                # Find all header rows (not just the first)
                                header_separator_idx = 1  # Default separator is at line 1
                                
                                # Find the actual separator line (the line with dashes/pipes)
                                for i, line in enumerate(lines):
                                    if all(c in '|-+: ' for c in line) and '|' in line:
                                        header_separator_idx = i
                                        break
                                
                                # Get all header rows
                                header_rows = [lines[i] for i in range(header_separator_idx)]
                                header_text = "\n".join(header_rows) + "\n" + lines[header_separator_idx]
                                
                                # Extract column-region mapping
                                column_mapping = extract_column_mapping(lines, header_separator_idx)
                                
                                for row_idx, row_content in enumerate(lines[header_separator_idx + 1:]):
                                    row = row_content.strip()
                                    if not row:
                                        continue
                                    # Include all header rows in each chunk
                                    row_md = f"{header_text}\n{row}"
                                    # Enhanced metadata with column mapping
                                    metadata = {
                                        "source": source,
                                        "is_table": True,
                                        "table_row_index": row_idx,
                                        "column_mapping": column_mapping
                                    }
                                    docs_url.append(Document(content=row_md, metadata=metadata))
                            else:
                                docs_url.append(Document(content=md, metadata={"source": source, "is_table": True}))
                        elif hasattr(elem, "text") and isinstance(elem.text, str):
                            txt = elem.text
                            docs_url.extend(_chunk_text_tokenwise(txt, {"source": source}))
                else:
                    click.echo(f"[warning] No elements extracted from '{source}'", err=True)

                # Basic fallback with BeautifulSoup if no docs_url
                if not docs_url:
                    click.echo("[info] Trying basic HTML extraction with BeautifulSoup...")
                    try:
                        from bs4 import BeautifulSoup
                        with open(tmp_path, "r", encoding="utf-8", errors="ignore") as fh:
                            html_content = fh.read()
                        soup = BeautifulSoup(html_content, "html.parser")
                        text = soup.get_text(separator="\n")
                        bs_chunks = _smart_chunk_text(text, chunk_size, overlap)
                        for idx, chunk in enumerate(bs_chunks):
                            docs_url.append(Document(content=chunk, metadata={"source": source, "chunk_index": idx}))
                        click.echo(f"[info] BeautifulSoup fallback loaded {len(docs_url)} chunks")
                    except Exception as e_bs:
                        click.echo(f"[warning] BeautifulSoup fallback failed: {e_bs}", err=True)

                if docs_url:
                    all_url_docs.extend(docs_url)
                    url_load_success = True
                    click.echo(f"[info] Successfully processed {len(docs_url)} chunks from URL using Unstructured.io")

                # Cleanup
                try:
                    _os.remove(tmp_path)
                except Exception:
                    pass

            except Exception as e2:
                click.echo(f"[warning] Unstructured URL fallback failed: {e2}", err=True)
                click.echo("[info] If you are trying to load a URL, please ensure you have required packages:", err=True)
                click.echo("    pip install langchain langchain-community unstructured requests bs4", err=True)

        # Return any documents we found
        if all_url_docs:
            click.echo(f"[info] Returning {len(all_url_docs)} total chunks from URL")
            return all_url_docs
        else:
            click.echo(f"[warning] Could not extract any content from URL: {source}", err=True)
            # Don't return an empty list - let the code continue to try other methods

        # Fallback to generic extractor

        # If source is a local PDF, use Unstructured.io for layout-aware parsing with fallback
    if os.path.isfile(source):
        ext = os.path.splitext(source)[1].lower()
        # PDF parsing
        if ext == '.pdf':
            # Try Poppler pdftotext for fast text extraction (no OCR)
            try:
                import subprocess
                # Extract plain text via poppler's pdftotext CLI
                proc = subprocess.run(
                    ["pdftotext", "-layout", "-enc", "UTF-8", source, "-"],
                    capture_output=True,
                    check=True,
                )
                text = proc.stdout.decode("utf-8", errors="ignore")
                # Chunk and return Document objects using tokenwise splitter
                return _chunk_text_tokenwise(text, {"source": source})
            except FileNotFoundError:
                # pdftotext not installed; fall back to existing PDF parser
                pass
            except Exception as e:
                click.echo(f"[warning] pdftotext extraction failed: {e}", err=True)
            # Fallback to Unstructured parser
            try:
                from unstructured.partition.pdf import partition_pdf
                from unstructured.documents.elements import Table
                elements = partition_pdf(source)
                docs_pdf: list[Document] = []
                for elem in elements:
                    if isinstance(elem, Table):
                        try:
                            md = elem.to_markdown()
                        except Exception:
                            md = elem.get_text()
                        # Split table into row-level chunks for better embeddings
                        lines = md.splitlines()
                        if len(lines) > 2:
                            # Find all header rows (not just the first)
                            header_separator_idx = 1  # Default separator is at line 1
                            
                            # Find the actual separator line (the line with dashes/pipes)
                            for i, line in enumerate(lines):
                                if all(c in '|-+: ' for c in line) and '|' in line:
                                    header_separator_idx = i
                                    break
                            
                            # Get all header rows
                            header_rows = [lines[i] for i in range(header_separator_idx)]
                            header_text = "\n".join(header_rows) + "\n" + lines[header_separator_idx]
                            
                            # Extract column-region mapping
                            column_mapping = extract_column_mapping(lines, header_separator_idx)
                            
                            for row_idx, row_content in enumerate(lines[header_separator_idx + 1:]):
                                row = row_content.strip()
                                if not row:
                                    continue
                                # Include all header rows in each chunk
                                row_md = f"{header_text}\n{row}"
                                # Enhanced metadata with column mapping
                                metadata = {
                                    "source": source,
                                    "is_table": True,
                                    "table_row_index": row_idx,
                                    "column_mapping": column_mapping
                                }
                                docs_pdf.append(Document(content=row_md, metadata=metadata))
                        else:
                            docs_pdf.append(Document(content=md, metadata={"source": source, "is_table": True}))
                    elif hasattr(elem, 'text') and isinstance(elem.text, str):
                        txt = elem.text
                        docs_pdf.extend(_chunk_text_tokenwise(txt, {"source": source}))
                return docs_pdf
            except Exception as e:
                click.echo(f"[warning] Unstructured PDF parse failed for '{source}': {e}", err=True)
                # Fallback: plain-text chunking
                try:
                    import io, os as _os
                    from unstructured.partition.text import partition_text
                    text_elems = partition_text(source)
                    texts = [e.text for e in text_elems if hasattr(e, 'text')]
                    return [d for t in texts for d in _chunk_text_tokenwise(t, {"source": source})]
                except Exception:
                    # Final fallback: read raw text
                    try:
                        with open(source, 'r', encoding='utf-8', errors='ignore') as fh:
                            full_text = fh.read()
                        return [Document(content=chunk, metadata={"source": source, "chunk_index": idx})
                                for idx, chunk in enumerate(_smart_chunk_text(full_text, chunk_size, overlap))]
                    except Exception:
                        pass
        # HTML parsing
        if ext in ('.html', '.htm'):
            try:
                # Import our improved table processing
                try:
                    from table_ingestion import process_html_file
                    
                    # Use our graph-based table processing with context preservation
                    docs_html = process_html_file(source)
                    if docs_html:
                        return docs_html
                except ImportError:
                    # If table_ingestion module is not available, log a warning
                    click.echo("[warning] table_ingestion module not found. Using standard processing.", err=True)
                
                # Fall back to enhanced standard processing if the module isn't available
                # or if it didn't find any tables
                from unstructured.partition.html import partition_html
                from unstructured.documents.elements import Table
                elements = partition_html(source)
                docs_html: list[Document] = []
                for elem in elements:
                    if isinstance(elem, Table):
                        try:
                            md = elem.to_markdown()
                        except Exception:
                            md = elem.get_text()
                        # Split table into row-level chunks for better embeddings
                        lines = md.splitlines()
                        if len(lines) > 2:
                            # Find all header rows (not just the first)
                            header_separator_idx = 1  # Default separator is at line 1
                            
                            # Find the actual separator line (the line with dashes/pipes)
                            for i, line in enumerate(lines):
                                if all(c in '|-+: ' for c in line) and '|' in line:
                                    header_separator_idx = i
                                    break
                            
                            # Get all header rows
                            header_rows = [lines[i] for i in range(header_separator_idx)]
                            header_text = "\n".join(header_rows) + "\n" + lines[header_separator_idx]
                            
                            # Extract column-region mapping
                            column_mapping = extract_column_mapping(lines, header_separator_idx)
                            
                            for row_idx, row_content in enumerate(lines[header_separator_idx + 1:]):
                                row = row_content.strip()
                                if not row:
                                    continue
                                # Include all header rows in each chunk
                                row_md = f"{header_text}\n{row}"
                                # Enhanced metadata with column mapping
                                metadata = {
                                    "source": source,
                                    "is_table": True,
                                    "table_row_index": row_idx,
                                    "column_mapping": column_mapping
                                }
                                docs_html.append(Document(content=row_md, metadata=metadata))
                        else:
                            # Fixed indentation error - this was at the wrong level
                            docs_html.append(Document(content=md, metadata={"source": source, "is_table": True}))
                    elif hasattr(elem, 'text') and isinstance(elem.text, str):
                        txt = elem.text
                        docs_html.extend(_chunk_text_tokenwise(txt, {"source": source}))
                return docs_html
            except Exception as e:
                click.echo(f"[warning] Unstructured HTML parse failed for '{source}': {e}", err=True)
                # Fallback: read raw HTML text
                try:
                    from bs4 import BeautifulSoup
                    with open(source, 'r', encoding='utf-8', errors='ignore') as fh:
                        soup = BeautifulSoup(fh, 'html.parser')
                    text = soup.get_text(separator='\n')
                    return [Document(content=chunk, metadata={"source": source, "chunk_index": idx})
                            for idx, chunk in enumerate(_smart_chunk_text(text, chunk_size, overlap))]
                except Exception:
                    pass


    # Try docling extract to get raw text, then chunk by tokens (fallback to char-chunks)
    # (definition moved to top of function so that it can be referenced from
    # earlier code paths � see above.)

    # Check if the source is a valid file or URL
    # Skip docling processing if source appears to be a flag passed incorrectly
    if source.startswith('--'):
        click.echo(f"[fatal] Invalid source '{source}' - appears to be a command line flag. Did you mean to use this as a parameter?", err=True)
        sys.exit(1)

    try:
        try:
            import docling.extract as dlextract
        except ImportError:
            import docling_core.extract as dlextract
        extractor = dlextract.TextExtractor(path=source, include_comments=True)
        extracted = extractor.run()
        documents: list[Document] = []
        for doc in extracted:
            if hasattr(doc, "text") and isinstance(doc.text, str):
                txt = doc.text
            elif hasattr(doc, "content") and isinstance(doc.content, str):
                txt = doc.content
            else:
                txt = str(doc)
            meta = getattr(doc, "metadata", {}) or {}
            documents.extend(_chunk_text_tokenwise(txt, meta))
        return documents
    except ImportError:
        # docling.extract not available; fall back to legacy loader
        pass
    except Exception as e:
        click.echo(f"[warning] docling extract failed: {e}", err=True)
        click.echo("[warning] Falling back to legacy loader...", err=True)
    # Otherwise, delegate to legacy docling if available
    try:
        docling = _lazy_import("docling")
    except SystemExit:
        # docling not installed – fall back to naive whitespace chunking of the entire file
        try:
            with open(source, "r", encoding="utf-8", errors="replace") as fh:
                full_text = fh.read()
        except Exception as e:
            click.echo(f"[warning] Could not read source '{source}': {e}", err=True)
            # Return empty list so caller can handle missing documents gracefully
            return []

        # Chunk with smarter boundaries and overlap
        text_chunks = _smart_chunk_text(full_text, chunk_size, overlap)
        return [
            Document(content=chunk, metadata={"source": source, "chunk_index": idx})
            for idx, chunk in enumerate(text_chunks)
        ]

    # Try old docling API: load() (pass chunk_size and overlap if supported)
    if hasattr(docling, "load"):
        try:
            dataset = docling.load(source, chunk_size=chunk_size, overlap=overlap)  # type: ignore[attr-defined]
        except TypeError:
            dataset = docling.load(source)  # type: ignore[attr-defined]
    # Try legacy API: DocumentSet
    elif hasattr(docling, "DocumentSet"):
        dataset = docling.DocumentSet(source)  # type: ignore
    # Try new API in submodule: DocumentConverter
    else:
        try:
            dc_mod = __import__("docling.document_converter", fromlist=["DocumentConverter"])
            converter = dc_mod.DocumentConverter()
            conv_res = converter.convert(source)
            doc_obj = conv_res.document
            if hasattr(doc_obj, "export_to_text") and callable(doc_obj.export_to_text):
                text = doc_obj.export_to_text()
            else:
                text = str(doc_obj)
            meta = {"source": source}
            # Chunk converted document into tokens
            return _chunk_text_tokenwise(text, meta)
        except Exception as e:
            click.echo(f"[warning] docling.document_converter failed for '{source}': {e}. Falling back to plain-text chunking.", err=True)
            try:
                with open(source, "r", encoding="utf-8", errors="replace") as fh:
                    full_text = fh.read()
            except Exception as e2:
                click.echo(f"[warning] Could not read source '{source}': {e2}", err=True)
                # Return empty list so caller can handle missing documents gracefully
                return []
            # Chunk with smarter boundaries and overlap
            chunks = _smart_chunk_text(full_text, chunk_size, overlap)
            return [Document(content=chunk, metadata={"source": source, "chunk_index": idx})
                    for idx, chunk in enumerate(chunks)]

    documents: list[Document] = []

    # Docling objects are generally iterable; we guard for attribute names.
    if hasattr(dataset, "documents"):
        iterable = dataset.documents
    else:
        iterable = dataset  # assume it's already iterable

    for doc in iterable:
        # heuristically determine textual content
        text = None

        # If doc has attribute 'text' or 'content', use it; otherwise str(doc)
        if hasattr(doc, "text") and isinstance(doc.text, str):
            text = doc.text
        elif hasattr(doc, "content") and isinstance(doc.content, str):
            text = doc.content
        else:
            text = str(doc)

        metadata = {}

        # If doc has id, title etc, capture them as metadata
        for key in ("id", "title", "name", "source", "path", "url"):
            if hasattr(doc, key):
                metadata[key] = getattr(doc, key)

        # store full raw json of doc (might not be serialisable). We attempt safe conversion.
        try:
            raw_json = json.loads(json.dumps(doc, default=str))
            metadata["raw"] = raw_json
        except (TypeError, ValueError):
            metadata["raw"] = str(doc)

        # Chunk legacy documents token-wise (fallback by char if token splitter missing)
        for chunk_doc in _chunk_text_tokenwise(text, metadata):
            documents.append(chunk_doc)

    return documents


def ensure_collection(client, collection_name: str, vector_size: int, distance: str = "Cosine") -> None:
    qdrant_client = _lazy_import("qdrant_client")
    from qdrant_client.http import models as rest

    existing_collections = {c.name for c in client.get_collections().collections}
    if collection_name in existing_collections:
        return

    click.echo(f"[info] Creating collection '{collection_name}' (size={vector_size}, distance={distance})")

    client.create_collection(
        collection_name=collection_name,
        vectors_config=rest.VectorParams(size=vector_size, distance=distance),
    )


# Dynamic worker pool for parallel embedding operations
class DynamicWorkerPool:
    """Thread-safe dynamic worker pool that adjusts the number of workers based on success/failure rates."""
    
    def __init__(self, initial_workers=20, min_workers=1, max_workers=50,
                 success_threshold=5, failure_threshold=2):
        """
        Initialize the dynamic worker pool.
        
        Args:
            initial_workers: Number of workers to start with
            min_workers: Minimum number of workers allowed
            max_workers: Maximum number of workers allowed
            success_threshold: Number of consecutive successes before increasing workers
            failure_threshold: Number of consecutive failures before decreasing workers
        """
        import threading
        self.current_workers = initial_workers
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.success_threshold = success_threshold
        self.failure_threshold = failure_threshold
        self.lock = threading.Lock()
        self.success_count = 0
        self.failure_count = 0
        self.consecutive_successes = 0
        self.consecutive_failures = 0
        self.adjustment_history = []
        
    def report_success(self):
        """
        Report a successful operation. Increases worker count if threshold is reached.
        
        Returns:
            bool: Whether the worker count was adjusted
        """
        with self.lock:
            self.success_count += 1
            self.consecutive_successes += 1
            self.consecutive_failures = 0
            
            # Increase workers after consecutive successes
            if self.consecutive_successes >= self.success_threshold and self.current_workers < self.max_workers:
                prev_workers = self.current_workers
                # Increase by 25% or at least 1 worker
                increase = max(1, self.current_workers // 4)
                self.current_workers = min(self.max_workers, self.current_workers + increase)
                self.consecutive_successes = 0
                
                # Record adjustment
                self.adjustment_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "previous": prev_workers,
                    "current": self.current_workers,
                    "reason": "success",
                    "total_success": self.success_count,
                    "total_failure": self.failure_count
                })
                return True
        return False
        
    def report_failure(self):
        """
        Report a failed operation. Decreases worker count if threshold is reached.
        
        Returns:
            bool: Whether the worker count was adjusted
        """
        with self.lock:
            self.failure_count += 1
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            
            # Decrease workers when failures occur
            if self.consecutive_failures >= self.failure_threshold and self.current_workers > self.min_workers:
                prev_workers = self.current_workers
                # Decrease by 30% or at least 1 worker
                decrease = max(1, self.current_workers // 3)
                self.current_workers = max(self.min_workers, self.current_workers - decrease)
                self.consecutive_failures = 0
                
                # Record adjustment
                self.adjustment_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "previous": prev_workers,
                    "current": self.current_workers,
                    "reason": "failure",
                    "total_success": self.success_count,
                    "total_failure": self.failure_count
                })
                return True
        return False
        
    def get_worker_count(self):
        """Get the current worker count in a thread-safe manner."""
        with self.lock:
            return self.current_workers
            
    def get_stats(self):
        """Get statistics about the worker pool."""
        with self.lock:
            return {
                "current_workers": self.current_workers,
                "success_count": self.success_count,
                "failure_count": self.failure_count,
                "consecutive_successes": self.consecutive_successes,
                "consecutive_failures": self.consecutive_failures,
                "adjustments": len(self.adjustment_history),
                "adjustment_history": self.adjustment_history
            }

def embed_and_upsert(
    client,
    collection: str,
    docs: List[Document],
    openai_client,
    batch_size: int = 100,
    model_name: str = "text-embedding-3-large",
    deterministic_id: bool = False,
    parallel: int = 15,
    initial_workers: int = 20,  # Start with 20 parallel workers
    min_workers: int = 1,       # Minimum number of workers
    max_workers: int = 50,      # Maximum number of workers
    dynamic_workers: bool = True,  # Enable/disable dynamic worker adjustment
):
    """Embed *docs* in batches and upsert them into Qdrant with optimized batch handling,
    retry logic, comprehensive metrics tracking, and dynamic worker adjustment."""
    
    # Initialize embedding metrics
    embedding_metrics = {
        "requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "retry_count": 0,
        "total_tokens": 0,
        "total_embedding_time": 0,
        "errors_by_type": {},
        "batch_sizes": [],
        "request_times": []
    }
    
    # Add embedding metrics to global diagnostics
    if "embedding_metrics" not in INGESTION_DIAGNOSTICS:
        INGESTION_DIAGNOSTICS["embedding_metrics"] = embedding_metrics
    
    # Estimate tokens in text (rough approximation)
    def estimate_tokens(text):
        # OpenAI models use ~4 chars per token on average
        return len(text) // 4
    
    # Estimate optimal batch size based on token counts
    def optimize_batch_for_tokens(docs, max_tokens=8000, max_docs=100):
        """Batch docs to stay under token limits while maximizing batch size"""
        batches = []
        current_batch = []
        current_tokens = 0
        
        for doc in docs:
            doc_tokens = estimate_tokens(doc.content)
            
            # If this document would exceed max_tokens or we've reached max docs per batch
            if (current_tokens + doc_tokens > max_tokens or
                len(current_batch) >= max_docs) and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            
            current_batch.append(doc)
            current_tokens += doc_tokens
            
            # Handle extra large documents that exceed max_tokens on their own
            if doc_tokens > max_tokens:
                logger.warning(f"Document exceeds token limit ({doc_tokens} tokens) - may cause API issues")
                
        # Add the last batch if not empty
        if current_batch:
            batches.append(current_batch)
            
        return batches

    # Determine which OpenAI binding style is active with more robust detection
    is_openai_v1 = False

    # First check for openai v1 API style
    if hasattr(openai_client, "embeddings"):
        # Verify it's actually the v1 API by checking for the 'create' method
        if hasattr(openai_client.embeddings, "create"):
            is_openai_v1 = True
    # Additional check for v1 API style with different attribute patterns
    elif hasattr(openai_client, "Embeddings") and hasattr(openai_client.Embeddings, "create"):
        is_openai_v1 = True

    logger.info(f"Using OpenAI {'v1' if is_openai_v1 else 'v0'} API for embeddings")
    click.echo(f"[info] Using OpenAI {'v1' if is_openai_v1 else 'v0'} API for embeddings")
    
    # Local helper to call the OpenAI embeddings API with exponential backoff retry
    import time
    import random
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    
    # Using module-level API error classes defined earlier
    
    # Function to categorize errors from API responses
    def categorize_embedding_error(e):
        """Categorize API errors for proper handling"""
        error_str = str(e).lower()
        error_type = type(e).__name__
        
        # Track error in metrics
        if error_type not in embedding_metrics["errors_by_type"]:
            embedding_metrics["errors_by_type"][error_type] = 0
        embedding_metrics["errors_by_type"][error_type] += 1
        
        # Rate limit errors
        if "rate limit" in error_str or "rate_limit" in error_str or "too many requests" in error_str:
            logger.warning(f"Rate limit error detected: {e}")
            return RateLimitError(f"Rate limit exceeded: {e}")
        
        # Authentication errors
        elif "auth" in error_str or "api key" in error_str or "unauthorized" in error_str:
            logger.error(f"Authentication error detected: {e}")
            return AuthenticationError(f"Authentication failed: {e}")
        
        # Service unavailable errors
        elif "503" in error_str or "service unavailable" in error_str or "server error" in error_str:
            logger.warning(f"Service unavailable error detected: {e}")
            return ServiceUnavailableError(f"Service temporarily unavailable: {e}")
        
        # Keep original error for other cases
        return e
    
    # Define which errors should use retry logic
    def is_retryable_error(exception):
        """Determine if an error should trigger a retry"""
        if isinstance(exception, (RateLimitError, ServiceUnavailableError)):
            return True
        
        error_str = str(exception).lower()
        return ("timeout" in error_str or
                "connection" in error_str or
                "network" in error_str or
                "5xx" in error_str or
                "500" in error_str or
                "503" in error_str or
                "retry" in error_str)
    
    # Decorate with retry logic
    # Decorate with retry logic that can be properly tested
    @retry(
        retry=retry_if_exception_type(lambda x: is_retryable_error(x)),
        stop=stop_after_attempt(5),  # Maximum 5 attempts
        wait=wait_exponential(multiplier=1, min=2, max=60),  # Exponential backoff: 2, 4, 8, 16, 32 seconds
        before_sleep=lambda retry_state: logger.info(
            f"Retrying embedding API call after error: {retry_state.outcome.exception()}, "
            f"attempt {retry_state.attempt_number}/5, "
            f"sleeping for {retry_state.next_action.sleep} seconds"
        )
    )
    def get_embeddings(texts, timeout=60):
        """Get embeddings with retry logic, timeout, and error tracking"""
        start_time = time.time()
        
        # Update metrics
        embedding_metrics["requests"] += 1
        embedding_metrics["batch_sizes"].append(len(texts))
        token_estimate = sum(estimate_tokens(text) for text in texts)
        embedding_metrics["total_tokens"] += token_estimate
        
        logger.info(f"Requesting embeddings for {len(texts)} texts ({token_estimate} est. tokens) using model '{model_name}'")
        click.echo(f"[debug] Requesting embeddings for {len(texts)} texts ({token_estimate} est. tokens) using model '{model_name}'")
        
        # Manual retry implementation to ensure tests can verify retry behavior
        max_retries = 3
        retry_count = 0
        
        while True:
            try:
                if is_openai_v1:
                    logger.debug(f"Using OpenAI v1 API for embeddings")
                    try:
                        resp = openai_client.embeddings.create(model=model_name, input=texts, timeout=timeout)
                    except TypeError as exc:
                        if "timeout" in str(exc):
                            logger.debug("Timeout not supported, retrying without timeout")
                            resp = openai_client.embeddings.create(model=model_name, input=texts)
                        else:
                            raise
                    embeddings = [record.embedding for record in resp.data]
                    
                    # Extract usage information if available
                    if hasattr(resp, 'usage') and resp.usage:
                        if hasattr(resp.usage, 'prompt_tokens'):
                            embedding_metrics["total_tokens"] = resp.usage.prompt_tokens
                        logger.debug(f"API reported token usage: {resp.usage}")
                    
                else:
                    logger.debug(f"Using OpenAI v0 API for embeddings")
                    resp = openai_client.Embedding.create(model=model_name, input=texts, request_timeout=timeout)
                    embeddings = [record["embedding"] for record in resp["data"]]
                    
                    # Extract usage information if available
                    if "usage" in resp and resp["usage"]:
                        if "prompt_tokens" in resp["usage"]:
                            embedding_metrics["total_tokens"] = resp["usage"]["prompt_tokens"]
                        logger.debug(f"API reported token usage: {resp['usage']}")
                
                elapsed = time.time() - start_time
                embedding_metrics["request_times"].append(elapsed)
                embedding_metrics["total_embedding_time"] += elapsed
                embedding_metrics["successful_requests"] += 1
                
                if embeddings:
                    logger.debug(f"Received embeddings with dimension: {len(embeddings[0])}")
                
                # Log performance metrics
                logger.info(f"Embedding batch complete: {len(texts)} texts in {elapsed:.2f}s "
                          f"({len(texts)/elapsed:.1f} texts/sec, {token_estimate/elapsed:.1f} tokens/sec)")
                
                return embeddings
                
            except Exception as e:
                elapsed = time.time() - start_time
                embedding_metrics["failed_requests"] += 1
                embedding_metrics["request_times"].append(elapsed)
                
                logger.warning(f"Embedding API call failed after {elapsed:.1f}s: {e}")
                click.echo(f"[warning] Embedding API call failed after {elapsed:.1f}s: {e}", err=True)
                
                # Categorize error for proper handling
                categorized_error = categorize_embedding_error(e)
                
                # Handle specific error types
                if isinstance(categorized_error, RateLimitError):
                    # Add extra delay for rate limit errors
                    extra_delay = random.uniform(1, 5)
                    logger.warning(f"Rate limit error, adding {extra_delay:.1f}s additional delay before retry")
                    time.sleep(extra_delay)
                
                # Manual retry logic for testing purposes
                retry_count += 1
                embedding_metrics["retry_count"] += 1
                
                if retry_count < max_retries and is_retryable_error(categorized_error):
                    logger.warning(f"Retrying embedding API call (attempt {retry_count+1}/{max_retries})")
                    # Don't actually sleep in tests
                    continue
                else:
                    # If we've exceeded max retries or it's a non-retryable error, reraise
                    raise categorized_error

    from qdrant_client.http import models as rest
    
    # Create token-aware batches
    click.echo(f"[info] Creating token-optimized batches from {len(docs)} documents")
    token_optimized_batches = optimize_batch_for_tokens(docs, max_tokens=8000, max_docs=batch_size)
    total_batches = len(token_optimized_batches)
    avg_batch_size = sum(len(batch) for batch in token_optimized_batches) / total_batches if total_batches else 0
    
    click.echo(f"[info] Created {total_batches} token-optimized batches (avg {avg_batch_size:.1f} docs/batch)")
    logger.info(f"Created {total_batches} token-optimized batches (avg {avg_batch_size:.1f} docs/batch)")

    # Function to process a batch with retries and robust error handling
    def process_batch(batch):
        texts = [d.content for d in batch]
        
        # Calculate token estimate for this batch
        token_estimate = sum(estimate_tokens(text) for text in texts)
        logger.info(f"Processing batch of {len(texts)} documents (~{token_estimate} tokens)")
        
        try:
            embeddings = get_embeddings(texts)
            
            if not embeddings:
                logger.error(f"Empty embeddings returned for batch of {len(texts)} documents")
                click.echo(f"[error] Empty embeddings returned for batch", err=True)
                return
                
            # Determine expected dimension
            expected_size = len(embeddings[0]) if embeddings else None

            # Build Qdrant points
            points = []
            for doc, vector in zip(batch, embeddings):
                metadata = doc.metadata.copy()
                content = doc.content
                if deterministic_id:
                    try:
                        meta_str = json.dumps(metadata, sort_keys=True, default=str)
                    except Exception:
                        meta_str = str(metadata)
                    id_input = meta_str + "\n" + content
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, id_input))
                else:
                    point_id = str(uuid.uuid4())
                payload = metadata
                payload["chunk_text"] = content
                # Add embedding metrics to payload for accountability
                payload["embedding_info"] = {
                    "model": model_name,
                    "token_estimate": estimate_tokens(content),
                    "timestamp": datetime.now().isoformat()
                }
                points.append(rest.PointStruct(id=point_id, vector=vector, payload=payload))

            # Verify vector dimensions
            if expected_size is not None:
                lengths = [len(v) for v in embeddings]
                if any(l != expected_size for l in lengths):
                    logger.warning(f"Vector dimension mismatch in batch: expected {expected_size}, got {set(lengths)}")
                    click.echo(f"[warning] Vector dimension mismatch in batch: expected {expected_size}, got {set(lengths)}", err=True)
                else:
                    logger.info(f"Batch vectors verified: {len(lengths)} vectors of dimension {expected_size}")
            else:
                logger.warning(f"No embeddings returned for batch")
                click.echo(f"[warning] No embeddings returned for batch", err=True)
                return

            # Upsert points with retry logic for database operations
            try:
                # Determine if parallel parameter is supported
                client_upsert_params = inspect.signature(client.upsert).parameters
                if "parallel" in client_upsert_params:
                    client.upsert(collection_name=collection, points=points, parallel=parallel)
                else:
                    client.upsert(collection_name=collection, points=points)
                logger.info(f"Successfully upserted batch of {len(points)} vectors to Qdrant")
            except Exception as db_error:
                logger.error(f"Database upsert failed: {db_error}")
                click.echo(f"[error] Database upsert failed: {db_error}", err=True)
                # Track database errors separately
                if "database_errors" not in embedding_metrics:
                    embedding_metrics["database_errors"] = []
                embedding_metrics["database_errors"].append(str(db_error))
                raise
                
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            click.echo(f"[error] Batch processing failed: {e}", err=True)
            # Add to INGESTION_DIAGNOSTICS
            INGESTION_DIAGNOSTICS["errors"].append({
                "timestamp": datetime.now().isoformat(),
                "component": "embedding_batch_processing",
                "error_type": type(e).__name__,
                "message": str(e),
                "batch_size": len(texts),
                "token_estimate": token_estimate
            })
            return

    # Execute batches in parallel with adaptive concurrency
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import inspect
    import time
    
    # Initialize dynamic worker pool
    worker_pool = DynamicWorkerPool(
        initial_workers=initial_workers if dynamic_workers else parallel,
        min_workers=min_workers,
        max_workers=max_workers
    )
    
    # Add worker pool info to metrics
    embedding_metrics["worker_pool"] = {
        "initial_workers": worker_pool.current_workers,
        "min_workers": worker_pool.min_workers,
        "max_workers": worker_pool.max_workers,
        "dynamic_enabled": dynamic_workers,
        "adjustments": []
    }
    
    # Modify process_batch to report success/failure to worker pool
    original_process_batch = process_batch
    
    def process_batch_with_tracking(batch):
        try:
            original_process_batch(batch)
            
            # Report success to worker pool
            if dynamic_workers and worker_pool.report_success():
                new_count = worker_pool.get_worker_count()
                logger.info(f"Increased worker count to {new_count} after successful embedding")
                embedding_metrics["worker_pool"]["adjustments"].append({
                    "timestamp": datetime.now().isoformat(),
                    "new_count": new_count,
                    "previous_count": new_count - (new_count - worker_pool.adjustment_history[-1]["previous"]),
                    "reason": "success"
                })
            return True
        except Exception as e:
            # Report failure to worker pool
            if dynamic_workers and worker_pool.report_failure():
                new_count = worker_pool.get_worker_count()
                logger.warning(f"Decreased worker count to {new_count} after embedding failure: {str(e)}")
                embedding_metrics["worker_pool"]["adjustments"].append({
                    "timestamp": datetime.now().isoformat(),
                    "new_count": new_count,
                    "previous_count": new_count + (worker_pool.adjustment_history[-1]["previous"] - new_count),
                    "reason": "failure",
                    "error_type": type(e).__name__
                })
            return False
    
    # Process batches in chunks to allow worker count adjustments
    chunk_size = min(100, max(10, total_batches // 5))  # Aim for 5-10 chunks
    batch_chunks = [token_optimized_batches[i:i+chunk_size] for i in range(0, len(token_optimized_batches), chunk_size)]
    
    click.echo(f"[info] Embedding & upserting in {total_batches} batch(es) using dynamic worker pool (starting with {worker_pool.get_worker_count()} workers)")
    logger.info(f"Embedding & upserting in {total_batches} batch(es) using dynamic worker pool (starting with {worker_pool.get_worker_count()} workers)")
    
    start_time = time.time()
    completed_batches = 0
    success_count = 0
    
    for chunk_idx, batch_chunk in enumerate(batch_chunks):
        current_workers = worker_pool.get_worker_count()
        chunk_desc = f"Chunk {chunk_idx+1}/{len(batch_chunks)}"
        
        click.echo(f"[info] Processing {chunk_desc} using {current_workers} workers ({len(batch_chunk)} batches)")
        logger.info(f"Processing {chunk_desc} using {current_workers} workers ({len(batch_chunk)} batches)")
        
        with ThreadPoolExecutor(max_workers=current_workers) as executor:
            futures = [executor.submit(process_batch_with_tracking, batch) for batch in batch_chunk]
            
            for future in tqdm(as_completed(futures), total=len(batch_chunk), desc=f"{chunk_desc} embedding"):
                completed_batches += 1
                
                # Check if the future raised an exception
                if future.exception():
                    logger.error(f"Batch processing raised exception: {future.exception()}")
                elif future.result() is True:  # Check explicit success result
                    success_count += 1
            
            # Log progress periodically
            if completed_batches % max(1, total_batches // 10) == 0:
                elapsed = time.time() - start_time
                progress_pct = (completed_batches / total_batches) * 100
                est_remaining = (elapsed / completed_batches) * (total_batches - completed_batches) if completed_batches > 0 else 0
                click.echo(f"[info] Progress: {completed_batches}/{total_batches} batches ({progress_pct:.1f}%), "
                          f"est. {est_remaining:.1f}s remaining")
    
    # Compute and log final statistics
    total_time = time.time() - start_time
    avg_time_per_batch = total_time / total_batches if total_batches else 0
    success_rate = (success_count / total_batches) * 100 if total_batches else 0
    
    embedding_metrics.update({
        "total_batches": total_batches,
        "successful_batches": success_count,
        "success_rate": success_rate,
        "total_processing_time": total_time,
        "avg_time_per_batch": avg_time_per_batch,
        "completed_at": datetime.now().isoformat()
    })
    
    # Add worker pool stats to metrics
    final_stats = worker_pool.get_stats()
    embedding_metrics["worker_pool"].update({
        "final_workers": final_stats["current_workers"],
        "total_adjustments": final_stats["adjustments"],
        "final_stats": final_stats
    })
    
    # Save metrics to INGESTION_DIAGNOSTICS
    INGESTION_DIAGNOSTICS["embedding_metrics"] = embedding_metrics
    
    # Log final statistics
    logger.info(f"Embedding complete: {success_count}/{total_batches} batches successful ({success_rate:.1f}%)")
    logger.info(f"Total embedding time: {total_time:.2f}s, avg {avg_time_per_batch:.2f}s per batch")
    logger.info(f"Dynamic worker pool: started={initial_workers}, ended={final_stats['current_workers']}, " +
               f"made {final_stats['adjustments']} adjustments")
    
    click.echo(f"[info] Embedding complete: {success_count}/{total_batches} batches successful ({success_rate:.1f}%)")
    click.echo(f"[info] Total embedding time: {total_time:.2f}s, avg {avg_time_per_batch:.2f}s per batch")
    click.echo(f"[info] Dynamic worker pool: started={initial_workers}, ended={final_stats['current_workers']}, " +
              f"made {final_stats['adjustments']} adjustments")
    
    if embedding_metrics["retry_count"] > 0:
        logger.info(f"Required {embedding_metrics['retry_count']} retries due to temporary errors")
        click.echo(f"[info] Required {embedding_metrics['retry_count']} retries due to temporary errors")


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--env-file",
    type=click.Path(dir_okay=False, readable=True),
    default=None,
    help="Path to .env file with environment variables (if present, loaded before other options)."
)
@click.option("--source", required=True, help="Path/URL/DSN pointing to the corpus to ingest.")
@click.option("--collection", default="mattermost_rag_store", show_default=True, help="Qdrant collection name to create/use.")
@click.option("--batch-size", type=int, default=128, show_default=True, help="Embedding batch size.")
@click.option("--openai-api-key", envvar="OPENAI_API_KEY", help="Your OpenAI API key (can also use env var OPENAI_API_KEY)")
@click.option("--qdrant-host", default="localhost", show_default=True, help="Qdrant host (ignored when --qdrant-url is provided).")
@click.option("--qdrant-port", type=int, default=6333, show_default=True, help="Qdrant port (ignored when --qdrant-url is provided).")
@click.option("--qdrant-url", help="Full Qdrant URL (e.g. https://*.qdrant.io:6333). Overrides host/port.")
@click.option("--qdrant-api-key", envvar="QDRANT_API_KEY", help="Qdrant API key if required (Cloud).")
@click.option("--distance", type=click.Choice(["Cosine", "Dot", "Euclid"], case_sensitive=False), default="Cosine", help="Vector distance metric.")
@click.option("--chunk-size", type=int, default=1000, show_default=True, help="Chunk size (tokens) for docling chunker.")
@click.option("--chunk-overlap", type=int, default=50, show_default=True, help="Overlap (tokens) between chunks.")
@click.option("--crawl-depth", type=int, default=0, show_default=True, help="When SOURCE is a URL, crawl hyperlinks up to this depth (0=no crawl).")
@click.option("--parallel", type=int, default=15, show_default=True, help="Number of parallel workers for Qdrant upsert.")
@click.option("--initial-workers", type=int, default=20, show_default=True, help="Starting number of workers for embedding processing.")
@click.option("--min-workers", type=int, default=1, show_default=True, help="Minimum number of workers allowed during dynamic adjustment.")
@click.option("--max-workers", type=int, default=50, show_default=True, help="Maximum number of workers allowed during dynamic adjustment.")
@click.option("--dynamic-workers/--no-dynamic-workers", default=True, show_default=True, help="Enable/disable dynamic worker adjustment based on success/failure.")
@click.option("--fast-chunking/--precise-chunking", default=True, show_default=True,
              help="Use fast heuristic-based semantic chunking (faster) or transformer-based semantic chunking (more precise but much slower).")
@click.option("--generate-summaries/--no-generate-summaries", default=True, show_default=True,
               help="Generate and index brief summaries of each chunk for multi-granularity retrieval.")
@click.option("--quality-checks/--no-quality-checks", default=True, show_default=True,
               help="Perform post-ingest quality checks on chunk sizes and entity consistency.")
@click.option("--rich-metadata/--no-rich-metadata", default=True, show_default=True,
               help="Extract rich metadata from document content for better retrieval context.",
              is_flag=True)
@click.option("--hierarchical-embeddings/--no-hierarchical-embeddings", default=True, show_default=True,
               help="Create hierarchical embeddings at document, section, and chunk levels.",
              is_flag=True)
@click.option("--entity-extraction/--no-entity-extraction", default=False, show_default=True,
               help="Enable entity extraction and normalization for better embeddings.",
              is_flag=True)
@click.option("--enhance-text-with-entities/--no-enhance-text-with-entities", default=False, show_default=True,
               help="Enhance document text with extracted entity information.",
              is_flag=True)
@click.option("--adaptive-chunking/--no-adaptive-chunking", default=False, show_default=True,
               help="Use content-aware adaptive chunking instead of fixed-size chunking.",
              is_flag=True)
@click.option("--deduplication/--no-deduplication", default=True, show_default=True,
              help="Enable duplicate detection and removal during ingestion.",
              is_flag=True)
@click.option("--similarity-threshold", type=float, default=0.85, show_default=True,
              help="Similarity threshold for near-duplicate detection (0-1).")
@click.option("--merge-duplicates/--no-merge-duplicates", default=True, show_default=True,
              help="Merge similar documents instead of removing them.")
@click.option("--validate-ingestion/--no-validate-ingestion", default=True, show_default=True,
              help="Run post-ingestion validation to verify embedding quality.")
@click.option("--run-test-queries/--no-run-test-queries", default=False, show_default=True,
              help="Run test queries after ingestion to verify retrieval.")
@click.option("--doc-embedding-model", default="text-embedding-3-large", show_default=True,
              help="OpenAI model to use for document-level embeddings.")
@click.option("--section-embedding-model", default="text-embedding-3-large", show_default=True,
              help="OpenAI model to use for section-level embeddings.")
@click.option("--chunk-embedding-model", default="text-embedding-3-large", show_default=True,
              help="OpenAI model to use for chunk-level embeddings.")
@click.option(
    "--bm25-index",
    type=click.Path(dir_okay=False, writable=True),
    default="mattermost_rag_store_bm25_index.json",
    show_default=True,
    help="Path to write BM25 index JSON mapping point IDs to chunk_text.")
# ------------------------------------------------------------------
# Language filtering & validation behaviour
# ------------------------------------------------------------------

@click.option(
    "--languages",
    default="en",
    show_default=True,
    help="Comma-separated list of ISO language codes to *keep*. Use 'all' to disable filtering.",
)
@click.option(
    "--lang-threshold",
    type=float,
    default=0.9,
    show_default=True,
    help="Minimum probability required from the language detector to trust the result (0-1).",
)
@click.option(
    "--fail-on-validation-error/--no-fail-on-validation-error",
    default=False,
    show_default=True,
    help="Exit with non-zero status if post-ingest validation fails.",
)
def cli(
    env_file: str | None,
    source: str,
    collection: str,
    batch_size: int,
    openai_api_key: str | None,
    qdrant_host: str,
    qdrant_port: int,
    qdrant_url: str | None,
    qdrant_api_key: str | None,
    distance: str,
    chunk_size: int,
    chunk_overlap: int,
    crawl_depth: int,
    parallel: int,
    initial_workers: int,
    min_workers: int,
    max_workers: int,
    dynamic_workers: bool,
    fast_chunking: bool,
    generate_summaries: bool,
    quality_checks: bool,
    rich_metadata: bool,
    hierarchical_embeddings: bool,
    entity_extraction: bool,
    enhance_text_with_entities: bool,
    adaptive_chunking: bool,
    deduplication: bool,
    similarity_threshold: float,
    merge_duplicates: bool,
    validate_ingestion: bool,
    run_test_queries: bool,
    doc_embedding_model: str,
    section_embedding_model: str,
    chunk_embedding_model: str,
    bm25_index: str | None,
    languages: str,
    lang_threshold: float,
    fail_on_validation_error: bool,
) -> None:
    """Ingest *SOURCE* into a Qdrant RAG database using OpenAI embeddings."""

    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # Load .env file (if present) BEFORE reading env vars / flags
    # ---------------------------------------------------------------------

    if env_file and os.path.isfile(env_file):
        try:
            dotenv = _lazy_import("dotenv")
            dotenv.load_dotenv(env_file, override=False)
            click.echo(f"[info] Environment variables loaded from {env_file}")
        except SystemExit:
            raise
        except Exception:  # pragma: no cover � edge-case, continue silently
            pass

    # Accept lower-case variants (e.g. `openai_api_key=`) for convenience
    if "OPENAI_API_KEY" not in os.environ and "openai_api_key" in os.environ:
        os.environ["OPENAI_API_KEY"] = os.environ["openai_api_key"]
    if "QDRANT_API_KEY" not in os.environ and "qdrant_api_key" in os.environ:
        os.environ["QDRANT_API_KEY"] = os.environ["qdrant_api_key"]

    # ------------------------------------------------------------------
    # Configure language filter from CLI
    # ------------------------------------------------------------------

    global _allowed_languages, _lang_prob_threshold
    if languages.lower() == "all":
        _allowed_languages = set()  # empty set � filter disabled
    else:
        _allowed_languages = {lang.strip().lower() for lang in languages.split(',') if lang.strip()}
    _lang_prob_threshold = max(0.0, min(1.0, lang_threshold))

    # ---------------------------------------------------------------------
    # Validate & set up dependencies
    # ---------------------------------------------------------------------

    # If the CLI flag wasn't provided, fall back to the environment
    if openai_api_key is None:
        openai_api_key = os.environ.get("OPENAI_API_KEY")

    if openai_api_key is None:
        click.echo(
            "[fatal] OPENAI_API_KEY is not set. Provide �openai-api-key, set it in the"
            " environment, or put it in the .env file.",
            err=True,
        )
        sys.exit(1)

    qdrant_client = _lazy_import("qdrant_client")

    # Build Qdrant client
    if qdrant_url:
        client = qdrant_client.QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    else:
        client = qdrant_client.QdrantClient(host=qdrant_host, port=qdrant_port, api_key=qdrant_api_key)

    # ---------------------------------------------------------------------
    # Load documents via docling
    # ---------------------------------------------------------------------

    click.echo(f"[info] Loading documents from source: {source} (chunk_size={chunk_size}, overlap={chunk_overlap}, crawl_depth={crawl_depth}, fast_chunking={fast_chunking})")
    # Pass configuration variables to the global scope for use in chunking functions
    global _use_fast_chunking, _adaptive_chunking
    _use_fast_chunking = fast_chunking
    _adaptive_chunking = adaptive_chunking
    documents = load_documents(source, chunk_size, chunk_overlap, crawl_depth)

    # ------------------------------------------------------------------
    # Language filtering (optional, controlled by �languages CLI flag)
    # ------------------------------------------------------------------

    if _allowed_languages and _lang_prob_threshold > 0.0:
        filtered_docs: list[Document] = []
        skipped = 0
        for doc in documents:
            lang, prob = _detect_language(doc.content[:4000])  # only look at first part for speed
            if lang is None:
                # Unable to detect � keep document (conservative)
                filtered_docs.append(doc)
                continue

            if lang in _allowed_languages and prob >= _lang_prob_threshold:
                filtered_docs.append(doc)
            else:
                skipped += 1
        if skipped:
            click.echo(f"[info] Language filter removed {skipped} documents (allowed={','.join(sorted(_allowed_languages))}, threshold={_lang_prob_threshold})")
        documents = filtered_docs
    click.echo(f"[info] Loaded {len(documents)} document(s)")

    if not documents:
        click.echo("[warning] No documents found � nothing to do.")
        return

    # -----------------------------------------------------------------
    # Enhanced metadata, quality checks, and optional summarization
    # -----------------------------------------------------------------

    # Apply entity extraction and normalization if enabled
    if entity_extraction:
        click.echo("[info] Applying entity extraction and normalization...")

        # Attempt to import entity_extraction module
        try:
            # Try to import using relative path first
            try:
                import entity_extraction
                from entity_extraction import extract_and_normalize_entities, enhance_text_with_entities, get_entity_metadata
                click.echo("[info] Using entity_extraction module from current directory")
            except ImportError:
                # Try to import using absolute path second
                import os
                import sys
                # Get the script directory
                script_dir = os.path.dirname(os.path.abspath(__file__))
                # Add the script directory to sys.path if not already there
                if script_dir not in sys.path:
                    sys.path.insert(0, script_dir)
                # Try importing again
                from entity_extraction import extract_and_normalize_entities, enhance_text_with_entities, get_entity_metadata
                click.echo("[info] Using entity_extraction module from script directory")

            # Process documents with entity extraction
            processed_documents = []
            for doc in tqdm(documents, desc="Extracting and normalizing entities"):
                # Extract entities from document content
                entity_data = extract_and_normalize_entities(doc.content)

                # Create metadata with entity information
                entity_metadata = get_entity_metadata(doc.content)

                # Update document metadata with entity information
                new_metadata = doc.metadata.copy()
                new_metadata.update(entity_metadata)

                # Optionally enhance text with entity annotations for better embeddings
                if enhance_text_with_entities:
                    enhanced_content = enhance_text_with_entities(doc.content, entity_data)
                    click.echo(f"[info] Enhanced document text with entity annotations")
                else:
                    enhanced_content = doc.content

                # Create new document with entity-enhanced content and metadata
                processed_doc = Document(
                    content=enhanced_content,
                    metadata=new_metadata
                )
                processed_documents.append(processed_doc)

            # Replace original documents with processed versions
            documents = processed_documents
            click.echo(f"[info] Entity extraction completed for {len(documents)} documents")

        except ImportError as e:
            click.echo(f"[warning] Entity extraction failed: {e}", err=True)
            click.echo("[info] Install entity_extraction.py module or disable �entity-extraction flag", err=True)
        except Exception as e:
            click.echo(f"[warning] Entity extraction encountered an error: {e}", err=True)
            click.echo("[info] Continuing without entity extraction", err=True)

    # Apply rich metadata extraction if enabled
    if rich_metadata:
        click.echo("[info] Applying rich metadata extraction...")

        # Attempt to import rich_metadata module
        try:
            # Try to import using relative path first
            try:
                import rich_metadata
                from rich_metadata import enrich_document_metadata
                click.echo("[info] Using rich_metadata module from current directory")
            except ImportError:
                # Try to import using absolute path second
                import os
                import sys
                script_dir = os.path.dirname(os.path.abspath(__file__))
                # Add the script directory to sys.path if not already there
                if script_dir not in sys.path:
                    sys.path.insert(0, script_dir)
                # Try importing again
                from rich_metadata import enrich_document_metadata
                click.echo("[info] Using rich_metadata module from script directory")

            # Process documents with rich metadata extraction
            enriched_documents = []
            for doc in tqdm(documents, desc="Extracting rich metadata"):
                # Convert Document class to dict format expected by enrich_document_metadata
                doc_dict = {
                    "content": doc.content,
                    "metadata": doc.metadata.copy()
                }
                # Enrich metadata
                enriched_doc_dict = enrich_document_metadata(doc_dict)
                # Convert back to Document class
                enriched_doc = Document(
                    content=enriched_doc_dict["content"],
                    metadata=enriched_doc_dict["metadata"]
                )
                enriched_documents.append(enriched_doc)

            # Replace original documents with enriched versions
            documents = enriched_documents
            click.echo(f"[info] Rich metadata extraction completed for {len(documents)} documents")

        except ImportError as e:
            click.echo(f"[warning] Rich metadata extraction failed: {e}", err=True)
            click.echo("[info] Install rich_metadata.py module or disable �rich-metadata flag", err=True)
        except Exception as e:
            click.echo(f"[warning] Rich metadata extraction encountered an error: {e}", err=True)
            click.echo("[info] Continuing with basic metadata", err=True)

    # Apply deduplication if enabled
    if deduplication:
        click.echo("[info] Applying document deduplication...")

        # Attempt to import deduplication module
        try:
            # Try to import using relative path first
            try:
                import deduplication
                from deduplication import deduplicate_documents
                click.echo("[info] Using deduplication module from current directory")
            except ImportError:
                # Try to import using absolute path second
                import os
                import sys
                script_dir = os.path.dirname(os.path.abspath(__file__))
                if script_dir not in sys.path:
                    sys.path.insert(0, script_dir)
                from deduplication import deduplicate_documents
                click.echo("[info] Using deduplication module from script directory")

            # Convert Document objects to dictionaries for deduplication
            doc_dicts = [{"content": doc.content, "metadata": doc.metadata} for doc in documents]

            # Apply deduplication
            click.echo(f"[info] Deduplicating {len(doc_dicts)} documents (threshold={similarity_threshold}, merge={merge_duplicates})")
            deduplicated_docs, stats = deduplicate_documents(
                doc_dicts,
                similarity_threshold=similarity_threshold,
                merge_similar=merge_duplicates
            )

            # Convert back to Document objects
            deduplicated = [Document(content=doc["content"], metadata=doc["metadata"]) for doc in deduplicated_docs]

            # Print deduplication stats
            click.echo(f"[info] Deduplication complete: {stats.total_documents} � {stats.unique_documents} documents")
            if stats.exact_duplicates > 0 or stats.near_duplicates > 0:
                click.echo(f"[info] Removed {stats.exact_duplicates} exact duplicates and {stats.near_duplicates} near-duplicates")
                click.echo(f"[info] Found {stats.duplicate_sets} duplicate clusters, largest had {stats.largest_cluster_size} documents")
                click.echo(f"[info] Saved approximately {stats.characters_saved/1024:.1f} KB by deduplication")

            # Replace original documents with deduplicated ones
            documents = deduplicated

        except ImportError as e:
            click.echo(f"[warning] Deduplication failed: {e}", err=True)
            click.echo("[info] Install deduplication.py module or disable �deduplication flag", err=True)
        except Exception as e:
            click.echo(f"[warning] Deduplication encountered an error: {e}", err=True)
            click.echo("[info] Continuing without deduplication", err=True)

    # Annotate each chunk with enhanced metadata
    for idx, doc in enumerate(documents):
        # Source file or URL
        doc.metadata.setdefault("source", source)
        # Section title: first line of chunk
        first_line = doc.content.split("\n", 1)[0].strip()
        doc.metadata.setdefault("section", first_line[:100])
        # Neighboring chunk indices for stitching
        doc.metadata.setdefault("neighbor_prev", idx - 1 if idx > 0 else None)
        doc.metadata.setdefault("neighbor_next", idx + 1 if idx < len(documents) - 1 else None)
        # Date detection: ISO or general date parsing
        if "date" not in doc.metadata:
            m = DATE_REGEX.search(doc.content)
            if m:
                # Found ISO-format date
                doc.metadata["date"] = m.group(1)
            else:
                # Try fuzzy parsing for other date formats
                try:
                    dt = _parse_date(doc.content, fuzzy=True)
                    # Only accept reasonable years
                    if dt.year and dt.year >= 1900:
                        doc.metadata["date"] = dt.date().isoformat()
                except Exception:
                    pass

    # Post-ingest quality checks on chunk sizes and filter out very small chunks
    if quality_checks:
        min_tokens = chunk_overlap
        max_tokens = chunk_size * 2
        filtered_documents = []
        small_chunks_count = 0

        for doc in documents:
            token_count = len(doc.content.split())
            if token_count < min_tokens:
                click.echo(
                    f"[warning] Chunk index={doc.metadata.get('chunk_index')} "
                    f"token_count={token_count} too small (<{min_tokens}) � will be filtered out",
                    err=True,
                )
                small_chunks_count += 1
                continue
            elif token_count > max_tokens:
                click.echo(
                    f"[warning] Chunk index={doc.metadata.get('chunk_index')} "
                    f"token_count={token_count} too large (>{max_tokens})",
                    err=True,
                )
            filtered_documents.append(doc)

        if small_chunks_count > 0:
            click.echo(f"[info] Filtered out {small_chunks_count} small chunks with fewer than {min_tokens} tokens")
            documents = filtered_documents

    # LLM-assisted summarization for multi-granularity indexing
    summary_docs: list[Document] = []
    if generate_summaries:
        click.echo(f"[info] Generating summaries for {len(documents)} chunks...")
        llm = get_openai_client(openai_api_key)

        # Process in smaller batches with timeouts and more detailed progress
        batch_size = 10
        total_docs = len(documents)
        successful_summaries = 0
        failed_summaries = 0

        for i in range(0, total_docs, batch_size):
            batch = documents[i:i+batch_size]
            click.echo(f"[info] Processing summary batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size} ({i+1}-{min(i+batch_size, total_docs)} of {total_docs})")

            for doc in batch:
                # Skip very short chunks
                if len(doc.content) < 200:
                    continue

                # Create a unique indicator for this doc
                doc_id = doc.metadata.get('chunk_index', 'unknown')
                try:
                    system_msg = {"role": "system", "content": "You are a helpful assistant."}
                    user_msg = {
                        "role": "user",
                        "content": f"Provide a concise (1-2 sentences) summary of the following text:\n\n{doc.content}",
                    }

                    # Use a timeout to prevent hanging
                    import threading
                    summary = None
                    error = None

                    def call_api():
                        nonlocal summary, error
                        try:
                            if hasattr(llm, "chat"):
                                resp = llm.chat.completions.create(
                                    model="gpt-4.1-nano", messages=[system_msg, user_msg], timeout=30
                                )
                                summary = resp.choices[0].message.content.strip()
                            else:
                                resp = llm.ChatCompletion.create(
                                    model="gpt-4.1-nano", messages=[system_msg, user_msg], request_timeout=30
                                )
                                summary = resp.choices[0].message.content.strip()  # type: ignore
                        except Exception as e:
                            error = str(e)

                    # Run API call with timeout
                    thread = threading.Thread(target=call_api)
                    thread.daemon = True
                    thread.start()
                    thread.join(30)  # Wait max 30 seconds

                    if thread.is_alive():
                        click.echo(f"[warning] Summary generation timed out for chunk {doc_id}", err=True)
                        failed_summaries += 1
                        continue

                    if error:
                        click.echo(f"[warning] Summary generation failed for chunk {doc_id}: {error}", err=True)
                        failed_summaries += 1
                        continue

                    if summary:
                        meta = doc.metadata.copy()
                        meta["is_summary"] = True
                        summary_docs.append(Document(content=summary, metadata=meta))
                        successful_summaries += 1
                    else:
                        click.echo(f"[warning] Empty summary generated for chunk {doc_id}", err=True)
                        failed_summaries += 1

                except Exception as e:
                    click.echo(f"[warning] Unexpected error in summary generation for chunk {doc_id}: {e}", err=True)
                    failed_summaries += 1

        # Report results
        click.echo(f"[info] Summary generation complete: {successful_summaries} successful, {failed_summaries} failed")

        # Make sure we continue with processing even if no summaries were generated
        if summary_docs:
            click.echo(f"[info] Adding {len(summary_docs)} summaries to the documents for indexing")
            documents.extend(summary_docs)
        else:
            click.echo("[warning] No summaries were generated, continuing with original chunks only", err=True)

    # ---------------------------------------------------------------------
    # Create collection if it does not exist
    # ---------------------------------------------------------------------

    VECTOR_SIZE = 3072  # text-embedding-3-large output dimension

    ensure_collection(client, collection, vector_size=VECTOR_SIZE, distance=distance)

    # ---------------------------------------------------------------------
    # Embed & upsert (with optional hierarchical embeddings)
    # ---------------------------------------------------------------------

    openai_client = get_openai_client(openai_api_key)

    if hierarchical_embeddings:
        try:
            # Import hierarchical embeddings module
            click.echo("[info] Using hierarchical embeddings at document, section, and chunk levels")
            try:
                # First try relative import
                import hierarchical_embeddings
            except ImportError:
                # Try absolute path if relative import fails
                import os
                import sys
                script_dir = os.path.dirname(os.path.abspath(__file__))
                if script_dir not in sys.path:
                    sys.path.insert(0, script_dir)
                import hierarchical_embeddings

            # Configure models
            hierarchical_embeddings.DOCUMENT_LEVEL_MODEL = doc_embedding_model
            hierarchical_embeddings.SECTION_LEVEL_MODEL = section_embedding_model
            hierarchical_embeddings.CHUNK_LEVEL_MODEL = chunk_embedding_model

            click.echo(f"[info] Using models: doc={doc_embedding_model}, section={section_embedding_model}, chunk={chunk_embedding_model}")

            # Convert our documents to the format expected by hierarchical_embeddings
            doc_list = [{"content": doc.content, "metadata": doc.metadata} for doc in documents]

            click.echo(f"[info] Creating hierarchical embeddings for {len(doc_list)} documents")

            # Create hierarchical embeddings
            hierarchical_data = hierarchical_embeddings.create_hierarchical_embeddings(
                doc_list,
                openai_client,
                batch_size=batch_size
            )

            # Prepare points for Qdrant
            points = hierarchical_embeddings.prepare_hierarchical_points_for_qdrant(hierarchical_data)

            # Statistics
            doc_count = len(hierarchical_data["documents"])
            section_count = len(hierarchical_data["sections"])
            chunk_count = len(hierarchical_data["chunks"])

            click.echo(f"[info] Created hierarchical structure with {doc_count} documents, {section_count} sections, and {chunk_count} chunks")

            # Upsert points into Qdrant
            from qdrant_client.http import models as rest

            # Batch points for upsert
            for i in range(0, len(points), batch_size):
                batch_points = points[i:i+batch_size]
                # Convert dict to PointStruct
                qdrant_points = [
                    rest.PointStruct(
                        id=p["id"],
                        vector=p["vector"],
                        payload=p["payload"]
                    ) for p in batch_points
                ]

                # Check if parallel parameter is supported
                import inspect
                client_upsert_params = inspect.signature(client.upsert).parameters
                if 'parallel' in client_upsert_params:
                    client.upsert(collection_name=collection, points=qdrant_points, parallel=parallel)
                else:
                    # Parallel kwarg not supported by this client version
                    client.upsert(collection_name=collection, points=qdrant_points)

                click.echo(f"[info] Upserted batch {i//batch_size + 1}/{(len(points) + batch_size - 1)//batch_size}")

            # Save hierarchical structure to a separate file
            structure_file = f"{collection}_hierarchical_structure.json"
            with open(structure_file, "w") as f:
                # Remove the large embeddings to keep file size reasonable
                save_data = {
                    "documents": [
                        {k: v for k, v in doc.items() if k != "embedding"}
                        for doc in hierarchical_data["documents"]
                    ],
                    "sections": [
                        {k: v for k, v in section.items() if k != "embedding"}
                        for section in hierarchical_data["sections"]
                    ],
                    "chunks": [
                        {k: v for k, v in chunk.items() if k != "embedding"}
                        for chunk in hierarchical_data["chunks"]
                    ],
                    "statistics": hierarchical_data["statistics"]
                }
                json.dump(save_data, f)

            click.echo(f"[info] Saved hierarchical structure to {structure_file}")

        except Exception as e:
            logger.error(f"Hierarchical embeddings failed: {e.__class__.__name__}: {e}")
            click.echo(f"[error] Hierarchical embeddings failed: {e.__class__.__name__}: {e}", err=True)
            
            # Add detailed error diagnostics
            INGESTION_DIAGNOSTICS["errors"].append({
                "timestamp": datetime.now().isoformat(),
                "component": "hierarchical_embeddings",
                "error_type": e.__class__.__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
                "fallback": "regular_embeddings"
            })
            
            # Log diagnostic information to help troubleshoot
            failure_details = {
                "error_type": e.__class__.__name__,
                "error_message": str(e),
                "error_traceback": traceback.format_exc(),
                "document_count": len(documents),
                "attempted_at": datetime.now().isoformat()
            }
            
            # Save diagnostics to file for later analysis
            try:
                with open(f"{collection}_embedding_failure_debug.json", "w") as f:
                    json.dump(failure_details, f, indent=2)
                logger.info(f"Saved embedding failure diagnostics to {collection}_embedding_failure_debug.json")
            except Exception as diag_e:
                logger.error(f"Failed to save embedding diagnostics: {diag_e}")
            
            click.echo("[info] Falling back to regular embeddings with enhanced error handling", err=True)
            
            # Fall back to regular embedding with retry logic
            try:
                embed_and_upsert(
                    client,
                    collection,
                    documents,
                    openai_client,
                    batch_size=batch_size,
                    deterministic_id=True,
                    parallel=parallel,
                    initial_workers=initial_workers,
                    min_workers=min_workers,
                    max_workers=max_workers,
                    dynamic_workers=dynamic_workers,
                )
            except Exception as embed_e:
                # Critical failure handling - try with smaller batch size and sequential processing
                logger.critical(f"Regular embedding failed: {embed_e}. Attempting with reduced batch size and sequential processing.")
                click.echo(f"[critical] Regular embedding failed: {embed_e}. Attempting with reduced batch size and sequential processing.", err=True)
                
                # Last attempt with minimal batch size
                reduced_batch = max(1, batch_size // 4)
                embed_and_upsert(
                    client,
                    collection,
                    documents,
                    openai_client,
                    batch_size=reduced_batch,
                    deterministic_id=True,
                    parallel=1,  # Sequential processing
                    initial_workers=min_workers,  # Start with minimum workers in fallback
                    min_workers=min_workers,
                    max_workers=max_workers,
                    dynamic_workers=dynamic_workers,
                )
    else:
        # Standard embedding approach
        embed_and_upsert(
            client,
            collection,
            documents,
            openai_client,
            batch_size=batch_size,
            deterministic_id=True,
            parallel=parallel,
            initial_workers=initial_workers,
            min_workers=min_workers,
            max_workers=max_workers,
            dynamic_workers=dynamic_workers,
        )

    click.secho(f"\n[success] Ingestion completed. Collection '{collection}' now holds the embeddings.", fg="green")

    # ---------------------------------------------------------------------
    # Run post-ingestion validation if enabled
    # ---------------------------------------------------------------------
    if validate_ingestion or run_test_queries:
        try:
            # Import validation module
            try:
                from ingest_validation import validate_ingestion as validate_fn
                from ingest_validation import run_test_queries as run_queries_fn
                click.echo("[info] Using ingest_validation module")
            except ImportError:
                # Try to import using absolute path
                import os
                import sys
                script_dir = os.path.dirname(os.path.abspath(__file__))
                if script_dir not in sys.path:
                    sys.path.insert(0, script_dir)
                from ingest_validation import validate_ingestion as validate_fn
                from ingest_validation import run_test_queries as run_queries_fn
                click.echo("[info] Using ingest_validation module from script directory")

            # Run validation
            if validate_ingestion:
                click.echo("\n[info] Running post-ingestion validation...")
                validation_summary = validate_fn(client, collection)

                # Print validation results
                click.secho(f"\nValidation Results:", fg="cyan")
                click.secho(f"Status: {validation_summary.overall_status}",
                           fg="green" if validation_summary.overall_status == "PASSED" else "yellow" if validation_summary.overall_status == "PARTIAL" else "red")
                click.echo(f"Tests Passed: {validation_summary.passed_tests}/{validation_summary.total_tests}")
                click.echo(f"Overall Score: {validation_summary.average_score:.2f}/1.00")

                # Print individual test results
                click.echo("\nTest Details:")
                for result in validation_summary.results:
                    status = "? PASS" if result.passed else "? FAIL"
                    click.echo(f"{status} [{result.test_name}] Score: {result.score:.2f} - {result.message}")

                # Optionally abort with non-zero exit code if validation failed
                if fail_on_validation_error and validation_summary.overall_status != "PASSED":
                    click.echo("[error] Validation did not pass and �fail-on-validation-error is set. Exiting.", err=True)
                    raise SystemExit(2)

                # Print critical issues
                if validation_summary.critical_issues:
                    click.secho("\nCritical Issues:", fg="red")
                    for issue in validation_summary.critical_issues:
                        click.echo(f"- {issue}")

            # Run test queries
            if run_test_queries:
                click.echo("\n[info] Running test queries to verify retrieval...")
                query_result = run_queries_fn(client, collection, openai_client)

                # Print query test results
                status = "? PASS" if query_result.passed else "? FAIL"
                click.secho(f"\nQuery Test: {status}", fg="green" if query_result.passed else "red")
                click.echo(f"Score: {query_result.score:.2f}")
                click.echo(f"Message: {query_result.message}")

                # Print detailed metrics
                details = query_result.details
                click.echo(f"Queries Run: {details.get('total_queries', 0)}")
                click.echo(f"Hit Rate: {details.get('hit_rate', 0):.2f}")
                if details.get('avg_position') is not None:
                    click.echo(f"Average Position: {details.get('avg_position', 0):.1f}")
                click.echo(f"Average Latency: {details.get('avg_latency', 0):.3f} seconds")

        except ImportError as e:
            click.echo(f"[warning] Validation failed to import: {e}", err=True)
            click.echo("[info] Install ingest_validation.py module to enable validation", err=True)
        except Exception as e:
            click.echo(f"[warning] Validation error: {e}", err=True)
            click.echo("[info] Validation failed but ingestion was successful", err=True)
    # ---------------------------------------------------------------------
    # Build BM25 index JSON mapping point IDs to chunk_text
    # ---------------------------------------------------------------------
    # Determine output path
    index_path = bm25_index or f"{collection}_bm25_index.json"
    click.echo(f"[info] Building BM25 index JSON at {index_path}")
    id2text: dict[str, str] = {}
    offset = None
    # Scroll through entire collection to collect chunk_text
    while True:
        records, offset = client.scroll(
            collection_name=collection,
            scroll_filter=None,
            limit=1000,
            offset=offset,
            with_payload=True,
        )
        if not records:
            break
        for rec in records:
            payload = getattr(rec, 'payload', {}) or {}
            text = payload.get("chunk_text")
            if isinstance(text, str) and text:
                id2text[rec.id] = text
        if offset is None:
            break
    try:
        with open(index_path, "w") as f:
            json.dump(id2text, f)
        click.secho(f"[success] BM25 index written to {index_path}", fg="green")
    except Exception as e:
        click.echo(f"[warning] Failed to write BM25 index: {e}", err=True)


if __name__ == "__main__":
    cli()