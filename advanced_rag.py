#!/usr/bin/env python3

"""
Advanced RAG Features - Implementation of enhanced RAG capabilities.

This module contains implementations of advanced RAG techniques from RAGIMPROVE.md:
1. Semantic Document Chunking - Chunk documents based on semantic topic boundaries
2. Query Expansion - Expand queries with LLM rewrites for better retrieval
3. RAG Self-Evaluation - Evaluate RAG system quality for continuous improvement
4. Contextual Compression - Focus retrieved documents on query-relevant parts
"""

import re
import json
import logging
import time
from typing import List, Dict, Any, Union

# Setup logging - use a thread-safe approach
import threading

# Lock for thread-safe logger initialization
_logger_lock = threading.RLock()

# Thread-local storage for loggers
_thread_local = threading.local()

def get_logger():
    """Get a thread-local logger instance to ensure thread safety."""
    if not hasattr(_thread_local, 'logger'):
        with _logger_lock:
            # Configure root logger only once
            if not logging.getLogger().handlers:
                logging.basicConfig(level=logging.INFO, 
                                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            # Create thread-local logger instance
            _thread_local.logger = logging.getLogger(f"{__name__}.{threading.get_ident()}")
    
    return _thread_local.logger

# For backwards compatibility - replace direct logger usage
logger = get_logger()

# Define types
Document = Dict[str, Any]  # Will include at least 'content' and 'metadata'
OpenAIClient = Any  # Type for OpenAI client (either v0 or v1)

# -------------------------------------------------------------------------------
# 1. SEMANTIC DOCUMENT CHUNKING
# -------------------------------------------------------------------------------

def semantic_chunk_text(text: str, max_chars: int = 1000, fast_mode: bool = True) -> List[str]:
    """
    Chunk text based on semantic topic boundaries.
    
    Args:
        text: The text to chunk
        max_chars: Maximum character length per chunk
        fast_mode: Use faster chunking method (default: True)
        
    Returns:
        List of semantically chunked text segments
    """
    # Get thread-local logger
    thread_logger = get_logger()
    
    # Fast mode uses a simpler, much faster approach
    if fast_mode:
        return _fast_semantic_chunk(text, max_chars)
        
    # Original full semantic chunking method (slower but more precise)
    try:
        from transformers import pipeline
        import torch
        
        thread_logger.info("Initializing semantic chunking with transformers")
        
        # Check for GPU availability
        device = 0 if torch.cuda.is_available() else -1
        thread_logger.info(f"Using device: {'GPU' if device == 0 else 'CPU'}")
        
        # Initialize zero-shot classification pipeline
        classifier = pipeline(
            "zero-shot-classification", 
            model="facebook/bart-large-mnli",
            device=device
        )
        
        # Split text into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        if not paragraphs:
            thread_logger.warning("No paragraphs found in text")
            return [text] if text else []
            
        thread_logger.info(f"Split text into {len(paragraphs)} paragraphs")
        
        # First pass: Group paragraphs by max_chars
        initial_chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_len = len(para)
            if current_length + para_len > max_chars and current_chunk:
                initial_chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_length = para_len
            else:
                current_chunk.append(para)
                current_length += para_len
                
        # Add the last chunk if it exists
        if current_chunk:
            initial_chunks.append("\n\n".join(current_chunk))
            
        thread_logger.info(f"Initial chunking resulted in {len(initial_chunks)} chunks")
        
        # If only one chunk, just return it
        if len(initial_chunks) <= 1:
            return initial_chunks
            
        # Second pass: Detect topics for chunks
        refined_chunks = []
        
        # Define possible topics
        candidate_topics = [
            "introduction", "background", "methodology", "results", 
            "discussion", "conclusion", "data analysis", "requirements",
            "implementation", "example", "reference", "appendix",
            "use case", "testing", "evaluation", "future work"
        ]
        
        # Process each chunk
        for i, chunk in enumerate(initial_chunks):
            if len(chunk) < 50:  # Very short chunks don't need classification
                refined_chunks.append({"text": chunk, "topic": "short_content"})
                continue
                
            # Truncate very long chunks for classification
            classification_text = chunk[:2000] if len(chunk) > 2000 else chunk
                
            try:
                # Classify chunk
                result = classifier(classification_text, candidate_topics)
                topic = result["labels"][0] 
                confidence = result["scores"][0]
                
                thread_logger.info(f"Chunk {i+1}: Topic '{topic}' (confidence: {confidence:.2f})")
                
                # Store chunk with topic information
                refined_chunks.append({
                    "text": chunk,
                    "topic": topic,
                    "confidence": confidence
                })
            except Exception as e:
                thread_logger.error(f"Topic classification failed for chunk {i+1}: {e}")
                refined_chunks.append({"text": chunk, "topic": "unknown"})
        
        # Optionally merge chunks with same topic (if they're small enough)
        merged_chunks = []
        current_merged = None
        
        for chunk in refined_chunks:
            if current_merged is None:
                current_merged = chunk.copy()
            elif (chunk["topic"] == current_merged["topic"] and 
                  len(current_merged["text"]) + len(chunk["text"]) <= max_chars * 1.2):
                # Merge chunks with same topic if they don't exceed max_chars by too much
                current_merged["text"] += "\n\n" + chunk["text"]
            else:
                merged_chunks.append(current_merged)
                current_merged = chunk.copy()
                
        if current_merged:
            merged_chunks.append(current_merged)
            
        thread_logger.info(f"Final semantic chunking produced {len(merged_chunks)} chunks")
        
        # Return just the text content
        return [chunk["text"] for chunk in merged_chunks]
        
    except Exception as e:
        thread_logger.error(f"Semantic chunking failed: {e}")
        # Fallback to fast chunking
        return _fast_semantic_chunk(text, max_chars)


def _fast_semantic_chunk(text: str, max_chars: int = 1000) -> List[str]:
    """
    A much faster semantic chunking approach using heuristics.
    
    This method breaks text on paragraph boundaries and clear semantic markers
    like headings, but doesn't use a large language model for classification.
    
    Args:
        text: The text to chunk
        max_chars: Maximum character length per chunk
        
    Returns:
        List of semantically chunked text segments
    """
    # Get thread-local logger
    thread_logger = get_logger()
    
    thread_logger.info("Using fast semantic chunking")
    
    # Quick check for empty text
    if not text or len(text.strip()) == 0:
        return []
    
    # Split text into paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    if not paragraphs:
        return [text]
    
    # Identify headings and section boundaries
    # Common heading patterns in markdown, restructured text and plain text
    heading_patterns = [
        r'^#+\s+.+$',                # Markdown heading
        r'^[A-Z][\w\s]+:$',          # Title with colon
        r'^[IVX]+\.\s+.+$',          # Roman numeral sections
        r'^\d+\.\d*\s+.+$',        # Numbered sections
        r'^[A-Z][A-Z\s]+$',          # ALL CAPS heading
        r'^={3,}$',                  # Underline style heading (===)
        r'^-{3,}$',                  # Underline style heading (---)
        r'^[A-Z][a-z]+\s+\d+:',      # "Section 1:" style
    ]
    
    heading_pattern = re.compile('|'.join(f'({pattern})' for pattern in heading_patterns), re.MULTILINE)
    
    # Identify paragraphs that are likely headings
    is_heading = [bool(heading_pattern.match(p)) for p in paragraphs]
    
    # Now build chunks based on headings and max_chars
    chunks = []
    current_chunk = []
    current_length = 0
    
    for i, para in enumerate(paragraphs):
        para_len = len(para)
        is_current_heading = is_heading[i]
        
        # Start a new chunk on heading or max_chars exceeded
        if (is_current_heading and current_chunk) or (current_length + para_len > max_chars and current_chunk):
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_length = para_len
        else:
            current_chunk.append(para)
            current_length += para_len + 4  # +4 for the paragraph separator
    
    # Add the last chunk
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    
    thread_logger.info(f"Fast semantic chunking produced {len(chunks)} chunks")
    return chunks


def _with_logger(func):
    """Decorator to inject thread-local logger into function."""
    def wrapper(*args, **kwargs):
        # Add thread_logger to kwargs
        kwargs['thread_logger'] = get_logger()
        return func(*args, **kwargs)
    return wrapper


@_with_logger
def _simple_chunk_text(text: str, max_chars: int, thread_logger=None) -> List[str]:
    """
    Simple fallback chunking method that splits text by paragraphs.
    
    Args:
        text: The text to chunk
        max_chars: Maximum character length per chunk
        
    Returns:
        List of text chunks
    """
    thread_logger.info("Using simple chunking as fallback")
    
    # Split text into paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    if not paragraphs:
        return [text] if text else []
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para_len = len(para)
        
        # If this paragraph alone exceeds max length, split it by sentences
        if para_len > max_chars:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_length = 0
                
            # Split long paragraph by sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            sentence_chunk = []
            sentence_length = 0
            
            for sentence in sentences:
                if sentence_length + len(sentence) > max_chars and sentence_chunk:
                    chunks.append(" ".join(sentence_chunk))
                    sentence_chunk = [sentence]
                    sentence_length = len(sentence)
                else:
                    sentence_chunk.append(sentence)
                    sentence_length += len(sentence)
                    
            if sentence_chunk:
                chunks.append(" ".join(sentence_chunk))
        
        # Normal case - add paragraph to current chunk if it fits
        elif current_length + para_len <= max_chars:
            current_chunk.append(para)
            current_length += para_len
        else:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_length = para_len
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    
    thread_logger.info(f"Simple chunking produced {len(chunks)} chunks")
    return chunks


# -------------------------------------------------------------------------------
# 2. QUERY EXPANSION
# -------------------------------------------------------------------------------

@_with_logger
def expand_query(query_text: str, openai_client: OpenAIClient, model: str = "gpt-4.1-mini", 
                 max_expansions: int = 4, thread_logger=None) -> List[str]:
    """
    Expand query with LLM rewrites for better retrieval.
    
    Args:
        query_text: The original query text to expand
        openai_client: OpenAI client (v0 or v1)
        model: The model to use for expansion
        max_expansions: Maximum number of expanded queries to return
        thread_logger: Thread-local logger (injected by decorator)
        
    Returns:
        List of expanded queries (including original query)
    """
    if not query_text or not query_text.strip():
        thread_logger.warning("Empty query provided for expansion")
        return [query_text] if query_text else []
        
    thread_logger.info(f"Expanding query: '{query_text}'")
    
    system_msg = {
        "role": "system", 
        "content": f"""You are a search query expansion expert. Generate {max_expansions} alternative versions 
of the user's search query, focusing on different aspects or using different terminology.
Each alternative should capture the same information need but use different wording.
DO NOT explain your process. Return ONLY a JSON array of strings with NO other text.
Ensure each query is a complete, well-formed question or request.
"""
    }
    
    user_msg = {
        "role": "user",
        "content": f"Original query: {query_text}"
    }
    
    try:
        # Detect OpenAI client version and use appropriate call
        if hasattr(openai_client, "chat") and hasattr(openai_client.chat, "completions"):
            # OpenAI Python v1.x
            thread_logger.info(f"Using OpenAI v1 API with model {model}")
            response = openai_client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[system_msg, user_msg],
                temperature=0.7,
                max_tokens=300
            )
            json_content = response.choices[0].message.content
        else:
            # OpenAI Python v0.x
            thread_logger.info(f"Using OpenAI v0 API with model {model}")
            response = openai_client.ChatCompletion.create(
                model=model,
                messages=[system_msg, user_msg],
                temperature=0.7,
                max_tokens=300
            )
            json_content = response.choices[0].message.content
        
        # Parse the JSON response
        try:
            # Extract JSON safely from potentially mixed text
            cleaned_json = extract_json_safely(json_content)
            
            # Parse the cleaned JSON content
            data = json.loads(cleaned_json)
            
            # Handle different possible JSON structures
            if isinstance(data, list):
                expanded_queries = data
            elif isinstance(data, dict):
                if "queries" in data:
                    expanded_queries = data["queries"]
                else:
                    # Try to get first list value in the dict
                    for key, value in data.items():
                        if isinstance(value, list):
                            expanded_queries = value
                            break
                    else:
                        expanded_queries = list(data.values())
            else:
                raise ValueError(f"Unexpected JSON structure: {type(data)}")
                
            # Filter out any non-string values
            expanded_queries = [q for q in expanded_queries if isinstance(q, str)]
            
            # Ensure we return at least the original query
            if not expanded_queries:
                expanded_queries = [query_text]
            elif query_text not in expanded_queries:
                expanded_queries.append(query_text)
                
            thread_logger.info(f"Generated {len(expanded_queries)} expanded queries")
            
            # Limit to max_expansions
            if len(expanded_queries) > max_expansions:
                expanded_queries = expanded_queries[:max_expansions-1] + [query_text]
                
            return expanded_queries
            
        except (json.JSONDecodeError, ValueError) as e:
            thread_logger.error(f"Failed to parse expansion JSON: {e}")
            thread_logger.error(f"Raw response: {json_content}")
            return [query_text]
            
    except Exception as e:
        thread_logger.error(f"Query expansion failed: {e}")
        return [query_text]  # Return original query on failure


# -------------------------------------------------------------------------------
# 3. RAG SELF-EVALUATION
# -------------------------------------------------------------------------------

@_with_logger
def evaluate_rag_quality(query: str, retrieved_chunks: Union[str, List[str]], 
                       generated_answer: str, openai_client: OpenAIClient,
                       model: str = "gpt-4.1-nano", thread_logger=None) -> Dict[str, Any]:
    """
    Evaluate the quality of RAG responses for continuous improvement.
    
    Args:
        query: The user query
        retrieved_chunks: The chunks retrieved by the system (either a string or list of strings)
        generated_answer: The answer generated from the retrieved chunks
        openai_client: OpenAI client (v0 or v1)
        model: The model to use for evaluation
        thread_logger: Thread-local logger (injected by decorator)
        
    Returns:
        Dictionary with evaluation scores and feedback
    """
    thread_logger.info(f"Evaluating RAG quality for query: '{query}'")
    
    # Format chunks if they're passed as a list
    if isinstance(retrieved_chunks, list):
        formatted_chunks = "\n\n---\n\n".join(retrieved_chunks)
    else:
        formatted_chunks = retrieved_chunks
    
    system_prompt = """You are an expert evaluator for Retrieval Augmented Generation (RAG) systems.
Your job is to assess the quality of retrieved chunks and the generated answer for a user query.

Score each of the following criteria on a scale of 1-10:

1. Relevance (1-10): How relevant are the retrieved chunks to the query?
   - 1: Completely irrelevant
   - 5: Somewhat relevant but missing key information
   - 10: Highly relevant, containing all needed information

2. Completeness (1-10): Do the retrieved chunks contain all information needed to answer?
   - 1: Missing critical information
   - 5: Contains partial information
   - 10: Contains all necessary information

3. Accuracy (1-10): Is the generated answer accurate based on the chunks?
   - 1: Contains major factual errors or contradictions
   - 5: Mostly accurate with minor errors
   - 10: Completely accurate

4. Hallucination (1-10): Does the answer contain information not found in the chunks?
   - 1: Severe hallucination (completely made up information)
   - 5: Some unfounded statements
   - 10: No hallucination (everything is supported by chunks)

5. Coherence (1-10): Is the answer well-structured, coherent and easy to understand?
   - 1: Incoherent or confusing
   - 5: Somewhat structured but could be clearer
   - 10: Well-structured and very clear

RETURN YOUR EVALUATION AS JSON with the following structure:
{
  "scores": {
    "relevance": X,
    "completeness": X,
    "accuracy": X,
    "hallucination": X,
    "coherence": X,
    "overall": X  // Weighted average, with hallucination weighted 2x
  },
  "feedback": {
    "strengths": ["Strength 1", "Strength 2"],
    "weaknesses": ["Weakness 1", "Weakness 2"],
    "improvement_suggestions": ["Suggestion 1", "Suggestion 2"]
  }
}
Do not include any text outside the JSON structure."""

    eval_prompt = f"""Evaluate this RAG system response:

QUERY:
{query}

RETRIEVED CHUNKS:
{formatted_chunks}

GENERATED ANSWER:
{generated_answer}

Please provide your evaluation following the criteria in the system prompt."""

    try:
        # Detect OpenAI client version and use appropriate call
        if hasattr(openai_client, "chat") and hasattr(openai_client.chat, "completions"):
            # OpenAI Python v1.x
            thread_logger.info(f"Using OpenAI v1 API with model {model} for evaluation")
            response = openai_client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": eval_prompt}
                ],
                temperature=0.2,  # Lower temperature for more consistent evaluations
                max_tokens=800
            )
            json_content = response.choices[0].message.content
        else:
            # OpenAI Python v0.x
            thread_logger.info(f"Using OpenAI v0 API with model {model} for evaluation")
            response = openai_client.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": eval_prompt}
                ],
                temperature=0.2,
                max_tokens=800
            )
            json_content = response.choices[0].message.content
        
        # Parse the evaluation JSON
        try:
            # Extract JSON safely from potentially mixed text
            cleaned_json = extract_json_safely(json_content)
            
            # Parse the cleaned JSON
            evaluation = json.loads(cleaned_json)
            
            # Add metadata
            evaluation["meta"] = {
                "query": query,
                "timestamp": get_timestamp(),
                "chunks_count": len(retrieved_chunks) if isinstance(retrieved_chunks, list) else 1,
                "answer_length": len(generated_answer)
            }
            
            thread_logger.info(f"RAG evaluation complete. Overall score: {evaluation.get('scores', {}).get('overall', 'N/A')}")
            return evaluation
            
        except json.JSONDecodeError as e:
            thread_logger.error(f"Failed to parse evaluation JSON: {e}")
            thread_logger.error(f"Raw response: {json_content}")
            return {
                "error": "Failed to parse evaluation response",
                "scores": {"overall": 0},
                "feedback": {"improvement_suggestions": ["Evaluation failed"]}
            }
            
    except Exception as e:
        thread_logger.error(f"RAG evaluation failed: {e}")
        return {
            "error": str(e),
            "scores": {"overall": 0},
            "feedback": {"improvement_suggestions": ["Evaluation system error"]}
        }


# -------------------------------------------------------------------------------
# 4. CONTEXTUAL COMPRESSION
# -------------------------------------------------------------------------------

@_with_logger
def contextual_compression(query: str, docs: List[Any], openai_client: OpenAIClient, 
                         model: str = "gpt-4.1-nano", thread_logger=None) -> List[Any]:
    """
    Focus retrieved documents on query-relevant parts to reduce hallucination.
    
    Args:
        query: The user query
        docs: List of retrieved documents (e.g. Qdrant search results)
        openai_client: OpenAI client (v0 or v1)
        model: The model to use for compression
        thread_logger: Thread-local logger (injected by decorator)
        
    Returns:
        List of documents with compressed text added
    """
    thread_logger.info(f"Performing contextual compression for query: '{query}'")
    
    if not docs:
        thread_logger.warning("No documents provided for compression")
        return docs
    
    compressed_docs = []
    start_time = time.time()
    
    # Determine if we have new OpenAI client or legacy
    is_v1 = hasattr(openai_client, "chat") and hasattr(openai_client.chat, "completions")
    
    for i, doc in enumerate(docs):
        # Extract document text
        payload = getattr(doc, 'payload', {}) or {}
        text = payload.get("chunk_text", "")
        
        if not text:
            thread_logger.warning(f"Document {i} has no chunk_text, skipping compression")
            compressed_docs.append(doc)
            continue
            
        # Skip very short texts that don't need compression
        if len(text) < 200:
            thread_logger.info(f"Document {i} is too short for compression ({len(text)} chars)")
            compressed_docs.append(doc)
            continue
            
        thread_logger.info(f"Compressing document {i} ({len(text)} chars)")
        
        # Create system and user messages
        system_msg = {
            "role": "system", 
            "content": """You are a document compression expert. Your task is to extract the parts of the 
document that are most relevant to the query, removing irrelevant information while 
preserving all query-relevant facts, context, and details.

Extract ONLY the sentences and paragraphs that directly relate to answering the query.
Maintain the original wording of the extracted parts. 
DO NOT add any new information, summaries or explanations."""
        }
        
        user_msg = {
            "role": "user",
            "content": f"""Query: {query}

Document:
{text}

Extract only the parts of this document that are most relevant to the query above.
Preserve the original wording of the important parts."""
        }
        
        try:
            # Call OpenAI API based on client version
            if is_v1:  # OpenAI v1 client
                response = openai_client.chat.completions.create(
                    model=model,
                    messages=[system_msg, user_msg],
                    temperature=0.1,  # Low temperature for more consistent results
                    max_tokens=min(len(text.split()) // 2, 1000),  # Limit token length to force compression
                    timeout=30  # 30 second timeout
                )
                compressed_text = response.choices[0].message.content.strip()
            else:  # OpenAI v0 client
                response = openai_client.ChatCompletion.create(
                    model=model,
                    messages=[system_msg, user_msg],
                    temperature=0.1,
                    max_tokens=min(len(text.split()) // 2, 1000),
                    request_timeout=30
                )
                compressed_text = response.choices[0].message.content.strip()  # type: ignore
                
            # Calculate compression ratio
            original_len = len(text)
            compressed_len = len(compressed_text)
            compression_ratio = compressed_len / original_len if original_len > 0 else 1.0
            
            thread_logger.info(f"Compressed doc {i}: {original_len} → {compressed_len} chars ({compression_ratio:.2%})")
            
            # Check for empty or too-short compression result
            if len(compressed_text) < 50 or compression_ratio < 0.1:
                thread_logger.warning(f"Compression too aggressive for doc {i}, using original")
                compressed_text = text
            
            # Save compressed text in document
            # We need to be careful here - Qdrant points aren't directly mutable
            # We'll create a new object with the same attributes plus our compressed text
            doc_dict = {}
            
            # Copy all attributes from the original document
            for attr in dir(doc):
                if not attr.startswith('_') and not callable(getattr(doc, attr)):
                    doc_dict[attr] = getattr(doc, attr)
            
            # Create a mutable copy of the payload to modify
            new_payload = payload.copy()
            
            # Add compressed text to payload
            new_payload["compressed_text"] = compressed_text
            new_payload["compression_ratio"] = compression_ratio
            
            # Create a new document object with the updated payload
            from types import SimpleNamespace
            new_doc = SimpleNamespace(**doc_dict)
            new_doc.payload = new_payload
            
            compressed_docs.append(new_doc)
            
        except Exception as e:
            thread_logger.error(f"Compression failed for document {i}: {e}")
            # Keep original document if compression fails
            compressed_docs.append(doc)
    
    elapsed_time = time.time() - start_time
    thread_logger.info(f"Contextual compression complete for {len(docs)} documents in {elapsed_time:.2f} seconds")
    
    return compressed_docs


def get_compressed_text(doc: Any) -> str:
    """
    Helper function to get the compressed text from a document,
    falling back to chunk_text if compressed_text is not available.
    
    Args:
        doc: Document object with payload
        
    Returns:
        Compressed text or original chunk text
    """
    payload = getattr(doc, 'payload', {}) or {}
    return payload.get("compressed_text", payload.get("chunk_text", ""))


# Helper function to get current timestamp
def get_timestamp():
    """Return current timestamp in string format"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def extract_json_safely(text: str) -> str:
    """
    Safely extract JSON content from a text string, handling various LLM response formats.
    
    Args:
        text: Text that may contain JSON (potentially with other text before/after)
        
    Returns:
        Cleaned text containing only the JSON portion
    """
    # First, try to find JSON enclosed in triple backticks
    if '```' in text:
        # Extract content between code blocks with json or JSON label
        match = re.search(r'```(?:json|JSON)?\n([\s\S]*?)\n```', text)
        if match:
            return match.group(1).strip()
            
    # Next, try to find JSON enclosed in single backticks
    if '`' in text:
        match = re.search(r'`([\s\S]*?)`', text)
        if match:
            candidate = match.group(1).strip()
            try:
                # Test if valid JSON
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass
    
    # Try to extract JSON by looking for brackets/braces patterns
    # Find a JSON object
    obj_match = re.search(r'(\{[\s\S]*\})', text)
    if obj_match:
        # Validate this is actually JSON by trying a stricter pattern
        strict_obj = re.search(r'(\{(?:[^{}]|"(?:\\.|[^"\\])*"|\{(?:[^{}]|"(?:\\.|[^"\\])*")*\})*\})', obj_match.group(0))
        if strict_obj:
            try:
                json.loads(strict_obj.group(0))
                return strict_obj.group(0)
            except json.JSONDecodeError:
                pass
    
    # Find a JSON array
    arr_match = re.search(r'(\[[\s\S]*\])', text)
    if arr_match:
        # Validate this is actually JSON by trying a stricter pattern
        strict_arr = re.search(r'(\[(?:[^\[\]]|"(?:\\.|[^"\\])*"|\[(?:[^\[\]]|"(?:\\.|[^"\\])*")*\])*\])', arr_match.group(0))
        if strict_arr:
            try:
                json.loads(strict_arr.group(0))
                return strict_arr.group(0)
            except json.JSONDecodeError:
                pass
    
    # If we couldn't find valid JSON, return the original text
    # It will still fail later, but at least we tried to clean it
    return text