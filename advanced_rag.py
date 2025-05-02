#!/usr/bin/env python3

"""
Advanced RAG Features - Implementation of enhanced RAG capabilities.

This module contains implementations of advanced RAG techniques from RAGIMPROVE.md:
1. Semantic Document Chunking - Chunk documents based on semantic topic boundaries
2. Query Expansion - Expand queries with LLM rewrites for better retrieval
3. RAG Self-Evaluation - Evaluate RAG system quality for continuous improvement

Usage:
    from advanced_rag import semantic_chunk_text, expand_query, evaluate_rag_quality
    
    # For semantic chunking
    chunks = semantic_chunk_text("Long document text here...", max_chars=1000)
    
    # For query expansion
    expanded_queries = expand_query("original query", openai_client)
    
    # For RAG evaluation
    evaluation = evaluate_rag_quality("query", "retrieved chunks", "generated answer", openai_client)
"""

import re
import os
import json
import logging
from typing import List, Dict, Any, Optional, Union, Tuple, Callable

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    # Fast mode uses a simpler, much faster approach
    if fast_mode:
        return _fast_semantic_chunk(text, max_chars)
        
    # Original full semantic chunking method (slower but more precise)
    try:
        from transformers import pipeline
        import torch
        
        logger.info("Initializing semantic chunking with transformers")
        
        # Check for GPU availability
        device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Using device: {'GPU' if device == 0 else 'CPU'}")
        
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
            logger.warning("No paragraphs found in text")
            return [text] if text else []
            
        logger.info(f"Split text into {len(paragraphs)} paragraphs")
        
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
            
        logger.info(f"Initial chunking resulted in {len(initial_chunks)} chunks")
        
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
                
                logger.info(f"Chunk {i+1}: Topic '{topic}' (confidence: {confidence:.2f})")
                
                # Store chunk with topic information
                refined_chunks.append({
                    "text": chunk,
                    "topic": topic,
                    "confidence": confidence
                })
            except Exception as e:
                logger.error(f"Topic classification failed for chunk {i+1}: {e}")
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
            
        logger.info(f"Final semantic chunking produced {len(merged_chunks)} chunks")
        
        # Return just the text content
        return [chunk["text"] for chunk in merged_chunks]
        
    except Exception as e:
        logger.error(f"Semantic chunking failed: {e}")
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
    logger.info("Using fast semantic chunking")
    
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
        r'^#+\s+.+$',                  # Markdown heading
        r'^[A-Z][\w\s]+:$',            # Title with colon
        r'^[IVX]+\.\s+.+$',            # Roman numeral sections
        r'^\d+\.\d*\s+.+$',            # Numbered sections
        r'^[A-Z][A-Z\s]+$',            # ALL CAPS heading
        r'^={3,}$', r'^-{3,}$',        # Underline style headings
        r'^[A-Z][a-z]+\s+\d+:',        # "Section 1:" style$',                  # Markdown heading
        r'^[A-Z][\w\s]+:$',            # Title with colon
        r'^[IVX]+\.\s+.+$',            # Roman numeral sections
        r'^\d+\.\d*\s+.+$',            # Numbered sections
        r'^[A-Z][A-Z\s]+$',            # ALL CAPS heading
        r'^={3,}$', r'^-{3,}$',        # Underline style headings
        r'^[A-Z][a-z]+\s+\d+:',        # "Section 1:" style
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
    
    logger.info(f"Fast semantic chunking produced {len(chunks)} chunks")
    return chunks


def _simple_chunk_text(text: str, max_chars: int) -> List[str]:
    """
    Simple fallback chunking method that splits text by paragraphs.
    
    Args:
        text: The text to chunk
        max_chars: Maximum character length per chunk
        
    Returns:
        List of text chunks
    """
    logger.info("Using simple chunking as fallback")
    
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
    
    logger.info(f"Simple chunking produced {len(chunks)} chunks")
    return chunks


# -------------------------------------------------------------------------------
# 2. QUERY EXPANSION
# -------------------------------------------------------------------------------

def expand_query(query_text: str, openai_client: OpenAIClient, model: str = "gpt-4.1-mini", 
                 max_expansions: int = 4) -> List[str]:
    """
    Expand query with LLM rewrites for better retrieval.
    
    Args:
        query_text: The original query text to expand
        openai_client: OpenAI client (v0 or v1)
        model: The model to use for expansion
        max_expansions: Maximum number of expanded queries to return
        
    Returns:
        List of expanded queries (including original query)
    """
    if not query_text or not query_text.strip():
        logger.warning("Empty query provided for expansion")
        return [query_text] if query_text else []
        
    logger.info(f"Expanding query: '{query_text}'")
    
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
            logger.info(f"Using OpenAI v1 API with model {model}")
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
            logger.info(f"Using OpenAI v0 API with model {model}")
            response = openai_client.ChatCompletion.create(
                model=model,
                messages=[system_msg, user_msg],
                temperature=0.7,
                max_tokens=300
            )
            json_content = response.choices[0].message.content
        
        # Parse the JSON response
        try:
            # Remove any non-JSON text that might be in the response
            json_string = re.search(r'(\[\s*".*"\s*\]|\{\s*".*"\s*\})', json_content, re.DOTALL)
            if json_string:
                json_content = json_string.group(0)
                
            data = json.loads(json_content)
            
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
                
            logger.info(f"Generated {len(expanded_queries)} expanded queries")
            
            # Limit to max_expansions
            if len(expanded_queries) > max_expansions:
                expanded_queries = expanded_queries[:max_expansions-1] + [query_text]
                
            return expanded_queries
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse expansion JSON: {e}")
            logger.error(f"Raw response: {json_content}")
            return [query_text]
            
    except Exception as e:
        logger.error(f"Query expansion failed: {e}")
        return [query_text]  # Return original query on failure


# -------------------------------------------------------------------------------
# 3. RAG SELF-EVALUATION
# -------------------------------------------------------------------------------

def evaluate_rag_quality(query: str, retrieved_chunks: Union[str, List[str]], 
                       generated_answer: str, openai_client: OpenAIClient,
                       model: str = "gpt-4.1-nano") -> Dict[str, Any]:
    """
    Evaluate the quality of RAG responses for continuous improvement.
    
    Args:
        query: The user query
        retrieved_chunks: The chunks retrieved by the system (either a string or list of strings)
        generated_answer: The answer generated from the retrieved chunks
        openai_client: OpenAI client (v0 or v1)
        model: The model to use for evaluation
        
    Returns:
        Dictionary with evaluation scores and feedback
    """
    logger.info(f"Evaluating RAG quality for query: '{query}'")
    
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
            logger.info(f"Using OpenAI v1 API with model {model} for evaluation")
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
            logger.info(f"Using OpenAI v0 API with model {model} for evaluation")
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
            evaluation = json.loads(json_content)
            
            # Add metadata
            evaluation["meta"] = {
                "query": query,
                "timestamp": get_timestamp(),
                "chunks_count": len(retrieved_chunks) if isinstance(retrieved_chunks, list) else 1,
                "answer_length": len(generated_answer)
            }
            
            logger.info(f"RAG evaluation complete. Overall score: {evaluation.get('scores', {}).get('overall', 'N/A')}")
            return evaluation
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse evaluation JSON: {e}")
            logger.error(f"Raw response: {json_content}")
            return {
                "error": "Failed to parse evaluation response",
                "scores": {"overall": 0},
                "feedback": {"improvement_suggestions": ["Evaluation failed"]}
            }
            
    except Exception as e:
        logger.error(f"RAG evaluation failed: {e}")
        return {
            "error": str(e),
            "scores": {"overall": 0},
            "feedback": {"improvement_suggestions": ["Evaluation system error"]}
        }


# Helper function to get current timestamp
def get_timestamp():
    """Return current timestamp in string format"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")