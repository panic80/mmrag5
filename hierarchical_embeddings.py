#!/usr/bin/env python3

"""
Hierarchical Embeddings for RAG Systems

This module provides hierarchical embedding capabilities for Retrieval-Augmented Generation (RAG) systems.
Hierarchical embeddings create representations at multiple granularities (document, section, chunk),
improving retrieval precision and recall by matching at the appropriate level of detail.

Features:
- Multi-level document hierarchy (document → section → chunk)
- Specialized embedding functions for each granularity level
- Embedding aggregation and parent-child relationships
- Structural metadata enrichment for better context

Usage:
    from hierarchical_embeddings import create_hierarchical_embeddings
    
    # Create hierarchical embeddings during ingestion
    hierarchical_docs = create_hierarchical_embeddings(documents, openai_client)
"""

import re
import logging
import json
from typing import List, Dict, Any, Optional, Union, Tuple
import time
from datetime import datetime
import uuid
import numpy as np
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define common types
Document = Dict[str, Any]  # Will include at least 'content' and 'metadata'
OpenAIClient = Any  # Type for OpenAI client (either v0 or v1)

# Constants
DOCUMENT_LEVEL_MODEL = "text-embedding-3-large"
SECTION_LEVEL_MODEL = "text-embedding-3-large"
CHUNK_LEVEL_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072  # For text-embedding-3-large

def is_openai_v1(client: OpenAIClient) -> bool:
    """Determine if we're using OpenAI Python v1.x client."""
    return (hasattr(client, "embeddings") and hasattr(client.embeddings, "create"))

def get_embeddings(client: OpenAIClient, texts: List[str], model: str = "text-embedding-3-large", 
                   timeout: int = 60) -> List[List[float]]:
    """
    Get embeddings for multiple texts using OpenAI's API.
    
    Args:
        client: OpenAI client (v0 or v1)
        texts: List of text strings to embed
        model: Embedding model name
        timeout: API timeout in seconds
        
    Returns:
        List of embedding vectors (each vector is a list of floats)
    """
    if not texts:
        logger.warning("Empty texts list provided for embedding")
        return []
        
    logger.info(f"Getting embeddings for {len(texts)} texts using model {model}")
    
    try:
        if is_openai_v1(client):
            # OpenAI Python v1.x
            try:
                embeddings_response = client.embeddings.create(
                    model=model,
                    input=texts,
                    timeout=timeout,
                )
                return [record.embedding for record in embeddings_response.data]
            except TypeError as exc:
                # Retry without timeout for compatibility with test clients
                if "timeout" in str(exc):
                    embeddings_response = client.embeddings.create(
                        model=model,
                        input=texts,
                    )
                    return [record.embedding for record in embeddings_response.data]
                else:
                    raise
        else:
            # OpenAI Python v0.x
            embeddings_response = client.Embedding.create(
                model=model, 
                input=texts,
                request_timeout=timeout
            )
            return [record["embedding"] for record in embeddings_response["data"]]
    except Exception as e:
        logger.error(f"Embedding API call failed: {e}")
        raise

def create_document_level_embedding(document_texts: List[str], client: OpenAIClient) -> List[List[float]]:
    """
    Create document-level embeddings for one or more documents.
    
    Document-level embeddings represent the overall semantics of a document
    and are used for high-level retrieval.
    
    Args:
        document_texts: List of full document texts
        client: OpenAI client
        
    Returns:
        List of embedding vectors for documents
    """
    logger.info(f"Creating document-level embeddings for {len(document_texts)} documents")
    
    # For document-level, we want to focus on overall document theme
    # We'll use a processed version that emphasizes key information
    processed_texts = []
    
    for doc_text in document_texts:
        # Extract the first ~1000 chars for title, intro, abstract
        intro = doc_text[:1000]
        
        # Extract section headings if present (common documentation pattern)
        headers = re.findall(r'^#{1,3}\s+(.+)$|^([A-Z][A-Za-z0-9\s]+:)$', doc_text, re.MULTILINE)
        headers_text = " ".join([h[0] or h[1] for h in headers if h[0] or h[1]])
        
        # Look for conclusion/summary sections
        conclusion_match = re.search(r'(?:Conclusion|Summary).*?$(.*?)(?:^#|\Z)', 
                                    doc_text, re.MULTILINE | re.DOTALL)
        conclusion = conclusion_match.group(1).strip() if conclusion_match else ""
        conclusion = conclusion[:1000] if conclusion else ""
        
        # Combine important parts for document-level embedding
        important_parts = f"{intro}\n\n{headers_text}\n\n{conclusion}"
        processed_texts.append(important_parts)
    
    # Get embeddings using document-level model
    return get_embeddings(client, processed_texts, model=DOCUMENT_LEVEL_MODEL)

def create_section_level_embeddings(section_texts: List[str], client: OpenAIClient) -> List[List[float]]:
    """
    Create section-level embeddings for sections within documents.
    
    Section-level embeddings represent cohesive subtopics and are
    an intermediate granularity between documents and chunks.
    
    Args:
        section_texts: List of section texts
        client: OpenAI client
        
    Returns:
        List of embedding vectors for sections
    """
    logger.info(f"Creating section-level embeddings for {len(section_texts)} sections")
    
    # For section-level embeddings, we'll focus on the content with emphasis on section structure
    processed_texts = []
    
    for section in section_texts:
        # Extract section heading if present
        header_match = re.match(r'^(#+\s+.+|[A-Z][A-Za-z0-9\s]+:)$', section, re.MULTILINE)
        header = header_match.group(1) if header_match else ""
        
        # Process section text - we emphasize the heading and first paragraph
        first_para = section.split('\n\n', 1)[0] if '\n\n' in section else section
        
        # Combine important parts with emphasis on structure
        processed_text = f"{header}\n{first_para}\n{section}"
        processed_texts.append(processed_text)
    
    # Get embeddings using section-level model
    return get_embeddings(client, processed_texts, model=SECTION_LEVEL_MODEL)

def group_chunks_into_sections(chunks: List[Document]) -> List[Dict[str, Any]]:
    """
    Group individual chunks into logical sections based on content and metadata.
    
    Args:
        chunks: List of document chunks
        
    Returns:
        List of sections, each containing the section text and member chunks
    """
    logger.info(f"Grouping {len(chunks)} chunks into sections")
    
    # Sort chunks by any available sequential indicators
    sorted_chunks = sorted(chunks, key=lambda x: (
        x.get('metadata', {}).get('source', ''),
        x.get('metadata', {}).get('chunk_index', 9999)
    ))
    
    sections = []
    current_section = {
        "chunks": [],
        "text": "",
        "metadata": {}
    }
    
    for i, chunk in enumerate(sorted_chunks):
        content = chunk.get('content', '')
        metadata = chunk.get('metadata', {})
        
        # Detect if this chunk starts a new section
        is_new_section = False
        
        # Check if chunk starts with a heading
        if re.match(r'^#+\s+|^[A-Z][A-Za-z0-9\s]+:$', content.strip(), re.MULTILINE):
            is_new_section = True
        
        # Check if chunk has special section metadata
        if metadata.get('is_section_start') or metadata.get('structure', {}).get('section_count', 0) > 0:
            is_new_section = True
            
        # Start a new section or add to current
        if is_new_section and current_section["chunks"]:
            # Finalize the current section before starting a new one
            current_section["text"] = "\n\n".join([c.get('content', '') for c in current_section["chunks"]])
            
            # Merge metadata from chunks, preferring values from first chunk
            combined_meta = {}
            for c in current_section["chunks"]:
                combined_meta.update(c.get('metadata', {}))
            
            current_section["metadata"] = combined_meta
            sections.append(current_section)
            
            # Start a new section
            current_section = {
                "chunks": [chunk],
                "text": "",
                "metadata": {}
            }
        else:
            # Add to current section
            current_section["chunks"].append(chunk)
    
    # Add the last section if it has chunks
    if current_section["chunks"]:
        current_section["text"] = "\n\n".join([c.get('content', '') for c in current_section["chunks"]])
        
        # Merge metadata from chunks
        combined_meta = {}
        for c in current_section["chunks"]:
            combined_meta.update(c.get('metadata', {}))
        
        current_section["metadata"] = combined_meta
        sections.append(current_section)
    
    logger.info(f"Created {len(sections)} sections from {len(chunks)} chunks")
    return sections

def group_sections_into_documents(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group sections into parent documents based on source and metadata.
    
    Args:
        sections: List of sections
        
    Returns:
        List of documents, each containing the full text and member sections
    """
    logger.info(f"Grouping {len(sections)} sections into documents")
    
    # Group by source
    docs_by_source = defaultdict(list)
    
    for section in sections:
        source = section.get('metadata', {}).get('source', 'unknown')
        docs_by_source[source].append(section)
    
    # Create document objects
    documents = []
    
    for source, source_sections in docs_by_source.items():
        # Sort sections by any sequential indicators
        sorted_sections = sorted(source_sections, 
                               key=lambda x: min(c.get('metadata', {}).get('chunk_index', 9999) 
                                               for c in x['chunks']))
        
        # Combine sections into a document
        document = {
            "sections": sorted_sections,
            "text": "\n\n".join(s["text"] for s in sorted_sections),
            "metadata": {
                "source": source,
                "section_count": len(sorted_sections),
                "chunk_count": sum(len(s["chunks"]) for s in sorted_sections)
            }
        }
        
        # Merge additional metadata from sections
        for section in sorted_sections:
            section_meta = section.get('metadata', {})
            for key, value in section_meta.items():
                if key not in document["metadata"] and key not in ['chunk_index', 'section']:
                    document["metadata"][key] = value
        
        documents.append(document)
    
    logger.info(f"Created {len(documents)} documents from {len(sections)} sections")
    return documents

def create_hierarchical_embeddings(
    chunks: List[Document], 
    client: OpenAIClient,
    batch_size: int = 16
) -> Dict[str, Any]:
    """
    Create hierarchical embeddings at document, section, and chunk levels.
    
    Args:
        chunks: List of document chunks (each with content and metadata)
        client: OpenAI client (v0 or v1)
        batch_size: Batch size for embedding API calls
        
    Returns:
        Dictionary with hierarchical structure and embeddings
    """
    logger.info(f"Creating hierarchical embeddings for {len(chunks)} chunks")
    start_time = time.time()
    
    # 1. Group chunks into sections
    sections = group_chunks_into_sections(chunks)
    
    # 2. Group sections into documents
    documents = group_sections_into_documents(sections)
    
    # 3. Create chunk-level embeddings (in batches)
    chunk_texts = [chunk.get('content', '') for chunk in chunks]
    chunk_embeddings = []
    
    for i in range(0, len(chunk_texts), batch_size):
        batch = chunk_texts[i:i+batch_size]
        batch_embeddings = get_embeddings(client, batch, model=CHUNK_LEVEL_MODEL)
        chunk_embeddings.extend(batch_embeddings)
    
    # 4. Create section-level embeddings (in batches)
    section_texts = [section["text"] for section in sections]
    section_embeddings = []
    
    for i in range(0, len(section_texts), batch_size):
        batch = section_texts[i:i+batch_size]
        batch_embeddings = get_embeddings(client, batch, model=SECTION_LEVEL_MODEL)
        section_embeddings.extend(batch_embeddings)
    
    # 5. Create document-level embeddings
    document_texts = [doc["text"] for doc in documents]
    document_embeddings = create_document_level_embedding(document_texts, client)
    
    # 6. Build the hierarchical structure with embeddings and relationships
    hierarchical_result = {
        "documents": [],
        "sections": [],
        "chunks": [],
        "statistics": {
            "document_count": len(documents),
            "section_count": len(sections),
            "chunk_count": len(chunks),
            "processing_time": time.time() - start_time
        }
    }
    
    # Add document information with embeddings
    for i, doc in enumerate(documents):
        doc_id = str(uuid.uuid4())
        hierarchical_result["documents"].append({
            "id": doc_id,
            "text": doc["text"],
            "metadata": doc["metadata"],
            "embedding": document_embeddings[i],
            "section_ids": []  # Will fill in with section IDs
        })
    
    # Add section information with embeddings and parent document reference
    for i, section in enumerate(sections):
        section_id = str(uuid.uuid4())
        
        # Find parent document
        doc_index = None
        source = section.get('metadata', {}).get('source', '')
        
        for j, doc in enumerate(hierarchical_result["documents"]):
            if doc["metadata"].get("source") == source:
                doc_index = j
                # Add this section to the document's section_ids
                hierarchical_result["documents"][j]["section_ids"].append(section_id)
                break
        
        hierarchical_result["sections"].append({
            "id": section_id,
            "text": section["text"],
            "metadata": section["metadata"],
            "embedding": section_embeddings[i],
            "document_id": hierarchical_result["documents"][doc_index]["id"] if doc_index is not None else None,
            "chunk_ids": []  # Will fill in with chunk IDs
        })
    
    # Add chunk information with embeddings and parent section reference
    for i, chunk in enumerate(chunks):
        chunk_id = str(uuid.uuid4())
        content = chunk.get('content', '')
        metadata = chunk.get('metadata', {})
        
        # Find parent section
        section_index = None
        source = metadata.get('source', '')
        chunk_index = metadata.get('chunk_index', 0)
        
        # Try to find the section this chunk belongs to
        for j, section in enumerate(hierarchical_result["sections"]):
            # Check if chunk belongs to this section
            if (section["metadata"].get("source") == source and
                any(c.get('metadata', {}).get('chunk_index') == chunk_index 
                    for c in sections[j]["chunks"])):
                section_index = j
                # Add this chunk to the section's chunk_ids
                hierarchical_result["sections"][j]["chunk_ids"].append(chunk_id)
                break
        
        hierarchical_result["chunks"].append({
            "id": chunk_id,
            "text": content,
            "metadata": metadata,
            "embedding": chunk_embeddings[i],
            "section_id": hierarchical_result["sections"][section_index]["id"] if section_index is not None else None
        })
    
    logger.info(f"Hierarchical embeddings created for {len(chunks)} chunks, {len(sections)} sections, {len(documents)} documents")
    return hierarchical_result

def prepare_hierarchical_points_for_qdrant(hierarchical_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert hierarchical embeddings into points for Qdrant.
    
    Args:
        hierarchical_data: Hierarchical embeddings structure
        
    Returns:
        List of points ready for Qdrant insertion
    """
    points = []
    
    # Add document-level points
    for doc in hierarchical_data["documents"]:
        doc_point = {
            "id": doc["id"],
            "vector": doc["embedding"],
            "payload": {
                "text": doc["text"][:10000],  # Truncate very long texts
                "metadata": doc["metadata"],
                "level": "document",
                "section_ids": doc["section_ids"]
            }
        }
        points.append(doc_point)
    
    # Add section-level points
    for section in hierarchical_data["sections"]:
        section_point = {
            "id": section["id"],
            "vector": section["embedding"],
            "payload": {
                "text": section["text"],
                "metadata": section["metadata"],
                "level": "section",
                "document_id": section["document_id"],
                "chunk_ids": section["chunk_ids"]
            }
        }
        points.append(section_point)
    
    # Add chunk-level points
    for chunk in hierarchical_data["chunks"]:
        chunk_point = {
            "id": chunk["id"],
            "vector": chunk["embedding"],
            "payload": {
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "level": "chunk",
                "section_id": chunk["section_id"]
            }
        }
        points.append(chunk_point)
    
    return points

def get_hierarchy_for_point(point_id: str, hierarchical_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get the full hierarchy for a specific point.
    
    Args:
        point_id: ID of the point to look up
        hierarchical_data: Hierarchical embeddings structure
        
    Returns:
        Dictionary with the point's ancestors and descendants
    """
    # Search in each level
    for chunk in hierarchical_data["chunks"]:
        if chunk["id"] == point_id:
            section_id = chunk["section_id"]
            section = next((s for s in hierarchical_data["sections"] if s["id"] == section_id), None)
            
            if section:
                document_id = section["document_id"]
                document = next((d for d in hierarchical_data["documents"] if d["id"] == document_id), None)
                
                return {
                    "point": chunk,
                    "ancestors": {
                        "section": section,
                        "document": document
                    },
                    "descendants": {}
                }
    
    for section in hierarchical_data["sections"]:
        if section["id"] == point_id:
            document_id = section["document_id"]
            document = next((d for d in hierarchical_data["documents"] if d["id"] == document_id), None)
            
            chunks = [c for c in hierarchical_data["chunks"] if c["section_id"] == point_id]
            
            return {
                "point": section,
                "ancestors": {
                    "document": document
                },
                "descendants": {
                    "chunks": chunks
                }
            }
    
    for document in hierarchical_data["documents"]:
        if document["id"] == point_id:
            sections = [s for s in hierarchical_data["sections"] if s["document_id"] == point_id]
            chunks = [c for c in hierarchical_data["chunks"] 
                     if c["section_id"] in [s["id"] for s in sections]]
            
            return {
                "point": document,
                "ancestors": {},
                "descendants": {
                    "sections": sections,
                    "chunks": chunks
                }
            }
    
    return {"error": f"Point ID {point_id} not found in hierarchy"}

# Main execution
if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Create hierarchical embeddings for documents")
    parser.add_argument("--input", help="Input file with documents to process", required=True)
    parser.add_argument("--output", help="Output file for hierarchical embeddings", required=True)
    parser.add_argument("--openai-api-key", help="OpenAI API key", required=False)
    
    args = parser.parse_args()
    
    # Get API key from args or environment
    api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key not provided. Use --openai-api-key or set OPENAI_API_KEY environment variable.")
        exit(1)
    
    try:
        # Import OpenAI
        try:
            import openai
        except ImportError:
            logger.error("OpenAI Python library not installed. Install with: pip install openai")
            exit(1)
        
        # Initialize client
        if hasattr(openai, "OpenAI"):
            # OpenAI Python v1.x
            client = openai.OpenAI(api_key=api_key)
        else:
            # OpenAI Python v0.x
            openai.api_key = api_key
            client = openai
        
        # Load input documents
        with open(args.input, 'r') as f:
            documents = json.load(f)
        
        # Create hierarchical embeddings
        result = create_hierarchical_embeddings(documents, client)
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Hierarchical embeddings saved to {args.output}")
        
    except Exception as e:
        logger.error(f"Error creating hierarchical embeddings: {e}")
        exit(1)