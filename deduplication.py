#!/usr/bin/env python3

"""
Duplicate Detection and Merging for RAG Systems

This module provides functionality for detecting and handling duplicate or near-duplicate
content in RAG systems. By identifying and either merging or removing duplicates, we can
improve embedding quality, reduce storage requirements, and improve retrieval relevance.

Features:
- Detect exact and near-duplicate documents
- Multiple similarity metrics (MinHash, Jaccard, cosine)
- Content merging strategies
- Metadata reconciliation
- Optimized for large document collections

Usage:
    from deduplication import deduplicate_documents, detect_duplicates
    
    # Detect and remove duplicates in a collection of documents
    deduplicated_docs = deduplicate_documents(documents)
    
    # Get clusters of similar documents
    duplicate_clusters = detect_duplicates(documents, threshold=0.85)
"""

import re
import logging
import json
import hashlib
from typing import Dict, Any, List, Optional, Set, Tuple, Callable
from collections import defaultdict
import math
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define document type (for type hints)
Document = Dict[str, Any]  # Will include at least 'content' and 'metadata'

@dataclass
class DuplicateStats:
    """Statistics about duplicate detection process."""
    total_documents: int
    unique_documents: int
    duplicate_sets: int
    exact_duplicates: int
    near_duplicates: int
    characters_saved: int
    largest_cluster_size: int

def compute_document_hash(doc: Document) -> str:
    """
    Compute a hash of document content for exact duplicate detection.
    
    Args:
        doc: Document to hash
        
    Returns:
        SHA-256 hash of normalized document content
    """
    content = doc.get("content", "")
    if not content:
        return ""
        
    # Normalize content (lowercase, remove extra whitespace)
    normalized = re.sub(r'\s+', ' ', content.lower()).strip()
    
    # Compute hash
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

def get_document_shingles(doc: Document, size: int = 3) -> Set[str]:
    """
    Create k-shingles (character n-grams) from document content.
    
    Args:
        doc: Document to process
        size: Size of each shingle
        
    Returns:
        Set of k-shingles
    """
    content = doc.get("content", "")
    if not content or len(content) < size:
        return set()
        
    # Normalize content
    normalized = re.sub(r'\s+', ' ', content.lower()).strip()
    
    # Generate shingles
    shingles = {normalized[i:i+size] for i in range(len(normalized) - size + 1)}
    return shingles

def compute_jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """
    Compute Jaccard similarity between two sets.
    
    Args:
        set1: First set
        set2: Second set
        
    Returns:
        Jaccard similarity (0-1)
    """
    if not set1 or not set2:
        return 0.0
        
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0

def minhash_signatures(shingles: Set[str], num_hashes: int = 100) -> List[int]:
    """
    Create MinHash signatures for efficient similarity estimation.
    
    Args:
        shingles: Set of document shingles
        num_hashes: Number of hash functions to use
        
    Returns:
        MinHash signature (list of integers)
    """
    if not shingles:
        return [0] * num_hashes
    
    import random
    
    # Create a list of hash functions (simulated with random parameters)
    # In a real implementation, we would use actual hash functions
    hash_functions = []
    for i in range(num_hashes):
        # Create hash function parameters (a, b for ax + b % c)
        a = random.randint(1, 2**31 - 1)
        b = random.randint(0, 2**31 - 1)
        c = 2**31 - 1  # Large prime
        hash_functions.append((a, b, c))
    
    # Initialize signature with maximum values
    signature = [float('inf')] * num_hashes
    
    # For each shingle, compute all hash values and keep the minimum
    for shingle in shingles:
        # Convert shingle to integer
        shingle_int = int.from_bytes(shingle.encode('utf-8'), byteorder='little')
        
        # Compute all hash values for this shingle
        for i, (a, b, c) in enumerate(hash_functions):
            hash_value = (a * shingle_int + b) % c
            signature[i] = min(signature[i], hash_value)
    
    return signature

def estimate_similarity_minhash(sig1: List[int], sig2: List[int]) -> float:
    """
    Estimate Jaccard similarity from MinHash signatures.
    
    Args:
        sig1: First MinHash signature
        sig2: Second MinHash signature
        
    Returns:
        Estimated Jaccard similarity (0-1)
    """
    if not sig1 or not sig2 or len(sig1) != len(sig2):
        return 0.0
    
    # Count how many signature components are equal
    matches = sum(1 for h1, h2 in zip(sig1, sig2) if h1 == h2)
    
    # Return fraction of matches
    return matches / len(sig1)

def compute_similarity(doc1: Document, doc2: Document, method: str = "shingles") -> float:
    """
    Compute similarity between two documents.
    
    Args:
        doc1: First document
        doc2: Second document
        method: Similarity method ('shingles', 'minhash', 'token')
        
    Returns:
        Similarity score (0-1)
    """
    if method == "shingles":
        # Use character shingles and Jaccard similarity
        shingles1 = get_document_shingles(doc1)
        shingles2 = get_document_shingles(doc2)
        return compute_jaccard_similarity(shingles1, shingles2)
    
    elif method == "minhash":
        # Use MinHash for faster similarity estimation
        shingles1 = get_document_shingles(doc1)
        shingles2 = get_document_shingles(doc2)
        sig1 = minhash_signatures(shingles1)
        sig2 = minhash_signatures(shingles2)
        return estimate_similarity_minhash(sig1, sig2)
    
    elif method == "token":
        # Use token sets (words) and Jaccard similarity
        content1 = doc1.get("content", "").lower()
        content2 = doc2.get("content", "").lower()
        tokens1 = set(re.findall(r'\b\w+\b', content1))
        tokens2 = set(re.findall(r'\b\w+\b', content2))
        return compute_jaccard_similarity(tokens1, tokens2)
    
    # Default fallback
    return 0.0

def find_exact_duplicates(documents: List[Document]) -> Dict[str, List[int]]:
    """
    Find exact duplicates using content hashing.
    
    Args:
        documents: List of documents
        
    Returns:
        Dictionary mapping hash to list of document indices
    """
    hash_to_indices = defaultdict(list)
    
    for i, doc in enumerate(documents):
        doc_hash = compute_document_hash(doc)
        if doc_hash:  # Only add if hash is valid
            hash_to_indices[doc_hash].append(i)
    
    # Filter to keep only hashes with multiple documents
    return {h: indices for h, indices in hash_to_indices.items() if len(indices) > 1}

def cluster_similar_documents(
    documents: List[Document], 
    threshold: float = 0.85, 
    similarity_method: str = "shingles",
    max_cluster_size: int = 100
) -> List[List[int]]:
    """
    Cluster similar (but not exact duplicate) documents.
    
    Args:
        documents: List of documents
        threshold: Similarity threshold (0-1)
        similarity_method: Method to compute similarity
        max_cluster_size: Maximum size for a single cluster
        
    Returns:
        List of document index clusters
    """
    clusters = []
    processed = set()
    
    # First find all similar pairs
    similar_pairs = []
    
    for i in range(len(documents)):
        if i in processed:
            continue
            
        # Find similar documents
        for j in range(i + 1, len(documents)):
            if j in processed:
                continue
                
            similarity = compute_similarity(documents[i], documents[j], similarity_method)
            
            if similarity >= threshold:
                similar_pairs.append((i, j, similarity))
    
    # Sort pairs by similarity (descending)
    similar_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Build clusters using transitive closure
    for i, j, similarity in similar_pairs:
        # Skip if either document is already in a too-large cluster
        in_cluster_i = None
        in_cluster_j = None
        
        for idx, cluster in enumerate(clusters):
            if i in cluster:
                in_cluster_i = idx
            if j in cluster:
                in_cluster_j = idx
        
        # Both documents are in different clusters
        if in_cluster_i is not None and in_cluster_j is not None and in_cluster_i != in_cluster_j:
            # Merge clusters if combined size is not too big
            if len(clusters[in_cluster_i]) + len(clusters[in_cluster_j]) <= max_cluster_size:
                clusters[in_cluster_i].extend(clusters[in_cluster_j])
                clusters.pop(in_cluster_j)
                
        # One document is in a cluster
        elif in_cluster_i is not None:
            if len(clusters[in_cluster_i]) < max_cluster_size:
                clusters[in_cluster_i].append(j)
                processed.add(j)
                
        elif in_cluster_j is not None:
            if len(clusters[in_cluster_j]) < max_cluster_size:
                clusters[in_cluster_j].append(i)
                processed.add(i)
                
        # Neither document is in a cluster
        else:
            clusters.append([i, j])
            processed.add(i)
            processed.add(j)
    
    return clusters

def merge_documents(docs: List[Document]) -> Document:
    """
    Merge multiple similar documents into one.
    
    Args:
        docs: List of documents to merge
        
    Returns:
        Merged document
    """
    if not docs:
        return {"content": "", "metadata": {}}
    
    if len(docs) == 1:
        return docs[0]
    
    # Use the longest document as the base
    base_doc = max(docs, key=lambda d: len(d.get("content", "")))
    base_content = base_doc.get("content", "")
    
    # Merge metadata
    merged_metadata = {}
    all_metadata = []
    
    for doc in docs:
        meta = doc.get("metadata", {})
        all_metadata.append(meta)
        
        # Add unique values to merged metadata
        for key, value in meta.items():
            if key not in merged_metadata:
                merged_metadata[key] = value
            elif key == "source" and value != merged_metadata[key]:
                # Special handling for source - keep as list
                if isinstance(merged_metadata[key], list):
                    if value not in merged_metadata[key]:
                        merged_metadata[key].append(value)
                else:
                    merged_metadata[key] = [merged_metadata[key], value]
    
    # Add special metadata for merged document
    merged_metadata["merged_from"] = len(docs)
    merged_metadata["is_merged"] = True
    
    return {
        "content": base_content,
        "metadata": merged_metadata
    }

def deduplicate_documents(
    documents: List[Document], 
    similarity_threshold: float = 0.85,
    similarity_method: str = "shingles",
    merge_similar: bool = True
) -> Tuple[List[Document], DuplicateStats]:
    """
    Detect and remove or merge duplicate documents.
    
    Args:
        documents: List of documents
        similarity_threshold: Threshold for near-duplicate detection
        similarity_method: Method to compute similarity
        merge_similar: Whether to merge similar documents (vs. keeping first)
        
    Returns:
        Tuple of (deduplicated documents, stats)
    """
    logger.info(f"Starting deduplication of {len(documents)} documents")
    
    if not documents:
        return [], DuplicateStats(0, 0, 0, 0, 0, 0, 0)
    
    # Initialize stats
    total_docs = len(documents)
    exact_duplicates = 0
    near_duplicates = 0
    duplicate_sets = 0
    original_chars = sum(len(doc.get("content", "")) for doc in documents)
    largest_cluster = 0
    
    # Find exact duplicates
    exact_dupe_clusters = find_exact_duplicates(documents)
    
    # Find similar documents
    similar_clusters = cluster_similar_documents(
        documents, 
        threshold=similarity_threshold,
        similarity_method=similarity_method
    )
    
    # Process exact duplicates first
    keep_indices = set(range(len(documents)))
    processed_indices = set()
    deduplicated_docs = []
    
    # Process exact duplicates
    for doc_hash, indices in exact_dupe_clusters.items():
        if not indices:
            continue
            
        duplicate_sets += 1
        largest_cluster = max(largest_cluster, len(indices))
        exact_duplicates += len(indices) - 1
        
        # Keep the first document, remove others
        keep_idx = indices[0]
        for idx in indices[1:]:
            keep_indices.discard(idx)
            processed_indices.add(idx)
            
        processed_indices.add(keep_idx)
    
    # Process similar clusters
    for cluster in similar_clusters:
        # Skip indices already processed as exact duplicates
        cluster = [idx for idx in cluster if idx not in processed_indices]
        
        if len(cluster) <= 1:
            continue
            
        duplicate_sets += 1
        largest_cluster = max(largest_cluster, len(cluster))
        near_duplicates += len(cluster) - 1
        
        if merge_similar:
            # Merge documents in this cluster
            to_merge = [documents[idx] for idx in cluster]
            merged_doc = merge_documents(to_merge)
            
            # Add merged document and mark all cluster indices as processed
            deduplicated_docs.append(merged_doc)
            for idx in cluster:
                keep_indices.discard(idx)
                processed_indices.add(idx)
        else:
            # Keep first document, discard others
            keep_idx = cluster[0]
            for idx in cluster[1:]:
                keep_indices.discard(idx)
                processed_indices.add(idx)
                
            processed_indices.add(keep_idx)
    
    # Add all remaining documents
    for idx in sorted(keep_indices):
        deduplicated_docs.append(documents[idx])
    
    # Calculate final stats
    final_chars = sum(len(doc.get("content", "")) for doc in deduplicated_docs)
    chars_saved = original_chars - final_chars
    
    stats = DuplicateStats(
        total_documents=total_docs,
        unique_documents=len(deduplicated_docs),
        duplicate_sets=duplicate_sets,
        exact_duplicates=exact_duplicates,
        near_duplicates=near_duplicates,
        characters_saved=chars_saved,
        largest_cluster_size=largest_cluster
    )
    
    logger.info(f"Deduplication complete: {stats.total_documents} → {stats.unique_documents} documents")
    logger.info(f"Removed {stats.exact_duplicates} exact duplicates and {stats.near_duplicates} near-duplicates")
    
    return deduplicated_docs, stats

def detect_duplicates(
    documents: List[Document],
    threshold: float = 0.85,
    similarity_method: str = "shingles"
) -> List[List[int]]:
    """
    Detect duplicate and near-duplicate documents.
    
    Args:
        documents: List of documents
        threshold: Similarity threshold for near-duplicates
        similarity_method: Method to compute similarity
        
    Returns:
        List of document index clusters
    """
    logger.info(f"Detecting duplicates in {len(documents)} documents")
    
    # Find exact duplicates
    exact_dupe_clusters = find_exact_duplicates(documents)
    
    # Find similar documents
    similar_clusters = cluster_similar_documents(
        documents, 
        threshold=threshold,
        similarity_method=similarity_method
    )
    
    # Combine all clusters
    all_clusters = [indices for indices in exact_dupe_clusters.values()]
    all_clusters.extend(similar_clusters)
    
    # Sort clusters by size (descending)
    all_clusters.sort(key=len, reverse=True)
    
    logger.info(f"Found {len(all_clusters)} duplicate clusters")
    return all_clusters

if __name__ == "__main__":
    """Command line interface for testing deduplication."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Detect and remove duplicate documents")
    parser.add_argument("--input", "-i", help="Input JSON file with documents", required=True)
    parser.add_argument("--output", "-o", help="Output file for deduplicated documents", required=False)
    parser.add_argument("--threshold", "-t", type=float, default=0.85, help="Similarity threshold (0-1)")
    parser.add_argument("--method", "-m", choices=["shingles", "minhash", "token"], default="shingles", 
                        help="Similarity method")
    parser.add_argument("--merge", action="store_true", help="Merge similar documents instead of removing")
    
    args = parser.parse_args()
    
    try:
        # Load documents
        with open(args.input, "r", encoding="utf-8") as f:
            documents = json.load(f)
            
        if not isinstance(documents, list):
            print("Error: Input file must contain a JSON array of documents")
            sys.exit(1)
            
        # Deduplicate documents
        deduplicated, stats = deduplicate_documents(
            documents,
            similarity_threshold=args.threshold,
            similarity_method=args.method,
            merge_similar=args.merge
        )
        
        # Print stats
        print(f"Total documents: {stats.total_documents}")
        print(f"Unique documents: {stats.unique_documents}")
        print(f"Duplicate sets: {stats.duplicate_sets}")
        print(f"Exact duplicates: {stats.exact_duplicates}")
        print(f"Near-duplicates: {stats.near_duplicates}")
        print(f"Characters saved: {stats.characters_saved}")
        print(f"Largest cluster size: {stats.largest_cluster_size}")
        
        # Save deduplicated documents
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(deduplicated, f, indent=2)
            print(f"Deduplicated documents saved to {args.output}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)