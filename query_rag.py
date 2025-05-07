#!/usr/bin/env python3
"""query_rag.py

Simple CLI to query a Qdrant RAG collection using OpenAI embeddings.
"""
from __future__ import annotations
import os
import sys
from typing import Sequence, Any

import click
from qdrant_client import QdrantClient

# Reuse OpenAI client helper from ingest_rag
from ingest_rag import get_openai_client
import math
import re

def _cosine_sim(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))

def _mmr_rerank(points: list[Any], mmr_lambda: float) -> list[Any]:
    """Apply Maximal Marginal Relevance to reorder points for diversity."""
    selected: list[Any] = []
    candidates = points.copy()
    # Stepwise selection
    while candidates:
        if not selected:
            # pick highest relevance
            best = max(candidates, key=lambda p: getattr(p, 'score', 0.0))
        else:
            best = None
            best_score = None
            for p in candidates:
                rel = getattr(p, 'score', 0.0)
                # novelty: max similarity to already selected
                # Compute novelty: maximum similarity to already selected
                try:
                    similarities = [
                        _cosine_sim(
                            (getattr(p, 'vector', None) or []),
                            (getattr(s, 'vector', None) or [])
                        )
                        for s in selected
                    ]
                    # Handle empty list case or all zero vectors
                    if not similarities or all(sim == 0 for sim in similarities):
                        nov = 0.0
                    else:
                        nov = max(similarities)
                except (ZeroDivisionError, ValueError):
                    # If cosine similarity fails due to division by zero
                    # (which happens when vectors are all zeros)
                    nov = 0.0
                mmr_score = mmr_lambda * rel - (1.0 - mmr_lambda) * nov
                if best is None or mmr_score > best_score:
                    best = p
                    best_score = mmr_score
        if best is None:
            break
        selected.append(best)
        candidates.remove(best)
    return selected


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--collection", default="rag_data", show_default=True, help="Qdrant collection name to query.")
@click.option(
    "--k",
    type=int,
    default=150,
    show_default=True,
    help="Number of nearest neighbors to retrieve.",
)
@click.option("--snippet/--no-snippet", default=True, help="Show a text snippet of each result.")
@click.option("--model", default="text-embedding-3-large", show_default=True, help="OpenAI embedding model to use.")
@click.option("--qdrant-host", default="localhost", show_default=True, help="Qdrant host (ignored if --qdrant-url is provided).")
@click.option("--qdrant-port", type=int, default=6333, show_default=True, help="Qdrant port (ignored if --qdrant-url is provided).")
@click.option("--qdrant-url", help="Full Qdrant URL (overrides host/port).")
@click.option("--qdrant-api-key", envvar="QDRANT_API_KEY", help="API key for Qdrant (if required).")
@click.option("--openai-api-key", envvar="OPENAI_API_KEY", help="OpenAI API key.")
@click.option(
    "--llm-model",
    default="gpt-4.1-mini",
    show_default=True,
    help=(
        "LLM model for answer generation (e.g. gpt-4.1-mini)."
        " Set to empty string to skip generation."
    ),
)
@click.option("--raw", is_flag=True, default=False,
              help="Show raw retrieval and answer (requires --llm-model).")
@click.option(
    "--hybrid/--no-hybrid",
    default=True,
    show_default=True,
    help="Enable hybrid BM25 + vector search (on by default; disable with --no-hybrid).",
)
@click.option(
    "--bm25-index",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help=(
        "Path to JSON file mapping point IDs to chunk_text for BM25 index. "
        "Defaults to '<collection>_bm25_index.json' if present."
    ),
)
@click.option("--alpha", type=float, default=0.5, show_default=True, help="Weight for vector scores in hybrid fusion (0.0-1.0).")
@click.option("--bm25-top", type=int, default=None, help="Number of top BM25 docs to consider (default: k).")
@click.option("--rrf-k", type=float, default=60.0, show_default=True, help="Reciprocal Rank Fusion k hyperparameter.")
@click.option("--rerank-top", type=int, default=20, show_default=True,
              help="Number of top retrieval results to re-rank using a cross-encoder (requires sentence-transformers).")
@click.option("--mmr-lambda", type=float, default=0.5, show_default=True,
              help="MMR diversity parameter (lambda: 0=full diversity, 1=full relevance). Used when deep search is enabled.")
@click.option("--deepsearch/--no-deepsearch", is_flag=True, default=False,
              help="Enable deep search (MMR re-ranking) for more diverse retrieval. Disabled by default.")
@click.option("--filter", "-f", "filters", multiple=True, help="Filter by payload key=value. Can be used multiple times.")
@click.option(
    "--use-expansion/--no-use-expansion",
    is_flag=True,
    default=True,
    show_default=True,
    help="Enable query expansion for better recall (requires advanced_rag).",
)
@click.option(
    "--max-expansions",
    type=int,
    default=3,
    show_default=True,
    help="Maximum number of query expansions to use.",
)
@click.option(
    "--hierarchical-search/--no-hierarchical-search",
    is_flag=True,
    default=False,
    show_default=True,
    help="Enable hierarchical search across document, section, and chunk levels.",
)
@click.option(
    "--level",
    type=click.Choice(["document", "section", "chunk", "auto"]),
    default="auto",
    show_default=True,
    help="Hierarchical embedding level to query (document, section, chunk, or auto).",
)
@click.option(
    "--evaluate",
    is_flag=True,
    default=False,
    show_default=True,
    help="Evaluate RAG quality and return feedback with results (requires advanced_rag).",
)
@click.option(
    "--compress/--no-compress",
    is_flag=True,
    default=False,
    show_default=True,
    help="Apply contextual compression to focus retrieved chunks on query-relevant parts.",
)
@click.argument("query", nargs=-1, required=True)
def main(
    collection: str,
    k: int,
    snippet: bool,
    model: str,
    qdrant_host: str,
    qdrant_port: int,
    qdrant_url: str | None,
    qdrant_api_key: str | None,
    openai_api_key: str | None,
    llm_model: str | None,
    raw: bool,
    hybrid: bool,
    bm25_index: str | None,
    alpha: float,
    bm25_top: int | None,
    rrf_k: float,
    rerank_top: int,
    mmr_lambda: float,
    deepsearch: bool,
    filters: Sequence[str],
    use_expansion: bool,
    max_expansions: int,
    hierarchical_search: bool,
    level: str,
    evaluate: bool,
    compress: bool,
    query: Sequence[str],
) -> None:
    """Embed QUERY with OpenAI and search a Qdrant RAG collection."""
    # Load .env file (if present) BEFORE reading API keys
    from pathlib import Path
    env_path = Path(".env")
    if env_path.is_file():
        try:
            import dotenv
            dotenv.load_dotenv(dotenv_path=str(env_path), override=False)
            click.echo(f"[info] Environment variables loaded from {env_path}")
        except ImportError:
            pass

    # Prepare the full query text
    query_text = " ".join(query)

    # Ensure OpenAI and Qdrant API keys
    if openai_api_key is None:
        openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key is None:
        click.echo("[fatal] OPENAI_API_KEY is not set (check .env or environment).", err=True)
        sys.exit(1)
    if qdrant_api_key is None:
        qdrant_api_key = os.environ.get("QDRANT_API_KEY")

    # Initialize OpenAI client
    openai_client = get_openai_client(openai_api_key)

    # Initialize Qdrant client (use HTTP by default)
    if qdrant_url:
        url = qdrant_url
    else:
        url = f"http://{qdrant_host}:{qdrant_port}"
    client = QdrantClient(url=url, api_key=qdrant_api_key)
    
    # Use query expansion if enabled
    expanded_queries = [query_text]
    if use_expansion:
        try:
            from advanced_rag import expand_query
            click.echo(f"[info] Using query expansion from advanced_rag module")
            
            # Expand the query
            expanded_queries = expand_query(query_text, openai_client, max_expansions=max_expansions)
            
            # Show the expanded queries
            if len(expanded_queries) > 1:
                click.echo("\nExpanded queries:")
                for i, exp_query in enumerate(expanded_queries):
                    click.echo(f"  {i+1}. {exp_query}")
        except ImportError as e:
            click.echo(f"[warning] Query expansion failed (advanced_rag module not found): {e}", err=True)
            click.echo("[info] To use query expansion, install the advanced_rag module")
        except Exception as e:
            click.echo(f"[warning] Query expansion failed: {e}", err=True)
            click.echo("[info] Continuing with original query only")

    # Process all queries (original and expanded if available)
    all_results = []
    
    for q_idx, q_text in enumerate(expanded_queries):
        if q_idx > 0:
            click.echo(f"\nProcessing expanded query {q_idx+1}: {q_text}")
            
        # Embed this query
        if hasattr(openai_client, "embeddings"):  # openai>=1.0 style
            resp = openai_client.embeddings.create(model=model, input=[q_text])
            vector = resp.data[0].embedding
        else:  # legacy style
            resp = openai_client.Embedding.create(model=model, input=[q_text])
            vector = resp["data"][0]["embedding"]  # type: ignore[index]
            
        # Store the current query vector for later use
        current_vector = vector
        
        # Build payload filter if specified (inside loop to use for each query)
        filter_obj = None
        if filters:
            from qdrant_client.http.models import Filter, FieldCondition, MatchValue
            conditions: list[FieldCondition] = []
            for f in filters:
                if "=" not in f:
                    click.echo(f"[fatal] Invalid filter '{f}'; must be key=value", err=True)
                    sys.exit(1)
                key, val = f.split("=", 1)
                conditions.append(FieldCondition(key=key, match=MatchValue(value=val)))
            filter_obj = Filter(must=conditions)
            if q_idx == 0:  # Only print this once
                click.echo(f"[info] Applying filters: {filters}")
                
        # Apply hierarchical search filter if enabled
        if hierarchical_search:
            from qdrant_client.http.models import Filter, FieldCondition, MatchValue
            
            # Create filter based on selected level
            # If filter_obj is already set from other filters, we'll add to it
            if filter_obj is None:
                filter_obj = Filter(must=[])
            
            # Auto mode tries to determine best level based on query length and complexity
            if level == "auto":
                # Use query characteristics to determine appropriate level
                # Short specific queries (likely target chunks)
                # Medium queries (likely target sections)
                # Broad conceptual queries (likely target documents)
                query_words = len(q_text.split())
                query_sentences = len(re.findall(r'[.!?]+', q_text)) + 1
                
                if "summary" in q_text.lower() or "overview" in q_text.lower() or query_words < 4:
                    selected_level = "document"
                elif query_words > 15 or query_sentences > 2:
                    selected_level = "section"
                else:
                    selected_level = "chunk"
                    
                if q_idx == 0:  # Only print this once
                    click.echo(f"[info] Auto-selected hierarchical search level: {selected_level}")
            else:
                selected_level = level
                
            # Add level filter to existing filter
            if hasattr(filter_obj, "must"):
                filter_obj.must.append(FieldCondition(
                    key="level",
                    match=MatchValue(value=selected_level)
                ))
            else:
                # In case filter_obj is an unexpected type, create a new one
                filter_obj = Filter(must=[
                    FieldCondition(key="level", match=MatchValue(value=selected_level))
                ])
                
        # Search in Qdrant (use new query_points if available)
        from qdrant_client.http.exceptions import UnexpectedResponse
        try:
            if hasattr(client, "query_points"):
                # New API returns a QueryResponse with .points attribute
                resp = client.query_points(
                    collection_name=collection,
                    query=current_vector,
                    limit=k,
                    with_payload=True,
                    with_vectors=True,
                    query_filter=filter_obj,
                )
                query_results = getattr(resp, 'points', [])
            else:
                # Fallback to deprecated search()
                query_results = client.search(
                    collection_name=collection,
                    query_vector=current_vector,
                    limit=k,
                    with_payload=True,
                    with_vectors=True,
                    query_filter=filter_obj,
                )
                
            # Add these results to our collection
            all_results.extend(query_results)
            
        except UnexpectedResponse as e:
            # likely missing collection
            msg = str(e)
            if "doesn't exist" in msg or "not found" in msg.lower():
                click.echo(f"[fatal] Collection '{collection}' not found.  Please ingest data first (e.g. ingest_rag.py --collection {collection}).", err=True)
                sys.exit(1)
            raise
    
    # Deduplicate results based on point ID
    from collections import defaultdict
    
    # Process all results to keep highest score for each unique ID
    unique_results = defaultdict(lambda: {"score": 0.0, "point": None})
    for point in all_results:
        point_id = point.id
        score = getattr(point, 'score', 0.0) or 0.0
        
        # If we already have this point, only keep the one with higher score
        if score > unique_results[point_id]["score"]:
            unique_results[point_id] = {"score": score, "point": point}
    
    # Sort by score (descending) and take top k
    scored = [entry["point"] for entry in sorted(
        unique_results.values(), 
        key=lambda x: x["score"], 
        reverse=True
    )][:k]
    if any(p is None for p in scored):
        click.echo(f"[debug] 'scored' contains {sum(1 for p in scored if p is None)} None values after initial creation from unique_results. IDs of None points: {[getattr(p, 'id', 'N/A_None_Point') for p in unique_results.values() if p['point'] is None]}", err=True)
    # Remove any None entries to prevent downstream errors
    scored = [p for p in scored if p is not None]
    
    if use_expansion and len(expanded_queries) > 1:
        click.echo(f"[info] Merged {len(all_results)} results from {len(expanded_queries)} queries into {len(scored)} unique results")
        
    # If hierarchical search is enabled, try to fetch related documents from hierarchy
    if hierarchical_search:
        try:
            # Check if we have the hierarchical structure available
            structure_file = f"{collection}_hierarchical_structure.json"
            if os.path.exists(structure_file):
                with open(structure_file, 'r') as f:
                    hierarchical_structure = json.load(f)
                
                if level != "document":  # For chunk or section level searches
                    # Enhance results by adding parent document information
                    # This will help generate more coherent answers by providing higher-level context
                    enhanced_points = []
                    doc_context_added = set()  # Track which documents we've already added
                    
                    for point in scored:
                        # Add the original result first
                        enhanced_points.append(point)
                        
                        # Check if this is a section or chunk and try to find its parent
                        point_id = point.id
                        payload = getattr(point, 'payload', {}) or {}
                        point_level = payload.get('level', '')
                        
                        # For sections, find parent document
                        if point_level == 'section':
                            doc_id = payload.get('document_id')
                            if doc_id and doc_id not in doc_context_added:
                                # Try to find the document in hierarchical_structure
                                document = next((d for d in hierarchical_structure.get('documents', []) 
                                                if d.get('id') == doc_id), None)
                                if document:
                                    # Create a simple namespace object for consistency
                                    from types import SimpleNamespace
                                    parent_doc = SimpleNamespace(
                                        id=document.get('id'),
                                        payload={'text': document.get('text', ''), 
                                                 'level': 'document',
                                                 'metadata': document.get('metadata', {})},
                                        score=0.3  # Lower score for context docs
                                    )
                                    enhanced_points.append(parent_doc)
                                    doc_context_added.add(doc_id)
                                    
                        # For chunks, find parent section and document
                        elif point_level == 'chunk':
                            section_id = payload.get('section_id')
                            if section_id:
                                # Find the section
                                section = next((s for s in hierarchical_structure.get('sections', []) 
                                              if s.get('id') == section_id), None)
                                if section:
                                    # Create section object
                                    from types import SimpleNamespace
                                    parent_section = SimpleNamespace(
                                        id=section.get('id'),
                                        payload={'text': section.get('text', ''),
                                                 'level': 'section',
                                                 'metadata': section.get('metadata', {})},
                                        score=0.4  # Medium score for parent sections
                                    )
                                    enhanced_points.append(parent_section)
                                    
                                    # Also try to find the document
                                    doc_id = section.get('document_id')
                                    if doc_id and doc_id not in doc_context_added:
                                        document = next((d for d in hierarchical_structure.get('documents', []) 
                                                        if d.get('id') == doc_id), None)
                                        if document:
                                            parent_doc = SimpleNamespace(
                                                id=document.get('id'),
                                                payload={'text': document.get('text', ''),
                                                         'level': 'document',
                                                         'metadata': document.get('metadata', {})},
                                                score=0.3  # Lower score for context docs
                                            )
                                            enhanced_points.append(parent_doc)
                                            doc_context_added.add(doc_id)
                    
                    # Replace scored with enhanced results, but keep the original limit
                    if enhanced_points:
                        click.echo(f"[info] Added hierarchical context from {len(doc_context_added)} parent documents")
                        scored = enhanced_points[:k]
                        
                elif level == "document":  # For document level searches
                    # Enhance with sections and chunks for more detailed retrieval
                    enhanced_points = []
                    
                    for point in scored:
                        # Add the original document first
                        enhanced_points.append(point)
                        
                        # Check if this is a document and find its sections and chunks
                        point_id = point.id
                        payload = getattr(point, 'payload', {}) or {}
                        point_level = payload.get('level', '')
                        
                        if point_level == 'document':
                            section_ids = payload.get('section_ids', [])
                            if section_ids:
                                # Find and add top sections (limited to avoid overwhelming)
                                sections_to_add = min(3, len(section_ids))
                                for i in range(sections_to_add):
                                    section_id = section_ids[i]
                                    section = next((s for s in hierarchical_structure.get('sections', []) 
                                                   if s.get('id') == section_id), None)
                                    if section:
                                        from types import SimpleNamespace
                                        child_section = SimpleNamespace(
                                            id=section.get('id'),
                                            payload={'text': section.get('text', ''),
                                                     'level': 'section', 
                                                     'metadata': section.get('metadata', {})},
                                            score=0.5  # Higher score for important sections
                                        )
                                        enhanced_points.append(child_section)
                    
                    # Replace scored with enhanced results, but keep the original limit
                    if enhanced_points:
                        click.echo(f"[info] Added hierarchical context from document sections")
                        scored = enhanced_points[:k]

        except Exception as e:
            click.echo(f"[warning] Hierarchical context enhancement failed: {e}", err=True)
            click.echo("[info] Continuing with standard search results", err=True)
        
    # Hybrid fusion (BM25 + vector) if enabled
    if hybrid:
        # Cache original scored list with vectors for MMR
        orig_scored = scored
        try:
            import json
            from rank_bm25 import BM25Okapi
        except ImportError:
            click.echo(
                "[fatal] rank_bm25 is required for hybrid search (pip install rank-bm25)",
                err=True,
            )
            sys.exit(1)
        from qdrant_client.http.models import Filter as QFilter, HasIdCondition
        # If no BM25 index path provided, try default '<collection>_bm25_index.json'
        if not bm25_index:
            default_idx = f"{collection}_bm25_index.json"
            if os.path.exists(default_idx):
                bm25_index = default_idx
                click.echo(f"[info] Loading BM25 index from {bm25_index}")

        # Load or build BM25 index mapping point IDs to chunk text
        # Attempt to load a pre-built BM25 index JSON; if missing or empty, rebuild from collection
        build_index = True
        if bm25_index:
            try:
                with open(bm25_index, "r") as f:
                    id2text = json.load(f)
                if id2text:
                    build_index = False
                    click.echo(f"[info] Loaded BM25 index from {bm25_index} ({len(id2text)} entries)")
                else:
                    click.echo(f"[warning] BM25 index file {bm25_index} is empty, rebuilding from collection", err=True)
            except Exception as e:
                click.echo(f"[warning] Failed to load BM25 index '{bm25_index}': {e}, rebuilding from collection", err=True)
                build_index = True
        if build_index:
            click.echo(f"[info] Building BM25 index from collection '{collection}' (this may take a while)...")
            id2text: dict[str, str] = {}
            offset = None
            # use same metadata filter if any
            scroll_filter = filter_obj if 'filter_obj' in locals() else None
            while True:
                records, offset = client.scroll(
                    collection_name=collection,
                    scroll_filter=scroll_filter,
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
        # Guard against empty corpus - rank_bm25 crashes with ZeroDivisionError
        if not id2text:
            click.echo(
                "[warning] No documents with 'chunk_text' payload - skipping BM25 component of hybrid search.",
                err=True,
            )
        else:
            ids = list(id2text.keys())
            tokenized = [id2text[_id].split() for _id in ids]
            # rank_bm25 expects at least one document; otherwise it raises ZeroDivisionError
            if not tokenized:
                click.echo(
                    "[warning] BM25 tokenization produced an empty corpus - skipping BM25 component of hybrid search.",
                    err=True,
                )
            else:
                bm25 = BM25Okapi(tokenized)
                # Compute BM25 rankings
                query_tokens = query_text.split()
                bm25_scores = bm25.get_scores(query_tokens)
                top_n = bm25_top or k
                bm25_sorted = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:top_n]
                bm25_rank = {ids[idx]: rank + 1 for rank, (idx, _) in enumerate(bm25_sorted)}

                # --- Debugging: Check for None in scored before processing ---
                if any(p is None for p in scored):
                    click.echo("[debug] Found None in 'scored' list before vector ranking.", err=True)
                    # Optional: Log the contents of scored for more details if needed
                    # click.echo(f"[debug] scored contents: {scored}", err=True)
                # --- End Debugging ---

                # Vector rankings (from initial Qdrant results)
                vec_rank = {point.id: rank for rank, point in enumerate(scored, start=1) if point is not None}

                # Reciprocal Rank Fusion
                fused_scores: dict[str, float] = {}
                for pid in set(vec_rank) | set(bm25_rank):
                    score_h = 0.0
                    if pid in vec_rank:
                        score_h += alpha * (1.0 / (rrf_k + vec_rank[pid]))
                    if pid in bm25_rank:
                        score_h += (1.0 - alpha) * (1.0 / (rrf_k + bm25_rank[pid]))
                    fused_scores[pid] = score_h

                # Select top-k fused results
                fused_ids = [pid for pid, _ in sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)][:k]

                # Build new scored list with payload lookup
                from types import SimpleNamespace
                payload_map = {point.id: getattr(point, 'payload', {}) or {} for point in scored if point is not None}
                # Fetch any BM25-only payloads
                for pid in fused_ids:
                    if pid not in payload_map:
                        fobj = QFilter(must=[HasIdCondition(has_id=[pid])])
                        records, _ = client.scroll(collection_name=collection, scroll_filter=fobj, with_payload=True, limit=1)
                        if records:
                            rec = records[0]
                            payload_map[pid] = getattr(rec, 'payload', {}) or {}
                        else:
                            payload_map[pid] = {}
                # Replace scored with fused SimpleNamespace objects
                # Preserve vector from original scored points
                vector_map = {p.id: getattr(p, 'vector', None) for p in orig_scored if p is not None}
                scored = [
                    SimpleNamespace(
                        id=pid,
                        payload=payload_map.get(pid, {}),
                        score=fused_scores.get(pid, 0.0),
                        vector=vector_map.get(pid),
                    )
                    for pid in fused_ids
                ]

        # If BM25 fusion skipped because of empty corpus, keep original 'scored'

        # Note: if no BM25 corpus was available, we simply keep the original
        # `scored` list coming from the pure-vector Qdrant search so that the
        # rest of the pipeline (answer generation, summaries, etc.) continues
        # to function as expected.
    # MMR re-ranking for diversity + relevance (deep search)
    if deepsearch:
        try:
            click.echo(f"[info] Applying MMR re-ranking (lambda={mmr_lambda})...")
            scored = _mmr_rerank(scored, mmr_lambda)
        except Exception as e:
            click.echo(f"[warning] MMR re-ranking failed: {e}", err=True)
    # Cross-encoder re-ranking if requested
    if rerank_top and rerank_top > 0:
        # Re-order top results using a cross-encoder model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            click.echo(
                "[fatal] sentence-transformers is required for --rerank-top (pip install sentence-transformers)",
                err=True,
            )
            sys.exit(1)
        # Load cross-encoder
        click.echo(f"[info] Preparing to re-rank up to {rerank_top} results with cross-encoder...")
        ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        # Determine how many candidates we have to rerank
        n_rerank = min(rerank_top, len(scored))
        if n_rerank > 0:
            click.echo(f"[info] Performing cross-encoder re-ranking on {n_rerank} candidates...")
            candidates = scored[:n_rerank]
            pairs = [
                (query_text, (getattr(p, 'payload', {}) or {}).get("chunk_text", ""))
                for p in candidates
            ]
            # Predict relevance scores and sort if pairs exist
            if pairs:
                rerank_scores = ce.predict(pairs)
                idxs = sorted(range(n_rerank), key=lambda i: rerank_scores[i], reverse=True)
                scored = [candidates[i] for i in idxs]
            else:
                click.echo("[warning] No text candidates available for cross-encoder re-ranking", err=True)
        else:
            click.echo(f"[info] --rerank-top specified but only {len(scored)} results available; skipping re-ranking", err=True)

    # Apply contextual compression if enabled
    if compress:
        try:
            from advanced_rag import contextual_compression, get_compressed_text
            click.echo(f"[info] Applying contextual compression to retrieved chunks...")
            
            # Compress the documents with respect to the query
            scored = contextual_compression(query_text, scored, openai_client)
            click.echo(f"[info] Compression complete for {len(scored)} documents")
            
        except ImportError as e:
            click.echo(f"[warning] Contextual compression failed (advanced_rag module not found): {e}", err=True)
            click.echo("[info] To use contextual compression, make sure the advanced_rag module is available")
        except Exception as e:
            click.echo(f"[warning] Contextual compression failed: {e}", err=True)
            click.echo("[info] Continuing with uncompressed chunks")

    # Handle no matches
    if not scored:
        if hybrid:
            click.echo("[warning] No hybrid results for query.", err=True)
        elif filters:
            click.echo(f"[warning] No results matched filters: {filters}", err=True)
        else:
            click.echo("[warning] No results found for query.", err=True)
        return
    # Branch display: raw retrieval + answer or summary-only
    if raw:
        # Show raw retrieval hits
        for idx, point in enumerate(scored, start=1):
            score = getattr(point, 'score', None)
            score_str = f"{score:.4f}" if score is not None else "N/A"
            payload: dict[str, Any] = getattr(point, 'payload', {}) or {}
            
            # Check if this result came from a specific expanded query
            origin = ""
            if use_expansion and len(expanded_queries) > 1:
                # We don't track which expansion provided each result, but we could add that in a future version
                pass
            
            # Add hierarchical level information if available
            level_info = ""
            if hierarchical_search:
                level_val = payload.get('level', '')
                if level_val:
                    # Format with different colors for different levels
                    if level_val == 'document':
                        level_info = click.style(" [DOCUMENT]", fg='bright_blue')
                    elif level_val == 'section':
                        level_info = click.style(" [SECTION]", fg='bright_green')
                    elif level_val == 'chunk':
                        level_info = click.style(" [CHUNK]", fg='bright_yellow')
                
            click.echo(f"[{idx}] id={point.id}  score={score_str}{origin}{level_info}")
            if snippet:
                # Use compressed_text if available, otherwise fall back to text or chunk_text
                snippet_text = ""
                if compress and "compressed_text" in payload:
                    snippet_text = str(payload.get("compressed_text", "")).replace("\n", " ")
                    # Add indication that this is compressed
                    compression_ratio = payload.get("compression_ratio", 1.0)
                    snippet_text = snippet_text[:200].strip()
                    if snippet_text:
                        click.echo(f"    compressed snippet [{compression_ratio:.1%}]: {snippet_text}...")
                else:
                    # For hierarchical results, check text field first (used by document/section levels)
                    if hierarchical_search and "text" in payload:
                        snippet_text = str(payload.get("text", "")).replace("\n", " ")
                    else:
                        snippet_text = str(payload.get("chunk_text", "")).replace("\n", " ")
                    
                    snippet_text = snippet_text[:200].strip()
                    
                    if snippet_text:
                        # Add level indication for hierarchical results
                        if hierarchical_search and "level" in payload:
                            level_val = payload.get("level", "")
                            level_tag = f"[{level_val.upper()}] " if level_val else ""
                            click.echo(f"    {level_tag}snippet: {snippet_text}...")
                        else:
                            click.echo(f"    snippet: {snippet_text}...")
        
        # If we used query expansion, show a summary of the merged results
        if use_expansion and len(expanded_queries) > 1:
            click.secho(f"\n[info] These results were merged from {len(expanded_queries)} different query expansions", fg="cyan")
            
        # Generate answer if LLM model is specified
        if llm_model:
            # Gather context passages with token limiting
            context_chunks: list[str] = []
            max_ctx_chars = 1000
            
            # Define token limits based on model
            model_token_limits = {
                "gpt-3.5-turbo": 4096,
                "gpt-3.5-turbo-16k": 16384,
                "gpt-4": 8192,
                "gpt-4-32k": 32768,
                "gpt-4-turbo": 128000,
                "gpt-4o": 128000,
            }
            
            # Determine max tokens based on model (reserving ~20% for prompt and response)
            max_context_tokens = 4000  # Default if model not recognized
            for model_prefix, limit in model_token_limits.items():
                if llm_model.startswith(model_prefix):
                    max_context_tokens = int(limit * 0.8)
                    break
            
            # Simple token estimator (approximate - 1 token ~ 4 chars in English)
            def estimate_tokens(text: str) -> int:
                return int(len(text) / 4) + 1
            
            # System and question tokens (approximate)
            system_prompt = (
                "You are a helpful assistant. "
                "Using ONLY the provided context, answer the question. "
                "Do not use any external information or general knowledge. "
                "Do NOT hallucinate or add content not present in the context. "
                "If the context does not contain enough information to answer, "
                "explicitly state 'I don't have enough information to answer that question.'"
            )
            
            base_tokens = estimate_tokens(system_prompt) + estimate_tokens(query_text) + 150  # Added overhead
            available_tokens = max(0, max_context_tokens - base_tokens)
            current_token_count = 0
            separator = "\n\n---\n\n"
            separator_tokens = estimate_tokens(separator)
            
            # Prioritize results by score for token allocation
            for point in sorted(scored, key=lambda p: getattr(p, 'score', 0.0), reverse=True):
                payload: dict[str, Any] = getattr(point, 'payload', {}) or {}
                
                # Use compressed text if available and compression is enabled
                if compress and "compressed_text" in payload:
                    text = payload.get("compressed_text", "")
                # For hierarchical results, check text field first (used by document/section levels)
                elif hierarchical_search and "text" in payload:
                    text = payload.get("text", "")
                else:
                    text = payload.get("chunk_text", "")
                    
                if isinstance(text, str) and text:
                    snippet = text.replace("\n", " ")[:max_ctx_chars]
                    snippet_tokens = estimate_tokens(snippet)
                    
                    # Check if adding this chunk would exceed the token limit
                    chunk_tokens_with_separator = snippet_tokens + (separator_tokens if context_chunks else 0)
                    if current_token_count + chunk_tokens_with_separator > available_tokens:
                        # Skip if this would exceed our token budget
                        continue
                    
                    context_chunks.append(snippet)
                    current_token_count += chunk_tokens_with_separator
            
            # Log token usage for debugging
            click.secho(f"\n[info] Using approximately {current_token_count} tokens for context out of {max_context_tokens} available", fg="cyan")
            
            context = separator.join(context_chunks)
            # Guide the model to use context but allow factual elaboration
            # Guide the model to use context strictly and avoid hallucination
            system_msg = {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Using ONLY the provided context, answer the question. "
                    "Do not use any external information or general knowledge. "
                    "Do NOT hallucinate or add content not present in the context. "
                    "If the context does not contain enough information to answer, "
                    "explicitly state 'I don't have enough information to answer that question.'"
                )
            }
            user_msg = {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query_text}"}
            if hasattr(openai_client, "chat"):
                chat_resp = openai_client.chat.completions.create(model=llm_model, messages=[system_msg, user_msg])
                answer = chat_resp.choices[0].message.content
            else:
                chat_resp = openai_client.ChatCompletion.create(model=llm_model, messages=[system_msg, user_msg])
                answer = chat_resp.choices[0].message.content  # type: ignore
            click.secho("\n[answer]", fg="green")
            click.echo(answer.strip())
        return
    # Default: summary-only
    # Generate brief summary using LLM with token limiting
    context_chunks: list[str] = []
    max_ctx_chars = 1000
    
    # Reuse token estimation logic from above
    # Define token limits based on model
    model_token_limits = {
        "gpt-3.5-turbo": 4096,
        "gpt-3.5-turbo-16k": 16384,
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-4-turbo": 128000,
        "gpt-4o": 128000,
    }
    
    # Determine max tokens based on model (reserving ~20% for prompt and response)
    max_context_tokens = 4000  # Default if model not recognized
    for model_prefix, limit in model_token_limits.items():
        if llm_model.startswith(model_prefix):
            max_context_tokens = int(limit * 0.8)
            break
    
    # Simple token estimator (approximate - 1 token ~ 4 chars in English)
    def estimate_tokens(text: str) -> int:
        return int(len(text) / 4) + 1
    
    # System and question tokens (approximate)
    system_prompt = (
        "You are a helpful assistant. Synthesize a concise answer using the provided context."
    )
    
    base_tokens = estimate_tokens(system_prompt) + estimate_tokens(query_text) + 150  # Added overhead
    available_tokens = max(0, max_context_tokens - base_tokens)
    current_token_count = 0
    separator = "\n\n---\n\n"
    separator_tokens = estimate_tokens(separator)
    
    # Prioritize results by score for token allocation
    for point in sorted(scored, key=lambda p: getattr(p, 'score', 0.0), reverse=True):
        payload: dict[str, Any] = getattr(point, 'payload', {}) or {}
        
        # Use compressed text if available and compression is enabled
        if compress and "compressed_text" in payload:
            text = payload.get("compressed_text", "")
        # For hierarchical results, check text field first (used by document/section levels)
        elif hierarchical_search and "text" in payload:
            text = payload.get("text", "")
        else:
            text = payload.get("chunk_text", "")
            
        if isinstance(text, str) and text:
            snippet = text.replace("\n", " ")[:max_ctx_chars]
            snippet_tokens = estimate_tokens(snippet)
            
            # Check if adding this chunk would exceed the token limit
            chunk_tokens_with_separator = snippet_tokens + (separator_tokens if context_chunks else 0)
            if current_token_count + chunk_tokens_with_separator > available_tokens:
                # Skip if this would exceed our token budget
                continue
            
            context_chunks.append(snippet)
            current_token_count += chunk_tokens_with_separator
    
    # Log token usage for debugging
    click.secho(f"\n[info] Using approximately {current_token_count} tokens for context out of {max_context_tokens} available", fg="cyan")
    
    context = separator.join(context_chunks)
    
    # Guide the model to produce a detailed, context-only summary
    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful assistant. "
            "Provide a comprehensive summary based ONLY on the provided context to address the question. "
            "Do not use any external knowledge or add information not present. "
            "If the context does not include necessary information, "
            "explicitly state 'I don't have enough information to answer that question.'"
        )
    }
    user_msg = {
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {query_text}"}
    
    if hasattr(openai_client, "chat"):
        chat_resp = openai_client.chat.completions.create(model=llm_model, messages=[system_msg, user_msg])
        summary = chat_resp.choices[0].message.content
    else:
        chat_resp = openai_client.ChatCompletion.create(model=llm_model, messages=[system_msg, user_msg])
        summary = chat_resp.choices[0].message.content  # type: ignore
    
    click.secho("\n[summary]", fg="green")
    click.echo(summary.strip())
    
    # Optional RAG quality evaluation
    if evaluate:
        try:
            from advanced_rag import evaluate_rag_quality
            click.echo("\nEvaluating RAG response quality...")
            
            evaluation = evaluate_rag_quality(
                query=query_text,
                retrieved_chunks=context_chunks,
                generated_answer=summary,
                openai_client=openai_client
            )
            
            click.secho("\n[RAG Quality Evaluation]", fg="cyan")
            
            # Display scores
            scores = evaluation.get('scores', {})
            if scores:
                for metric, score in scores.items():
                    # Format the score and highlight poor scores in red
                    if isinstance(score, (int, float)):
                        score_color = "red" if score < 5 else "green"
                        click.secho(f"{metric.capitalize()}: {score}/10", fg=score_color)
                    else:
                        click.echo(f"{metric.capitalize()}: {score}")
            
            # Display feedback
            feedback = evaluation.get('feedback', {})
            if feedback:
                strengths = feedback.get('strengths', [])
                if strengths:
                    click.secho("\nStrengths:", fg="green")
                    for s in strengths:
                        click.echo(f"  - {s}")
                
                weaknesses = feedback.get('weaknesses', [])
                if weaknesses:
                    click.secho("\nAreas for Improvement:", fg="yellow")
                    for w in weaknesses:
                        click.echo(f"  ! {w}")
                
                suggestions = feedback.get('improvement_suggestions', [])
                if suggestions:
                    click.secho("\nSuggestions:", fg="blue")
                    for s in suggestions:
                        click.echo(f"  -> {s}")
                
        except ImportError:
            click.echo("[warning] RAG evaluation failed - advanced_rag module not found", err=True)
            click.echo("[info] To use RAG evaluation, install the advanced_rag module")
        except Exception as e:
            click.echo(f"[warning] RAG evaluation failed: {e}", err=True)

if __name__ == "__main__":  # pragma: no cover
    main()