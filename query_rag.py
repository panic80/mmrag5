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
                nov = max(
                    _cosine_sim(
                        (getattr(p, 'vector', None) or []),
                        (getattr(s, 'vector', None) or [])
                    )
                    for s in selected
                )
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
@click.option("--rerank-top", type=int, default=0, show_default=True,
              help="Number of top retrieval results to re-rank using a cross-encoder (requires sentence-transformers).")
@click.option("--mmr-lambda", type=float, default=0.5, show_default=True,
              help="MMR diversity parameter (lambda: 0=full diversity, 1=full relevance). Always applied by default.")
@click.option("--filter", "-f", "filters", multiple=True, help="Filter by payload key=value. Can be used multiple times.")
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
    filters: Sequence[str],
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

    # Embed the query
    if hasattr(openai_client, "embeddings"):  # openai>=1.0 style
        resp = openai_client.embeddings.create(model=model, input=[query_text])
        vector = resp.data[0].embedding
    else:  # legacy style
        resp = openai_client.Embedding.create(model=model, input=[query_text])
        vector = resp["data"][0]["embedding"]  # type: ignore[index]

    # Build payload filter if specified
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
        click.echo(f"[info] Applying filters: {filters}")
    # Search in Qdrant (use new query_points if available)
    from qdrant_client.http.exceptions import UnexpectedResponse
    try:
        if hasattr(client, "query_points"):
            # New API returns a QueryResponse with .points attribute
            resp = client.query_points(
                collection_name=collection,
                query=vector,
                limit=k,
                with_payload=True,
                with_vectors=True,
                query_filter=filter_obj,
            )
            scored = getattr(resp, 'points', [])
        else:
            # Fallback to deprecated search()
            scored = client.search(
                collection_name=collection,
                query_vector=vector,
                limit=k,
                with_payload=True,
                with_vectors=True,
                query_filter=filter_obj,
            )
    except UnexpectedResponse as e:
        # likely missing collection
        msg = str(e)
        if "doesn't exist" in msg or "not found" in msg.lower():
            click.echo(f"[fatal] Collection '{collection}' not found.  Please ingest data first (e.g. ingest_rag.py --collection {collection}).", err=True)
            sys.exit(1)
        raise
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
        if bm25_index:
            # load pre-built JSON
            with open(bm25_index, "r") as f:
                id2text = json.load(f)
        else:
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
        # Guard against empty corpus – rank_bm25 crashes with ZeroDivisionError
        if not id2text:
            click.echo(
                "[warning] No documents with 'chunk_text' payload – skipping BM25 component of hybrid search.",
                err=True,
            )
        else:
            ids = list(id2text.keys())
            tokenized = [id2text[_id].split() for _id in ids]
            # rank_bm25 expects at least one document; otherwise it raises ZeroDivisionError
            if not tokenized:
                click.echo(
                    "[warning] BM25 tokenization produced an empty corpus – skipping BM25 component of hybrid search.",
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

                # Vector rankings (from initial Qdrant results)
                vec_rank = {point.id: rank for rank, point in enumerate(scored, start=1)}

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
                payload_map = {point.id: getattr(point, 'payload', {}) or {} for point in scored}
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
                vector_map = {p.id: getattr(p, 'vector', None) for p in orig_scored}
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
        # `scored` list coming from the pure‑vector Qdrant search so that the
        # rest of the pipeline (answer generation, summaries, etc.) continues
        # to function as expected.
    # MMR re-ranking for diversity + relevance
    try:
        if mmr_lambda is not None:
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
        # Load a default cross-encoder model
        click.echo(f"[info] Re-ranking top {rerank_top} results with cross-encoder...")
        ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        # Select initial candidates
        n_rerank = min(rerank_top, len(scored))
        candidates = scored[:n_rerank]
        # Prepare (query, passage) pairs for scoring
        pairs = [
            (query_text, (getattr(p, 'payload', {}) or {}).get("chunk_text", ""))
            for p in candidates
        ]
        # Predict relevance scores and sort
        rerank_scores = ce.predict(pairs)
        idxs = sorted(range(n_rerank), key=lambda i: rerank_scores[i], reverse=True)
        # Rebuild scored list in reranked order
        scored = [candidates[i] for i in idxs]

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
            click.echo(f"[{idx}] id={point.id}  score={score_str}")
            if snippet:
                snippet_text = str(payload.get("chunk_text", "")).replace("\n", " ")
                snippet_text = snippet_text[:200].strip()
                if snippet_text:
                    click.echo(f"    snippet: {snippet_text}…")
        # Generate answer if LLM model is specified
        if llm_model:
            # Gather context passages
            context_chunks: list[str] = []
            max_ctx_chars = 1000
            for point in scored:
                payload: dict[str, Any] = getattr(point, 'payload', {}) or {}
                text = payload.get("chunk_text")
                if isinstance(text, str) and text:
                    snippet = text.replace("\n", " ")[:max_ctx_chars]
                    context_chunks.append(snippet)
            context = "\n\n---\n\n".join(context_chunks)
            # Guide the model to use context but allow factual elaboration
            system_msg = {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Answer the question using the provided context as the primary source. "
                    "You may use your general knowledge to elaborate, but clearly indicate any information not present in the context."
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
    # Generate brief summary using LLM
    context_chunks: list[str] = []
    max_ctx_chars = 1000
    for point in scored:
        payload: dict[str, Any] = getattr(point, 'payload', {}) or {}
        text = payload.get("chunk_text")
        if isinstance(text, str) and text:
            snippet = text.replace("\n", " ")[:max_ctx_chars]
            context_chunks.append(snippet)
    context = "\n\n---\n\n".join(context_chunks)
    # Guide the model to produce a detailed summary based on context
    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful assistant. "
            "Provide a comprehensive summary of the context that addresses the question. "
            "Include all relevant details from the context. "
            "If any answer elements are not found in the context, state 'I don't know.'"
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

if __name__ == "__main__":  # pragma: no cover
    main()