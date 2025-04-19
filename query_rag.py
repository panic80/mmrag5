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


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--collection", default="rag_data", show_default=True, help="Qdrant collection name to query.")
@click.option("--k", type=int, default=5, show_default=True, help="Number of nearest neighbors to retrieve.")
@click.option("--snippet/--no-snippet", default=True, help="Show a text snippet of each result.")
@click.option("--model", default="text-embedding-3-large", show_default=True, help="OpenAI embedding model to use.")
@click.option("--qdrant-host", default="localhost", show_default=True, help="Qdrant host (ignored if --qdrant-url is provided).")
@click.option("--qdrant-port", type=int, default=6333, show_default=True, help="Qdrant port (ignored if --qdrant-url is provided).")
@click.option("--qdrant-url", help="Full Qdrant URL (overrides host/port).")
@click.option("--qdrant-api-key", envvar="QDRANT_API_KEY", help="API key for Qdrant (if required).")
@click.option("--openai-api-key", envvar="OPENAI_API_KEY", help="OpenAI API key.")
@click.option("--llm-model", default=None, help="LLM model for answer generation (e.g. gpt-4). Omit to skip generation.")
@click.option("--hybrid/--no-hybrid", default=False, help="Enable hybrid BM25 + vector search.")
@click.option("--bm25-index", type=click.Path(exists=True, dir_okay=False), default=None, help="Path to JSON file mapping point IDs to chunk_text for BM25 index.")
@click.option("--alpha", type=float, default=0.5, show_default=True, help="Weight for vector scores in hybrid fusion (0.0-1.0).")
@click.option("--bm25-top", type=int, default=None, help="Number of top BM25 docs to consider (default: k).")
@click.option("--rrf-k", type=float, default=60.0, show_default=True, help="Reciprocal Rank Fusion k hyperparameter.")
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
     hybrid: bool,
     bm25_index: str | None,
     alpha: float,
     bm25_top: int | None,
     rrf_k: float,
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
        try:
            import json
            from rank_bm25 import BM25Okapi
        except ImportError:
            click.echo("[fatal] rank_bm25 is required for hybrid search (pip install rank_bm25)", err=True)
            sys.exit(1)
        from qdrant_client.http.models import Filter as QFilter, HasIdCondition

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
        ids = list(id2text.keys())
        tokenized = [id2text[_id].split() for _id in ids]
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
        scored = [
            SimpleNamespace(id=pid, payload=payload_map.get(pid, {}), score=fused_scores.get(pid, 0.0))
            for pid in fused_ids
        ]

    # Handle no matches
    if not scored:
        if hybrid:
            click.echo("[warning] No hybrid results for query.", err=True)
        elif filters:
            click.echo(f"[warning] No results matched filters: {filters}", err=True)
        else:
            click.echo("[warning] No results found for query.", err=True)
        return
    # Display results
    for idx, point in enumerate(scored, start=1):
        # Score formatting
        score = getattr(point, 'score', None)
        score_str = f"{score:.4f}" if score is not None else "N/A"
        # Payload metadata
        payload: dict[str, Any] = getattr(point, 'payload', {}) or {}
        click.echo(f"[{idx}] id={point.id}  score={score_str}")
        for key, val in payload.items():
            click.echo(f"    {key}: {val}")
        # Optionally show a snippet of the chunk from stored payload
        if snippet:
            snippet_text = str(payload.get("chunk_text", "")).replace("\n", " ")
            snippet_text = snippet_text[:200].strip()
            if snippet_text:
                click.echo(f"    snippet: {snippet_text}…")
    # If an LLM model is specified, generate an answer using the retrieved chunks as context
    if llm_model:
        # Gather context passages (truncated) from stored payloads to fit context window
        context_chunks: list[str] = []
        max_ctx_chars = 1000  # limit per chunk
        for point in scored:
            payload: dict[str, Any] = getattr(point, 'payload', {}) or {}
            text = payload.get("chunk_text")
            if isinstance(text, str) and text:
                # replace newlines and truncate
                snippet = text.replace("\n", " ")[:max_ctx_chars]
                context_chunks.append(snippet)
        # Build prompt
        context = "\n\n---\n\n".join(context_chunks)
        system_msg = {"role": "system", "content": "You are a helpful assistant."}
        user_msg = {"role": "user", "content": f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {query_text}"}
        # Call chat completion
        if hasattr(openai_client, "chat"):
            chat_resp = openai_client.chat.completions.create(
                model=llm_model, messages=[system_msg, user_msg]
            )
            answer = chat_resp.choices[0].message.content
        else:
            chat_resp = openai_client.ChatCompletion.create(
                model=llm_model, messages=[system_msg, user_msg]
            )
            answer = chat_resp.choices[0].message.content  # type: ignore
        click.secho("\n[answer]", fg="green")
        click.echo(answer.strip())


if __name__ == "__main__":  # pragma: no cover
    main()