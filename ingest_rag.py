#!/usr/bin/env python3

"""ingest_rag.py

CLI utility for building a Retrieval‑Augmented Generation (RAG) vector
database with Qdrant.

The tool is intentionally *source‑agnostic* and *schema‑agnostic* – it relies
on the `docling` project to ingest documents from *any* kind of data source
(local files, URLs, databases, etc.).  After the documents are loaded, their
content is embedded with OpenAI’s `text-embedding-3-large` model and stored in
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
    Your OpenAI API key.  It can also be passed explicitly with the
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
from dataclasses import asdict, dataclass, field
from typing import Iterable, List, Sequence

import click
from tqdm.auto import tqdm
# deterministic UUID generation does not require hashlib

# Optional dependencies – import lazily so that the error message is clearer.


def _lazy_import(name: str):
    try:
        return __import__(name)
    except ImportError as exc:  # pragma: no cover – dev convenience
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
# Utilities
# ---------------------------------------------------------------------------


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


def get_openai_client(api_key: str):
    openai = _lazy_import("openai")
    # lazily set api key; property name changed in v1 openai lib
    try:
        openai.api_key = api_key
    except AttributeError:  # openai>=1.0
        client = openai.OpenAI(api_key=api_key)
        return client
    return openai


# ---------------------------------------------------------------------------
# Main ingestion routine
# ---------------------------------------------------------------------------


def load_documents(source: str, chunk_size: int = 500, overlap: int = 50) -> List[Document]:
    """
    Use *docling* to load and chunk documents from *source*.

    The function tries to stay *schema-agnostic*.  For every item docling
    yields, we keep its raw representation as the ``payload`` (metadata), and
    attempt to locate a reasonable textual representation for embedding.
    """
    # If source is a URL, fetch HTML and extract text without docling
    if source.lower().startswith(("http://", "https://")):
        # Fetch HTML content
        import requests
        from bs4 import BeautifulSoup

        try:
            response = requests.get(source)
            response.raise_for_status()
        except Exception as e:
            click.echo(f"[fatal] Failed to fetch URL '{source}': {e}", err=True)
            sys.exit(1)

        # Parse HTML and extract text
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator="\n")
        # Chunk text naively
        text_chunks = chunk_text(text, chunk_size)
        # Build Document objects
        return [
            Document(content=chunk, metadata={"source": source, "chunk_index": idx})
            for idx, chunk in enumerate(text_chunks)
        ]


    # Otherwise, attempt to use docling extract + chunk pipeline for local files
    try:
        # Use new docling modules; fall back to core package
        try:
            import docling.extract as dlextract
            import docling.chunk as dlchunk
        except ImportError:
            import docling_core.extract as dlextract
            import docling_core.chunk as dlchunk
        extractor = dlextract.TextExtractor(path=source, include_comments=True)
        extracted = extractor.run()
        chunker = dlchunk.Chunker(chunk_size=chunk_size, overlap=overlap)
        chunked_docs = chunker.run(extracted)
        documents: list[Document] = []
        for doc in chunked_docs:
            # determine textual content
            if hasattr(doc, "text") and isinstance(doc.text, str):
                text = doc.text
            elif hasattr(doc, "content") and isinstance(doc.content, str):
                text = doc.content
            else:
                text = str(doc)
            # copy any metadata dict
            metadata = getattr(doc, "metadata", {}) or {}
            documents.append(Document(content=text, metadata=dict(metadata)))
        return documents
    except ImportError:
        # docling.extract or docling.chunk not available; fall back to legacy API
        pass
    except Exception as e:
        click.echo(f"[warning] docling extract/chunk pipeline failed: {e}", err=True)
        click.echo("[warning] Falling back to legacy docling loader...", err=True)
    # Otherwise, delegate to legacy docling if available
    try:
        docling = _lazy_import("docling")
    except SystemExit:
        # docling not installed – fall back to naïve whitespace chunking of the
        # entire file.  This avoids a hard dependency on docling when the user
        # simply wants to ingest a plain‑text transcript.
        try:
            with open(source, "r", encoding="utf-8", errors="replace") as fh:
                full_text = fh.read()
        except Exception as e:
            click.echo(f"[fatal] Could not read source '{source}': {e}", err=True)
            sys.exit(1)

        text_chunks = chunk_text(full_text, chunk_size)
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
        # Try new API in submodule: DocumentConverter, fallback to plain-text on failure
        try:
            dc_mod = __import__("docling.document_converter", fromlist=["DocumentConverter"])
            converter = dc_mod.DocumentConverter()
            conv_res = converter.convert(source)
            # Extract textual content from the converted document
            doc_obj = conv_res.document
            if hasattr(doc_obj, "export_to_text") and callable(doc_obj.export_to_text):
                text = doc_obj.export_to_text()
            else:
                text = str(doc_obj)
            # Return single-chunk document
            return [Document(content=text, metadata={"source": source, "chunk_index": 0})]
        except Exception as e:
            # Fallback: treat as plain-text, naive whitespace chunking
            click.echo(f"[warning] docling.document_converter failed for '{source}': {e}. Falling back to plain-text chunking.", err=True)
            try:
                with open(source, "r", encoding="utf-8", errors="replace") as fh:
                    full_text = fh.read()
            except Exception as e2:
                click.echo(f"[fatal] Could not read source '{source}': {e2}", err=True)
                sys.exit(1)
            # Split on whitespace into chunks
            text_chunks = chunk_text(full_text, chunk_size)
            return [Document(content=chunk, metadata={"source": source, "chunk_index": idx})
                    for idx, chunk in enumerate(text_chunks)]

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

        documents.append(Document(content=text, metadata=metadata))

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


def embed_and_upsert(
    client,
    collection: str,
    docs: List[Document],
    openai_client,
    batch_size: int = 100,
    model_name: str = "text-embedding-3-large",
    deterministic_id: bool = False,
):
    """Embed *docs* in batches and upsert them into Qdrant."""

    # Determine which OpenAI binding style is active
    is_openai_v1 = hasattr(openai_client, "embeddings")  # v1+ has .embeddings.create

    from qdrant_client.http import models as rest

    doc_iter = tqdm(iter_batches(docs, batch_size), total=(len(docs) + batch_size - 1) // batch_size, desc="Embedding & upserting")

    for batch in doc_iter:
        texts = [d.content for d in batch]

        if is_openai_v1:
            embeddings_response = openai_client.embeddings.create(model=model_name, input=texts)
            embeddings = [record.embedding for record in embeddings_response.data]
        else:  # old openai<=0.28 style
            embeddings_response = openai_client.Embedding.create(model=model_name, input=texts)
            embeddings = [record["embedding"] for record in embeddings_response["data"]]

        points = []
        for doc, vector in zip(batch, embeddings):
            metadata = doc.metadata.copy()
            content = doc.content
            if deterministic_id:
                # Compute a deterministic UUID5 from metadata and content
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
            points.append(
                rest.PointStruct(id=point_id, vector=vector, payload=payload)
            )

        client.upsert(collection_name=collection, points=points)


# ---------------------------------------------------------------------------
# Command‑line interface
# ---------------------------------------------------------------------------


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--env-file", type=click.Path(exists=True, dir_okay=False, readable=True), default=".env", show_default=True, help="Path to .env file with environment variables (if present, loaded before other options).")
@click.option("--source", required=True, help="Path/URL/DSN pointing to the corpus to ingest.")
@click.option("--collection", default="rag_data", show_default=True, help="Qdrant collection name to create/use.")
@click.option("--batch-size", type=int, default=100, show_default=True, help="Embedding batch size.")
@click.option("--openai-api-key", envvar="OPENAI_API_KEY", help="Your OpenAI API key (can also use env var OPENAI_API_KEY)")
@click.option("--qdrant-host", default="localhost", show_default=True, help="Qdrant host (ignored when --qdrant-url is provided).")
@click.option("--qdrant-port", type=int, default=6333, show_default=True, help="Qdrant port (ignored when --qdrant-url is provided).")
@click.option("--qdrant-url", help="Full Qdrant URL (e.g. https://*.qdrant.io:6333). Overrides host/port.")
@click.option("--qdrant-api-key", envvar="QDRANT_API_KEY", help="Qdrant API key if required (Cloud).")
@click.option("--distance", type=click.Choice(["Cosine", "Dot", "Euclid"], case_sensitive=False), default="Cosine", help="Vector distance metric.")
@click.option("--chunk-size", type=int, default=500, show_default=True, help="Chunk size (characters) for docling chunker.")
@click.option("--chunk-overlap", type=int, default=50, show_default=True, help="Overlap (characters) between chunks.")
@click.option(
    "--bm25-index",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Path to write BM25 index JSON mapping point IDs to chunk_text."
         " Defaults to '<collection>_bm25_index.json'.",
)
def cli(
    env_file: str,
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
    bm25_index: str | None,
):
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
        except Exception:  # pragma: no cover – edge‑case, continue silently
            pass

    # Accept lower‑case variants (e.g. `openai_api_key=`) for convenience
    if "OPENAI_API_KEY" not in os.environ and "openai_api_key" in os.environ:
        os.environ["OPENAI_API_KEY"] = os.environ["openai_api_key"]
    if "QDRANT_API_KEY" not in os.environ and "qdrant_api_key" in os.environ:
        os.environ["QDRANT_API_KEY"] = os.environ["qdrant_api_key"]

    # ---------------------------------------------------------------------
    # Validate & set up dependencies
    # ---------------------------------------------------------------------

    # If the CLI flag wasn't provided, fall back to the environment
    if openai_api_key is None:
        openai_api_key = os.environ.get("OPENAI_API_KEY")

    if openai_api_key is None:
        click.echo(
            "[fatal] OPENAI_API_KEY is not set. Provide --openai-api-key, set it in the"\
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

    click.echo(f"[info] Loading documents from source: {source} (chunk_size={chunk_size}, overlap={chunk_overlap})")
    documents = load_documents(source, chunk_size, chunk_overlap)
    click.echo(f"[info] Loaded {len(documents)} document(s)")

    if not documents:
        click.echo("[warning] No documents found – nothing to do.")
        return

    # ---------------------------------------------------------------------
    # Create collection if it does not exist
    # ---------------------------------------------------------------------

    VECTOR_SIZE = 3072  # text-embedding-3-large output dimension

    ensure_collection(client, collection, vector_size=VECTOR_SIZE, distance=distance)

    # ---------------------------------------------------------------------
    # Embed & upsert
    # ---------------------------------------------------------------------

    openai_client = get_openai_client(openai_api_key)

    embed_and_upsert(client, collection, documents, openai_client, batch_size=batch_size)

    click.secho(f"\n[success] Ingestion completed. Collection '{collection}' now holds the embeddings.", fg="green")
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
