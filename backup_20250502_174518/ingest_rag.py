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

# Core dependencies
import click
from tqdm.auto import tqdm
import re
from dateutil.parser import parse as _parse_date

# Regex to detect ISO dates (YYYY-MM-DD) in text
DATE_REGEX = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
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
    
def _smart_chunk_text(text: str, max_chars: int, overlap: int = 0) -> list[str]:
    """
    Chunk text on paragraph and sentence boundaries up to max_chars,
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
    # apply overlap (character-level)
    if overlap > 0 and len(chunks) > 1:
        overlapped: list[str] = []
        for idx, chunk in enumerate(chunks):
            if idx == 0:
                overlapped.append(chunk)
            else:
                prev = overlapped[idx-1]
                # take last overlap chars from previous chunk
                ov = prev[-overlap:] if len(prev) >= overlap else prev
                overlapped.append(f"{ov} {chunk}")
        return overlapped
    return chunks


def semantic_chunk_text(text: str, max_chars: int, overlap: int = 0) -> list[str]:
    """
    Chunk text based on semantic topic boundaries.
    
    Args:
        text: The text to chunk
        max_chars: Maximum character length per chunk
        overlap: Overlap between chunks (not used in semantic chunking)
        
    Returns:
        List of semantically chunked text segments
    """
    # Import from advanced_rag if available, otherwise fall back to regular chunking
    try:
        from advanced_rag import semantic_chunk_text as advanced_semantic_chunk
        click.echo("[info] Using semantic chunking from advanced_rag module")
        chunks = advanced_semantic_chunk(text, max_chars=max_chars)
        
        # Verify we got valid chunks
        if chunks and all(isinstance(c, str) for c in chunks):
            click.echo(f"[info] Semantic chunking produced {len(chunks)} chunks")
            return chunks
        else:
            click.echo("[warning] Semantic chunking failed to produce valid chunks, falling back to regular chunking", err=True)
            return _smart_chunk_text(text, max_chars, overlap)
            
    except (ImportError, Exception) as e:
        click.echo(f"[warning] Semantic chunking not available or failed: {e}", err=True)
        click.echo("[info] Falling back to regular chunking", err=True)
        return _smart_chunk_text(text, max_chars, overlap)


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


def load_documents(source: str, chunk_size: int = 500, overlap: int = 50, crawl_depth: int = 0) -> List[Document]:
    """
    Use *docling* to load and chunk documents from *source*.

    The function tries to stay *schema-agnostic*.  For every item docling
    yields, we keep its raw representation as the ``payload`` (metadata), and
    attempt to locate a reasonable textual representation for embedding.
    """
    # ---------------------------------------------------------------------
    # Helper: chunk arbitrary raw *text* into smaller passages.  We keep the
    # helper nested so it can implicitly capture the *chunk_size* / *overlap*
    # parameters from the enclosing ``load_documents`` call, but we define it
    # right at the beginning of the function so that every subsequent code
    # path can freely reference it.
    # ---------------------------------------------------------------------

    def _chunk_text_tokenwise(text: str, metadata: dict[str, object]) -> List[Document]:
        """Split *text* into token-aware chunks, falling back to character
        splitting if the preferred tokenisers are unavailable.  A small helper
        that always returns a ``List[Document]`` with a ``chunk_index``
        injected into the copied *metadata* for downstream processing."""

        docs_out: list[Document] = []

        # 1. Try semantic chunking first (from advanced_rag)
        try:
            # Use semantic chunking for better context-aware chunks
            chunks = semantic_chunk_text(text, chunk_size, overlap)
            click.echo("[info] Used semantic chunking for text")
        except Exception as e:
            click.echo(f"[warning] Semantic chunking failed, trying fallbacks: {e}", err=True)
            # 2. Try docling's GPT-aware splitter.
            try:
                from docling.text import TextSplitter

                splitter = TextSplitter.from_model(  # type: ignore[attr-defined]
                    model="gpt-4.1-mini",
                    chunk_size=chunk_size,
                    chunk_overlap=overlap,
                )
                chunks = splitter.split(text)  # type: ignore[attr-defined]
            except Exception:
                # 3. Try LangChain's recursive character splitter.
                try:
                    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

                    lc_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=overlap,
                        separators=["\n\n", "\n", " "],
                    )
                    chunks = lc_splitter.split_text(text)
                except Exception:
                    # 4. Absolute last-ditch fallback – a very dumb char splitter.
                    chunks = _smart_chunk_text(text, chunk_size, overlap)

        for idx, chunk in enumerate(chunks):
            new_meta = dict(metadata)
            new_meta["chunk_index"] = idx
            docs_out.append(Document(content=chunk, metadata=new_meta))

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
            # The URL loader we need was moved to the latter.  We therefore try
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
                try:  # New ≥0.1.0 structure
                    from langchain_community.document_loaders import UnstructuredURLLoader
                    click.echo("[info] Using langchain-community.document_loaders.UnstructuredURLLoader")
                except ImportError:
                    pass
                    
            if UnstructuredURLLoader is None and have_langchain:
                try:  # Legacy structure (≤0.0.x)
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
                "Please install with: pip install langchain-community unstructured",
                err=True,
            )
        except Exception as e:
            click.echo(f"[warning] LangChain URL load failed for '{source}': {e}", err=True)
        
        # Fallback: Unstructured.io partition of remote HTML
        if not url_load_success:
            click.echo("[info] Trying fallback URL loader with Unstructured.io...")
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
                
                # Partition HTML into elements
                click.echo("[info] Partitioning HTML content...")
                elements = partition_html(tmp_path)
                
                if not elements:
                    click.echo(f"[warning] No elements extracted from '{source}'", err=True)
                else:
                    click.echo(f"[info] Extracted {len(elements)} elements from HTML")
                    
                    docs_url: list[Document] = []
                    for elem in elements:
                        if isinstance(elem, Table):
                            try:
                                md = elem.to_markdown()
                            except Exception:
                                md = elem.get_text()
                            docs_url.append(Document(content=md, metadata={"source": source, "is_table": True}))
                        elif hasattr(elem, 'text') and isinstance(elem.text, str):
                            txt = elem.text
                            docs_url.extend(_chunk_text_tokenwise(txt, {"source": source}))
                    
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
    # earlier code paths – see above.)

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
        # docling not installed – fall back to naïve whitespace chunking of the
        # entire file.  This avoids a hard dependency on docling when the user
        # simply wants to ingest a plain‑text transcript.
        try:
            with open(source, "r", encoding="utf-8", errors="replace") as fh:
                full_text = fh.read()
        except Exception as e:
            click.echo(f"[fatal] Could not read source '{source}': {e}", err=True)
            sys.exit(1)

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
                click.echo(f"[fatal] Could not read source '{source}': {e2}", err=True)
                sys.exit(1)
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


def embed_and_upsert(
    client,
    collection: str,
    docs: List[Document],
    openai_client,
    batch_size: int = 100,
    model_name: str = "text-embedding-3-large",
    deterministic_id: bool = False,
    parallel: int = 15,
):
    """Embed *docs* in batches and upsert them into Qdrant."""

    # Determine which OpenAI binding style is active
    is_openai_v1 = hasattr(openai_client, "embeddings")  # v1+ has .embeddings.create

    from qdrant_client.http import models as rest

    doc_iter = tqdm(iter_batches(docs, batch_size), total=(len(docs) + batch_size - 1) // batch_size, desc="Embedding & upserting")
    
    import time
    from concurrent.futures import ThreadPoolExecutor, TimeoutError

    def get_embeddings(texts, timeout=60):
        """Get embeddings with a timeout to prevent hanging"""
        start_time = time.time()
        try:
            if is_openai_v1:
                embeddings_response = openai_client.embeddings.create(
                    model=model_name, 
                    input=texts,
                    timeout=timeout
                )
                return [record.embedding for record in embeddings_response.data]
            else:  # old openai<=0.28 style
                embeddings_response = openai_client.Embedding.create(
                    model=model_name, 
                    input=texts,
                    request_timeout=timeout
                )
                return [record["embedding"] for record in embeddings_response["data"]]
        except Exception as e:
            elapsed = time.time() - start_time
            click.echo(f"[warning] Embedding API call failed after {elapsed:.1f}s: {e}", err=True)
            raise

    for batch in doc_iter:
        texts = [d.content for d in batch]
        
        # Use a thread with timeout to prevent hanging
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(get_embeddings, texts)
            try:
                embeddings = future.result(timeout=120)  # 2 minute timeout
            except TimeoutError:
                click.echo(f"[error] Embedding API call timed out after 120 seconds", err=True)
                click.echo(f"[info] Retrying with smaller batch...", err=True)
                
                # Try with half the batch size if it's big enough
                if len(texts) > 1:
                    mid = len(texts) // 2
                    try:
                        # Process first half
                        embeddings_first = get_embeddings(texts[:mid], timeout=60)
                        # Process second half
                        embeddings_second = get_embeddings(texts[mid:], timeout=60)
                        # Combine results
                        embeddings = embeddings_first + embeddings_second
                    except Exception as e:
                        click.echo(f"[error] Retry failed: {e}", err=True)
                        click.echo(f"[warning] Skipping this batch", err=True)
                        continue
                else:
                    click.echo(f"[warning] Batch size already minimal, skipping this text", err=True)
                    continue

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

        # Upsert points with configurable parallel workers (default=15);
        # Check if parallel parameter is supported by the client version
        import inspect
        client_upsert_params = inspect.signature(client.upsert).parameters
        if 'parallel' in client_upsert_params:
            client.upsert(collection_name=collection, points=points, parallel=parallel)
        else:
            # Parallel kwarg not supported by this client version
            client.upsert(collection_name=collection, points=points)


# ---------------------------------------------------------------------------
# Command‑line interface
# ---------------------------------------------------------------------------


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--env-file",
    type=click.Path(dir_okay=False, readable=True),
    default=None,
    help="Path to .env file with environment variables (if present, loaded before other options)."
)
@click.option("--source", required=True, help="Path/URL/DSN pointing to the corpus to ingest.")
@click.option("--collection", default="rag_data", show_default=True, help="Qdrant collection name to create/use.")
@click.option("--batch-size", type=int, default=100, show_default=True, help="Embedding batch size.")
@click.option("--openai-api-key", envvar="OPENAI_API_KEY", help="Your OpenAI API key (can also use env var OPENAI_API_KEY)")
@click.option("--qdrant-host", default="localhost", show_default=True, help="Qdrant host (ignored when --qdrant-url is provided).")
@click.option("--qdrant-port", type=int, default=6333, show_default=True, help="Qdrant port (ignored when --qdrant-url is provided).")
@click.option("--qdrant-url", help="Full Qdrant URL (e.g. https://*.qdrant.io:6333). Overrides host/port.")
@click.option("--qdrant-api-key", envvar="QDRANT_API_KEY", help="Qdrant API key if required (Cloud).")
@click.option("--distance", type=click.Choice(["Cosine", "Dot", "Euclid"], case_sensitive=False), default="Cosine", help="Vector distance metric.")
@click.option("--chunk-size", type=int, default=500, show_default=True, help="Chunk size (tokens) for docling chunker.")
@click.option("--chunk-overlap", type=int, default=50, show_default=True, help="Overlap (tokens) between chunks.")
@click.option("--crawl-depth", type=int, default=0, show_default=True, help="When SOURCE is a URL, crawl hyperlinks up to this depth (0=no crawl).")
@click.option("--parallel", type=int, default=15, show_default=True, help="Number of parallel workers for Qdrant upsert.")
@click.option("--generate-summaries/--no-generate-summaries", default=False, show_default=True,
              help="Generate and index brief summaries of each chunk for multi-granularity retrieval.")
@click.option("--quality-checks/--no-quality-checks", default=False, show_default=True,
              help="Perform post-ingest quality checks on chunk sizes and entity consistency.")
@click.option(
    "--bm25-index",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Path to write BM25 index JSON mapping point IDs to chunk_text."
         " Defaults to '<collection>_bm25_index.json'.",
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
    generate_summaries: bool,
    quality_checks: bool,
    bm25_index: str | None,
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

    click.echo(f"[info] Loading documents from source: {source} (chunk_size={chunk_size}, overlap={chunk_overlap}, crawl_depth={crawl_depth})")
    documents = load_documents(source, chunk_size, chunk_overlap, crawl_depth)
    click.echo(f"[info] Loaded {len(documents)} document(s)")

    if not documents:
        click.echo("[warning] No documents found – nothing to do.")
        return

    # -----------------------------------------------------------------
    # Enhanced metadata, quality checks, and optional summarization
    # -----------------------------------------------------------------
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
                    f"token_count={token_count} too small (<{min_tokens}) - will be filtered out",
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
    # Embed & upsert
    # ---------------------------------------------------------------------

    openai_client = get_openai_client(openai_api_key)

    embed_and_upsert(
        client,
        collection,
        documents,
        openai_client,
        batch_size=batch_size,
        parallel=parallel,
    )

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
