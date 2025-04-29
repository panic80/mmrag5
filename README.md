 # RAG Toolkit Documentation

 ## Table of Contents
 1. [Introduction](#introduction)
 2. [High-Level Architecture](#high-level-architecture)
 3. [Core Components](#core-components)
 4. [Ingestion Pipeline](#ingestion-pipeline)
 5. [Query Pipeline](#query-pipeline)
 6. [Flask Server & Slash‑Command Integration](#flask-server--slash-command-integration)
 7. [Installation & Setup](#installation--setup)
 8. [Usage Examples](#usage-examples)
 9. [Docker / docker‑compose Deployment](#docker--docker-compose-deployment)
 10. [Testing](#testing)
 11. [Future Improvements](#future-improvements)

 ---

 ## Introduction

 This project provides a **Retrieval‑Augmented Generation (RAG)** system that lets you:
 - **Ingest** arbitrary documents (local files, URLs, PDF, Mattermost channels, databases, etc.)
 - **Embed** document chunks using OpenAI's `text-embedding-3-large` model
 - **Store** embeddings in a Qdrant vector database
 - **Query** the collection via CLI or a simple REST API
 - **(Optional)** Fuse with BM25 rankings and generate LLM answers

 Use cases include knowledge‑base search, chatbot assistants with up‑to‑date context, and automated FAQs.

 ---

 ## High-Level Architecture

 ```mermaid
 flowchart LR
   subgraph Ingestion
     A["Source: File / URL / Channel"] --> B["load_documents()"]
     B --> C["Chunk & Metadata"]
     C --> D["OpenAI Embedding API"]
     D --> E["QdrantClient.upsert()"]
     E --> F["Vector Collection"]
   end

   subgraph Query
     U["User Query"] --> V["Embed Query via OpenAI"]
     V --> W["QdrantClient.search()"]
     W --> X["Retrieve Top-K Chunks"]
     X --> Y["Display Snippets"]
     X --> Z["LLM Answer Generation (optional)"]
   end
 ```

 ---

 ## Core Components

 - **ingest_rag.py**: CLI to ingest any source into Qdrant, using docling for extraction & chunking.
 - **query_rag.py**: CLI to query the vector store, supporting hybrid BM25 + vector search and optional LLM answer generation.
 - **server.py**: Flask app exposing `/health`, `/inject`, `/ask` endpoints for Mattermost integration.
 - **Dockerfile & docker-compose.yml**: Containerize Qdrant and the app, exposing ports 6333 and 5000.
 - **requirements.txt**: Python dependencies (`qdrant-client`, `openai`, `docling-core`, `flask`, etc.).

 ---

 ## Ingestion Pipeline

 ```mermaid
 sequenceDiagram
   participant User
   participant CLI as ingest_rag.py
   participant OpenAI
   participant Qdrant

   User->>CLI: python ingest_rag.py --source <SRC> --collection my_rag
   CLI->>CLI: load_documents(source)
   CLI->>OpenAI: embeddings.create(model, chunks)
   OpenAI-->>CLI: vectors
   CLI->>Qdrant: upsert(points)
   Qdrant-->>CLI: OK
   CLI-->>User: "[success] Ingestion completed…"
 ```

 **Steps**:
 1. **Load & Chunk**: arbitrary source → `Document(content, metadata)`
 2. **Batch & Embed**: call OpenAI embedding API
 3. **Upsert**: push vectors + payload to Qdrant

 ---

 ## Query Pipeline

 ```mermaid
 flowchart LR
   U[User Query] --> E[Embed via OpenAI]
   E --> Q[Qdrant Search]
   Q --> R[Format & Display]
   R --> O{LLM Model?}
   O -- Yes --> L[ChatCompletion using context]
   O -- No --> Z[Done]
 ```

 - **Hybrid Search**: Builds BM25 index over stored chunks and fuses with vector ranks using Reciprocal Rank Fusion.
 - **LLM Answer**: Aggregates top‑K chunks as context and calls OpenAI ChatCompletion.

 ---

 ## Flask Server & Slash‑Command Integration

 - **Endpoints**:
   - `GET/POST /health` → `{ "status": "ok" }`
   - `POST /inject` → Ingest channel or provided sources
   - `POST /ask` → Query the RAG collection
 - **Token Validation**: Uses `SLASH_TOKEN`, `SLASH_TOKEN_INJECT`, `SLASH_TOKEN_ASK`.
 - **Mattermost Integration**:
   - Requires environment variables: `MATTERMOST_URL`, `MATTERMOST_TOKEN`, `SLASH_TOKEN`, `SLASH_TOKEN_INJECT`, `SLASH_TOKEN_ASK` (see [Installation & Setup](#installation--setup)).
   - Configure slash commands in Mattermost:
     - `/inject` → `https://<your-app-url>/inject`
     - `/ask`    → `https://<your-app-url>/ask`
   - Use commands in chat:
     - `/inject --collection my_collection_name`  (ingest current channel or provided sources)
     - `/ask What is our refund policy?`          (query the RAG collection)

 ---

 ## Installation & Setup

 1. Clone the repo and `cd` into it.
 2. Create a `.env` with:
    ```ini
    OPENAI_API_KEY=...
    QDRANT_API_KEY=...    # if using managed Qdrant Cloud
    SLASH_TOKEN=...
    SLASH_TOKEN_INJECT=...
    SLASH_TOKEN_ASK=...
    MATTERMOST_URL=...
    MATTERMOST_TOKEN=...
    ```
 3. (Optional) Create & activate a Python virtualenv.
 4. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

 ---

 ## Usage Examples

 **Ingest Local Folder**:
 ```bash
 python ingest_rag.py --source ./my_docs --collection knowledge
 ```

 **Ingest PDF from URL**:
 ```bash
 python ingest_rag.py --source https://example.com/report.pdf
 ```

 **Query CLI** (default top-150 chunks):
 ```bash
 python query_rag.py --collection knowledge "What is the refund policy?"
 ```

 **Hybrid + LLM Answer**:
 ```bash
 python query_rag.py --hybrid --bm25-top 10 --alpha 0.6 --llm-model gpt-4 "Summarize policy"
 ```

 **Slash‑Command (Mattermost)**:
 - Configure `/inject` and `/ask` in Mattermost pointing to this service.

 ### Advanced Examples

 **Hybrid Search with BM25 Fusion**:
 ```bash
 python query_rag.py \
   --collection knowledge \
   --hybrid \
   --bm25-index bm25_index.json \
   --alpha 0.7 \
   --bm25-top 10 \
   --rrf-k 50 \
   "Explain our refund policy."
 ```

 **Filtered Query**:
 ```bash
 python query_rag.py \
   --collection reports \
   -f author=Alice \
   -f type=financial \
   "latest quarterly earnings"
 ```

 **Ingest with Custom Chunking & Distance**:
 ```bash
 python ingest_rag.py \
   --source ./manuals \
   --collection product_docs \
   --chunk-size 1000 \
   --chunk-overlap 200 \
   --distance Euclid \
   --batch-size 50
 ```

 **Mattermost `/inject` Sling Command with Purge**:
 ```bash
 /inject --collection knowledge --purge
 ```

 ## CLI Flags

 ### ingest_rag.py
 - `--env-file <path>`         Path to .env file (default: `.env`).
 - `--source <path|URL|DSN>`  Source to ingest (required).
 - `--collection <name>`      Qdrant collection name (default: `rag_data`).
 - `--batch-size <int>`       Number of documents per embedding batch (default: 100).
 - `--openai-api-key <key>`   OpenAI API key (or use `OPENAI_API_KEY` env var).
 - `--qdrant-host <host>`     Qdrant host (default: `localhost`).
 - `--qdrant-port <port>`     Qdrant port (default: 6333).
 - `--qdrant-url <url>`       Full Qdrant URL (overrides host/port).
 - `--qdrant-api-key <key>`   Qdrant API key (or use `QDRANT_API_KEY` env var).
 - `--distance <metric>`      Vector distance: `Cosine`, `Dot`, `Euclid` (default: `Cosine`).
 - `--chunk-size <int>`       Max characters per chunk (default: 500).
 - `--chunk-overlap <int>`    Overlap characters between chunks (default: 50).
- `--bm25-index <path>`       Path to write BM25 JSON index mapping point IDs to chunk_text (default: `<collection>_bm25_index.json`).

 ### query_rag.py
 - `--collection <name>`      Qdrant collection to query (default: `rag_data`).
 - `--k <int>`                Number of nearest neighbors to retrieve (default: 150).
 - `--snippet/--no-snippet`   Show or hide text snippets (default: show).
 - `--model <name>`           OpenAI embedding model (default: `text-embedding-3-large`).
 - `--qdrant-host <host>`     Qdrant host (default: `localhost`).
 - `--qdrant-port <port>`     Qdrant port (default: 6333).
 - `--qdrant-url <url>`       Full Qdrant URL (overrides host/port).
 - `--qdrant-api-key <key>`   Qdrant API key (or use `QDRANT_API_KEY` env var).
 - `--openai-api-key <key>`   OpenAI API key (or use `OPENAI_API_KEY` env var).
 - `--llm-model <name>`       LLM for answer generation (default: `gpt-4.1-mini`; set to empty to skip).
 - `--raw`                    Show raw retrieval hits and full answer (requires `--llm-model`).
 - `--hybrid/--no-hybrid`     Enable hybrid BM25 + vector search (on by default; disable with `--no-hybrid`).
 - `--bm25-index <path>`      Path to JSON BM25 index file (defaults to `<collection>_bm25_index.json`).
 - `--alpha <float>`          Weight for vector score in fusion (default: 0.5).
 - `--bm25-top <int>`         Number of top BM25 docs (default: same as `k`).
 - `--rrf-k <float>`          RRF hyperparameter (default: 60.0).
 - `-f, --filter key=value`   Filter by payload key=value (can repeat).
 - `<query>`                  Query text (positional argument).

 ---
 ## Automated Evaluation

 Use the included `evaluate_rag.py` to run automated RAG evaluation with RAGAS, Constitutional-Judge, and LangGraph.

 1. Install evaluation packages:
 ```bash
 pip install ragas constitutional-judge langgraph
 ```

 2. Prepare a test file (`tests/eval_cases.jsonl`) with one JSON object per line:
 ```jsonl
 {"query": "What is our refund policy?", "ground_truth": "Our refund policy is ...", "relevant_doc_ids": ["id1","id2",...]}
 ```

 3. Run evaluation:
 ```bash
 python evaluate_rag.py \
   --test-file tests/eval_cases.jsonl \
   --collection my_collection \
   --qdrant-url http://localhost:6333 \
   --openai-api-key $OPENAI_API_KEY \
   --framework ragas \
   --framework cj \
   --framework langgraph
 ```

 The script will output average scores per framework.
 ---

 ## Docker / docker‑compose Deployment

 ```bash
 docker-compose up -d
 ```

 - Qdrant: `localhost:6333`
 - App: `localhost:5000`

 ---

 ## Testing

 ```bash
 pytest -q
 ```

 ---

 ## Future Improvements

 - Authentication & RBAC
 - UI Dashboard for ingest & query
 - Support for other vector stores (Weaviate, Pinecone)
 - Fine‑tuned or private embedding models
 - Caching layer & monitoring
 - Incremental ingestion & differential updates
 - Enhanced semantic chunking