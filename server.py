import os
import subprocess
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/inject",  methods=["GET", "POST"])
@app.route("/inject/", methods=["GET", "POST"])
@app.route("/injest",  methods=["GET", "POST"])
@app.route("/injest/", methods=["GET", "POST"])
@app.route("/ask",     methods=["GET", "POST"])
@app.route("/ask/",   methods=["GET", "POST"])
@app.route("/",        methods=["GET", "POST"])
def handle_slash():
    try:
        # Mattermost will send either GET?text=... or POST form text=...
        if request.method == "GET":
            text = request.args.get("text", "")
        else:
            text = request.form.get("text", "")
        # DEBUG: log incoming request values for debugging trigger and params
        app.logger.info(f"Slash command hit: path={request.path}, values={request.values.to_dict()}, form={request.form.to_dict()}")

        # Determine which slash command was invoked and validate its token
        trigger = request.values.get("command") or request.values.get("trigger_word") or ""
        trigger = trigger.strip()
        cmd_name = trigger.lstrip("/").lower()

        # Validate slash-command token (supporting per-command overrides)
        req_token = request.values.get("token")
        generic_token = os.environ.get("SLASH_TOKEN")
        inject_token = os.environ.get("SLASH_TOKEN_INJECT", generic_token)
        ask_token = os.environ.get("SLASH_TOKEN_ASK", generic_token)
        # Pick the expected token; alias 'injest' to same as 'inject'
        if cmd_name in ("inject", "injest"):
            expected_token = inject_token
        elif cmd_name == "ask":
            expected_token = ask_token
        else:
            expected_token = generic_token
        if expected_token and req_token != expected_token:
            return jsonify({"text": "Invalid token."}), 403

        # 'ask' triggers a RAG query
        if cmd_name == "ask":
            if not text:
                return jsonify({"text": "No text provided."}), 400
            import shlex
            args = shlex.split(text)
            cmd = ["python3", "-m", "query_rag"]
            qdrant_url = os.environ.get("QDRANT_URL")
            if qdrant_url and not any(arg.startswith("--qdrant-url") for arg in args):
                cmd += ["--qdrant-url", qdrant_url]
            cmd += args
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                answer = result.stdout.strip()
            except subprocess.CalledProcessError as e:
                return jsonify({"error": e.stderr.strip()}), 500
            return answer, 200, {"Content-Type": "text/plain"}

        # 'inject' ingests the current channel into the RAG collection
        elif cmd_name in ("inject", "injest"):
            import shlex
            import requests
            import uuid
            # Re‑use helper utilities and collection management from ingest_rag
            from ingest_rag import get_openai_client, load_documents, embed_and_upsert, chunk_text as _legacy_chunk_text, ensure_collection
            from qdrant_client import QdrantClient

            # Parse optional collection override; remaining args become sources
            args = shlex.split(text or "")
            # Detect --purge to clear and recreate the entire collection
            purge = False
            if "--purge" in args:
                purge = True
                args = [a for a in args if a != "--purge"]
            collection = "rag_data"
            if "--collection" in args:
                idx = args.index("--collection")
                if idx + 1 < len(args):
                    collection = args[idx + 1]
                # remove the flag and its value
                args.pop(idx)
                args.pop(idx)
            elif "-c" in args:
                idx = args.index("-c")
                if idx + 1 < len(args):
                    collection = args[idx + 1]
                args.pop(idx)
                args.pop(idx)

            # If sources (e.g. URLs or paths) were provided, ingest them instead of channel
            if args:
                # Initialize OpenAI and Qdrant clients
                openai_api_key = os.environ.get("OPENAI_API_KEY")
                if not openai_api_key:
                    return jsonify({"text": "Error: OPENAI_API_KEY not set."}), 200
                qdrant_url_env = os.environ.get("QDRANT_URL", "http://localhost:6333")
                qdrant_api_key = os.environ.get("QDRANT_API_KEY")
                openai_client = get_openai_client(openai_api_key)
                client = QdrantClient(url=qdrant_url_env, api_key=qdrant_api_key)
                # Purge existing collection if requested
                if purge:
                    try:
                        client.delete_collection(collection_name=collection)
                    except Exception as e:
                        return jsonify({"text": f"❌ Failed to purge collection '{collection}': {e}"}), 200
                    try:
                        ensure_collection(client, collection, vector_size=3072)
                    except Exception as e:
                        return jsonify({"text": f"❌ Failed to recreate collection '{collection}': {e}"}), 200

                total_chunks = 0
                # Helper imports for URL-based downloads
                import tempfile, urllib.parse
                for source in args:
                    local_source = source
                    # If source is a remote URL and looks like a PDF, download it first
                    if source.lower().startswith(("http://", "https://")):
                        parsed = urllib.parse.urlparse(source)
                        ext = os.path.splitext(parsed.path)[1].lower()
                        if ext == ".pdf":
                            # Download PDF to temporary file
                            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                            tmp_path = tmp_file.name
                            tmp_file.close()
                            try:
                                resp = requests.get(source, stream=True, timeout=60)
                                resp.raise_for_status()
                                with open(tmp_path, "wb") as f:
                                    for chunk in resp.iter_content(chunk_size=8192):
                                        f.write(chunk)
                                local_source = tmp_path
                            except Exception as e:
                                # Cleanup on failure
                                try:
                                    os.remove(tmp_path)
                                except Exception:
                                    pass
                                return jsonify({"text": f"❌ Failed to download {source}: {e}"}), 200
                    # Load documents from local file or other source via docling
                    docs = load_documents(local_source, chunk_size=500)
                    count = len(docs)
                    total_chunks += count
                    # Embed and upsert
                    embed_and_upsert(client, collection, docs, openai_client, batch_size=64, deterministic_id=True)
                    # Clean up temporary file if downloaded
                    if local_source != source:
                        try:
                            os.remove(local_source)
                        except Exception:
                            pass

                return f"Ingested {total_chunks} chunks from {len(args)} source(s) into '{collection}'.", 200, {"Content-Type": "text/plain"}

            # Mattermost API settings
            mattermost_url = os.environ.get("MATTERMOST_URL")
            mattermost_token = os.environ.get("MATTERMOST_TOKEN")
            if not mattermost_url or not mattermost_token:
                return jsonify({"text": "Error: MATTERMOST_URL and MATTERMOST_TOKEN must be set."}), 200

            channel_id = request.values.get("channel_id")
            headers = {"Authorization": f"Bearer {mattermost_token}"}
            per_page = 200
            page = 0
            messages: list[str] = []
            # Paginate through posts
            while True:
                resp = requests.get(
                    f"{mattermost_url}/api/v4/channels/{channel_id}/posts",
                    params={"page": page, "per_page": per_page},
                    headers=headers,
                )
                if resp.status_code != 200:
                    return jsonify({"text": f"Error fetching posts: {resp.status_code} {resp.text}"}), 200
                data = resp.json()
                posts = data.get("posts", {})
                order = data.get("order", [])
                if not order:
                    break
                for pid in order:
                    post = posts.get(pid)
                    if post and post.get("message"):
                        messages.append(post["message"])
                if len(order) < per_page:
                    break
                page += 1
            if not messages:
                return "No messages to ingest.", 200, {"Content-Type": "text/plain"}

            # ------------------------------------------------------------------
            # Use *docling* for extraction & chunking of the channel transcript
            # ------------------------------------------------------------------

            # Write the full channel transcript to a temporary .txt file so that
            # the existing `ingest_rag.load_documents()` helper – which relies
            # on docling for rich extraction / chunking – can be reused without
            # duplicating its logic here.

            import tempfile

            with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tmp_file:
                tmp_path = tmp_file.name
                # Concatenate all messages with a newline so that the transcript
                # roughly resembles a plain‑text document.  The message index is
                # preserved later via metadata on each chunk.
                for line in messages:
                    tmp_file.write(line.rstrip("\n") + "\n")

            # Leverage the high‑level loader which delegates to docling when
            # available (and gracefully falls back to a whitespace splitter
            # otherwise).  This keeps the /inject endpoint behaviour aligned
            # with the standalone *ingest_rag* CLI.
            from ingest_rag import load_documents, Document

            try:
                try:
                    docs_temp = load_documents(tmp_path, chunk_size=500, overlap=50)
                except BaseException:  # pragma: no cover – docling may be missing
                    # Fallback: simple whitespace chunking (legacy behaviour)
                    from ingest_rag import chunk_text as _legacy_chunk_text

                    docs_temp = []
                    for idx, chunk in enumerate(_legacy_chunk_text(
                        open(tmp_path, "r", encoding="utf-8", errors="ignore").read(),
                        max_chars=500,
                    )):
                        docs_temp.append(Document(content=chunk, metadata={"chunk_index": idx}))
            finally:
                # Clean up the temporary file regardless of success/failure.
                import os as _os
                try:
                    _os.remove(tmp_path)
                except FileNotFoundError:
                    pass

            # Enrich every generated Document with Mattermost‑specific metadata
            # (channel id + dummy message/chunk indexes when available).
            docs_list: list[Document] = []
            for global_idx, doc in enumerate(docs_temp):
                meta = doc.metadata.copy()
                meta.setdefault("source_channel", channel_id)
                # Best‑effort mapping of chunk to original message: we store the
                # *approximate* message index via integer division assuming a
                # 1‑to‑1 mapping between input messages and chunks – this is
                # mainly for traceability and is not guaranteed to be exact
                # when docling merges/splits lines differently.  The original
                # (docling‑generated) metadata is preserved.
                meta.setdefault("message_index", global_idx)
                meta.setdefault("chunk_index", 0)  # docling already indexes its chunks
                docs_list.append(Document(content=doc.content, metadata=meta))

            # Initialize OpenAI and Qdrant clients
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                return jsonify({"text": "Error: OPENAI_API_KEY not set."}), 200
            qdrant_url_env = os.environ.get("QDRANT_URL", "http://localhost:6333")
            qdrant_api_key = os.environ.get("QDRANT_API_KEY")
            openai_client = get_openai_client(openai_api_key)
            client = QdrantClient(url=qdrant_url_env, api_key=qdrant_api_key)
            # Purge existing collection if requested
            if purge:
                try:
                    client.delete_collection(collection_name=collection)
                except Exception as e:
                    return jsonify({"text": f"❌ Failed to purge collection '{collection}': {e}"}), 200
                try:
                    ensure_collection(client, collection, vector_size=3072)
                except Exception as e:
                    return jsonify({"text": f"❌ Failed to recreate collection '{collection}': {e}"}), 200

            # Embed and upsert all chunks (incremental)
            embed_and_upsert(client, collection, docs_list, openai_client, batch_size=64, deterministic_id=True)

            return (
                f"Ingested {len(docs_list)} chunks from {len(messages)} messages into '{collection}'.",
                200,
                {"Content-Type": "text/plain"},
            )

            # Unknown command
            return jsonify({"text": f"Unknown command '{trigger}'."}), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Return exception message to Mattermost
        return jsonify({"text": f"Error: {e}"}), 200

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    app.run(host=host, port=port)
