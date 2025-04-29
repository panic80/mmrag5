import os
import subprocess
import threading
from flask import Flask, request, jsonify
import requests

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
        # Context for asynchronous replies
        response_url = request.values.get("response_url")  # Slack-style callback URL
        channel_id = request.values.get("channel_id")      # Mattermost channel ID
        mattermost_url = os.environ.get("MATTERMOST_URL")  # e.g. https://your-mattermost-server
        mattermost_token = os.environ.get("MATTERMOST_TOKEN")
        # (Using Personal Access Token for Mattermost REST API)

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
            # Build command
            args = shlex.split(text)
            cmd = ["python3", "-m", "query_rag"]
            qdrant_url = os.environ.get("QDRANT_URL")
            if qdrant_url and not any(arg.startswith("--qdrant-url") for arg in args):
                cmd += ["--qdrant-url", qdrant_url]
            cmd += args
            # Asynchronous execution
            def run_and_post():
                # Launch the query subprocess and stream output back to Mattermost
                try:
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                except Exception as e:
                    error_msg = f"Failed to start query: {e}"
                    try:
                        if mattermost_url and mattermost_token and channel_id:
                            headers = {"Authorization": f"Bearer {mattermost_token}"}
                            requests.post(
                                f"{mattermost_url}/api/v4/posts",
                                headers=headers,
                                json={"channel_id": channel_id, "message": error_msg},
                            )
                        else:
                            app.logger.error("Missing Mattermost URL/token/channel_id for posting error message")
                    except Exception:
                        app.logger.exception("Failed to post error message")
                    return

                # Stream subprocess output line by line
                if proc.stdout:
                    for raw_line in proc.stdout:
                        line = raw_line.rstrip()
                        try:
                            if mattermost_url and mattermost_token and channel_id:
                                headers = {"Authorization": f"Bearer {mattermost_token}"}
                                requests.post(
                                    f"{mattermost_url}/api/v4/posts",
                                    headers=headers,
                                    json={"channel_id": channel_id, "message": line},
                                )
                            else:
                                app.logger.error("Missing Mattermost URL/token/channel_id for posting progress")
                        except Exception:
                            app.logger.exception("Failed to post query output line")
                # Wait for process to exit and report non-zero exit code if any
                retcode = proc.wait()
                if retcode != 0:
                    error_msg = f"Query process exited with code {retcode}"
                    try:
                        if mattermost_url and mattermost_token and channel_id:
                            headers = {"Authorization": f"Bearer {mattermost_token}"}
                            requests.post(
                                f"{mattermost_url}/api/v4/posts",
                                headers=headers,
                                json={"channel_id": channel_id, "message": error_msg},
                            )
                        else:
                            app.logger.error("Missing Mattermost URL/token/channel_id for posting error code")
                    except Exception:
                        app.logger.exception("Failed to post exit code message")
            # Always run asynchronously
            threading.Thread(target=run_and_post, daemon=True).start()
            # Immediate acknowledgement
            return jsonify({"text": "Processing your query..."}), 200

        # 'inject' ingests the current channel into the RAG collection
        elif cmd_name in ("inject", "injest"):
            import shlex
            def run_inject():
                import requests
                import uuid
                import os
                import tempfile
                import urllib.parse
                from ingest_rag import get_openai_client, load_documents, embed_and_upsert, ensure_collection
                from qdrant_client import QdrantClient

                # Helper to post messages (Slack via response_url or Mattermost via REST API)
                def post(msg: str):
                    try:
                        # Mattermost posts take priority when properly configured
                        if mattermost_url and mattermost_token and channel_id:
                            hdr = {"Authorization": f"Bearer {mattermost_token}"}
                            requests.post(
                                f"{mattermost_url}/api/v4/posts",
                                headers=hdr,
                                json={"channel_id": channel_id, "message": msg},
                            )
                        # Fallback to Slack-style response_url if available
                        elif response_url:
                            requests.post(
                                response_url,
                                json={"response_type": "in_channel", "text": msg},
                            )
                        else:
                            app.logger.error("No callback available to post message: %s", msg)
                    except Exception:
                        app.logger.exception("Failed to post message: %s", msg)

                try:
                    args = shlex.split(text or "")
                    purge = False
                    if "--purge" in args:
                        purge = True
                        args = [a for a in args if a != "--purge"]
                    collection = "rag_data"
                    # Handle collection override
                    if "--collection" in args:
                        idx = args.index("--collection")
                        if idx + 1 < len(args):
                            collection = args[idx + 1]
                        args.pop(idx); args.pop(idx)
                    elif "-c" in args:
                        idx = args.index("-c")
                        if idx + 1 < len(args):
                            collection = args[idx + 1]
                        args.pop(idx); args.pop(idx)

                    # Initialize Qdrant & OpenAI clients
                    openai_api_key = os.environ.get("OPENAI_API_KEY")
                    qdrant_url_env = os.environ.get("QDRANT_URL", "http://localhost:6333")
                    qdrant_api_key = os.environ.get("QDRANT_API_KEY")
                    openai_client = get_openai_client(openai_api_key)
                    client = QdrantClient(url=qdrant_url_env, api_key=qdrant_api_key)

                    # Handle purge or ensure collection
                    if purge:
                        try:
                            client.delete_collection(collection_name=collection)
                            post(f"✅ Purged existing collection '{collection}'.")
                        except Exception as e:
                            post(f"⚠️ Purge skipped: collection '{collection}' may not exist. ({e})")
                        # Recreate empty collection
                        ensure_collection(client, collection, vector_size=3072)
                        # Purge-only: do not proceed to re-ingest
                        return
                    else:
                        ensure_collection(client, collection, vector_size=3072)

                    # No explicit sources ⇒ ingest the current channel transcript.

                    if not args:
                        # Fetch channel messages
                        if not mattermost_url:
                            post("❌ MATTERMOST_URL is not configured – unable to fetch channel messages.")
                            return
                        msgs: list[str] = []
                        hdrs = {"Authorization": f"Bearer {mattermost_token}"} if mattermost_token else {}
                        per_page = 200
                        page = 0
                        while True:
                            resp_ct = requests.get(
                                f"{mattermost_url}/api/v4/channels/{channel_id}/posts",
                                params={"page": page, "per_page": per_page},
                                headers=hdrs,
                            )
                            if resp_ct.status_code != 200:
                                post(
                                    f"❌ Error fetching posts: {resp_ct.status_code} {resp_ct.text}"
                                )
                                return

                            data_ct = resp_ct.json()
                            posts = data_ct.get("posts", {})
                            order = data_ct.get("order", [])

                            if not order:
                                break

                            for pid in order:
                                p = posts.get(pid)
                                if p and p.get("message"):
                                    msgs.append(p["message"])

                            if len(order) < per_page:
                                break
                            page += 1

                        if not msgs:
                            post("No messages to ingest – channel is empty.")
                            return

                        # 2. Dump the transcript to a temporary file so that we can
                        #    delegate rich extraction / chunking to docling via
                        #    `load_documents()`.
                        import tempfile, os as _os

                        with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tmp_fp:
                            tmp_path = tmp_fp.name
                            tmp_fp.write("\n".join(m.rstrip("\n") for m in msgs))

                        # 3. Let `load_documents()` (docling) do the heavy lifting.
                        try:
                            try:
                                docs_temp = load_documents(tmp_path, chunk_size=500, overlap=50)
                            except BaseException:
                                # Fallback: cheap whitespace splitter as a last resort
                                from ingest_rag import chunk_text as _chunk_text, Document as _Document

                                docs_temp = []
                                for idx, chunk in enumerate(_chunk_text("\n".join(msgs), max_chars=500)):
                                    docs_temp.append(_Document(content=chunk, metadata={"chunk_index": idx}))
                        finally:
                            try:
                                _os.remove(tmp_path)
                            except FileNotFoundError:
                                pass

                        # 4. Enrich each Document with Mattermost‑specific metadata
                        from ingest_rag import Document as _Doc

                        docs: list[_Doc] = []
                        for g_idx, doc in enumerate(docs_temp):
                            meta = doc.metadata.copy()
                            meta.setdefault("source_channel", channel_id)
                            # Best‑effort mapping back to original message index – useful
                            # for traceability when showing snippets.
                            meta.setdefault("message_index", g_idx)
                            docs.append(_Doc(content=doc.content, metadata=meta))

                        total_chunks = len(docs)
                        post(
                            f"Chunked channel into {total_chunks} chunk(s) from {len(msgs)} "
                            "messages – embedding & upserting…"
                        )

                        embed_and_upsert(
                            client,
                            collection,
                            docs,
                            openai_client,
                            batch_size=64,
                            deterministic_id=True,
                        )

                        post(
                            f"✅ Ingested {total_chunks} chunks from {len(msgs)} messages into "
                            f"'{collection}'."
                        )
                        return

                    # Ingest provided sources
                    total_sources = len(args)
                    post(f"Starting ingestion of {total_sources} source(s) into '{collection}'...")
                    total_chunks = 0
                    for idx, source in enumerate(args, start=1):
                        local_src = source
                        # Download remote source to local file so docling can extract & chunk
                        if source.lower().startswith(("http://", "https://")):
                            parsed = urllib.parse.urlparse(source)
                            ext = os.path.splitext(parsed.path)[1].lower()
                            # Remote PDF: stream to temp .pdf file
                            if ext == ".pdf":
                                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                                tmp_path = tmp.name; tmp.close()
                                try:
                                    resp = requests.get(source, stream=True, timeout=60)
                                    resp.raise_for_status()
                                    with open(tmp_path, "wb") as f:
                                        for chunk in resp.iter_content(chunk_size=8192):
                                            f.write(chunk)
                                    local_src = tmp_path
                                except Exception as e:
                                    post(f"❌ Download failed: {e}")
                                    return
                            else:
                                # Remote HTML/other: fetch content to temp .html file
                                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
                                tmp_path = tmp.name; tmp.close()
                                try:
                                    resp = requests.get(source, timeout=60)
                                    resp.raise_for_status()
                                    # Write as text for HTML extraction
                                    with open(tmp_path, "w", encoding="utf-8", errors="replace") as f:
                                        f.write(resp.text)
                                    local_src = tmp_path
                                except Exception as e:
                                    post(f"❌ Download failed: {e}")
                                    return
                        # Load, extract and chunk via docling
                        docs = load_documents(local_src, chunk_size=500)
                        cnt = len(docs); total_chunks += cnt
                        embed_and_upsert(client, collection, docs, openai_client, batch_size=64, deterministic_id=True)
                        post(f"[{idx}/{total_sources}] Processed {cnt} chunks from '{source}'. Total: {total_chunks} chunks.")
                        # Cleanup temp PDF
                        if local_src != source:
                            try: os.remove(local_src)
                            except: pass
                    post(f"Ingested {total_chunks} chunks from {total_sources} source(s) into '{collection}'.")
                except BaseException as e:
                    # Catch SystemExit raised by _lazy_import as well as regular exceptions
                    post(f"❌ Ingestion failed: {e}")
            # Parse flags early for immediate ack
            import shlex
            args = shlex.split(text or "")
            purge_flag = "--purge" in args
            # Launch ingestion thread and immediately acknowledge
            threading.Thread(target=run_inject, daemon=True).start()
            # Inform user if purge was requested
            if purge_flag:
                # Acknowledge purge-only
                ack_msg = "Purging existing collection. Progress will be posted shortly."
            else:
                ack_msg = "Ingestion started... progress will be posted shortly."
            return jsonify({"text": ack_msg}), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Return exception message to Mattermost
        return jsonify({"text": f"Error: {e}"}), 200

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    app.run(host=host, port=port)
