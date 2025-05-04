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
    # Bare GET to root path can be used as a health check without slash payloads
    if request.path == "/" and request.method == "GET":
        return jsonify({"status": "ok"}), 200
    try:
        # Parse incoming payload (JSON or form)
        payload = request.get_json(silent=True) or request.values
        # Extract text parameter
        text = payload.get("text", "")
        # DEBUG: log incoming request values
        try:
            app.logger.info(f"Slash command hit: path={request.path}, payload={payload}")
        except Exception:
            pass
        # Context for asynchronous replies
        response_url = payload.get("response_url")
        channel_id = payload.get("channel_id")
        mattermost_url = os.environ.get("MATTERMOST_URL")  # e.g. https://your-mattermost-server
        mattermost_token = os.environ.get("MATTERMOST_TOKEN")
        # (Using Personal Access Token for Mattermost REST API)

        # Determine which slash command was invoked and validate its token
        trigger = payload.get("command") or payload.get("trigger_word") or ""
        trigger = trigger.strip()
        cmd_name = trigger.lstrip("/").lower()

        # Validate slash-command token (supporting per-command overrides)
        req_token = payload.get("token")
        generic_token = os.environ.get("SLASH_TOKEN")
        inject_token = os.environ.get("SLASH_TOKEN_INJECT", generic_token)
        ask_token = os.environ.get("SLASH_TOKEN_ASK", generic_token)
        # Pick the expected token; alias 'injest' to same as 'inject'.
        if cmd_name in ("inject", "injest"):
            expected_token = inject_token
        elif cmd_name == "ask":
            expected_token = ask_token
        else:
            expected_token = generic_token
        # ------------------------------------------------------------------
        # Harden token validation
        # ------------------------------------------------------------------

        # 1. Refuse the request outright if *no* token is configured for the
        #    invoked command – this prevents accidentally exposing the
        #    endpoint when the admin forgot to set the environment variable.
        if not expected_token:
            # Service unavailable until the administrator configures a token.
            return jsonify({"text": "slash token not set"}), 503

        # 2. Explicitly reject mismatching tokens.
        if req_token != expected_token:
            return jsonify({"text": "Invalid token."}), 403

        # 'ask' triggers a RAG query
        if cmd_name == "ask":
            if not text:
                return jsonify({"text": "No text provided."}), 400
            import shlex
            # Build command
            args = shlex.split(text)
            cmd = ["python3", "-m", "query_rag"]
            
            # Add the --use-expansion flag by default if not explicitly specified
            if not any(arg == "--use-expansion" or arg == "--no-use-expansion" for arg in args):
                cmd.append("--use-expansion")
                
            # Add Qdrant URL if available
            qdrant_url = os.environ.get("QDRANT_URL")
            if qdrant_url and not any(arg.startswith("--qdrant-url") for arg in args):
                cmd += ["--qdrant-url", qdrant_url]
                
            cmd += args
            # Asynchronous execution
            def run_and_post():
                # Launch the query subprocess and stream output back to Mattermost
                # Track whether REST calls are still worth trying for this request
                rest_usable: dict[str, bool] = {"ok": bool(mattermost_url and mattermost_token and channel_id)}

                def _post_message(txt: str):
                    """Post *txt* to the originating channel.

                    1. Try the Mattermost REST API (only once it succeeds).
                    2. If it is unavailable or replies 401/403, permanently
                       disable further REST attempts for this request and fall
                       back to `response_url` so we don’t spam the log.
                    """

                    # Fast-path: if we already know REST is unusable, skip it.
                    if rest_usable["ok"]:
                        try:
                            hdrs = {"Authorization": f"Bearer {mattermost_token}"}
                            resp = requests.post(
                                f"{mattermost_url}/api/v4/posts",
                                headers=hdrs,
                                json={"channel_id": channel_id, "message": txt},
                                timeout=10,
                            )
                            if resp.status_code in (200, 201):
                                return  # success
                            # On auth errors, stop trying REST for the rest of this request
                            if resp.status_code in (401, 403):
                                rest_usable["ok"] = False
                            app.logger.warning(
                                "Posting via REST API failed with %s – falling back to response_url",
                                resp.status_code,
                            )
                        except Exception:
                            rest_usable["ok"] = False
                            app.logger.exception("REST API post failed – falling back to response_url")

                    # Fallback: use response_url (Slack/Mattermost compatible)
                    if response_url:
                        try:
                            requests.post(
                                response_url,
                                json={"response_type": "in_channel", "text": txt},
                                timeout=10,
                            )
                            return
                        except Exception:
                            app.logger.exception("Failed to post via response_url")

                try:
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                except Exception as e:
                    _post_message(f"Failed to start query: {e}")
                    return

                # Stream subprocess output line by line
                # -------------------------------------------------------------
                # Collect full stdout so we can extract only the final answer
                # (plus minimal context) and post a single tidy message.
                # -------------------------------------------------------------
                all_lines: list[str] = []
                if proc.stdout:
                    for raw_line in proc.stdout:
                        all_lines.append(raw_line.rstrip())

                # Wait for process termination
                retcode = proc.wait()

                # -------------------------------------------------------------
                # Parse output – look for [answer] or [summary] header produced
                # by query_rag.py and capture everything that follows it.
                # -------------------------------------------------------------
                answer_text = None
                header_idx = None
                for idx, ln in enumerate(all_lines):
                    if ln.strip().lower().startswith("[answer]") or ln.strip().lower().startswith("[summary]"):
                        header_idx = idx
                        break
                if header_idx is not None:
                    # Take all lines after the header until another header or end
                    answer_body: list[str] = []
                    for ln in all_lines[header_idx + 1 :]:
                        if ln.startswith("["):  # another section starts
                            break
                        answer_body.append(ln)
                    answer_text = "\n".join(answer_body).strip()

                if not answer_text:
                    # Fallback: use last 20 lines of output
                    answer_text = "\n".join(all_lines[-20:]).strip()

                # Compose final message
                final_msg = f"**Q:** {text}\n\n**A:**\n{answer_text}"

                _post_message(final_msg)
                if retcode != 0:
                    _post_message(f"Query process exited with code {retcode}")
            # Always run asynchronously
            threading.Thread(target=run_and_post, daemon=True).start()
            # Immediate acknowledgement
            return jsonify({"text": "Processing your query..."}), 200

        # 'inject' ingests the current channel into the RAG collection
        elif cmd_name in ("inject", "injest"):
            import shlex
            def run_inject():
                """Handle the /inject (or /injest) command in a background thread.

                All progress lines are buffered and posted at the end as **one**
                consolidated message so the channel isn’t flooded.
                """
                import requests
                import shlex
                import uuid
                import os
                import sys
                import tempfile
                import urllib.parse
                from ingest_rag import get_openai_client, load_documents, embed_and_upsert, ensure_collection, Document
                from qdrant_client import QdrantClient

                # ------------------------------------------------------------------
                # Buffering: collect all progress lines and send once at the end
                # ------------------------------------------------------------------

                buffered: list[str] = []

                def post(msg: str):
                    """Collect progress lines instead of posting immediately."""
                    buffered.append(msg)

                # Helper to send the final combined message (reuses logic from /ask)
                rest_usable: dict[str, bool] = {"ok": bool(mattermost_url and mattermost_token and channel_id)}

                def _post_combined():
                    if not buffered:
                        return
                    joined = "\n".join(buffered)
                    MAX_LEN = 3500

                    # Send in chunks to stay below Mattermost's limit
                    def _send(msg_part: str):
                        if rest_usable["ok"]:
                            try:
                                hdrs = {"Authorization": f"Bearer {mattermost_token}"}
                                resp = requests.post(
                                    f"{mattermost_url}/api/v4/posts",
                                    headers=hdrs,
                                    json={"channel_id": channel_id, "message": msg_part},
                                    timeout=10,
                                )
                                if resp.status_code in (200, 201):
                                    return True
                                if resp.status_code in (401, 403):
                                    rest_usable["ok"] = False
                            except Exception:
                                rest_usable["ok"] = False
                        # Fallback to response_url
                        if response_url:
                            try:
                                requests.post(
                                    response_url,
                                    json={"response_type": "in_channel", "text": msg_part},
                                    timeout=10,
                                )
                                return True
                            except Exception:
                                app.logger.exception("Failed to post combined message via response_url")
                        return False

                    for start in range(0, len(joined), MAX_LEN):
                        _send(joined[start : start + MAX_LEN])
                    return


                # ----------------------------------------------------------
                # Main ingestion logic – any return path will still execute
                # the *finally* block below to post the combined message.
                # ----------------------------------------------------------
                try:
                    args = shlex.split(text or "")
                    # Default parallel upsert workers
                    parallel_var = 15
                    # Parse slash-command flags for injection
                    # Known flags: parallel, chunk-size, chunk-overlap, crawl-depth,
                    # purge, collection, generate-summaries, quality-checks
                    chunk_size_var = 500
                    chunk_overlap_var = 50
                    crawl_depth_var = 0
                    clean_args: list[str] = []
                    gen_summaries_flag = False
                    qc_flag = False
                    i = 0
                    while i < len(args):
                        # Parallel upsert workers
                        if args[i] == "--parallel" and i + 1 < len(args):
                            try:
                                parallel_var = int(args[i+1])
                            except ValueError:
                                pass
                            i += 2
                            continue
                        # Chunking parameters
                        if args[i] == "--chunk-size" and i + 1 < len(args):
                            try:
                                chunk_size_var = int(args[i+1])
                            except ValueError:
                                pass
                            i += 2
                        elif args[i] == "--chunk-overlap" and i + 1 < len(args):
                            try:
                                chunk_overlap_var = int(args[i+1])
                            except ValueError:
                                pass
                            i += 2
                        elif args[i] in ("--crawl-depth", "--depth-crawl") and i + 1 < len(args):
                            try:
                                crawl_depth_var = int(args[i+1])
                            except ValueError:
                                pass
                            i += 2
                        # Purge flag
                        elif args[i] == "--purge":
                            clean_args.append(args[i])
                            i += 1
                        # Summarization flags
                        elif args[i] == "--generate-summaries":
                            gen_summaries_flag = True
                            i += 1
                        elif args[i] == "--no-generate-summaries":
                            gen_summaries_flag = False
                            i += 1
                        # Quality-check flags - handle common typos
                        elif args[i] in ("--quality-checks", "--quality-check", "--qualith-checks", "--qualty-checks"):
                            qc_flag = True
                            if args[i] != "--quality-checks":
                                post(f"Note: Interpreted '{args[i]}' as '--quality-checks'")
                            i += 1
                        elif args[i] in ("--no-quality-checks", "--no-quality-check"):
                            qc_flag = False
                            i += 1
                        # Rich metadata flags
                        elif args[i] == "--rich-metadata":
                            clean_args.append(args[i])
                            i += 1
                        elif args[i] == "--no-rich-metadata":
                            # Skip this flag
                            i += 1
                        else:
                            clean_args.append(args[i])
                            i += 1
                    args = clean_args
                    # Handle purge flag
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
                    # If summaries or quality checks requested, delegate to ingest_rag CLI
                    if gen_summaries_flag or qc_flag:
                        # Determine list of sources (args) or channel transcript
                        sources = list(args)
                        tmp_path = None
                        if not sources:
                            # Fetch channel messages to temp file
                            if not mattermost_url:
                                post("❌ MATTERMOST_URL is not configured – unable to fetch channel messages.")
                                return
                            msgs = []
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
                                    post(f"❌ Error fetching posts: {resp_ct.status_code} {resp_ct.text}")
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
                            tmp = tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False)
                            tmp_path = tmp.name; tmp.write("\n".join(m.rstrip("\n") for m in msgs)); tmp.close()
                            sources = [tmp_path]
                        # Run ingest_rag CLI for each source
                        for src in sources:
                            # Invoke the ingest_rag CLI using the same Python interpreter
                            cmd = [sys.executable, "-u", "-m", "ingest_rag", "--source", src, "--collection", collection,
                                   "--chunk-size", str(chunk_size_var), "--chunk-overlap", str(chunk_overlap_var),
                                   "--crawl-depth", str(crawl_depth_var)]
                            
                            # Check if we should add the generate-summaries flag
                            if gen_summaries_flag:
                                # Check if OPENAI_API_KEY is valid - if not, skip summaries
                                openai_key = os.environ.get("OPENAI_API_KEY")
                                if not openai_key or len(openai_key.strip()) < 10:
                                    post("[warning] OPENAI_API_KEY is missing or invalid - skipping summary generation")
                                else:
                                    cmd.append("--generate-summaries")
                            
                            # Add quality checks flag
                            if qc_flag:
                                cmd.append("--quality-checks")
                                
                            # Add rich metadata flag if present in cleaned args or in args
                            if "--rich-metadata" in clean_args:
                                # Make sure rich-metadata comes after --source to avoid treating it as a source
                                # Move any existing occurrence to ensure it's after the source
                                cmd = [arg for arg in cmd if arg != "--rich-metadata"]
                                # Add it right after the source parameter
                                src_idx = cmd.index("--source") + 2
                                cmd.insert(src_idx, "--rich-metadata")
                                
                            # pass Qdrant URL from env if set
                            qurl = os.environ.get("QDRANT_URL")
                            if qurl and not any(a.startswith("--qdrant-url") for a in cmd):
                                cmd.extend(["--qdrant-url", qurl])
                            # pass parallel worker count
                            cmd.extend(["--parallel", str(parallel_var)])
                            # Execute and stream output
                            try:
                                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                            except Exception as e:
                                post(f"❌ Failed to start ingestion subprocess: {e}")
                                continue
                            if proc.stdout:
                                for line in proc.stdout:
                                    post(line.rstrip())
                            ret = proc.wait()
                            if ret != 0:
                                post(f"❌ Ingestion subprocess exited with code {ret}")
                        # Cleanup temp file
                        if tmp_path:
                            try:
                                os.remove(tmp_path)
                            except OSError:
                                pass
                        return

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
                                post(f"❌ Error fetching posts: {resp_ct.status_code} {resp_ct.text}")
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

                        # Ingest the current channel transcript using the ingest_rag CLI
                        import tempfile as _tempfile, os as _os, shlex

                        # Dump the transcript to a temporary file
                        tmp = _tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False)
                        tmp_path = tmp.name
                        tmp.write("\n".join(m.rstrip("\n") for m in msgs))
                        tmp.close()

                        # Build the ingestion command
                        # Invoke the ingest_rag CLI on the channel transcript
                        cmd = [
                            sys.executable, "-u", "-m", "ingest_rag",
                            "--source", tmp_path,
                            "--collection", collection,
                            "--chunk-size", str(chunk_size_var),
                            "--chunk-overlap", str(chunk_overlap_var),
                            "--crawl-depth", str(crawl_depth_var),
                        ]
                        # Add rich metadata flag if specified
                        if "--rich-metadata" in clean_args:
                            # Make sure rich-metadata comes after --source to avoid treating it as a source
                            # Move any existing occurrence to ensure it's after the source
                            cmd = [arg for arg in cmd if arg != "--rich-metadata"]
                            # Add it right after the source parameter
                            src_idx = cmd.index("--source") + 2
                            cmd.insert(src_idx, "--rich-metadata")
                            
                        # Include Qdrant URL if provided
                        qurl = os.environ.get("QDRANT_URL")
                        if qurl and not any(a.startswith("--qdrant-url") for a in cmd):
                            cmd.extend(["--qdrant-url", qurl])
                        # Include parallel worker count
                        cmd.extend(["--parallel", str(parallel_var)])

                        # Execute and stream output
                        try:
                            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                        except Exception as e:
                            post(f"❌ Failed to start ingestion subprocess: {e}")
                            return
                        if proc.stdout:
                            for line in proc.stdout:
                                post(line.rstrip())
                        ret = proc.wait()
                        if ret != 0:
                            post(f"❌ Ingestion subprocess exited with code {ret}")

                        # Cleanup temporary file
                        try:
                            _os.remove(tmp_path)
                        except OSError:
                            pass
                        return

                    # Filter out the --rich-metadata flag from args before treating them as sources
                    sources = [arg for arg in args if arg != "--rich-metadata"]
                    total_sources = len(sources)
                    post(f"Starting ingestion of {total_sources} source(s) into '{collection}'...")
                    total_chunks = 0
                    for idx, source in enumerate(sources, start=1):
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
                                # Remote HTML or other URL: use URL directly for crawling
                                local_src = source
                        # Initial status: crawling and collecting chunks for this source
                        post(f"[{idx}/{total_sources}] Fetching and chunking '{source}' (depth={crawl_depth_var})...")
                        
                        # Enhanced URL loading with better error handling
                        try:
                            # Apply rich metadata if flag is present
                            apply_rich_metadata = "--rich-metadata" in clean_args
                            # Get all chunks (may include multiple pages)
                            docs = load_documents(local_src, chunk_size=chunk_size_var, overlap=chunk_overlap_var, crawl_depth=crawl_depth_var)
                            
                            # If rich metadata is requested, apply it directly to the documents
                            if apply_rich_metadata and docs:
                                post(f"[{idx}/{total_sources}] Applying rich metadata extraction...")
                                try:
                                    from rich_metadata import enrich_document_metadata
                                    enriched_docs = []
                                    for doc in docs:
                                        # Convert Document class to dict format expected by enrich_document_metadata
                                        doc_dict = {
                                            "content": doc.content,
                                            "metadata": doc.metadata.copy()
                                        }
                                        # Enrich metadata
                                        enriched_doc_dict = enrich_document_metadata(doc_dict)
                                        # Convert back to Document class
                                        enriched_doc = Document(
                                            content=enriched_doc_dict["content"],
                                            metadata=enriched_doc_dict["metadata"]
                                        )
                                        enriched_docs.append(enriched_doc)
                                    
                                    # Replace original documents with enriched versions
                                    docs = enriched_docs
                                    post(f"[{idx}/{total_sources}] Rich metadata extraction completed for {len(docs)} documents")
                                except Exception as e:
                                    post(f"[warning] Rich metadata extraction failed: {e}")
                            
                            # Check if any documents were loaded
                            if not docs:
                                post(f"⚠️ No documents loaded from '{source}'. This may indicate an issue with the URL or with langchain/unstructured packages.")
                                post("If this is a URL, make sure langchain and unstructured packages are installed with: pip install langchain langchain-community unstructured")
                                continue
                            
                            post(f"[{idx}/{total_sources}] Successfully loaded {len(docs)} chunks from '{source}'")
                            
                            # Group chunks by originating page URL in metadata
                            pages: dict[str, list] = {}
                            for doc in docs:
                                page = doc.metadata.get("source", source)
                                pages.setdefault(page, []).append(doc)
                        except Exception as e:
                            post(f"❌ Failed to load documents from '{source}': {e}")
                            post("If this is a URL, try installing required packages with: pip install langchain langchain-community unstructured requests bs4")
                            continue
                        # Process each page separately for visibility
                        for pg_idx, (page_url, pg_docs) in enumerate(pages.items(), start=1):
                            pg_count = len(pg_docs)
                            post(f"[{idx}/{total_sources}] Page {pg_idx}/{len(pages)}: '{page_url}' → {pg_count} chunks")
                            post(f"[{idx}/{total_sources}] Embedding & upserting {pg_count} chunks from page {pg_idx}...")
                            embed_and_upsert(
                                client,
                                collection,
                                pg_docs,
                                openai_client,
                                batch_size=16,
                                deterministic_id=True,
                                parallel=parallel_var,
                            )
                            total_chunks += pg_count
                            post(f"[{idx}/{total_sources}] Done page {pg_idx}: total chunks so far: {total_chunks}")
                        # Cleanup temp PDF
                        if local_src != source:
                            try: os.remove(local_src)
                            except: pass
                    post(f"Ingested {total_chunks} chunks from {total_sources} source(s) into '{collection}'.")
                except BaseException as e:
                    # Catch SystemExit raised by _lazy_import as well as regular exceptions
                    post(f"❌ Ingestion failed: {e}")
                finally:
                    # Flush buffered progress lines (success or error)
                    _post_combined()
            # Parse flags early for immediate handling of purge
            import shlex
            args = shlex.split(text or "")
            purge_flag = "--purge" in args
            # Determine target collection: override via flag or env var
            collection = os.environ.get("QDRANT_COLLECTION_NAME", "rag_data")
            if "--collection" in args:
                idx = args.index("--collection")
                if idx + 1 < len(args):
                    collection = args[idx + 1]
            elif "-c" in args:
                idx = args.index("-c")
                if idx + 1 < len(args):
                    collection = args[idx + 1]
            # Handle purge synchronously and return
            if purge_flag:
                from qdrant_client import QdrantClient
                qdrant_url_env = os.environ.get("QDRANT_URL", "http://localhost:6333")
                qdrant_api_key = os.environ.get("QDRANT_API_KEY")
                client = QdrantClient(url=qdrant_url_env, api_key=qdrant_api_key)
                try:
                    client.delete_collection(collection_name=collection)
                    msg = f"✅ Purged existing collection '{collection}'."
                except Exception as e:
                    msg = f"⚠️ Purge skipped: collection '{collection}' may not exist. ({e})"
                # Recreate empty collection
                try:
                    from ingest_rag import ensure_collection
                    ensure_collection(client, collection, vector_size=3072)
                except Exception:
                    pass
                return jsonify({"text": msg}), 200
            # Launch ingestion thread and immediately acknowledge
            threading.Thread(target=run_inject, daemon=True).start()
            return jsonify({"text": "Ingestion started... progress will be posted shortly."}), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Return exception message to Mattermost
        return jsonify({"text": f"Error: {e}"}), 200

def check_url_dependencies():
    """Check if URL handling dependencies are installed."""
    missing_deps = []
    try:
        import langchain_community
    except ImportError:
        missing_deps.append("langchain-community")
    
    try:
        import langchain
    except ImportError:
        missing_deps.append("langchain")
    
    try:
        import unstructured
    except ImportError:
        missing_deps.append("unstructured")
    
    try:
        import bs4
    except ImportError:
        missing_deps.append("bs4")
    
    if missing_deps:
        print("WARNING: Missing packages for URL handling:", ", ".join(missing_deps))
        print("To install: pip install " + " ".join(missing_deps))
        print("Without these packages, /inject <URL> may not work properly.")
    else:
        print("URL handling dependencies: OK")

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    check_url_dependencies()
    app.run(host=host, port=port)
