import os
import subprocess
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/", methods=["GET", "POST"])
def handle_slash():
    # Mattermost will send either GET?text=... or POST form text=...
    if request.method == "GET":
        text = request.args.get("text", "")
    else:
        text = request.form.get("text", "")

    # Verify slash command token for authenticity
    slash_token = os.environ.get("SLASH_TOKEN")
    req_token = request.values.get("token")
    if slash_token and req_token != slash_token:
        return jsonify({"text": "Invalid token."}), 403

    # Determine which slash command was invoked
    command = request.values.get("command", "").strip()

    # '/ask' triggers a RAG query
    if command == "/ask":
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

    # '/inject' ingests the current channel into the RAG collection
    if command == "/inject":
        import shlex
        import requests
        import uuid
        from ingest_rag import chunk_text, iter_batches, get_openai_client
        from qdrant_client import QdrantClient

        # Parse optional collection override
        args = shlex.split(text or "")
        collection = "rag_data"
        if "--collection" in args:
            idx = args.index("--collection")
            if idx + 1 < len(args):
                collection = args[idx + 1]
        elif "-c" in args:
            idx = args.index("-c")
            if idx + 1 < len(args):
                collection = args[idx + 1]

        # Mattermost API settings
        mattermost_url = os.environ.get("MATTERMOST_URL")
        mattermost_token = os.environ.get("MATTERMOST_TOKEN")
        if not mattermost_url or not mattermost_token:
            return jsonify({"text": "MATTERMOST_URL and MATTERMOST_TOKEN must be set."}), 500

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
                return jsonify({"text": f"Failed to fetch posts: {resp.text}"}), 500
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

        # Chunk messages and prepare documents
        docs: list[tuple[str, dict]] = []
        for msg_idx, msg in enumerate(messages):
            chunks = chunk_text(msg, max_chars=500)
            for chunk_idx, chunk in enumerate(chunks):
                docs.append((chunk, {"source_channel": channel_id, "message_index": msg_idx, "chunk_index": chunk_idx}))

        # Initialize OpenAI and Qdrant clients
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            return jsonify({"text": "OPENAI_API_KEY not set"}), 500
        qdrant_url_env = os.environ.get("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.environ.get("QDRANT_API_KEY")
        openai_client = get_openai_client(openai_api_key)
        client = QdrantClient(url=qdrant_url_env, api_key=qdrant_api_key)

        # Embed and upsert in batches
        batch_size = 64
        total_chunks = len(docs)
        for batch in iter_batches(docs, batch_size):
            texts = [doc for doc, _ in batch]
            # Create embeddings
            try:
                if hasattr(openai_client, "embeddings"):
                    emb_resp = openai_client.embeddings.create(model="text-embedding-3-large", input=texts)
                    vectors = [d.embedding for d in emb_resp.data]
                else:
                    emb_resp = openai_client.Embedding.create(model="text-embedding-3-large", input=texts)
                    vectors = [d["data"][0]["embedding"] for d in emb_resp.data]
            except Exception as e:
                return jsonify({"text": f"Embedding error: {e}"}), 500
            points = []
            for (chunk_text_val, meta), vec in zip(batch, vectors):
                payload = meta.copy()
                payload["chunk_text"] = chunk_text_val
                points.append({"id": str(uuid.uuid4()), "vector": vec, "payload": payload})
            try:
                client.upsert(collection_name=collection, points=points)
            except Exception as e:
                return jsonify({"text": f"Upsert error: {e}"}), 500

        return f"Ingested {total_chunks} chunks from {len(messages)} messages into '{collection}'.", 200, {"Content-Type": "text/plain"}

    # Unknown command
    return jsonify({"text": f"Unknown command '{command}'."}), 400

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    app.run(host=host, port=port)
