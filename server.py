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

    if not text:
        return jsonify({"text": "No text provided."}), 400
    # Verify slash command token to ensure request authenticity
    slash_token = os.environ.get("SLASH_TOKEN")
    req_token = request.values.get("token")
    if slash_token and req_token != slash_token:
        return jsonify({"text": "Invalid token."}), 403

    # Prepare and invoke the query_rag CLI with proper argument parsing
    import shlex
    # Split the incoming text (allows flags and quoted strings)
    args = shlex.split(text)
    # Build the base command to call query_rag
    cmd = ["python3", "-m", "query_rag"]
    # Inject default QDRANT_URL from environment if not overridden in args
    qdrant_url = os.environ.get("QDRANT_URL")
    if qdrant_url and not any(arg.startswith("--qdrant-url") for arg in args):
        cmd += ["--qdrant-url", qdrant_url]
    # Append all user-provided args (flags and query terms)
    cmd += args
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        answer = result.stdout.strip()
    except subprocess.CalledProcessError as e:
        # Return error details for debugging
        return jsonify({"error": e.stderr.strip()}), 500

    # Mattermost will accept a plain‐text body
    return answer, 200, {"Content-Type": "text/plain"}

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    app.run(host=host, port=port)
