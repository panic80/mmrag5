#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_rag_stack.sh  –  Qdrant + Docling + Flask worker                      |
# ---------------------------------------------------------------------------
# This installer provisions a self-contained RAG stack on an Ubuntu/Debian    |
# server:                                                                     |
#   • Qdrant       – vector database                                         |
#   • Flask worker (server.py in your repo) – slash commands + ingestion     |
#                                                                             |
# Everything runs in Docker, supervised by systemd.                           |
# ---------------------------------------------------------------------------
# Usage                                                                       |
#   echo "OPENAI_API_KEY=sk-..." > .env                                       |
#   sudo bash setup_rag_stack.sh                                                 
# Optional env vars you may set in .env beforehand:                           |
#   QDRANT_API_KEY, QDRANT_URL, MATTERMOST_URL, MATTERMOST_TOKEN, SLASH_TOKEN |
# ---------------------------------------------------------------------------

set -euo pipefail

## --------------------------- defaults / config -----------------------------
# By convention, this script assumes you run it from the root of your project
# so that your code files (server.py, ingest_rag.py, Dockerfile, etc.) are
# in the current directory. It will deploy into the current directory.
STACK_HOME="$(pwd)"
# The folder "worker/" will be created under $STACK_HOME for building the app image.
WORKER_PORT="5000"     # host ↔ flask worker
QDRANT_PORT="6333"     # host ↔ qdrant:6333
QDRANT_IMAGE="qdrant/qdrant:latest"
WORKER_IMAGE="rag-worker:latest"

# Keep track of the directory from which the script was invoked – we’ll copy
# its contents into $STACK_HOME/worker.
INVOKE_DIR="$(pwd)"

# ---------------------------- root check -----------------------------------
if [[ $EUID -ne 0 ]]; then
  echo "❌ Please run as root (sudo)." >&2
  exit 1
fi

# ---------------------------- load .env ------------------------------------
if [[ -f .env ]]; then
  # shellcheck disable=SC2046,SC2155
  export $(grep -v '^#' .env | xargs -d '\n')
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "❌ OPENAI_API_KEY is missing (set it or put it in .env)." >&2
  exit 1
fi

# ---------------------- install docker if missing --------------------------
if ! command -v docker &>/dev/null; then
  echo "🔧 Installing Docker Engine …"
  apt-get update -qq
  apt-get install -y ca-certificates curl gnupg lsb-release >/dev/null
  install -d -m 0755 /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/$(. /etc/os-release && echo "$ID")/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/$(. /etc/os-release && echo "$ID") $(lsb_release -cs) stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -qq
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin docker-buildx-plugin >/dev/null
fi

# ----------------------- prepare stack directory ---------------------------
mkdir -p "$STACK_HOME"
cd "$STACK_HOME"

# Ensure .env present in stack dir
if [[ ! -f .env ]]; then
  echo "OPENAI_API_KEY=${OPENAI_API_KEY}" > .env
fi

# --------------------------- copy local code -------------------------------
if [[ ! -d worker ]]; then
  echo "📦 Copying local worker code from ${INVOKE_DIR}"
  mkdir -p worker
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --exclude 'worker' --exclude 'qdrant_data' "${INVOKE_DIR}/" worker/
  else
    cp -a "${INVOKE_DIR}"/. worker/
  fi
fi

# ------------------- build images ------------------------------------------

# Build worker image
echo "🐳 Building worker image (${WORKER_IMAGE}) …"
# Prefer BuildKit, but fall back to classic builder when buildx unavailable.
if docker buildx version >/dev/null 2>&1; then
  BUILDKIT_ENV="DOCKER_BUILDKIT=1"
else
  echo "⚠️  docker-buildx-plugin not available – building without BuildKit"
  BUILDKIT_ENV="DOCKER_BUILDKIT=0"
fi

eval "$BUILDKIT_ENV docker build -t \"$WORKER_IMAGE\" ./worker"

# ---------------------- write docker-compose.yml ---------------------------
cat > docker-compose.yml <<YML
version: "3.9"

services:
  qdrant:
    image: ${QDRANT_IMAGE}
    restart: unless-stopped
    environment:
      - QDRANT_LOG_LEVEL=INFO
    ports:
      - "${QDRANT_PORT}:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  app:
    image: ${WORKER_IMAGE}
    depends_on:
      - qdrant
    restart: unless-stopped
    env_file:
      - .env
    environment:
      - QDRANT_URL=http://qdrant:6333
    ports:
      - "${WORKER_PORT}:5000"
    command: python3 server.py

volumes:
  qdrant_data:
YML

# ------------------ helper wrapper: workerctl ------------------------------
cat > /usr/local/bin/workerctl <<WRAP
#!/usr/bin/env bash
cd "${STACK_HOME}" || exit 1
docker compose "$@"
WRAP
chmod +x /usr/local/bin/workerctl

## ---------------------------- launch stack --------------------------------
echo "🚀 Starting Docker Compose services…"
docker compose down || true
docker compose up -d

# ----------------------------- summary -------------------------------------
IP=$(hostname -I | awk '{print $1}')
cat <<SUMMARY

✅ RAG stack is running

Qdrant : http://${IP}:${QDRANT_PORT}
Worker : http://${IP}:${WORKER_PORT}/health

Use 'workerctl logs -f' to follow logs.
SUMMARY
