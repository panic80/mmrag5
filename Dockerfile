# Dockerfile
FROM python:3.11-slim

WORKDIR /app

        # Install your Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

        # Copy your code
COPY . .

        # If you're going to expose a web service (e.g. a Flask slack‑slash 
   # handler),
        # replace the CMD below with the appropriate command.
        # For pure CLI, you might not even need CMD.
#CMD ["python3", "-m", "query_rag", "--help"]
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 5000
CMD ["python3", "server.py"]

