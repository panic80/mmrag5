 # RAG Toolkit (Explain Like I’m Five)

 Imagine you have a huge pile of books, and you want to find the exact line that answers your question right away.

 ## What does this tool do?
 1. It reads your documents (PDFs, text files, websites).
 2. It chops them into small pieces (called *chunks*).
 3. It asks OpenAI to turn each chunk into a secret code (a list of numbers).
 4. It stores all the codes in Qdrant (a fast special database).
 5. When you ask a question, it turns your question into a code and finds the closest chunks.

 ## Why use this?
 - Normal text search looks for words, but might miss the real meaning.
 - These codes (**embeddings**) find similar ideas even if words change.

 ## How do you use it?

 ### 1. Install
 ```bash
 git clone <repo-url>
 cd <repo-directory>
 pip install -r requirements.txt
 ```

 ### 2. Add your secret keys
 Create a file named `.env`:
 ```ini
 OPENAI_API_KEY=your_openai_key_here
 QDRANT_API_KEY=your_qdrant_key_if_any
 ```

 ### 3. Ingest your documents
 ```bash
 python ingest_rag.py \
   --source ./my_documents_folder \
   --collection my_collection_name
 ```
 This reads and stores all chunks in the Qdrant database.

 ### 4. Ask a question
 ```bash
 python query_rag.py \
   --collection my_collection_name \
   "What is our refund policy?"
 ```
 You’ll get the best matching text snippets.

 ## Super Simple Diagram
 ```text
 [Documents] -> [Chunks] -> [Embeddings] -> [Qdrant]
                                ^
                                |
           [Your Question] -> [Embed] -> [Find closest chunks]
 ```

 ## A few grown‑up options
- `--hybrid`: mix regular word search with embedding search (on by default; disable with `--no-hybrid`).
 - `--bm25-index <file>`: reuse a saved word‑search index.
- `--llm-model <model>`: specify the LLM to use (default: `gpt-4.1-mini`; set to empty to skip).
- `--raw`: show raw retrieval hits and full answer (requires `--llm-model`).
- `--chunk-size` / `--chunk-overlap`: control how big or overlapping each piece is.
- `--crawl-depth <int>`: when SOURCE is a URL, crawl hyperlinks up to this depth (default: 0=no crawl).

 And that’s it! You now have a magic question‑answering box for your documents.
 
 ## Using from Mattermost (Chat Commands)

 You can also use this tool right inside your Mattermost chat:

 1. **Set up secrets** in your `.env` or environment:
    ```ini
    MATTERMOST_URL=https://your-mattermost-server
    MATTERMOST_TOKEN=your_mattermost_token
    SLASH_TOKEN=common_slash_token
    SLASH_TOKEN_INJECT=inject_command_token
    SLASH_TOKEN_ASK=ask_command_token
    ```

 2. **Create slash commands** in Mattermost:
    - **Command**: `/inject`  → **URL**: `https://your-app-url/inject`
    - **Command**: `/ask`     → **URL**: `https://your-app-url/ask`

 3. **Use in chat**:
    - To ingest messages in the current channel:
      ```
      /inject --collection my_collection_name
      ```
    - To ask a question:
      ```
      /ask What is our refund policy?
      ```

 Make sure your app (server.py) is running and reachable at the URLs you configured.