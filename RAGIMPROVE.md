# Advanced RAG Improvements

This document outlines advanced techniques to improve Retrieval-Augmented Generation (RAG) systems across ingestion, extraction, chunking, embedding, retrieval, and evaluation.

## Table of Contents

- [Smarter Document Chunking](#smarter-document-chunking)
- [Query Expansion for Better Retrieval](#query-expansion-for-better-retrieval)
- [Reranking for Higher Relevance](#reranking-for-higher-relevance)
- [Local Embedding Models](#local-embedding-models)
- [Advanced OCR for Document Extraction](#advanced-ocr-for-document-extraction)
- [Contextual Compression](#contextual-compression)
- [Parallel Document Processing](#parallel-document-processing)
- [RAG Self-Evaluation](#rag-self-evaluation)
- [Additional Techniques](#additional-techniques)

## Smarter Document Chunking

Semantic chunking divides documents based on topic boundaries rather than arbitrary character limits.

```python
def semantic_chunk_text(text, max_chars=1000):
    """Chunk text based on semantic topic boundaries."""
    try:
        from transformers import pipeline
        
        # Initialize zero-shot classification pipeline
        classifier = pipeline("zero-shot-classification", 
                             model="facebook/bart-large-mnli")
        
        # Split text into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Process paragraphs into semantic chunks
        initial_chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_len = len(para)
            if current_length + para_len > max_chars and current_chunk:
                initial_chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_length = para_len
            else:
                current_chunk.append(para)
                current_length += para_len
                
        # Add the last chunk if it exists
        if current_chunk:
            initial_chunks.append("\n\n".join(current_chunk))
            
        # Group chunks by detected topic
        refined_chunks = []
        for chunk in initial_chunks:
            topics = ["introduction", "background", "methodology", "results", 
                     "discussion", "conclusion"]
            result = classifier(chunk, topics)
            # Store topic information with chunk
            refined_chunks.append({
                "text": chunk,
                "topic": result["labels"][0],
                "confidence": result["scores"][0]
            })
            
        return [c["text"] for c in refined_chunks]
    except Exception as e:
        # Fallback to regular chunking
        return _smart_chunk_text(text, max_chars)
```

## Hierarchy-Aware Chunking

This approach preserves document structure like headers and sections.

```python
def hierarchy_chunk_text(text, max_chars=1000, overlap=50):
    """Chunk text while preserving document hierarchy (headers, sections)."""
    # Regex for detecting headers
    header_pattern = re.compile(r'^(#+)\s+(.+)$', re.MULTILINE)
    
    # Split text into sections based on headers
    sections = []
    current_level = 0
    current_header = "Introduction"
    current_content = []
    
    for line in text.split('\n'):
        header_match = header_pattern.match(line)
        if header_match:
            # We found a header, save previous section if it exists
            if current_content:
                sections.append({
                    "level": current_level,
                    "header": current_header,
                    "content": "\n".join(current_content)
                })
            
            # Start new section
            current_level = len(header_match.group(1))
            current_header = header_match.group(2)
            current_content = []
        else:
            current_content.append(line)
    
    # Add the last section
    if current_content:
        sections.append({
            "level": current_level,
            "header": current_header,
            "content": "\n".join(current_content)
        })
    
    # Chunk each section while preserving headers
    chunks = []
    for section in sections:
        # For short sections, keep as is
        if len(section["content"]) <= max_chars:
            chunks.append(f"# {section['header']}\n\n{section['content']}")
            continue
        
        # For longer sections, chunk content but keep header with each chunk
        content_chunks = _smart_chunk_text(section["content"], max_chars - len(section["header"]) - 10, overlap)
        for i, chunk in enumerate(content_chunks):
            if i == 0:
                # First chunk gets the full header
                chunks.append(f"# {section['header']}\n\n{chunk}")
            else:
                # Continuation chunks get header with continuation marker
                chunks.append(f"# {section['header']} (continued {i+1})\n\n{chunk}")
    
    return chunks
```

## Query Expansion for Better Retrieval

Expand the original query into multiple variations to improve recall.

```python
def expand_query(query_text, openai_client, model="gpt-4.1-mini"):
    """Expand query with LLM rewrites for better retrieval."""
    system_msg = {
        "role": "system", 
        "content": "You are a helpful assistant specialized in search query expansion. "
                 "Your task is to generate 3-5 alternative versions of a search query, "
                 "each focusing on different aspects or using different terminology. "
                 "Return ONLY the expanded queries as a JSON array, with no other text."
    }
    
    user_msg = {
        "role": "user",
        "content": f"Original query: {query_text}"
    }
    
    try:
        if hasattr(openai_client, "chat"):
            response = openai_client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[system_msg, user_msg]
            )
            content = response.choices[0].message.content
        else:
            response = openai_client.ChatCompletion.create(
                model=model,
                messages=[system_msg, user_msg]
            )
            content = response.choices[0].message.content
        
        import json
        expanded = json.loads(content)
        if "queries" in expanded:
            return expanded["queries"]
        else:
            return list(expanded.values())[0]
            
    except Exception as e:
        return [query_text]  # Return original query on failure
```

## Reranking for Higher Relevance

Use a cross-encoder model to rerank retrieved documents by relevance.

```python
def rerank_results(query, results, k=10):
    """Rerank search results using a cross-encoder model."""
    try:
        from sentence_transformers import CrossEncoder
        
        # Load cross-encoder for reranking
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Prepare passage pairs
        passage_pairs = []
        for result in results:
            text = result.payload.get("chunk_text", "")
            if text:
                passage_pairs.append([query, text])
        
        # Score all pairs
        scores = cross_encoder.predict(passage_pairs)
        
        # Add scores to results
        for i, result in enumerate(results[:len(scores)]):
            result.rerank_score = float(scores[i])
        
        # Sort by reranking score
        reranked = sorted(results, key=lambda x: getattr(x, 'rerank_score', 0.0), reverse=True)
        
        return reranked[:k]
    except Exception as e:
        # Return original results if reranking fails
        return results[:k]
```

## Local Embedding Models

Use local embedding models to reduce API costs and latency.

```python
def get_local_embeddings(texts, model_name="BAAI/bge-large-en-v1.5"):
    """Use local embedding models instead of OpenAI API."""
    try:
        from sentence_transformers import SentenceTransformer
        
        # Load the model (only once)
        if not hasattr(get_local_embeddings, "model"):
            get_local_embeddings.model = SentenceTransformer(model_name)
            
        # Handle both single strings and lists
        if isinstance(texts, str):
            texts = [texts]
            
        # Generate embeddings
        embeddings = get_local_embeddings.model.encode(
            texts, 
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 10
        )
        
        # Format similar to OpenAI response
        return [{"embedding": emb.tolist()} for emb in embeddings]
    except Exception as e:
        # Fall back to OpenAI if local embedding fails
        openai_client = get_openai_client(os.environ.get("OPENAI_API_KEY"))
        if hasattr(openai_client, "embeddings"):
            resp = openai_client.embeddings.create(model="text-embedding-3-large", input=texts)
            return [{"embedding": rec.embedding} for rec in resp.data]
        else:
            resp = openai_client.Embedding.create(model="text-embedding-3-large", input=texts)
            return [{"embedding": rec["embedding"]} for rec in resp["data"]]
```

## Domain-Specific Embeddings

Fine-tune an embedding model on your specific domain.

```python
def finetune_embeddings_model(texts, model_name="BAAI/bge-small-en-v1.5"):
    """Fine-tune an embedding model on domain-specific data."""
    try:
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader
        import torch
        
        # Load base model
        model = SentenceTransformer(model_name)
        
        # Prepare training examples (using contrastive learning approach)
        train_examples = []
        
        # Group texts by their metadata (assuming similar metadata = similar content)
        text_groups = {}
        for doc in texts:
            key = doc.metadata.get("source", "unknown")
            if key not in text_groups:
                text_groups[key] = []
            text_groups[key].append(doc.content)
        
        # Create positive pairs (texts from same source)
        for source, group_texts in text_groups.items():
            if len(group_texts) < 2:
                continue
                
            for i in range(len(group_texts)):
                for j in range(i+1, min(i+5, len(group_texts))):
                    train_examples.append(InputExample(texts=[group_texts[i], group_texts[j]], label=1.0))
        
        # Create training dataloader
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        
        # Use cosine similarity loss
        train_loss = losses.CosineSimilarityLoss(model)
        
        # Train for a few epochs
        model.fit(train_dataloader=train_dataloader, epochs=1, warmup_steps=100)
        
        # Save the fine-tuned model
        model.save("fine_tuned_embeddings")
        
        return model
    except Exception as e:
        return None
```

## Advanced OCR for Document Extraction

Extract text from images and scanned PDFs.

```python
def extract_text_with_ocr(file_path):
    """Extract text from images and scanned PDFs using OCR."""
    try:
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            from pdf2image import convert_from_path
            import pytesseract
            
            # Convert PDF pages to images
            pages = convert_from_path(file_path, dpi=300)
            text_parts = []
            
            for i, page in enumerate(pages):
                # Extract text from each page
                page_text = pytesseract.image_to_string(page)
                text_parts.append(f"Page {i+1}:\n{page_text}")
                
            return "\n\n".join(text_parts)
            
        elif ext in ['.jpg', '.jpeg', '.png', '.tiff']:
            import pytesseract
            from PIL import Image
            
            image = Image.open(file_path)
            return pytesseract.image_to_string(image)
            
        return None
        
    except Exception as e:
        click.echo(f"[warning] OCR failed: {e}", err=True)
        return None
```

## Table Extraction

Extract structured data from tables in PDFs and HTML.

```python
def extract_tables_from_pdf(file_path):
    """Extract tables from PDF as structured data."""
    try:
        import tabula
        import pandas as pd
        
        # Extract all tables from the PDF
        tables = tabula.read_pdf(file_path, pages='all', multiple_tables=True)
        
        results = []
        for i, table in enumerate(tables):
            # Convert table to markdown format
            md_table = table.to_markdown(index=False)
            results.append(f"Table {i+1}:\n{md_table}")
        
        return results
    except Exception as e:
        return []
```

## Contextual Compression

Compress retrieved documents to focus on query-relevant content.

```python
def contextual_compression(query, docs):
    """Focus retrieved documents on query-relevant parts."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        # Load summarization model
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
        
        compressed_docs = []
        for doc in docs:
            # Extract document text
            text = doc.payload.get("chunk_text", "")
            if not text:
                compressed_docs.append(doc)
                continue
                
            # Create prompt focusing on query
            prompt = f"Query: {query}\n\nDocument: {text}\n\nExtract the parts most relevant to the query:"
            
            # Generate focused extract
            inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
            outputs = model.generate(
                inputs.input_ids, 
                max_length=200, 
                min_length=50,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
            
            # Save compressed text
            compressed_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            doc_copy = doc
            doc_copy.payload["compressed_text"] = compressed_text
            compressed_docs.append(doc_copy)
        
        return compressed_docs
    except Exception as e:
        # Return original docs if compression fails
        return docs
```

## Parallel Document Processing

Process documents concurrently for faster ingestion.

```python
def process_documents_parallel(documents, processor_fn, max_workers=8):
    """Process documents in parallel for faster ingestion."""
    from concurrent.futures import ThreadPoolExecutor
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all documents for processing
        future_to_doc = {executor.submit(processor_fn, doc): doc for doc in documents}
        
        # Collect results as they complete
        for future in future_to_doc:
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                doc = future_to_doc[future]
                source = doc.metadata.get("source", "unknown")
                click.echo(f"[warning] Processing failed for {source}: {e}", err=True)
    
    return results
```

## Embedding Caching

Cache embeddings to avoid redundant computation.

```python
def cached_embedding(text, model_name, cache_dir=".embedding_cache"):
    """Cache embeddings to avoid redundant computation."""
    import hashlib
    import os
    import json
    import numpy as np
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a hash of the text and model name for the cache key
    cache_key = hashlib.md5(f"{text}:{model_name}".encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"{cache_key}.json")
    
    # Check if embedding is already cached
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                cached = json.load(f)
            return np.array(cached["embedding"])
        except Exception:
            # If loading fails, recompute
            pass
    
    # Compute embedding
    openai_client = get_openai_client(os.environ.get("OPENAI_API_KEY"))
    if hasattr(openai_client, "embeddings"):  # openai>=1.0 style
        resp = openai_client.embeddings.create(model=model_name, input=[text])
        embedding = resp.data[0].embedding
    else:  # legacy style
        resp = openai_client.Embedding.create(model=model_name, input=[text])
        embedding = resp["data"][0]["embedding"]
    
    # Cache the result
    try:
        with open(cache_path, 'w') as f:
            json.dump({"text": text, "model": model_name, "embedding": embedding}, f)
    except Exception:
        pass
    
    return np.array(embedding)
```

## RAG Self-Evaluation

Evaluate RAG system quality for continuous improvement.

```python
def evaluate_rag_quality(query, retrieved_chunks, generated_answer, openai_client):
    """Evaluate the quality of RAG responses for continuous improvement."""
    system_prompt = """You are an evaluator for RAG (Retrieval Augmented Generation) systems.
Assess the quality of the retrieved chunks and the generated answer for a given query.
Score on a scale of 1-10 for:
1. Relevance: How relevant are the retrieved chunks to the query?
2. Completeness: Do the chunks contain all information needed to answer?
3. Accuracy: Is the generated answer accurate based on the chunks?
4. Hallucination: Does the answer contain information not in the chunks?
Return your evaluation as JSON with scores and brief explanations."""

    eval_prompt = f"""
Query: {query}

Retrieved chunks:
{retrieved_chunks}

Generated answer:
{generated_answer}

Provide your evaluation:
"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4.1-nano",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": eval_prompt}
            ]
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return {"error": str(e)}
```

## Additional Techniques

### Multi-Vector Retrieval

Store multiple embeddings per document for better retrieval.

```python
def multi_vector_embed(document, model):
    """Create multiple embeddings for different sections of a document."""
    # Split document into sections (headings, paragraphs, etc.)
    sections = split_into_sections(document)
    
    # Embed each section separately
    embeddings = []
    for section in sections:
        embedding = model.encode(section, normalize_embeddings=True)
        embeddings.append({
            "vector": embedding.tolist(),
            "text": section,
            "metadata": document.metadata
        })
    
    return embeddings
```

### Parent-Child Document Relationships

Maintain hierarchical relationships between documents and chunks.

```python
def create_hierarchical_chunks(document, chunk_size=500, overlap=50):
    """Create chunks while maintaining hierarchical relationships."""
    # Create a parent document record
    parent_id = str(uuid.uuid4())
    parent_record = {
        "id": parent_id,
        "content": document.content[:1000] + "...",  # Truncated content
        "metadata": document.metadata,
        "is_parent": True,
        "children": []
    }
    
    # Create child chunks
    chunks = _smart_chunk_text(document.content, chunk_size, overlap)
    child_records = []
    
    for i, chunk in enumerate(chunks):
        child_id = str(uuid.uuid4())
        child_record = {
            "id": child_id,
            "content": chunk,
            "metadata": {**document.metadata, "chunk_index": i, "parent_id": parent_id},
            "is_parent": False
        }
        parent_record["children"].append(child_id)
        child_records.append(child_record)
    
    return {"parent": parent_record, "children": child_records}
```

### Hybrid Search Implementation

Combine vector and keyword search for better results.

```python
def hybrid_search(query, client, collection, bm25_index_path, k=10, alpha=0.7):
    """Perform hybrid search combining vector and keyword search."""
    import json
    from rank_bm25 import BM25Okapi
    import numpy as np
    
    # Vector search
    vector_results = vector_search(client, collection, query, k=k*2)
    
    # Keyword search using BM25
    with open(bm25_index_path, 'r') as f:
        id2text = json.load(f)
    
    # Extract corpus and document IDs
    corpus = []
    doc_ids = []
    for doc_id, text in id2text.items():
        if text and isinstance(text, str):
            corpus.append(text.lower().split())
            doc_ids.append(doc_id)
    
    # Create BM25 index
    bm25 = BM25Okapi(corpus)
    
    # Tokenize query and get scores
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Get top scores and IDs
    top_indices = np.argsort(bm25_scores)[::-1][:k*2]
    bm25_results = []
    
    for idx in top_indices:
        doc_id = doc_ids[idx]
        for result in vector_results:
            if result.id == doc_id:
                # Add BM25 score to existing vector result
                result.bm25_score = float(bm25_scores[idx])
                bm25_results.append(result)
                break
    
    # Combine scores - normalize first
    combined_results = []
    max_vector = max([r.score for r in vector_results]) if vector_results else 1.0
    max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 else 1.0
    
    # Combine unique results from both sources
    all_results = {r.id: r for r in vector_results}
    
    # Calculate combined scores
    for result_id, result in all_results.items():
        vector_score = result.score / max_vector if max_vector > 0 else 0
        bm25_score = getattr(result, 'bm25_score', 0) / max_bm25 if max_bm25 > 0 else 0
        
        # Weighted combination
        combined_score = alpha * vector_score + (1 - alpha) * bm25_score
        result.combined_score = combined_score
        combined_results.append(result)
    
    # Sort by combined score
    combined_results = sorted(combined_results, key=lambda x: x.combined_score, reverse=True)
    
    return combined_results[:k]
```

## Implementation Strategy

To improve your RAG system:

1. **Start with chunking improvements**: Implement semantic or hierarchy-aware chunking
2. **Add retrieval enhancements**: Implement query expansion and reranking
3. **Optimize performance**: Add parallel processing and embedding caching
4. **Improve extraction**: Add OCR and table extraction for richer content
5. **Enable evaluation**: Add self-evaluation to continuously improve the system

Each technique can be implemented incrementally, starting with the ones that address your most pressing challenges.