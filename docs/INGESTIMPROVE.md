# Ingestion-Side RAG Improvements for Reducing Hallucination

This document outlines advanced ingestion techniques to improve the quality of retrieved content in RAG systems, focusing on reducing hallucination by improving data quality at the source.

## Table of Contents

- [Content Pre-processing and Cleaning](#content-pre-processing-and-cleaning)
- [Metadata Enrichment and Entity Extraction](#metadata-enrichment-and-entity-extraction)
- [Multi-granularity Indexing](#multi-granularity-indexing)
- [Quality Filtering](#quality-filtering)
- [Named Entity Recognition and Fact Extraction](#named-entity-recognition-and-fact-extraction)
- [Knowledge Graph Construction](#knowledge-graph-construction)
- [Consistency Checking](#consistency-checking)
- [Confidence Scoring](#confidence-scoring)

## Content Pre-processing and Cleaning

Improve the quality of ingested content by removing noise and normalizing formats.

```python
def clean_document_content(text):
    """Clean and normalize document content for higher quality indexing."""
    # Remove duplicated whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKC', text)
    
    # Remove boilerplate content (headers, footers, etc)
    text = remove_boilerplate(text)
    
    # Fix common OCR errors and misspellings
    text = fix_common_errors(text)
    
    # Handle special characters and formatting
    text = normalize_formatting(text)
    
    # Filter out non-informative sections (e.g., "About Us" generic content)
    text = filter_non_informative(text)
    
    return text
```

## Metadata Enrichment and Entity Extraction

Extract structured information from documents to provide better context and filtering options.

```python
def enrich_document_metadata(document):
    """Extract and enrich document metadata for better retrieval context."""
    text = document.content
    metadata = document.metadata.copy()
    
    # Extract and normalize dates
    dates = extract_dates(text)
    if dates:
        metadata["dates"] = dates
        metadata["date_range"] = {"start": min(dates), "end": max(dates)}
    
    # Extract named entities 
    entities = extract_entities(text)
    if entities:
        metadata["entities"] = entities
    
    # Extract numeric data and units
    numeric_data = extract_numeric_data(text)
    if numeric_data:
        metadata["numeric_data"] = numeric_data
    
    # Topic classification
    topics = classify_topics(text)
    if topics:
        metadata["topics"] = topics
    
    # Sentiment analysis
    sentiment = analyze_sentiment(text)
    if sentiment:
        metadata["sentiment"] = sentiment
    
    # Factual claim extraction
    claims = extract_factual_claims(text)
    if claims:
        metadata["factual_claims"] = claims
    
    # Confidence scoring for extracted info
    metadata["extraction_confidence"] = calculate_extraction_confidence(
        text, entities, dates, numeric_data
    )
    
    return Document(content=text, metadata=metadata)
```

## Multi-granularity Indexing

Store documents at multiple levels of granularity (full document, sections, and chunks) to provide better context during retrieval.

```python
def create_multi_granularity_index(document, chunk_size=1000, overlap=100):
    """Create document representations at multiple granularity levels."""
    doc_text = document.content
    doc_metadata = document.metadata.copy()
    
    # Level 1: Document Summary
    summary = generate_document_summary(doc_text)
    summary_doc = Document(
        content=summary,
        metadata={**doc_metadata, "level": "summary", "parent_id": doc_metadata.get("id")}
    )
    
    # Level 2: Section-level chunks
    sections = split_into_sections(doc_text)
    section_docs = []
    for i, section in enumerate(sections):
        section_metadata = {
            **doc_metadata,
            "level": "section",
            "section_index": i,
            "parent_id": doc_metadata.get("id"),
            "section_title": extract_section_title(section)
        }
        section_docs.append(Document(content=section, metadata=section_metadata))
    
    # Level 3: Fine-grained chunks
    chunk_docs = []
    for i, section in enumerate(sections):
        chunks = semantic_chunk_text(section, chunk_size, overlap)
        for j, chunk in enumerate(chunks):
            chunk_metadata = {
                **doc_metadata,
                "level": "chunk",
                "section_index": i,
                "chunk_index": j,
                "parent_id": doc_metadata.get("id"),
                "section_title": extract_section_title(section)
            }
            chunk_docs.append(Document(content=chunk, metadata=chunk_metadata))
    
    return [summary_doc] + section_docs + chunk_docs
```

## Quality Filtering

Implement filters to remove low-quality or uninformative content before indexing.

```python
def filter_low_quality_chunks(chunks, min_quality_score=0.6):
    """Filter out low-quality chunks based on multiple criteria."""
    filtered_chunks = []
    
    for chunk in chunks:
        # Calculate information density
        info_density = calculate_information_density(chunk.content)
        
        # Check for duplicate or near-duplicate content
        is_duplicate = check_is_duplicate(chunk.content, filtered_chunks)
        
        # Assess relevance to document's main topics
        topic_relevance = assess_topic_relevance(
            chunk.content, chunk.metadata.get("document_topics", [])
        )
        
        # Check for minimum length
        sufficient_length = len(chunk.content.split()) >= 50
        
        # Calculate overall quality score
        quality_score = calculate_quality_score(
            info_density, is_duplicate, topic_relevance, sufficient_length
        )
        
        # Add quality metrics to metadata
        chunk.metadata["quality"] = {
            "info_density": info_density,
            "is_duplicate": is_duplicate,
            "topic_relevance": topic_relevance,
            "sufficient_length": sufficient_length,
            "overall_score": quality_score
        }
        
        # Filter based on quality
        if quality_score >= min_quality_score and not is_duplicate:
            filtered_chunks.append(chunk)
    
    return filtered_chunks
```

## Named Entity Recognition and Fact Extraction

Extract and index specific facts to enhance retrieval precision and fact-checking.

```python
def extract_and_index_facts(document):
    """Extract factual statements and named entities for more precise retrieval."""
    import spacy
    
    # Load NER model
    nlp = spacy.load("en_core_web_trf")
    
    text = document.content
    doc = nlp(text)
    
    # Extract named entities
    entities = {}
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        entities[ent.label_].append({
            "text": ent.text,
            "start": ent.start_char,
            "end": ent.end_char,
            "context": text[max(0, ent.start_char - 50):min(len(text), ent.end_char + 50)]
        })
    
    # Extract key numeric facts
    numeric_facts = extract_numeric_facts(doc)
    
    # Extract date references
    date_references = extract_date_references(doc)
    
    # Extract relationships between entities
    relationships = extract_entity_relationships(doc)
    
    # Extract quoted statements
    quotes = extract_quotes(text)
    
    # Create new metadata with extracted information
    enhanced_metadata = {
        **document.metadata,
        "entities": entities,
        "numeric_facts": numeric_facts,
        "date_references": date_references,
        "entity_relationships": relationships,
        "quotes": quotes
    }
    
    return Document(content=document.content, metadata=enhanced_metadata)
```

## Knowledge Graph Construction

Build a knowledge graph during ingestion to capture entity relationships and improve fact retrieval.

```python
def build_knowledge_graph_during_ingestion(documents):
    """Construct a knowledge graph from ingested documents to capture relationships."""
    import networkx as nx
    
    # Initialize knowledge graph
    graph = nx.DiGraph()
    
    # Extract entities and relationships from each document
    for doc in documents:
        entities = extract_entities(doc.content)
        relationships = extract_relationships(doc.content, entities)
        
        # Add entities to graph
        for entity in entities:
            if not graph.has_node(entity["id"]):
                graph.add_node(
                    entity["id"],
                    type=entity["type"],
                    name=entity["name"],
                    mentions=[{
                        "doc_id": doc.metadata.get("id"),
                        "context": entity["context"]
                    }]
                )
            else:
                # Update existing entity with new mention
                graph.nodes[entity["id"]]["mentions"].append({
                    "doc_id": doc.metadata.get("id"),
                    "context": entity["context"]
                })
        
        # Add relationships to graph
        for rel in relationships:
            if graph.has_node(rel["source"]) and graph.has_node(rel["target"]):
                graph.add_edge(
                    rel["source"],
                    rel["target"],
                    type=rel["type"],
                    confidence=rel["confidence"],
                    source_doc=doc.metadata.get("id"),
                    context=rel["context"]
                )
    
    # Store graph in a format that can be used for retrieval enhancement
    nx.write_gpickle(graph, "knowledge_graph.gpickle")
    
    # Convert graph to document metadata for Qdrant storage
    for node_id in graph.nodes:
        node_data = graph.nodes[node_id]
        edges = list(graph.out_edges(node_id, data=True))
        
        # Create entity document with relationship information
        entity_doc = Document(
            content=f"Entity: {node_data['name']} ({node_data['type']})",
            metadata={
                "entity_id": node_id,
                "entity_type": node_data["type"],
                "entity_name": node_data["name"],
                "relationships": [{
                    "target": target,
                    "type": data["type"],
                    "confidence": data.get("confidence", 1.0)
                } for _, target, data in edges],
                "mentions": node_data.get("mentions", []),
                "is_entity_node": True
            }
        )
        documents.append(entity_doc)
    
    return documents
```

## Consistency Checking

Check for contradictory information within documents during ingestion and flag inconsistencies.

```python
def check_consistency_during_ingestion(documents):
    """Identify and flag inconsistencies between document chunks."""
    # Collect facts by subject
    fact_groups = {}
    
    # Extract facts from each document
    for doc in documents:
        facts = extract_facts(doc.content)
        
        for fact in facts:
            subject = fact["subject"]
            if subject not in fact_groups:
                fact_groups[subject] = []
            
            fact_groups[subject].append({
                "predicate": fact["predicate"],
                "object": fact["object"],
                "confidence": fact["confidence"],
                "source_doc": doc.metadata.get("id"),
                "context": fact["context"]
            })
    
    # Check for contradictions within each subject group
    contradictions = []
    for subject, facts in fact_groups.items():
        # Group facts by predicate
        by_predicate = {}
        for fact in facts:
            pred = fact["predicate"]
            if pred not in by_predicate:
                by_predicate[pred] = []
            by_predicate[pred].append(fact)
        
        # Look for conflicting values for the same predicate
        for predicate, pred_facts in by_predicate.items():
            if len(pred_facts) > 1:
                # Check if objects conflict
                objects = [f["object"] for f in pred_facts]
                if len(set(objects)) > 1 and not are_compatible(objects):
                    contradictions.append({
                        "subject": subject,
                        "predicate": predicate,
                        "conflicting_facts": pred_facts
                    })
    
    # Update document metadata with consistency information
    for doc in documents:
        doc_id = doc.metadata.get("id")
        doc_contradictions = []
        
        for contradiction in contradictions:
            # Check if this document is involved in the contradiction
            involved = any(
                fact["source_doc"] == doc_id 
                for fact in contradiction["conflicting_facts"]
            )
            if involved:
                doc_contradictions.append(contradiction)
        
        if doc_contradictions:
            doc.metadata["contradictions"] = doc_contradictions
            doc.metadata["has_consistency_issues"] = True
        else:
            doc.metadata["has_consistency_issues"] = False
    
    return documents
```

## Confidence Scoring

Add confidence scores to ingested information to help guide the retrieval process.

```python
def add_confidence_scoring(documents):
    """Add confidence scores to documents based on various quality metrics."""
    for doc in documents:
        # Calculate different confidence dimensions
        source_confidence = assess_source_reliability(doc.metadata.get("source", ""))
        content_quality = assess_content_quality(doc.content)
        factual_density = assess_factual_density(doc.content)
        citation_score = assess_citation_presence(doc.content)
        
        # Check for hedging language that indicates uncertainty
        certainty_score = assess_certainty(doc.content)
        
        # Check for temporal relevance (recency)
        recency_score = assess_temporal_relevance(doc.metadata.get("date"))
        
        # Calculate verification score (are facts independently verifiable)
        verifiability = assess_verifiability(doc.content)
        
        # Calculate overall confidence score as weighted average
        overall_confidence = calculate_weighted_confidence(
            source_confidence, content_quality, factual_density,
            citation_score, certainty_score, recency_score, verifiability
        )
        
        # Add confidence metadata
        doc.metadata["confidence"] = {
            "overall": overall_confidence,
            "source_reliability": source_confidence,
            "content_quality": content_quality,
            "factual_density": factual_density,
            "citation_presence": citation_score,
            "certainty": certainty_score,
            "recency": recency_score,
            "verifiability": verifiability
        }
    
    return documents
```

## Implementation Strategy

To improve your RAG system's ingestion process for reducing hallucination:

1. **Data Cleaning and Normalization**: Implement pre-processing to remove noise and normalize content
2. **Metadata Enrichment**: Extract rich metadata during ingestion to provide better context
3. **Multi-granularity Indexing**: Index at document, section, and chunk levels for better context preservation
4. **Fact Extraction**: Extract and verify key factual statements during ingestion
5. **Knowledge Graph Construction**: Build a graph of relationships between entities for consistency
6. **Quality Filtering**: Filter out low-quality content that may lead to hallucination
7. **Confidence Scoring**: Add confidence scores to help the retrieval system prioritize reliable information

Each improvement can be added incrementally to your existing ingestion pipeline. Focus on the techniques that address your most common hallucination issues first.