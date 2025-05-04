#!/usr/bin/env python3

"""
Rich Metadata Extraction Module for RAG Ingestion

This module provides enhanced metadata extraction for documents during the RAG ingestion process.
Extracting rich, structured metadata helps provide better context for retrieval and reduces
hallucination by giving the LLM more reliable information about the content's source and nature.

Usage:
    from rich_metadata import enrich_document_metadata
    
    # Enrich document metadata during ingestion
    enriched_doc = enrich_document_metadata(document)
"""

import re
import os
import logging
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, date
import unicodedata

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define common types
Document = Dict[str, Any]  # Will include at least 'content' and 'metadata'

# Regular expressions for entity extraction
DATE_PATTERN = re.compile(r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\b", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
URL_PATTERN = re.compile(r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\.-]*(?:\?\S+)?")
VERSION_PATTERN = re.compile(r"\bv?(\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9.-]+)?)\b")
PHONE_PATTERN = re.compile(r"\b(?:\+\d{1,3}\s?)?(?:\(\d{1,4}\)\s?)?(?:\d{1,4}[-.\s]?){1,4}\d{1,4}\b")

# Common named entities to look for
ORGANIZATION_INDICATORS = [
    "Inc", "LLC", "Ltd", "Corporation", "Corp", "Company", "Co", "GmbH",
    "Limited", "Association", "Foundation", "Institute", "University", "College"
]

# Function to extract and normalize dates
def extract_dates(text: str) -> List[str]:
    """
    Extract and normalize dates from text.
    
    Args:
        text: Document text
        
    Returns:
        List of ISO format dates (YYYY-MM-DD)
    """
    dates = []
    
    # Find all date patterns
    date_matches = DATE_PATTERN.findall(text)
    
    for date_str in date_matches:
        try:
            # Try parsing with multiple formats
            for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", 
                      "%d %b %Y", "%d %B %Y", "%b %d %Y", "%B %d %Y"]:
                try:
                    parsed_date = datetime.strptime(date_str, fmt).date()
                    
                    # Only accept reasonable years 
                    if 1900 <= parsed_date.year <= datetime.now().year + 1:
                        dates.append(parsed_date.isoformat())
                        break
                except ValueError:
                    continue
        except Exception:
            # Skip problematic dates
            pass
    
    # Remove duplicates while preserving order
    unique_dates = []
    seen = set()
    for d in dates:
        if d not in seen:
            seen.add(d)
            unique_dates.append(d)
            
    return unique_dates

# Function to extract document type
def detect_document_type(text: str, filename: Optional[str] = None) -> str:
    """
    Detect the document type based on content and filename.
    
    Args:
        text: Document text
        filename: Optional filename
        
    Returns:
        Document type classification
    """
    # Check filename extension if available
    if filename:
        ext = os.path.splitext(filename)[1].lower()
        if ext in ['.pdf', '.doc', '.docx']:
            return "document"
        elif ext in ['.csv', '.xlsx', '.xls']:
            return "spreadsheet"
        elif ext in ['.md', '.txt', '.rst']:
            return "text"
        elif ext in ['.ppt', '.pptx']:
            return "presentation"
        elif ext in ['.py', '.js', '.java', '.cpp', '.go', '.rs']:
            return "code"
        elif ext in ['.html', '.htm']:
            return "webpage"
            
    # Check content patterns
    if re.search(r'^#\s+|\n#{1,6}\s+', text):
        return "markdown"
    elif re.search(r'<html|<body|<div|<p>|<h[1-6]', text):
        return "html"
    elif re.search(r'def\s+\w+\s*\(|class\s+\w+\s*[:\(]|import\s+\w+|from\s+\w+\s+import', text):
        return "code"
    elif re.search(r'\b(?:Dear|To Whom It May Concern|Sincerely|Regards)[,:]', text):
        return "letter"
    elif re.search(r'\b(?:Abstract|Introduction|Methodology|Results|Conclusion|References)\b', text):
        return "academic"
    elif re.search(r'\b(?:Table of Contents|Chapter \d+)\b', text):
        return "book"
    
    # Default to general text
    return "text"

# Function to extract key entities
def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract named entities from text using pattern matching.
    
    Args:
        text: Document text
        
    Returns:
        Dictionary of entity types and values
    """
    entities = {
        "emails": list(set(EMAIL_PATTERN.findall(text))),
        "urls": list(set(URL_PATTERN.findall(text))),
        "versions": list(set(VERSION_PATTERN.findall(text))),
        "phones": list(set(PHONE_PATTERN.findall(text))),
        "organizations": []
    }
    
    # Extract potential organization names
    for org_indicator in ORGANIZATION_INDICATORS:
        # Find organization names with the indicator
        pattern = rf"\b[A-Z][A-Za-z0-9\s&'-]+\s+{re.escape(org_indicator)}\b"
        orgs = re.findall(pattern, text)
        if orgs:
            entities["organizations"].extend(orgs)
    
    # Deduplicate organizations
    entities["organizations"] = list(set(entities["organizations"]))
    
    return {k: v for k, v in entities.items() if v}  # Remove empty lists

# Function to extract numeric data with units
def extract_numeric_data(text: str) -> List[Dict[str, str]]:
    """
    Extract numeric values with their units.
    
    Args:
        text: Document text
        
    Returns:
        List of numeric data points with values and units
    """
    # Common units and their patterns
    unit_patterns = [
        (r'\b(\d+(?:\.\d+)?)\s*(GB|MB|KB|TB|Bytes|B)\b', 'data_size'),
        (r'\b(\d+(?:\.\d+)?)\s*(kg|g|mg|pounds|lbs|oz)\b', 'weight'),
        (r'\b(\d+(?:\.\d+)?)\s*(km|m|cm|mm|miles|inches|ft|foot|feet)\b', 'distance'),
        (r'\b(\d+(?:\.\d+)?)\s*(hours|hour|hrs|hr|minutes|mins|seconds|secs)\b', 'time'),
        (r'\b(\d+(?:\.\d+)?)\s*(°C|°F|K|degrees Celsius|degrees Fahrenheit|deg C|deg F)\b', 'temperature'),
        (r'\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)|(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:USD|EUR|JPY|GBP)', 'currency'),
        (r'\b(\d+(?:\.\d+)?)\s*(percent|pct|%)\b', 'percentage')
    ]
    
    numeric_data = []
    
    for pattern, data_type in unit_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                # Some regex patterns have multiple capturing groups
                if match[0]:  # Use first non-empty group as value
                    value = match[0]
                    unit = match[1]
                else:
                    value = match[1]
                    unit = match[2] if len(match) > 2 else ""
            else:
                # Simple string value
                parts = match.split()
                if len(parts) >= 2:
                    value = parts[0]
                    unit = parts[1]
                else:
                    value = match
                    unit = ""
                    
            numeric_data.append({
                "value": value,
                "unit": unit,
                "type": data_type
            })
    
    return numeric_data

# Function to classify document topics
def classify_topics(text: str) -> List[str]:
    """
    Classify document topics based on keyword frequency.
    
    Args:
        text: Document text
        
    Returns:
        List of likely topics
    """
    # Common topic keywords
    topic_keywords = {
        "technology": ["software", "hardware", "computer", "algorithm", "programming", "code", "development", "api"],
        "finance": ["budget", "investment", "financial", "revenue", "profit", "loss", "market", "stock", "fund"],
        "healthcare": ["patient", "medical", "health", "treatment", "diagnosis", "hospital", "clinic", "doctor"],
        "education": ["student", "learning", "teaching", "course", "curriculum", "school", "university", "education"],
        "business": ["company", "business", "corporate", "strategy", "management", "customer", "service", "product"],
        "legal": ["law", "legal", "regulation", "compliance", "contract", "agreement", "policy", "rights"],
        "science": ["research", "experiment", "hypothesis", "data", "analysis", "scientific", "theory"],
        "marketing": ["brand", "campaign", "advertising", "market", "consumer", "customer", "promotion"]
    }
    
    # Convert text to lowercase for case-insensitive matching
    lower_text = text.lower()
    
    # Count keyword occurrences for each topic
    topic_scores = {}
    for topic, keywords in topic_keywords.items():
        score = sum(lower_text.count(keyword) for keyword in keywords)
        if score > 0:
            topic_scores[topic] = score
    
    # Sort topics by score (descending)
    sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return top topics (those with at least 20% of the highest score)
    if sorted_topics:
        max_score = sorted_topics[0][1]
        threshold = max_score * 0.2
        return [topic for topic, score in sorted_topics if score >= threshold]
    
    return []

# Function to analyze text sentiment
def analyze_sentiment(text: str) -> Dict[str, float]:
    """
    Simple rule-based sentiment analysis.
    
    Args:
        text: Document text
        
    Returns:
        Dictionary with sentiment scores
    """
    # Simple sentiment wordlists
    positive_words = ["good", "great", "excellent", "positive", "outstanding", "beneficial", 
                      "successful", "improve", "advantage", "useful", "helpful", "best",
                      "recommend", "happy", "pleased", "impressive", "perfect", "wonderful"]
    
    negative_words = ["bad", "poor", "negative", "terrible", "horrible", "worst", "failure",
                      "problem", "issue", "difficult", "disappointing", "fail", "disadvantage",
                      "damage", "risk", "harm", "severe", "trouble", "wrong", "difficult"]
    
    # Convert to lowercase and tokenize
    lower_text = text.lower()
    words = re.findall(r'\b\w+\b', lower_text)
    
    # Count positive and negative words
    positive_count = sum(words.count(word) for word in positive_words)
    negative_count = sum(words.count(word) for word in negative_words)
    
    # Calculate total and sentiment score
    total_sentiment_words = positive_count + negative_count
    if total_sentiment_words == 0:
        return {"sentiment": "neutral", "score": 0.0}
    
    # Calculate normalized scores
    normalized_positive = positive_count / total_sentiment_words
    normalized_negative = negative_count / total_sentiment_words
    
    # Overall sentiment score (-1 to 1 scale)
    sentiment_score = (normalized_positive - normalized_negative)
    
    # Categorize sentiment
    if sentiment_score > 0.25:
        sentiment = "positive"
    elif sentiment_score < -0.25:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    return {
        "sentiment": sentiment,
        "score": round(sentiment_score, 2),
        "positive_ratio": round(normalized_positive, 2),
        "negative_ratio": round(normalized_negative, 2)
    }

# Function to extract document structure information
def extract_document_structure(text: str) -> Dict[str, Any]:
    """
    Extract structural elements from the document.
    
    Args:
        text: Document text
        
    Returns:
        Dictionary with document structure information
    """
    structure = {
        "section_count": 0,
        "sections": [],
        "has_lists": False,
        "list_count": 0,
        "paragraph_count": 0
    }
    
    # Detect sections (headers)
    headers = re.findall(r'^#+\s+(.+)$|^([A-Z][A-Za-z0-9\s]+:)$', text, re.MULTILINE)
    if headers:
        structure["section_count"] = len(headers)
        structure["sections"] = [h[0] or h[1] for h in headers]
    
    # Detect numbered lists
    numbered_lists = re.findall(r'^\s*\d+\.\s+\w+', text, re.MULTILINE)
    
    # Detect bullet lists
    bullet_lists = re.findall(r'^\s*[-*•]\s+\w+', text, re.MULTILINE)
    
    structure["has_lists"] = len(numbered_lists) > 0 or len(bullet_lists) > 0
    structure["list_count"] = len(numbered_lists) + len(bullet_lists)
    
    # Count paragraphs (text blocks separated by blank lines)
    paragraphs = re.split(r'\n\s*\n', text)
    structure["paragraph_count"] = len([p for p in paragraphs if p.strip()])
    
    return structure

# Function to extract potential references or citations
def extract_references(text: str) -> List[str]:
    """
    Extract references or citations from the document.
    
    Args:
        text: Document text
        
    Returns:
        List of potential references
    """
    references = []
    
    # Look for common citation patterns
    
    # Harvard style (Author, Year)
    harvard_refs = re.findall(r'\(([A-Z][a-z]+(?:[-\s][A-Z][a-z]+)*)(?:[\s,]+(?:et al\.?|and)[\s,]+[A-Z][a-z]+)*[\s,]+(\d{4})\)', text)
    if harvard_refs:
        references.extend([f"{author}, {year}" for author, year in harvard_refs])
    
    # IEEE style [1], [2], etc.
    ieee_refs = re.findall(r'\[(\d+)\]', text)
    if ieee_refs:
        references.extend([f"[{ref}]" for ref in ieee_refs])
    
    # APA style section
    if re.search(r'^References\s*$|^Bibliography\s*$|^Works Cited\s*$|^Sources\s*$', text, re.MULTILINE):
        # Find the references section
        ref_sections = re.split(r'^References\s*$|^Bibliography\s*$|^Works Cited\s*$|^Sources\s*$', text, flags=re.MULTILINE)
        if len(ref_sections) > 1:
            ref_text = ref_sections[-1]  # Get the text after the header
            # Split by lines and identify potential references
            lines = ref_text.split('\n')
            current_ref = ""
            for line in lines:
                line = line.strip()
                if not line:
                    if current_ref:
                        references.append(current_ref)
                        current_ref = ""
                elif re.match(r'^[A-Z]', line):  # New reference likely starts with capital letter
                    if current_ref:
                        references.append(current_ref)
                    current_ref = line
                else:
                    if current_ref:
                        current_ref += " " + line
            # Add the last reference if any
            if current_ref:
                references.append(current_ref)
    
    # Remove duplicates while preserving order
    unique_refs = []
    seen = set()
    for ref in references:
        if ref not in seen:
            seen.add(ref)
            unique_refs.append(ref)
            
    return unique_refs

# Main metadata enrichment function
def enrich_document_metadata(document: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract rich metadata from document content.
    
    Args:
        document: Document with content and metadata fields
        
    Returns:
        Updated document with enriched metadata
    """
    if not document or "content" not in document:
        logger.warning("Cannot enrich document: no content field found")
        return document
    
    text = document["content"]
    metadata = document.get("metadata", {}).copy()
    
    logger.info("Enriching document metadata")
    
    # Get document source from metadata if available
    source = metadata.get("source", "")
    
    try:
        # 1. Document file type and classification
        doc_type = detect_document_type(text, source)
        metadata["document_type"] = doc_type
        
        # 2. Extract and normalize dates
        dates = extract_dates(text)
        if dates:
            metadata["dates"] = dates
            metadata["date_range"] = {"earliest": min(dates), "latest": max(dates)}
        
        # 3. Extract named entities
        entities = extract_entities(text)
        if entities:
            metadata["entities"] = entities
        
        # 4. Extract numeric data and units
        numeric_data = extract_numeric_data(text)
        if numeric_data:
            metadata["numeric_data"] = numeric_data
        
        # 5. Topic classification
        topics = classify_topics(text)
        if topics:
            metadata["topics"] = topics
        
        # 6. Sentiment analysis
        sentiment = analyze_sentiment(text)
        if sentiment:
            metadata["sentiment"] = sentiment
        
        # 7. Document structure analysis
        structure = extract_document_structure(text)
        if structure:
            metadata["structure"] = structure
        
        # 8. References/citations
        references = extract_references(text)
        if references:
            metadata["references"] = references
            metadata["references_count"] = len(references)
            
        # 9. Text statistics
        words = re.findall(r'\b\w+\b', text)
        metadata["statistics"] = {
            "word_count": len(words),
            "character_count": len(text),
            "line_count": text.count('\n') + 1
        }
        
        # 10. Content hash for duplicate detection
        import hashlib
        content_hash = hashlib.md5(text.encode()).hexdigest()
        metadata["content_hash"] = content_hash
        
        # 11. Processing timestamp
        metadata["processed_at"] = datetime.now().isoformat()
        
        # Final extraction quality score
        extraction_confidence = calculate_extraction_confidence(metadata)
        metadata["extraction_confidence"] = extraction_confidence
        
        logger.info(f"Metadata enrichment completed with confidence score {extraction_confidence}")
        
    except Exception as e:
        logger.error(f"Error enriching document metadata: {e}")
        # Add error information but don't fail
        metadata["metadata_error"] = str(e)
    
    # Create new document with enriched metadata
    return {
        "content": document["content"],
        "metadata": metadata
    }

# Helper function to calculate confidence in metadata extraction
def calculate_extraction_confidence(metadata: Dict[str, Any]) -> float:
    """
    Calculate a confidence score for the quality of metadata extraction.
    
    Args:
        metadata: Extracted metadata dictionary
        
    Returns:
        Confidence score (0.0 to 1.0)
    """
    # Start with base confidence
    confidence = 0.6
    
    # Add confidence for presence of different metadata types
    if "document_type" in metadata:
        confidence += 0.05
    
    if "dates" in metadata:
        confidence += 0.05
    
    if "entities" in metadata:
        # More entities = higher confidence
        entity_count = sum(len(values) for values in metadata["entities"].values())
        confidence += min(0.1, entity_count * 0.01)
    
    if "topics" in metadata:
        confidence += min(0.1, len(metadata["topics"]) * 0.02)
    
    if "sentiment" in metadata:
        # Higher confidence for non-neutral sentiment
        if metadata["sentiment"].get("sentiment") != "neutral":
            confidence += 0.05
    
    if "references" in metadata and metadata.get("references_count", 0) > 0:
        confidence += 0.1
    
    # Reduce confidence for very short texts
    word_count = metadata.get("statistics", {}).get("word_count", 0)
    if word_count < 50:
        confidence -= 0.2
    elif word_count < 100:
        confidence -= 0.1
    
    # Cap confidence at 1.0
    return min(1.0, max(0.1, confidence))

# Batch processing function for multiple documents
def enrich_documents_batch(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process a batch of documents with metadata enrichment.
    
    Args:
        documents: List of documents to process
        
    Returns:
        List of documents with enriched metadata
    """
    enriched_docs = []
    
    for doc in documents:
        enriched_doc = enrich_document_metadata(doc)
        enriched_docs.append(enriched_doc)
    
    return enriched_docs

# If run as a script, demonstrate functionality
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enrich document metadata")
    parser.add_argument("--input", help="Input text file to process", required=True)
    parser.add_argument("--output", help="Output JSON file for enriched metadata", required=False)
    
    args = parser.parse_args()
    
    try:
        # Read input file
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Create document object
        doc = {
            "content": text,
            "metadata": {
                "source": args.input
            }
        }
        
        # Process document
        enriched_doc = enrich_document_metadata(doc)
        
        # Print or save results
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(enriched_doc["metadata"], f, indent=2)
            print(f"Enriched metadata saved to {args.output}")
        else:
            print(json.dumps(enriched_doc["metadata"], indent=2))
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)