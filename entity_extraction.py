#!/usr/bin/env python3

"""
Entity Extraction and Normalization for RAG Systems

This module provides advanced entity extraction and normalization functionality to
improve embedding quality by identifying and normalizing key entities in document content.
This helps create more accurate and consistent embeddings by standardizing entity references
and providing additional structured metadata.

Features:
- Named entity recognition (NER) for common entity types
- Entity normalization (consistent representation)
- Coreference resolution (linking pronouns to entities)
- Entity-aware text enhancement for embeddings
- Structured entity metadata for improved retrieval

Usage:
    from entity_extraction import extract_and_normalize_entities, enhance_text_with_entities
    
    # Extract and normalize entities in a document
    entities = extract_and_normalize_entities(document_text)
    
    # Enhance text with entity information for better embeddings
    enhanced_text = enhance_text_with_entities(document_text, entities)
"""

import re
import logging
import json
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict
import unicodedata
import string

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define entity types
ENTITY_TYPES = [
    "PERSON", "ORGANIZATION", "LOCATION", "DATE", "TIME", 
    "MONEY", "PERCENT", "PRODUCT", "EVENT", "URL", "EMAIL"
]

# Regular expressions for entity extraction
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
URL_PATTERN = re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\.-]*(?:\?\S+)?')
DATE_PATTERN = re.compile(r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\b', re.IGNORECASE)
MONEY_PATTERN = re.compile(r'\$\s*\d+(?:[,.]\d+)*(?:\s*(?:thousand|million|billion|trillion))?|\d+(?:[,.]\d+)*\s*(?:USD|EUR|GBP|JPY|CHF|CAD|AUD|CNY|HKD|NZD|KRW)')
PERCENT_PATTERN = re.compile(r'\b\d+(?:\.\d+)?%\b')
VERSION_PATTERN = re.compile(r'\bv?(?:\d+\.){1,3}\d+(?:-[a-zA-Z0-9.-]+)?\b')
PHONE_PATTERN = re.compile(r'\b(?:\+\d{1,3}\s?)?(?:\(\d{1,4}\)\s?)?(?:\d{1,4}[-.\s]?){1,4}\d{1,4}\b')

# Common organization indicators for improving organization detection
ORG_INDICATORS = [
    "Inc", "LLC", "Ltd", "Limited", "Corp", "Corporation", "Company", "Co", 
    "GmbH", "AG", "Foundation", "Institute", "University", "College", "School",
    "Association", "Society", "Partners", "Group", "Team", "Department"
]

# Common location indicators
LOCATION_INDICATORS = [
    "Street", "Avenue", "Road", "Boulevard", "Lane", "Drive", "Place", "Square",
    "Highway", "Bridge", "Park", "City", "Town", "Village", "County", "District",
    "State", "Province", "Country", "Region", "Area", "Valley", "Mountain", "Island",
    "River", "Lake", "Ocean", "Sea", "Bay", "Coast", "Port", "Airport"
]

# A list of common pronouns for coreference resolution
PRONOUNS = [
    "he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs",
    "it", "its", "this", "that", "these", "those"
]

# Dictionary of common entity abbreviations for normalization
COMMON_ABBREVIATIONS = {
    "USA": "United States of America",
    "US": "United States",
    "UK": "United Kingdom",
    "EU": "European Union",
    "UN": "United Nations",
    "CEO": "Chief Executive Officer",
    "CFO": "Chief Financial Officer",
    "CTO": "Chief Technology Officer",
    "VP": "Vice President",
    "Dr.": "Doctor",
    "Prof.": "Professor",
    "Mr.": "Mister",
    "Mrs.": "Missus",
    "MS": "Microsoft",
    "FB": "Facebook",
    "AMZN": "Amazon",
    "GOOG": "Google",
    "AAPL": "Apple",
    "IBM": "International Business Machines"
}

def normalize_text(text: str) -> str:
    """
    Normalize text for consistent entity extraction.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Replace newlines with spaces
    text = text.replace('\n', ' ')
    
    # Normalize Unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    # Remove punctuation at word boundaries
    text = re.sub(r'\b[^\w\s]\b', '', text)
    
    return text

def extract_entities_rule_based(text: str) -> Dict[str, List[str]]:
    """
    Extract entities using pattern matching rules.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary of extracted entities by type
    """
    entities = defaultdict(set)
    
    # Extract emails
    emails = EMAIL_PATTERN.findall(text)
    entities["EMAIL"].update(emails)
    
    # Extract URLs
    urls = URL_PATTERN.findall(text)
    entities["URL"].update(urls)
    
    # Extract dates
    dates = DATE_PATTERN.findall(text)
    entities["DATE"].update(dates)
    
    # Extract money references
    money = MONEY_PATTERN.findall(text)
    entities["MONEY"].update(money)
    
    # Extract percentages
    percentages = PERCENT_PATTERN.findall(text)
    entities["PERCENT"].update(percentages)
    
    # Extract versions
    versions = VERSION_PATTERN.findall(text)
    entities["VERSION"].update(versions)
    
    # Extract phone numbers
    phones = PHONE_PATTERN.findall(text)
    entities["PHONE"].update(phones)
    
    # Extract organizations (basic heuristic approach)
    for org_indicator in ORG_INDICATORS:
        org_pattern = re.compile(r'\b([A-Z][A-Za-z0-9\s&\'-]+)\s+' + re.escape(org_indicator) + r'\b')
        orgs = org_pattern.findall(text)
        entities["ORGANIZATION"].update([f"{org} {org_indicator}" for org in orgs])
    
    # Extract locations (basic approach)
    for loc_indicator in LOCATION_INDICATORS:
        loc_pattern = re.compile(r'\b([A-Z][A-Za-z0-9\s&\'-]+)\s+' + re.escape(loc_indicator) + r'\b')
        locs = loc_pattern.findall(text)
        entities["LOCATION"].update([f"{loc} {loc_indicator}" for loc in locs])
    
    # Convert sets to lists for JSON serialization
    return {k: list(v) for k, v in entities.items() if v}

def try_spacy_extraction(text: str) -> Dict[str, List[str]]:
    """
    Try to extract entities using spaCy if available.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary of extracted entities by type, or empty dict if spaCy unavailable
    """
    try:
        import spacy
        
        # Load the spaCy model (requires a pre-downloaded model)
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("SpaCy model not found. Attempting to download...")
            spacy.cli.download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
        
        # Process the text
        doc = nlp(text)
        
        # Extract named entities
        entities = defaultdict(set)
        for ent in doc.ents:
            # Map spaCy entity types to our types
            ent_type = ent.label_
            if ent_type == "PERSON" or ent_type == "PER":
                entities["PERSON"].add(ent.text)
            elif ent_type == "ORG":
                entities["ORGANIZATION"].add(ent.text)
            elif ent_type in ["GPE", "LOC"]:
                entities["LOCATION"].add(ent.text)
            elif ent_type == "DATE":
                entities["DATE"].add(ent.text)
            elif ent_type == "TIME":
                entities["TIME"].add(ent.text)
            elif ent_type in ["MONEY", "CARDINAL"]:
                entities["MONEY"].add(ent.text)
            elif ent_type == "PERCENT":
                entities["PERCENT"].add(ent.text)
            elif ent_type == "PRODUCT":
                entities["PRODUCT"].add(ent.text)
            elif ent_type == "EVENT":
                entities["EVENT"].add(ent.text)
        
        # Convert sets to lists for JSON serialization
        return {k: list(v) for k, v in entities.items() if v}
        
    except ImportError:
        logger.warning("spaCy not available. Using rule-based entity extraction only.")
        return {}
    except Exception as e:
        logger.error(f"Error in spaCy extraction: {e}")
        return {}

def normalize_entities(entities: Dict[str, List[str]]) -> Dict[str, List[Dict[str, str]]]:
    """
    Normalize extracted entities for consistent representation.
    
    Args:
        entities: Dictionary of extracted entities by type
        
    Returns:
        Dictionary of normalized entities with original forms
    """
    normalized = defaultdict(list)
    
    for entity_type, entity_list in entities.items():
        for entity in entity_list:
            # Skip empty entities
            if not entity.strip():
                continue
                
            # Create base normalized entity
            norm_entity = {
                "original": entity,
                "normalized": entity
            }
            
            # Apply type-specific normalization
            if entity_type == "PERSON":
                # Normalize person names (remove titles, normalize spacing)
                norm_text = entity
                for title in ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof."]:
                    norm_text = norm_text.replace(f"{title} ", "")
                norm_entity["normalized"] = " ".join(norm_text.split())
                
            elif entity_type == "ORGANIZATION":
                # Check for abbreviations
                norm_text = entity
                for abbr, full in COMMON_ABBREVIATIONS.items():
                    if entity == abbr:
                        norm_text = full
                        break
                    elif entity == full:
                        norm_entity["abbreviation"] = abbr
                norm_entity["normalized"] = norm_text
                
            elif entity_type == "DATE":
                # Normalize dates to ISO format if possible
                try:
                    from dateutil import parser
                    try:
                        parsed_date = parser.parse(entity)
                        norm_entity["normalized"] = parsed_date.strftime("%Y-%m-%d")
                        norm_entity["iso_date"] = parsed_date.isoformat()[:10]
                    except Exception:
                        pass
                except ImportError:
                    pass
            
            # Add to normalized entities
            normalized[entity_type].append(norm_entity)
    
    return dict(normalized)

def resolve_coreferences_simple(text: str, entities: Dict[str, List[Dict[str, str]]]) -> str:
    """
    Perform simple rule-based coreference resolution.
    
    Args:
        text: Original text
        entities: Dictionary of normalized entities
        
    Returns:
        Text with some pronouns replaced by their referents
    """
    # This is a very basic implementation - full coreference resolution 
    # typically requires much more sophisticated NLP models
    
    resolved_text = text
    
    # Extract all person and organization entities
    people = []
    for person in entities.get("PERSON", []):
        people.append(person["normalized"])
    
    orgs = []
    for org in entities.get("ORGANIZATION", []):
        orgs.append(org["normalized"])
    
    # Simple pronoun resolution - replace pronouns with most recently mentioned entity
    # This is overly simplistic but helps in some cases
    sentences = re.split(r'(?<=[.!?])\s+', text)
    last_entity = None
    last_entity_type = None
    
    for i, sentence in enumerate(sentences):
        # Check for entities in this sentence
        found_entity = False
        for person in people:
            if person in sentence:
                last_entity = person
                last_entity_type = "PERSON"
                found_entity = True
                break
                
        if not found_entity:
            for org in orgs:
                if org in sentence:
                    last_entity = org
                    last_entity_type = "ORGANIZATION"
                    found_entity = True
                    break
        
        # Replace pronouns in the next sentence if we have a potential referent
        if i < len(sentences) - 1 and last_entity:
            next_sentence = sentences[i + 1]
            # Replace pronouns based on entity type
            if last_entity_type == "PERSON":
                # Very basic - would need to consider gender in a real implementation
                for pronoun in ["he", "him", "his", "she", "her", "they", "them", "their"]:
                    # Only replace pronouns at word boundaries
                    next_sentence = re.sub(r'\b' + pronoun + r'\b', f"{last_entity}", next_sentence, flags=re.IGNORECASE)
            elif last_entity_type == "ORGANIZATION":
                for pronoun in ["it", "its", "they", "them", "their"]:
                    next_sentence = re.sub(r'\b' + pronoun + r'\b', f"{last_entity}", next_sentence, flags=re.IGNORECASE)
                    
            sentences[i + 1] = next_sentence
    
    # Reconstruct the text
    resolved_text = " ".join(sentences)
    return resolved_text

def extract_and_normalize_entities(text: str) -> Dict[str, Any]:
    """
    Extract and normalize entities from text.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary with extracted and normalized entities
    """
    logger.info("Extracting and normalizing entities")
    
    # Extract entities with rule-based patterns
    rule_entities = extract_entities_rule_based(text)
    
    # Try spaCy extraction if available
    spacy_entities = try_spacy_extraction(text)
    
    # Merge entities from both approaches
    merged_entities = defaultdict(set)
    
    for entity_type, entities in rule_entities.items():
        merged_entities[entity_type].update(entities)
        
    for entity_type, entities in spacy_entities.items():
        merged_entities[entity_type].update(entities)
    
    # Convert sets to lists
    merged_dict = {k: list(v) for k, v in merged_entities.items() if v}
    
    # Normalize entities
    normalized_entities = normalize_entities(merged_dict)
    
    # Add entity statistics
    entity_count = sum(len(entities) for entities in normalized_entities.values())
    
    return {
        "entities": normalized_entities,
        "stats": {
            "total_entities": entity_count,
            "entity_types": list(normalized_entities.keys()),
            "type_counts": {k: len(v) for k, v in normalized_entities.items()}
        }
    }

def enhance_text_with_entities(text: str, entity_data: Dict[str, Any]) -> str:
    """
    Enhance text with entity information for better embeddings.
    
    Args:
        text: Original text
        entity_data: Entity data from extract_and_normalize_entities
        
    Returns:
        Enhanced text with entity annotations
    """
    logger.info("Enhancing text with entity information")
    
    # Skip if no entities found
    if not entity_data or "entities" not in entity_data:
        return text
    
    entities = entity_data["entities"]
    
    # Start with original text
    enhanced_text = text
    
    # Attempt basic coreference resolution
    enhanced_text = resolve_coreferences_simple(enhanced_text, entities)
    
    # Add an entity summary section at the beginning
    entity_summary = []
    
    for entity_type, entity_list in entities.items():
        if entity_list:
            entity_summary.append(f"{entity_type}:")
            for entity in entity_list[:5]:  # Limit to top 5 entities per type
                if isinstance(entity, dict):
                    orig = entity.get("original", "")
                    norm = entity.get("normalized", "")
                    if orig and norm and orig != norm:
                        entity_summary.append(f"- {orig} ({norm})")
                    else:
                        entity_summary.append(f"- {orig or norm}")
                else:
                    entity_summary.append(f"- {entity}")
            
            # Add indication if more entities were omitted
            if len(entity_list) > 5:
                entity_summary.append(f"  (and {len(entity_list) - 5} more)")
    
    # Only add summary if we found entities
    if entity_summary:
        summary_text = "\n".join(entity_summary)
        enhanced_text = f"ENTITY SUMMARY:\n{summary_text}\n\n{enhanced_text}"
    
    return enhanced_text

def get_entity_metadata(text: str) -> Dict[str, Any]:
    """
    Extract entity metadata for document in structured format.
    
    Args:
        text: Document text
        
    Returns:
        Dictionary with entity metadata for embedding
    """
    # Extract and normalize entities
    entity_data = extract_and_normalize_entities(text)
    
    # Format entity metadata for document
    metadata = {
        "entity_count": entity_data.get("stats", {}).get("total_entities", 0),
        "entity_types": entity_data.get("stats", {}).get("entity_types", []),
    }
    
    # Add key entities by type
    entities = entity_data.get("entities", {})
    for entity_type, entity_list in entities.items():
        # Get list of normalized entities (up to 10 per type)
        normalized_items = []
        for item in entity_list[:10]:
            if isinstance(item, dict):
                normalized_items.append(item.get("normalized", ""))
            else:
                normalized_items.append(item)
        
        # Add to metadata
        key = f"{entity_type.lower()}_entities"
        metadata[key] = normalized_items
    
    return metadata

if __name__ == "__main__":
    """Command line interface for testing entity extraction."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Extract and normalize entities from text")
    parser.add_argument("--input", "-i", help="Input file path", required=False)
    parser.add_argument("--output", "-o", help="Output file path for JSON results", required=False)
    parser.add_argument("--text", "-t", help="Text to process", required=False)
    parser.add_argument("--enhance", "-e", action="store_true", help="Enhance text with entity annotations")
    
    args = parser.parse_args()
    
    # Get input text
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        print("Either --input or --text must be provided")
        parser.print_help()
        sys.exit(1)
    
    # Extract and normalize entities
    entity_data = extract_and_normalize_entities(text)
    
    # Enhance text if requested
    if args.enhance:
        enhanced_text = enhance_text_with_entities(text, entity_data)
        print("\nEnhanced text:")
        print("-" * 40)
        print(enhanced_text)
    
    # Output results
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(entity_data, f, indent=2)
        print(f"Entity data written to {args.output}")
    else:
        print("\nExtracted entities:")
        print("-" * 40)
        print(json.dumps(entity_data, indent=2))