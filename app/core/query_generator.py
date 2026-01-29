"""
Query Generator Module

Generates multiple search queries from a claim using NER (Named Entity Recognition).
Creates diverse queries to maximize evidence coverage.
Model is lazy-loaded via model_registry to prevent import-time memory allocation.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


def generate_queries(claim: str, keywords: list[str] | None = None) -> List[str]:
    """
    Generate multiple search queries from a claim.
    
    Uses Named Entity Recognition to identify key entities (people, organizations,
    locations) and optional keywords from KeyBERT to create diverse queries
    for better evidence coverage.
    
    Args:
        claim: The claim to generate queries for
        keywords: Optional list of keywords extracted by KeyBERT
        
    Returns:
        List of unique search query strings
    """
    from app.core.model_registry import get_spacy_nlp
    
    nlp = get_spacy_nlp()
    doc = nlp(claim)
    entities = [ent.text for ent in doc.ents]
    
    logger.debug(f"Extracted entities: {entities}")
    if keywords:
        logger.debug(f"Using keywords: {keywords}")

    base = claim.lower()

    queries = [
        base,
        base + " fact check",
        base + " true or false",
        base + " hoax",
        base + " authenticity check",
    ]

    # Entity-based queries for better coverage
    for e in entities:
        queries.append(f"{e} {base}")
        queries.append(f"{base} {e} false")
        queries.append(f"{e} controversy")
        queries.append(f"{e} news verification")

    # Keyword-based queries (from KeyBERT)
    if keywords:
        for kw in keywords[:3]:
            queries.append(f"{kw} fact check")
            queries.append(f"{kw} {base}")
            queries.append(f"{kw} news")

    unique_queries = list(set(queries))
    # Limit to 10 queries max for performance
    unique_queries = unique_queries[:10]
    logger.info(f"Generated {len(unique_queries)} queries from claim")
    
    return unique_queries
