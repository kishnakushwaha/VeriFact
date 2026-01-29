"""
Claim Extractor Module

Extracts claims from URLs or text using NLP techniques.
Uses Spacy for sentence segmentation and KeyBERT for keyword extraction.
Models are lazy-loaded via model_registry to prevent import-time memory allocation.
"""

from trafilatura import fetch_url, extract
import nltk
import logging
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

# Download required NLTK data (lightweight, ~5MB)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)


def extract_text_from_url(url: str) -> str:
    """
    Fetch and clean article content from a URL.
    
    Args:
        url: The URL of the article to extract text from
        
    Returns:
        Cleaned article text, or empty string if extraction fails
    """
    try:
        html = fetch_url(url)
        if not html:
            logger.warning(f"Failed to fetch HTML from {url}")
            return ""
        text = extract(html)
        return text or ""
    except Exception as e:
        logger.error(f"Error extracting text from {url}: {e}")
        return ""


def clean_text(text: str) -> str:
    """Normalize whitespace and formatting."""
    if not text:
        return ""
    return " ".join(text.split())


def score_sentence_importance(sentence: str, doc) -> float:
    """
    Score sentence importance based on multiple factors.
    
    Args:
        sentence: The sentence to score
        doc: The spacy doc containing entities
        
    Returns:
        Importance score (higher = more important)
    """
    from app.core.model_registry import get_spacy_nlp
    nlp = get_spacy_nlp()
    
    score = 0.0
    
    # Contains named entities (+2 per entity type)
    sentence_ents = [ent for ent in doc.ents if ent.text in sentence]
    score += len(set(ent.label_ for ent in sentence_ents)) * 2.0
    
    # Contains numbers/statistics (+1)
    sentence_doc = nlp(sentence)
    if any(tok.like_num for tok in sentence_doc):
        score += 1.0
    
    # Good length: not too short, not too long (+1)
    if 30 < len(sentence) < 200:
        score += 1.0
    
    # Contains quotation marks (likely a claim) (+1.5)
    if '"' in sentence or "'" in sentence:
        score += 1.5
    
    return score


def extract_claim_from_text(text: str) -> Tuple[str, List[str]]:
    """
    Extracts the main claim by choosing the most important sentence.
    Uses sentence importance scoring instead of always taking the first sentence.
    Also extracts keywords using KeyBERT for enhanced query generation.
    
    Args:
        text: The article text to extract claim from
        
    Returns:
        Tuple of (claim_sentence, keywords_list)
    """
    from app.core.model_registry import get_spacy_nlp, get_keybert_model
    
    text = clean_text(text)
    if not text:
        return "", []

    nlp = get_spacy_nlp()
    doc = nlp(text[:5000])
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]

    if not sentences:
        return text[:500], []

    # Score each sentence for importance
    scored_sentences = []
    for idx, sent in enumerate(sentences[:10]):
        score = score_sentence_importance(sent, doc)
        position_bonus = max(0, 3 - idx) * 0.5
        total_score = score + position_bonus
        scored_sentences.append((sent, total_score))
    
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    claim = scored_sentences[0][0]

    # Extract keywords using KeyBERT
    try:
        kw_model = get_keybert_model()
        keywords_raw = kw_model.extract_keywords(
            claim, 
            top_n=5,
            keyphrase_ngram_range=(1, 2),
            stop_words='english'
        )
        keywords = [kw[0] for kw in keywords_raw]
    except Exception as e:
        logger.warning(f"KeyBERT extraction failed: {e}")
        keywords = []

    return claim, keywords