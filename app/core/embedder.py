"""
Embedder Module - Semantic Similarity with Accuracy Improvements

Uses Sentence-BERT (SBERT) to compute semantic similarity between claims and evidence.
Converts text to 384-dimensional vectors (MiniLM) and compares using cosine similarity.

Accuracy improvements:
- Returns top-N matching sentences (not just top-1)
- LRU cache for claim embeddings
"""

from sentence_transformers import util
from functools import lru_cache
import logging
from typing import Tuple, List, Optional
import hashlib

logger = logging.getLogger(__name__)


# LRU cache for claim embeddings (bounded to 32 entries)
@lru_cache(maxsize=32)
def _encode_claim_cached(claim_hash: str, claim: str):
    """Cache claim embeddings by hash to avoid re-encoding same claims."""
    from app.core.model_registry import get_sbert_model
    sbert_model = get_sbert_model()
    return sbert_model.encode(claim, convert_to_tensor=True)


def encode_claim(claim: str):
    """Get cached claim embedding."""
    claim_hash = hashlib.md5(claim.encode()).hexdigest()
    return _encode_claim_cached(claim_hash, claim)


def get_best_matching_sentences(
    claim: str, 
    sentences: List[str],
    top_n: int = 3
) -> List[Tuple[str, float]]:
    """
    Find the top N sentences most semantically similar to the claim.
    
    Args:
        claim: The claim to fact-check
        sentences: List of candidate sentences from an article
        top_n: Number of top matches to return
        
    Returns:
        List of (sentence, similarity_score) tuples, sorted by score descending
    """
    from app.core.model_registry import get_sbert_model
    
    if not sentences:
        return []

    sbert_model = get_sbert_model()
    
    # Use cached claim embedding
    claim_emb = encode_claim(claim)
    sent_embs = sbert_model.encode(sentences, convert_to_tensor=True, batch_size=32)

    # Compute cosine similarity
    cosine_scores = util.pytorch_cos_sim(claim_emb, sent_embs)[0]

    # Convert to Python list
    scores_list = cosine_scores.cpu().tolist()

    # Create list of (sentence, score) tuples
    scored_sentences = list(zip(sentences, scores_list))
    
    # Sort by score descending and return top N
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    
    return scored_sentences[:top_n]


def get_best_matching_sentence(
    claim: str, 
    sentences: List[str]
) -> Tuple[Optional[str], float, List[float]]:
    """
    Find the sentence most semantically similar to the claim.
    (Backwards compatible wrapper)
    
    Args:
        claim: The claim to fact-check
        sentences: List of candidate sentences from an article
        
    Returns:
        Tuple of:
            - best_sentence: The most similar sentence (or None)
            - best_score: Similarity score (0-1)
            - all_scores: List of all similarity scores
    """
    from app.core.model_registry import get_sbert_model
    
    if not sentences:
        return None, 0, []

    sbert_model = get_sbert_model()
    
    # Use cached claim embedding
    claim_emb = encode_claim(claim)
    sent_embs = sbert_model.encode(sentences, convert_to_tensor=True, batch_size=32)

    # Compute cosine similarity
    cosine_scores = util.pytorch_cos_sim(claim_emb, sent_embs)[0]

    # Convert to Python list
    scores_list = cosine_scores.cpu().tolist()

    # Get index of highest match
    best_idx = scores_list.index(max(scores_list))

    best_sentence = sentences[best_idx]
    best_score = scores_list[best_idx]

    return best_sentence, best_score, scores_list
