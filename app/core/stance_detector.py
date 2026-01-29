"""
Stance Detector Module with Optimizations

Uses zero-shot classification to determine if evidence supports, refutes, 
or discusses a claim.

Default model: DeBERTa-v3 (~700MB, faster, sufficient accuracy)
Alternative: BART-MNLI (set STANCE_MODEL=bart, ~1.6GB, higher accuracy)

Optimizations applied:
- Confidence calibration (Rank 13)
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

MODELS = {
    "deberta": {
        "name": "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
        "display": "DeBERTa-v3",
        "size": "~700MB"
    },
    "bart": {
        "name": "facebook/bart-large-mnli",
        "display": "BART-large",
        "size": "~1.6GB"
    }
}


def get_current_model() -> str:
    """Return the currently loaded model name."""
    from app.core.model_registry import get_nli_classifier
    _, model_key = get_nli_classifier()
    return MODELS.get(model_key, MODELS["deberta"])["display"]


# Optimization #13: Confidence calibration
def calibrate_confidence(raw_score: float, temperature: float = 1.3) -> float:
    """Apply temperature scaling for better calibrated confidence scores."""
    return raw_score ** (1 / temperature)


def detect_stance(evidence_sentence: str, claim: str) -> Dict:
    """
    Performs zero-shot stance detection using NLI.
    
    Uses natural language inference to determine if the evidence
    supports, refutes, or is neutral towards the claim.
    
    Args:
        evidence_sentence: The sentence from the article (premise)
        claim: The claim to fact-check (used to form hypotheses)
    
    Returns:
        dict: {label: str, confidence: float}
    """
    from app.core.model_registry import get_nli_classifier
    
    if not evidence_sentence or not claim:
        return {"label": "neutral", "confidence": 0}
    
    premise = evidence_sentence.strip()
    claim = claim.strip()
    
    try:
        nli_classifier, _ = get_nli_classifier()
        
        hypotheses = [
            f"This supports the claim: {claim}",
            f"This contradicts the claim: {claim}",
            f"This is unrelated to the claim: {claim}"
        ]
        
        result = nli_classifier(
            premise, 
            hypotheses,
            multi_label=False
        )
        
        label_map = {
            hypotheses[0]: "supports",
            hypotheses[1]: "refutes",
            hypotheses[2]: "neutral"
        }
        
        top_hypothesis = result["labels"][0]
        label = label_map.get(top_hypothesis, "neutral")
        raw_confidence = float(result["scores"][0])
        
        # Apply confidence calibration
        confidence = calibrate_confidence(raw_confidence)
        
        if label == "neutral":
            label = "discusses"
        
        return {
            "label": label,
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Stance detection error: {e}")
        return {"label": "discusses", "confidence": 0}


def batch_detect_stance(premises: List[str], claim: str) -> List[Dict]:
    """
    Batch stance detection for multiple premises.
    
    Args:
        premises: List of evidence sentences
        claim: The claim to check against
        
    Returns:
        List of stance results
    """
    return [detect_stance(premise, claim) for premise in premises]
