"""
Singleton Model Registry - Lazy Loading with Optimizations

All ML models are loaded on-demand at first use, not at import time.
This prevents memory exhaustion during Gunicorn worker spawn.

Optimizations applied:
- Torch gradients disabled globally (Rank 2)
- Tokenizer parallelism disabled (Rank 9)
- Models set to eval mode after loading
"""

import os
import logging
from threading import Lock

# Optimization #9: Disable tokenizer parallelism before any imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Optimization #2: Disable torch gradients globally
import torch
torch.set_grad_enabled(False)

logger = logging.getLogger(__name__)

# Thread-safe locks for singleton initialization
_locks = {
    "spacy": Lock(),
    "keybert": Lock(),
    "sbert": Lock(),
    "nli": Lock(),
}

# Singleton instances (None until first access)
_instances = {
    "spacy": None,
    "keybert": None,
    "sbert": None,
    "nli": None,
    "nli_model_key": None,
}


def get_spacy_nlp():
    """Lazy-load spaCy model (singleton)."""
    if _instances["spacy"] is None:
        with _locks["spacy"]:
            if _instances["spacy"] is None:
                import spacy
                logger.info("Loading spaCy model (en_core_web_sm)...")
                _instances["spacy"] = spacy.load("en_core_web_sm")
                logger.info("✓ spaCy model loaded")
    return _instances["spacy"]


def get_keybert_model():
    """Lazy-load KeyBERT model (singleton)."""
    if _instances["keybert"] is None:
        with _locks["keybert"]:
            if _instances["keybert"] is None:
                from keybert import KeyBERT
                logger.info("Loading KeyBERT model...")
                _instances["keybert"] = KeyBERT(model="all-MiniLM-L6-v2")
                logger.info("✓ KeyBERT model loaded")
    return _instances["keybert"]


def get_sbert_model():
    """Lazy-load Sentence-BERT model (singleton)."""
    if _instances["sbert"] is None:
        with _locks["sbert"]:
            if _instances["sbert"] is None:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading Sentence-BERT model (all-MiniLM-L6-v2)...")
                model = SentenceTransformer("all-MiniLM-L6-v2")
                model.eval()  # Optimization #2: Set to eval mode
                _instances["sbert"] = model
                logger.info("✓ Sentence-BERT model loaded")
    return _instances["sbert"]


def get_nli_classifier():
    """Lazy-load NLI classifier (singleton). Returns (classifier, model_key)."""
    if _instances["nli"] is None:
        with _locks["nli"]:
            if _instances["nli"] is None:
                from transformers import pipeline
                
                model_key = os.getenv("STANCE_MODEL", "bart").lower()
                
                MODELS = {
                    "deberta": "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
                    "bart": "facebook/bart-large-mnli",
                }
                
                if model_key not in MODELS:
                    logger.warning(f"Unknown model '{model_key}', using deberta")
                    model_key = "deberta"
                
                model_name = MODELS[model_key]
                logger.info(f"Loading NLI classifier ({model_key}: {model_name})...")
                
                try:
                    _instances["nli"] = pipeline(
                        "zero-shot-classification",
                        model=model_name,
                        device=-1
                    )
                    _instances["nli_model_key"] = model_key
                    logger.info(f"✓ NLI classifier loaded ({model_key})")
                except Exception as e:
                    if model_key == "deberta":
                        raise RuntimeError(f"Failed to load NLI model: {e}")
                    logger.warning(f"BART failed, trying DeBERTa: {e}")
                    _instances["nli"] = pipeline(
                        "zero-shot-classification",
                        model=MODELS["deberta"],
                        device=-1
                    )
                    _instances["nli_model_key"] = "deberta"
                    logger.info("✓ NLI classifier loaded (deberta fallback)")
    
    return _instances["nli"], _instances["nli_model_key"]


def are_models_loaded() -> bool:
    """Check if all models are loaded (for health check)."""
    return all([
        _instances.get("spacy") is not None,
        _instances.get("sbert") is not None,
        _instances.get("nli") is not None
    ])


def warmup_all_models():
    """Load all models for warmup endpoint."""
    get_spacy_nlp()
    get_sbert_model()
    get_keybert_model()
    get_nli_classifier()
    return {
        "spacy": _instances["spacy"] is not None,
        "sbert": _instances["sbert"] is not None,
        "keybert": _instances["keybert"] is not None,
        "nli": _instances["nli"] is not None
    }
