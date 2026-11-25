"""Inference wrapper for the trained query-type classifier.

Runs locally: no network call, no rate limit, microsecond latency.

It sees only the query text, never the document schema. The type of operation
does not depend on which document is open, and feeding the schema in would
bind the model to one table set. Whether the data exists at all is decided
separately by the scope gate.

Returns None when no model has been trained, so the pipeline still runs.
"""
import os

from core import lang
from generation.dataset_core import TAXONOMY

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# The Ukrainian model predates the language profiles and keeps its original
# file name; training a second language must not overwrite the artefact the
# phase-2 numbers were measured on.
_LEGACY_PATH = os.path.join(_DATA_DIR, "query_classifier.joblib")


def model_path(code: str = None) -> str:
    code = code or lang.active_lang()
    path = os.path.join(_DATA_DIR, f"query_classifier_{code}.joblib")
    if code == "uk" and not os.path.exists(path) and os.path.exists(_LEGACY_PATH):
        return _LEGACY_PATH
    return path


_cache = {}


def load_model(path: str = None):
    """Lazy load, cached per path."""
    path = path or model_path()
    if path in _cache:
        return _cache[path]
    model = None
    if os.path.exists(path):
        import joblib
        model = joblib.load(path)
    _cache[path] = model
    return model


def is_available() -> bool:
    return load_model() is not None


def classify(query: str, path: str = None):
    """-> a label from TAXONOMY, or None if no model is trained."""
    model = load_model(path)
    if model is None:
        return None
    return str(model.predict([query])[0])


def classify_with_confidence(query: str, path: str = None):
    """-> (label, probability) or (None, None).

    The probability is not consumed yet; it is there for an escalation policy
    that accounts for classifier uncertainty.
    """
    model = load_model(path)
    if model is None:
        return None, None
    label = str(model.predict([query])[0])
    try:
        proba = model.predict_proba([query])[0]
        conf = float(max(proba))
    except (AttributeError, NotImplementedError):
        conf = None
    return label, conf


def describe() -> dict:
    return {
        "available": is_available(),
        "lang": lang.active_lang(),
        "model_path": model_path(),
        "classes": len(TAXONOMY),
        "needs_network": False,
        "needs_schema": False,
    }
