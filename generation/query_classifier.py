"""Inference wrapper for the trained query-type classifier.

Runs locally: no network call, no rate limit, microsecond latency.

It sees only the query text, never the document schema. The type of operation
does not depend on which document is open, and feeding the schema in would
bind the model to one table set. Whether the data exists at all is decided
separately by the scope gate.

Returns None when no model has been trained, so the pipeline still runs.
"""
import os

from generation.dataset_core import TAXONOMY

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "query_classifier.joblib",
)

_model = None
_load_attempted = False


def load_model(path: str = None):
    """Lazy load, cached per process."""
    global _model, _load_attempted
    path = path or MODEL_PATH
    if path == MODEL_PATH:
        if _load_attempted:
            return _model
        _load_attempted = True
    if not os.path.exists(path):
        return None
    import joblib
    model = joblib.load(path)
    if path == MODEL_PATH:
        _model = model
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
        "model_path": MODEL_PATH,
        "classes": len(TAXONOMY),
        "needs_network": False,
        "needs_schema": False,
    }
