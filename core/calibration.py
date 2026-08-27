"""Confidence calibration phi(v) over the verification signal vector.

Falls back to a weighted heuristic while there are too few episodes to train
on, and uses a logistic regression once episodes accumulate.
"""
import os

import numpy as np

_FEATURES = ["v_type_ok", "v_schema_ok", "v_range_ok", "v_cross_table", "v_semantic_ok"]
_HEURISTIC_WEIGHTS = {"v_type_ok": 0.35, "v_schema_ok": 0.15, "v_range_ok": 0.30, "v_cross_table": 0.10, "v_semantic_ok": 0.10}

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "calibration_model.joblib")
_model = None


def _signals_to_vector(signals: dict) -> np.ndarray:
    # None (signal not applicable) maps to 0.5: neither penalty nor reward
    return np.array([[1.0 if signals.get(f) is True else (0.0 if signals.get(f) is False else 0.5) for f in _FEATURES]])


def _heuristic_confidence(signals: dict) -> float:
    score, total_w = 0.0, 0.0
    for k, w in _HEURISTIC_WEIGHTS.items():
        val = signals.get(k)
        if val is None:
            continue
        score += w * (1.0 if val else 0.0)
        total_w += w
    return score / total_w if total_w else 0.5


def load_model(path: str = None):
    global _model
    path = path or _MODEL_PATH
    if path == _MODEL_PATH and _model is not None:
        return _model
    if os.path.exists(path):
        import joblib
        model = joblib.load(path)
        if path == _MODEL_PATH:
            _model = model
        return model
    return None


def confidence(signals: dict, path: str = None) -> float:
    model = load_model(path)
    if model is None:
        return _heuristic_confidence(signals)
    x = _signals_to_vector(signals)
    return float(model.predict_proba(x)[0, 1])


def fit(episodes: list[dict], path: str = None) -> dict:
    """Fit the calibrator on collected episodes.

    `path` allows training an isolated model without overwriting the one the
    pipeline loads by default. Returns training metrics.
    """
    from sklearn.linear_model import LogisticRegression
    import joblib

    path = path or _MODEL_PATH
    X = np.vstack([_signals_to_vector(e["signals"]) for e in episodes])
    y = np.array([1 if e["correct"] else 0 for e in episodes])

    if len(set(y.tolist())) < 2:
        return {"trained": False, "reason": "both classes required"}

    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X, y)
    joblib.dump(model, path)

    if path == _MODEL_PATH:
        global _model
        _model = model
    train_acc = float(model.score(X, y))
    return {"trained": True, "n_episodes": len(episodes), "train_accuracy": train_acc, "coef": dict(zip(_FEATURES, model.coef_[0].tolist()))}
