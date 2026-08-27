"""Trains the query-type classifier.

Character n-gram TF-IDF with a linear SVM: character n-grams handle Ukrainian
inflection without a lemmatiser, and a linear model keeps inference in the
microsecond range, which is the reason for training a local model at all.

Splitting is done by whole seed groups, so paraphrases of one seed never
straddle train and validation.

This script never touches the held-out live queries; those are reserved for a
single final evaluation.
"""
import json
import os

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DATASET_PATH = os.path.join(DATA_DIR, "query_type_dataset.json")
MODEL_PATH = os.path.join(DATA_DIR, "query_classifier.joblib")

RANDOM_STATE = 42
VAL_FRACTION = 0.2


def load_dataset(path: str = None) -> list:
    path = path or DATASET_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Датасет не знайдено: {path}\n"
            "Спочатку згенеруй його:\n"
            "    python -m generation.dataset_core            # лише ядро, без ключа\n"
            "    python -m generation.dataset_paraphrase      # ядро + LLM-шар (потрібен GROQ_API_KEY)"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_pipeline():
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            sublinear_tf=True,
            min_df=2,
            lowercase=True,
        )),
        # Wrapped to expose predict_proba, which LinearSVC lacks.
        ("clf", CalibratedClassifierCV(LinearSVC(C=1.0, random_state=RANDOM_STATE), cv=3)),
    ])


def split_by_seed(data: list, val_fraction: float = VAL_FRACTION, random_state: int = RANDOM_STATE):
    """Split by whole seed groups."""
    from sklearn.model_selection import GroupShuffleSplit

    X = np.array([d["query"] for d in data])
    y = np.array([d["type"] for d in data])
    groups = np.array([d["seed_id"] for d in data])

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_state)
    train_idx, val_idx = next(splitter.split(X, y, groups))

    overlap = set(groups[train_idx]) & set(groups[val_idx])
    assert not overlap, f"leak: {len(overlap)} seed groups on both sides"

    return (X[train_idx], y[train_idx], X[val_idx], y[val_idx],
            np.array([d["source"] for d in data])[val_idx])


def train(data: list = None, save: bool = True) -> dict:
    from sklearn.metrics import accuracy_score, classification_report

    data = data or load_dataset()
    X_tr, y_tr, X_val, y_val, src_val = split_by_seed(data)

    model = build_pipeline()
    model.fit(X_tr, y_tr)

    pred = model.predict(X_val)
    acc = float(accuracy_score(y_val, pred))

    # Accuracy per source is an early signal of whether the model learned
    # only the generator's templates.
    per_source = {}
    for s in sorted(set(src_val.tolist())):
        mask = src_val == s
        if mask.sum():
            per_source[s] = {"n": int(mask.sum()),
                             "accuracy": float(accuracy_score(y_val[mask], pred[mask]))}

    if save:
        import joblib
        os.makedirs(DATA_DIR, exist_ok=True)
        joblib.dump(model, MODEL_PATH)

    return {
        "n_total": len(data),
        "n_train": len(X_tr),
        "n_val": len(X_val),
        "n_seed_groups": len({d["seed_id"] for d in data}),
        "val_accuracy": acc,
        "per_source": per_source,
        "report": classification_report(y_val, pred, zero_division=0),
        "model_path": MODEL_PATH if save else None,
    }


if __name__ == "__main__":
    m = train()
    print("=" * 78)
    print("ТРЕНУВАННЯ КЛАСИФІКАТОРА ТИПУ ЗАПИТУ")
    print("=" * 78)
    print(f"Датасет: {m['n_total']} прикладів, {m['n_seed_groups']} seed-груп")
    print(f"Розбиття по групах: train {m['n_train']} / validation {m['n_val']}")
    print(f"\nValidation accuracy: {m['val_accuracy']:.1%}")

    print("\nТочність на validation за джерелом прикладу:")
    for s, d in m["per_source"].items():
        print(f"  {s:18s} n={d['n']:4d}  accuracy={d['accuracy']:.1%}")

    print("\n" + m["report"])
    print(f"Модель збережено: {m['model_path']}")
    print("\nЖиві запити тут не використовувались: вони зарезервовані для")
    print("одноразової фінальної оцінки.")
