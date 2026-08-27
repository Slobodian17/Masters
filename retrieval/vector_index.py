"""Retrieval index 3/3: semantic similarity.

The lexical and temporal indexes fail when a query uses synonyms, which is
where embeddings help. Two backends are supported, following the same
fallback pattern as the PDF parser:

  1. sentence-transformers plus ChromaDB;
  2. character n-gram TF-IDF on scikit-learn.

The fallback is a working path, not a placeholder: character n-grams handle
Ukrainian morphology reasonably and need neither a GPU nor a large download.
The heavy stack is enabled explicitly through an environment variable so the
same command yields the same numbers on any machine.
"""
import os

import config  # noqa: F401 - loads .env

_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Enabled explicitly rather than by package presence, for reproducibility.
USE_EMBEDDINGS_ENV = "VKSK_USE_EMBEDDINGS"

_backend = None


def embeddings_enabled() -> bool:
    return os.environ.get(USE_EMBEDDINGS_ENV, "").strip().lower() in ("1", "true", "yes")


def _table_units(name: str, df) -> list:
    """Table -> one indexing unit per row.

    Encoding a whole table as a single vector makes long tables lose to short
    ones, since the averaged vector resembles no specific query. Rows are
    indexed separately and pooled by maximum, which also matches the
    granularity of the structural index.
    """
    labels = [str(v) for v in df.iloc[:, 0].tolist()] if df.shape[1] else []
    headers = " ".join(str(c) for c in df.columns)
    units = [f"{name}: {lbl}" for lbl in labels if str(lbl).strip()]
    units.append(f"{name}: {headers}")  # separate unit for period queries
    return units


def _build_units(tables: dict) -> tuple:
    """-> (texts, owners), where owner is the table each unit belongs to."""
    texts, owners = [], []
    for name, df in tables.items():
        for unit in _table_units(name, df):
            texts.append(unit)
            owners.append(name)
    return texts, owners


def _max_pool(sims, owners: list, names: list) -> dict:
    out = {n: 0.0 for n in names}
    for s, owner in zip(sims, owners):
        v = max(0.0, min(1.0, float(s)))
        if v > out[owner]:
            out[owner] = v
    return out


# ---------- backend 1: sentence-transformers + ChromaDB ----------

class _EmbeddingBackend:
    name = "sentence-transformers+chromadb"

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(_MODEL_NAME)
        self._collection = None
        self._names = []

    def index(self, tables: dict):
        import chromadb
        import numpy as np

        self._names = list(tables.keys())
        texts, self._owners = _build_units(tables)
        vecs = self._model.encode(texts, normalize_embeddings=True)

        client = chromadb.EphemeralClient()
        try:
            client.delete_collection("tables")
        except Exception:
            pass
        self._collection = client.create_collection("tables", metadata={"hnsw:space": "cosine"})
        self._collection.add(
            ids=[f"u{i}" for i in range(len(texts))],
            embeddings=[np.asarray(v, dtype=float).tolist() for v in vecs],
            documents=texts,
            metadatas=[{"table": o} for o in self._owners],
        )
        self._n_units = len(texts)

    def score_all(self, query: str) -> dict:
        import numpy as np

        qv = np.asarray(self._model.encode([query], normalize_embeddings=True)[0], dtype=float)
        res = self._collection.query(query_embeddings=[qv.tolist()], n_results=self._n_units)
        sims, owners = [], []
        for dist, meta in zip(res["distances"][0], res["metadatas"][0]):
            # chromadb returns 1 - cosine as distance
            sims.append(1.0 - float(dist))
            owners.append(meta["table"])
        return _max_pool(sims, owners, self._names)


# ---------- backend 2: character n-grams + TF-IDF ----------

class _TfidfBackend:
    name = "char-ngram-tfidf (sklearn)"

    def __init__(self):
        self._vectorizer = None
        self._matrix = None
        self._names = []

    def index(self, tables: dict):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._names = list(tables.keys())
        texts, self._owners = _build_units(tables)
        self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self._matrix = self._vectorizer.fit_transform(texts)

    def score_all(self, query: str) -> dict:
        from sklearn.metrics.pairwise import cosine_similarity

        qv = self._vectorizer.transform([query])
        sims = cosine_similarity(qv, self._matrix)[0]
        return _max_pool(sims, self._owners, self._names)


def get_backend():
    """Return the active backend, built once per process."""
    global _backend
    if _backend is not None:
        return _backend
    if embeddings_enabled():
        try:
            _backend = _EmbeddingBackend()
            return _backend
        except Exception as e:  # package missing or model unavailable
            print(f"[vector_index] embeddings недоступні ({type(e).__name__}: {e}); "
                  f"працюю на TF-IDF-fallback")
    _backend = _TfidfBackend()
    return _backend


def reset_backend():
    """Reset the cached backend; used by tests that switch modes."""
    global _backend
    _backend = None


def rank(query: str, tables: dict) -> list:
    if not tables:
        return []
    backend = get_backend()
    backend.index(tables)
    scores = backend.score_all(query)
    return sorted(((n, scores.get(n, 0.0)) for n in tables), key=lambda kv: (-kv[1], kv[0]))


def score(query: str, df, name: str = "table") -> float:
    return dict(rank(query, {name: df})).get(name, 0.0)


def backend_name() -> str:
    return get_backend().name
