"""Public entry point for hybrid table retrieval.

Called *before* `pipeline.run` rather than inside it, so recall can be measured
without running the whole correction loop, and so the with/without-retrieval
ablation is a single argument.
"""
from retrieval import rrf_fusion, structural_index, temporal_index, vector_index

# Default number of tables handed to the generator. One, because the point of
# retrieval is narrowing: with top_k equal to the table count the module
# degenerates into a no-op.
DEFAULT_TOP_K = 1

INDEXES = {
    "structural": structural_index.rank,
    "temporal": temporal_index.rank,
    "vector": vector_index.rank,
}


def rank_tables(query: str, tables: dict, indexes: dict = None) -> dict:
    """Full ranking result: fused list plus each index's contribution."""
    indexes = indexes or INDEXES
    rankings = {name: fn(query, tables) for name, fn in indexes.items()}
    return rrf_fusion.fuse_with_detail(rankings)


def select_tables(query: str, tables: dict, top_k: int = DEFAULT_TOP_K,
                  indexes: dict = None) -> list:
    """Query plus all document tables -> names of the selected ones."""
    if not tables:
        return []
    if len(tables) <= top_k:
        return list(tables.keys())
    fused = rank_tables(query, tables, indexes)["fused"]
    return [name for name, _ in fused[:top_k]]


def explain(query: str, tables: dict, top_k: int = DEFAULT_TOP_K) -> dict:
    """Same decision, broken down per index, for the UI and error analysis."""
    detail = rank_tables(query, tables)
    return {
        "query": query,
        "selected": [n for n, _ in detail["fused"][:top_k]],
        "fused": detail["fused"],
        "per_index": detail["per_index"],
        "vector_backend": vector_index.backend_name(),
    }
