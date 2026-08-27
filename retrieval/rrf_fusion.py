"""Reciprocal Rank Fusion over the three retrieval indexes.

RRF is used because it has no tunable weights: any hand-set weighting would be
fitted on the same data the result is measured on. It also works on ranks
rather than raw scores, so the indexes need no calibration between their
different scales.

This is the fixed baseline a learned re-ranker has to beat, so the formula
stays untouched.
"""
K = 60  # canonical value from the original paper, not tuned here


def _ranks(scored: list) -> dict:
    """Map scored items to competition ranks, so ties share a rank.

    Without this, alphabetical order would silently decide cases where an
    index has no preference.
    """
    out, prev_score, prev_rank = {}, None, 0
    for i, (name, s) in enumerate(sorted(scored, key=lambda kv: -kv[1]), start=1):
        if prev_score is not None and s == prev_score:
            out[name] = prev_rank
        else:
            out[name] = i
            prev_rank, prev_score = i, s
    return out


def fuse(rankings: dict, k: int = K) -> list:
    """Fuse per-index rankings into one ordered list."""
    totals, per_source = {}, {}
    for source, scored in rankings.items():
        ranks = _ranks(scored)
        per_source[source] = ranks
        for name, r in ranks.items():
            totals[name] = totals.get(name, 0.0) + 1.0 / (k + r)
    return sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))


def fuse_with_detail(rankings: dict, k: int = K) -> dict:
    """Fuse and keep each index's contribution, for provenance and debugging."""
    fused = fuse(rankings, k)
    detail = {}
    for source, scored in rankings.items():
        ranks = _ranks(scored)
        scores = dict(scored)
        detail[source] = {n: {"rank": ranks[n], "score": round(scores.get(n, 0.0), 4)} for n in ranks}
    return {"fused": fused, "per_index": detail}
