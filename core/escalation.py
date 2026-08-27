"""Rule-based escalation policy (variant 7a)."""
from collections import Counter

L0, L1, L2, L3 = "L0", "L1", "L2", "L3"
ACCEPT, ABSTAIN = "ACCEPT", "ABSTAIN"


def escalation_policy(confidence_score, threshold, exception_type, trace, i, max_iter):
    """Choose the next action.

    With `exception_type` set, the L0 repair has already failed, so the choice
    is between L1, L2 and ABSTAIN. With it None, the code ran cleanly and the
    decision follows the confidence score.
    """
    if i >= max_iter:
        return ABSTAIN

    if exception_type is not None:
        exc_counts = Counter(t.get("exception_type") for t in trace if t.get("exception_type"))
        return L1 if exc_counts.get(exception_type, 0) <= 1 else L2

    if confidence_score is not None and confidence_score >= threshold:
        return ACCEPT

    consecutive_l2 = 0
    for t in reversed(trace):
        if t.get("level") == L2:
            consecutive_l2 += 1
        else:
            break
    if consecutive_l2 >= 2:
        return L3
    return L2
