"""Conformal, risk-controlled confidence calibration.

The logistic calibrator produces a score phi and a hand-picked threshold, but
"phi >= 0.75 means the answer is reliable" has no grounding: phi is a model
output, not a frequency. The question "what share of accepted answers is
actually wrong?" stays unanswered.

This module computes the threshold instead of guessing it, so that

    Pr[answer accepted AND wrong] <= alpha

holds with a finite-sample, distribution-free guarantee that assumes only
exchangeability between the calibration and test sets.

Split-conformal selective prediction:

  1. Take a calibration set of (signals, correct) episodes, separate from the
     data the scorer was fitted on.
  2. Score each episode.
  3. Find the smallest threshold tau whose empirical joint error, with the
     conformal correction, stays within alpha:

         (#{phi_i >= tau and not correct_i} + 1) / (n + 1) <= alpha

  4. Accept at inference exactly when phi >= tau.

The joint probability is bounded rather than the conditional one: guaranteeing
Pr[wrong | accepted] <= alpha is impossible without further assumptions, since
a single accepted-and-wrong answer makes the conditional rate 100% at any tau.
The conditional rate is reported as diagnostics only.

The guarantee is meaningful only when the calibration set contains errors. On
an error-free set any threshold satisfies the inequality and tau collapses to
zero, which is correct but empty; `calibrate` flags that case rather than
returning a plausible-looking threshold.
"""
import math

import numpy as np

DEFAULT_ALPHA = 0.05


# ------------------------- helpers -------------------------

def _scores(episodes: list, scorer) -> np.ndarray:
    return np.array([float(scorer(e["signals"])) for e in episodes])


def _labels(episodes: list) -> np.ndarray:
    return np.array([1 if e["correct"] else 0 for e in episodes], dtype=int)


# ------------------------- calibration -------------------------

def calibrate(episodes: list, alpha: float = DEFAULT_ALPHA, scorer=None) -> dict:
    """Compute the threshold tau with the joint-error guarantee.

    `episodes` is the calibration set, separate from the scorer's training
    data. Returns tau plus diagnostics and a degeneracy flag.
    """
    if scorer is None:
        from core.calibration import confidence as scorer

    n = len(episodes)
    if n == 0:
        return {"alpha": alpha, "tau": 1.0, "n": 0, "n_errors": 0,
                "degenerate": True,
                "reason": "empty calibration set"}

    phi = _scores(episodes, scorer)
    correct = _labels(episodes)
    n_errors = int((correct == 0).sum())

    # Candidate thresholds: the observed scores, plus 0 and just above the
    # maximum (the "accept nothing" option).
    candidates = np.unique(np.concatenate([[0.0], phi, [phi.max() + 1e-9]]))

    tau, achieved = None, None
    for t in candidates:  # ascending, so the first hit is the smallest tau
        accepted = phi >= t
        joint_errors = int(((correct == 0) & accepted).sum())
        bound = (joint_errors + 1) / (n + 1)
        if bound <= alpha:
            tau, achieved = float(t), float(bound)
            break

    if tau is None:
        # even the strictest threshold cannot reach alpha on this set
        return {"alpha": alpha, "tau": float(phi.max() + 1e-9), "n": n,
                "n_errors": n_errors, "degenerate": False, "achievable": False,
                "reason": (f"alpha={alpha} unreachable: the smallest bound "
                           f"is {1 / (n + 1):.4f}. Needs a larger calibration "
                           f"set or a larger alpha"),
                "min_possible_alpha": 1 / (n + 1)}

    accepted = phi >= tau
    n_accepted = int(accepted.sum())
    cond_err = (float(((correct == 0) & accepted).sum() / n_accepted)
                if n_accepted else 0.0)

    out = {
        "alpha": alpha,
        "tau": tau,
        "n": n,
        "n_errors": n_errors,
        "n_accepted": n_accepted,
        "coverage": n_accepted / n,
        "joint_error_bound": achieved,          # guaranteed upper bound
        "conditional_error_rate": cond_err,     # diagnostics, not guaranteed
        "achievable": True,
        "degenerate": n_errors == 0,
    }
    if out["degenerate"]:
        out["reason"] = (
            "No errors in the calibration set, so any threshold satisfies the "
            "inequality and tau collapses to 0. The guarantee is formally "
            "correct but empty."
        )
    return out


def accept(signals: dict, tau: float, scorer=None) -> bool:
    """Accept or reject against the conformal threshold."""
    if scorer is None:
        from core.calibration import confidence as scorer
    return float(scorer(signals)) >= tau


# ------------------------- calibration quality metrics -------------------------

def brier_score(episodes: list, scorer=None) -> float:
    """Mean squared deviation of phi from actual correctness; lower is
    better. Captures calibration and resolution together."""
    if scorer is None:
        from core.calibration import confidence as scorer
    if not episodes:
        return float("nan")
    phi = _scores(episodes, scorer)
    y = _labels(episodes)
    return float(np.mean((phi - y) ** 2))


def expected_calibration_error(episodes: list, n_bins: int = 10, scorer=None) -> dict:
    """Expected calibration error over confidence bins.

    Measures calibration proper: whether phi = 0.8 really means 80%.
    """
    if scorer is None:
        from core.calibration import confidence as scorer
    if not episodes:
        return {"ece": float("nan"), "bins": []}

    phi = _scores(episodes, scorer)
    y = _labels(episodes)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece, bins = 0.0, []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (phi >= lo) & (phi < hi) if i < n_bins - 1 else (phi >= lo) & (phi <= hi)
        k = int(mask.sum())
        if not k:
            continue
        conf = float(phi[mask].mean())
        acc = float(y[mask].mean())
        ece += (k / len(episodes)) * abs(conf - acc)
        bins.append({"lo": float(lo), "hi": float(hi), "n": k,
                     "mean_confidence": conf, "accuracy": acc,
                     "gap": abs(conf - acc)})
    return {"ece": float(ece), "bins": bins}


def risk_coverage_curve(episodes: list, scorer=None) -> dict:
    """Risk-coverage curve and its area (AURC), lower being better.

    For every threshold: coverage is the accepted share, risk the error rate
    among accepted.
    """
    if scorer is None:
        from core.calibration import confidence as scorer
    if not episodes:
        return {"aurc": float("nan"), "points": []}

    phi = _scores(episodes, scorer)
    y = _labels(episodes)
    order = np.argsort(-phi)  # most confident first
    y_sorted = y[order]

    points, risks = [], []
    errors = 0
    for k in range(1, len(y_sorted) + 1):
        errors += int(y_sorted[k - 1] == 0)
        risk = errors / k
        cov = k / len(y_sorted)
        points.append({"coverage": cov, "risk": risk,
                       "threshold": float(phi[order][k - 1])})
        risks.append(risk)

    return {"aurc": float(np.mean(risks)), "points": points}


# ------------------------- comparison with the logistic scorer -------------------------

def compare_with_logreg(cal_episodes: list, test_episodes: list,
                        alpha: float = DEFAULT_ALPHA,
                        heuristic_threshold: float = 0.75) -> dict:
    """Compare the conformal threshold with the hand-picked one.

    Both are evaluated on the same test set, or the comparison is meaningless.
    """
    from core.calibration import confidence as scorer

    cal = calibrate(cal_episodes, alpha=alpha, scorer=scorer)
    tau = cal["tau"]

    phi_test = _scores(test_episodes, scorer)
    y_test = _labels(test_episodes)
    n = len(test_episodes)

    def _eval(threshold):
        acc_mask = phi_test >= threshold
        k = int(acc_mask.sum())
        joint = float(((y_test == 0) & acc_mask).sum() / n) if n else 0.0
        cond = float(((y_test == 0) & acc_mask).sum() / k) if k else 0.0
        return {"threshold": float(threshold), "coverage": (k / n) if n else 0.0,
                "joint_error": joint, "conditional_error": cond, "n_accepted": k}

    return {
        "alpha": alpha,
        "calibration": cal,
        "conformal": _eval(tau),
        "heuristic": _eval(heuristic_threshold),
        "brier": brier_score(test_episodes, scorer),
        "ece": expected_calibration_error(test_episodes, scorer=scorer)["ece"],
        "aurc": risk_coverage_curve(test_episodes, scorer=scorer)["aurc"],
        "guarantee_held": _eval(tau)["joint_error"] <= alpha,
    }


def describe(result: dict) -> str:
    """Human-readable calibration summary for the UI and reports."""
    if result.get("degenerate"):
        return (f"tau={result['tau']:.3f} (degenerate: {result['n_errors']} errors "
                f"in {result['n']} episodes, guarantee is empty)")
    if not result.get("achievable", True):
        return f"alpha={result['alpha']} unreachable: {result.get('reason', '')}"
    return (f"tau={result['tau']:.3f}, coverage {result['coverage']:.0%}, "
            f"Pr[accepted and wrong] <= {result['joint_error_bound']:.3f} "
            f"(target alpha={result['alpha']})")
