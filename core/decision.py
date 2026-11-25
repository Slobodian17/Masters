"""Diagnostics for one accept/reject decision of the verification cascade.

`rejection recall` on its own is a number without a cause. To say what the
invariant signal catches beyond the other four, every rejection has to record
WHICH signal decided it, and that is not simply "the signal that was False":
the cascade rejects on a weighted score, so a single false signal may be
harmless while two together are decisive.

Attribution is therefore counterfactual and exact rather than heuristic. A
signal is pivotal when flipping it alone to True would have carried the score
over the threshold. When no single flip suffices, `min_flips` reports how many
were needed together — a rejection with min_flips=2 has no single cause, and
saying otherwise would be an invention.

A veto signal is the one exception: when it fires it IS the reason, whatever
the score would have been. Invariants are that kind of signal, which is the
point of adding them.

Nothing here calls an LLM or touches a table; it reads a finished `run()`
result.
"""
import itertools

from core import calibration

SIGNAL_KEYS = tuple(calibration._FEATURES)

# Why the pipeline stopped without answering.
OUT_OF_SCOPE = "OUT_OF_SCOPE"       # gate refused before any generation
EXCEPTION = "EXCEPTION"             # code never ran cleanly
LOW_CONFIDENCE = "LOW_CONFIDENCE"   # a value was produced and rejected
MAX_ITER = "MAX_ITER"               # repair budget exhausted
VETO = "VETO"                       # a veto signal refuted the value


def _with(signals: dict, flips) -> dict:
    out = dict(signals)
    for k in flips:
        out[k] = True
    return out


def analyse(signals: dict, threshold: float, confidence_value: float = None) -> dict:
    """Attribute an accept/reject decision to individual signals.

    -> {"confidence", "accepted", "negative", "neutral",
        "pivotal_negative", "pivotal_neutral", "min_flips", "min_flip_sets"}

    `negative` lists signals that were False. `neutral` lists signals that were
    None, which is not a failure: None means the signal did not apply, and it
    is kept apart so that "no evidence" is never reported as "evidence
    against".

    Pivotal signals are split the same way, and the split is not cosmetic. A
    neutral signal can be pivotal too, because giving it support raises the
    score -- but it did not cause the rejection, it merely failed to rescue
    it. Only `pivotal_negative` answers "which signal fired".

    The score is always recomputed here rather than taken from the caller,
    so that the baseline and the counterfactuals come from the same function.
    Mixing a recorded score with freshly computed alternatives would attribute
    a decision that was never taken -- the calibrator can be retrained between
    the run and the analysis. `confidence_value`, when given, is compared
    against the recomputed one and any disagreement is reported rather than
    resolved silently.
    """
    conf = calibration.confidence(signals)
    accepted = conf >= threshold
    mismatch = (confidence_value is not None
                and abs(conf - confidence_value) > 1e-9)

    negative = [k for k in SIGNAL_KEYS if signals.get(k) is False]
    neutral = [k for k in SIGNAL_KEYS if signals.get(k) is None]

    if accepted:
        return {"confidence": conf, "recorded_confidence": confidence_value,
                "confidence_mismatch": mismatch, "accepted": True,
                "negative": negative, "neutral": neutral,
                "pivotal_negative": [], "pivotal_neutral": [],
                "min_flips": 0, "min_flip_sets": []}

    candidates = negative + neutral
    pivotal = [k for k in candidates
               if calibration.confidence(_with(signals, [k])) >= threshold]

    min_flips, min_sets = 0, []
    for size in range(1, len(candidates) + 1):
        sets = [c for c in itertools.combinations(candidates, size)
                if calibration.confidence(_with(signals, c)) >= threshold]
        if sets:
            min_flips, min_sets = size, [list(s) for s in sets]
            break

    return {"confidence": conf, "recorded_confidence": confidence_value,
            "confidence_mismatch": mismatch, "accepted": False,
            "negative": negative, "neutral": neutral,
            "pivotal_negative": [k for k in pivotal if k in negative],
            "pivotal_neutral": [k for k in pivotal if k in neutral],
            "min_flips": min_flips, "min_flip_sets": min_sets}


def _last_verification_step(trace: list):
    """Last trace entry that carries a produced value, or None."""
    for step in reversed(trace or []):
        if "signals" in step:
            return step
    return None


def outcome(result: dict, threshold: float) -> dict:
    """Flatten one `run()` result into a decision record.

    The candidate value matters as much as the final answer. On ABSTAIN the
    pipeline throws the value away, but whether that value was wrong is
    exactly what rejection precision is made of, so it is recovered from the
    trace here rather than being lost.
    """
    trace = result.get("trace") or []
    status = result.get("status")
    step = _last_verification_step(trace)

    rec = {
        "status": status,
        "iterations": len(trace),
        "answer": result.get("answer") if status == "ACCEPT" else None,
        "candidate": step.get("result") if step else None,
        "has_candidate": step is not None,
        "confidence": result.get("confidence") if status == "ACCEPT"
                      else (step.get("confidence") if step else None),
        "signals": step.get("signals") if step else None,
        "veto": (step or {}).get("veto"),
        "abstain_reason": result.get("abstain_reason"),
    }

    if rec["abstain_reason"] is None and status == "ABSTAIN":
        # Derive it from the shape of the trace when the pipeline did not say.
        if not trace:
            rec["abstain_reason"] = OUT_OF_SCOPE
        elif step is None:
            rec["abstain_reason"] = EXCEPTION
        elif len(trace) > 0 and (step or {}).get("veto", {}) and step["veto"].get("fired"):
            rec["abstain_reason"] = VETO
        else:
            rec["abstain_reason"] = LOW_CONFIDENCE

    if rec["signals"]:
        a = analyse(rec["signals"], threshold, rec["confidence"])
        rec.update({
            "negative_signals": a["negative"],
            "neutral_signals": a["neutral"],
            "pivotal_signals": a["pivotal_negative"],
            "pivotal_neutral": a["pivotal_neutral"],
            "min_flips": a["min_flips"],
            "min_flip_sets": a["min_flip_sets"],
            "confidence_mismatch": a["confidence_mismatch"],
        })
    else:
        rec.update({"negative_signals": [], "neutral_signals": [],
                    "pivotal_signals": [], "pivotal_neutral": [],
                    "min_flips": None, "min_flip_sets": [],
                    "confidence_mismatch": False})

    if rec["veto"] and rec["veto"].get("fired"):
        rec["abstain_reason"] = VETO
        rec["pivotal_signals"] = ["v_invariant"]

    return rec


# Refusals that never reached the cascade. Keeping them out of rejection
# precision is not bookkeeping: the scope gate and the cascade answer
# different questions -- "can this document answer at all" against "is this
# particular value right" -- and a rate that mixes them describes neither.
# A gate refusal has no candidate to be right or wrong about, and an
# exhausted repair budget produced no value either.
PRE_CASCADE_REASONS = (OUT_OF_SCOPE, EXCEPTION)


def rejection_metrics(records: list) -> dict:
    """Rejection precision and recall for the CASCADE alone.

    Each record needs `rejected` (bool) and `candidate_correct` (bool or None);
    `abstain_reason` is used to separate the mechanisms when present.

        recall    = rejected and wrong / all wrong
        precision = rejected and wrong / all rejected

    Records with no candidate are excluded from both and counted by reason,
    never silently assumed wrong. `coverage` says what share of the input the
    cascade actually got to judge, so a flattering rate measured on a third of
    the data cannot be read as if it applied to all of it.
    """
    scored = [r for r in records if r.get("candidate_correct") is not None]
    undecidable = [r for r in records if r.get("candidate_correct") is None]

    by_reason = {}
    for r in undecidable:
        key = r.get("abstain_reason") or "OTHER"
        by_reason[key] = by_reason.get(key, 0) + 1

    wrong = [r for r in scored if not r["candidate_correct"]]
    rejected = [r for r in scored if r["rejected"]]
    caught = [r for r in rejected if not r["candidate_correct"]]

    n = len(records)
    return {
        "n": n,
        "n_scored": len(scored),
        "n_undecidable": len(undecidable),
        "undecidable_by_reason": by_reason,
        "n_gate_refusals": by_reason.get(OUT_OF_SCOPE, 0),
        "n_no_code": by_reason.get(EXCEPTION, 0),
        "coverage": len(scored) / n if n else None,
        "n_wrong": len(wrong),
        "n_rejected": len(rejected),
        "recall": len(caught) / len(wrong) if wrong else None,
        "precision": len(caught) / len(rejected) if rejected else None,
        "false_rejections": len(rejected) - len(caught),
    }
