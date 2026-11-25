"""Unit tests for the decision diagnostics.

This module is measuring apparatus, not pipeline. The FinQA audit made the
case for testing it as strictly as the pipeline itself: both bugs found there
were in the measuring code, not in the system being measured, and both moved
the numbers in a plausible direction where nothing would have looked wrong.

Run:  python -m tests.test_decision
"""
import sys

from core import calibration, decision

PASS, FAIL = [], []
T = 0.7


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append(f"{name}: got {got!r}, want {want!r}")


def sig(**kw):
    """Signal vector; anything unnamed is True."""
    out = {k: True for k in decision.SIGNAL_KEYS}
    out.update(kw)
    return out


def test_accept_has_no_cause():
    a = decision.analyse(sig(), T)
    check("all true -> accepted", a["accepted"], True)
    check("all true -> no pivotal", a["pivotal_negative"], [])
    check("all true -> zero flips", a["min_flips"], 0)


def test_negative_and_neutral_are_not_the_same():
    """None is absence of evidence, False is evidence against."""
    a = decision.analyse(sig(v_cross_table=None), T)
    check("neutral is not negative", a["negative"], [])
    check("neutral is listed", a["neutral"], ["v_cross_table"])

    b = decision.analyse(sig(v_range_ok=False), T)
    check("negative is listed", b["negative"], ["v_range_ok"])
    check("negative is not neutral", b["neutral"], [])


def test_pivotal_split():
    """A neutral signal can cross the threshold without having caused anything.

    v_range_ok=False with v_cross_table=None sits below the threshold. Two
    different single changes lift it: fixing the failing range signal, or
    getting support from the neutral one. Only the first is a cause.
    """
    s = sig(v_range_ok=False, v_cross_table=None)
    a = decision.analyse(s, T)
    check("below threshold", a["accepted"], False)
    check("cause is the negative one", a["pivotal_negative"], ["v_range_ok"])
    check("rescue is the neutral one", a["pivotal_neutral"], ["v_cross_table"])
    check("one change suffices", a["min_flips"], 1)


def test_no_single_cause():
    """When no single flip crosses the threshold, none may be named as THE cause."""
    s = sig(v_type_ok=False, v_range_ok=False, v_cross_table=None,
            v_semantic_ok=False)
    a = decision.analyse(s, T)
    check("rejected", a["accepted"], False)
    if a["min_flips"] == 1:
        FAIL.append("no_single_cause: fixture no longer needs two flips")
    else:
        check("no single pivotal", a["pivotal_negative"], [])
        check("needs more than one", a["min_flips"] >= 2, True)


def test_weak_signals_cannot_reject():
    """Recorded property of the current weighting, not an endorsement of it.

    v_schema_ok and v_semantic_ok together carry 0.25 of the weight, so both
    failing still scores 0.75 and is accepted at threshold 0.70. The cascade
    as it stands cannot reject on those two alone -- which is part of the case
    for a veto signal that does not go through the weighted score.

    With v_cross_table neutral as well the score drops to 0.722 and is still
    accepted, so the property is not an artefact of one vector.
    """
    s = sig(v_schema_ok=False, v_semantic_ok=False)
    conf = calibration.confidence(s)
    check("two weak signals still accept", conf >= T, True)
    check("score with cross_table supporting", round(conf, 3), 0.75)

    s2 = sig(v_schema_ok=False, v_semantic_ok=False, v_cross_table=None)
    conf2 = calibration.confidence(s2)
    check("still accepts with cross_table neutral", conf2 >= T, True)
    check("score with cross_table neutral", round(conf2, 3), 0.722)


def test_outcome_recovers_rejected_candidate():
    """The value behind a rejection must survive into the record.

    Rejection precision is built from it. The pipeline discards it.
    """
    result = {
        "status": "ABSTAIN",
        "abstain_reason": "LOW_CONFIDENCE",
        "trace": [{"iter": 0, "result": 608.07,
                   "signals": sig(v_range_ok=False, v_semantic_ok=False),
                   "confidence": 0.6, "level": "L2"}],
    }
    o = decision.outcome(result, T)
    check("candidate recovered", o["candidate"], 608.07)
    check("has_candidate", o["has_candidate"], True)
    check("reason kept", o["abstain_reason"], "LOW_CONFIDENCE")
    check("two signals were against", o["negative_signals"],
          ["v_range_ok", "v_semantic_ok"])
    # Either one alone would have carried it over 0.70, so both are causes.
    check("both are causes", o["pivotal_signals"],
          ["v_range_ok", "v_semantic_ok"])
    check("no disagreement with the record", o["confidence_mismatch"], False)


def test_confidence_mismatch_is_reported():
    """A recorded score that the calibrator no longer reproduces must show.

    The calibrator can be retrained between a run and its analysis. Silently
    preferring either number would attribute a decision nobody took.
    """
    s = sig(v_range_ok=False, v_semantic_ok=False)
    a = decision.analyse(s, T, confidence_value=0.556)
    check("recomputed, not taken on trust", round(a["confidence"], 3), 0.6)
    check("record preserved", a["recorded_confidence"], 0.556)
    check("disagreement flagged", a["confidence_mismatch"], True)


def test_outcome_reasons():
    gate = {"status": "ABSTAIN", "abstain_reason": "OUT_OF_SCOPE", "trace": []}
    o = decision.outcome(gate, T)
    check("gate reason", o["abstain_reason"], "OUT_OF_SCOPE")
    check("gate has no candidate", o["has_candidate"], False)
    check("gate spent no iterations", o["iterations"], 0)

    exc = {"status": "ABSTAIN", "trace": [
        {"iter": 0, "exception_type": "KeyError", "error": "x", "level": "L1"},
        {"iter": 1, "exception_type": "KeyError", "error": "x", "level": "ABSTAIN"}]}
    o = decision.outcome(exc, T)
    check("exception reason derived", o["abstain_reason"], "EXCEPTION")
    check("exception has no candidate", o["has_candidate"], False)

    acc = {"status": "ACCEPT", "answer": 42.0, "confidence": 1.0,
           "trace": [{"iter": 0, "result": 42.0, "signals": sig(),
                      "confidence": 1.0, "level": "ACCEPT"}]}
    o = decision.outcome(acc, T)
    check("accept reason is none", o["abstain_reason"], None)
    check("accept answer", o["answer"], 42.0)


def test_veto_overrides_score():
    """A veto is the reason on its own, whatever the weighted score says."""
    result = {"status": "ABSTAIN", "trace": [
        {"iter": 0, "result": 999.0, "signals": sig(), "confidence": 1.0,
         "veto": {"fired": True, "expected": 100.0, "given": 999.0},
         "level": "ABSTAIN"}]}
    o = decision.outcome(result, T)
    check("veto is the reason", o["abstain_reason"], "VETO")
    check("veto is the cause", o["pivotal_signals"], ["v_invariant"])
    check("veto beats a perfect score", o["confidence"], 1.0)


def test_rejection_metrics():
    """Undecidable records are excluded from both rates, never assumed wrong."""
    recs = [
        {"rejected": True, "candidate_correct": False},   # caught
        {"rejected": True, "candidate_correct": False},   # caught
        {"rejected": True, "candidate_correct": True},    # false rejection
        {"rejected": False, "candidate_correct": False},  # missed
        {"rejected": False, "candidate_correct": True},   # correctly kept
        {"rejected": True, "candidate_correct": None},    # no candidate
    ]
    m = decision.rejection_metrics(recs)
    check("scored excludes undecidable", m["n_scored"], 5)
    check("undecidable counted", m["n_undecidable"], 1)
    check("wrong count", m["n_wrong"], 3)
    check("recall", round(m["recall"], 4), round(2 / 3, 4))
    check("precision", round(m["precision"], 4), round(2 / 3, 4))
    check("false rejections", m["false_rejections"], 1)

    empty = decision.rejection_metrics([])
    check("empty is None not zero", empty["recall"], None)
    check("empty precision is None", empty["precision"], None)


def main():
    for fn in (test_accept_has_no_cause, test_negative_and_neutral_are_not_the_same,
               test_pivotal_split, test_no_single_cause, test_weak_signals_cannot_reject,
               test_outcome_recovers_rejected_candidate,
               test_confidence_mismatch_is_reported, test_outcome_reasons,
               test_veto_overrides_score, test_rejection_metrics):
        fn()

    print(f"passed: {len(PASS)}")
    if FAIL:
        print(f"FAILED: {len(FAIL)}")
        for f in FAIL:
            print("  -", f)
        return 1
    print("all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
