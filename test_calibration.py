"""Checks that the calibrator actually trains.

The frozen test set yields only correct episodes, so the logistic model has a
single class and cannot fit; this uses synthetic episodes of both classes to
confirm the training path itself works, writing to an isolated model file.
"""
from core import calibration

synthetic_episodes = [
    {"signals": {"v_type_ok": True, "v_schema_ok": True, "v_range_ok": True, "v_cross_table": True, "v_semantic_ok": True}, "correct": True},
    {"signals": {"v_type_ok": True, "v_schema_ok": True, "v_range_ok": True, "v_cross_table": None, "v_semantic_ok": True}, "correct": True},
    {"signals": {"v_type_ok": True, "v_schema_ok": True, "v_range_ok": True, "v_cross_table": None, "v_semantic_ok": True}, "correct": True},
    {"signals": {"v_type_ok": True, "v_schema_ok": False, "v_range_ok": True, "v_cross_table": None, "v_semantic_ok": False}, "correct": False},
    {"signals": {"v_type_ok": True, "v_schema_ok": True, "v_range_ok": False, "v_cross_table": None, "v_semantic_ok": False}, "correct": False},
    {"signals": {"v_type_ok": True, "v_schema_ok": True, "v_range_ok": True, "v_cross_table": None, "v_semantic_ok": False}, "correct": False},
    {"signals": {"v_type_ok": True, "v_schema_ok": True, "v_range_ok": True, "v_cross_table": True, "v_semantic_ok": True}, "correct": True},
    {"signals": {"v_type_ok": True, "v_schema_ok": True, "v_range_ok": True, "v_cross_table": None, "v_semantic_ok": True}, "correct": True},
]

DEMO_PATH = "calibration_model_DEMO_ONLY.joblib"  # isolated from the pipeline's model

if __name__ == "__main__":
    metrics = calibration.fit(synthetic_episodes, path=DEMO_PATH)
    print("Результат тренування:", metrics)

    print("\nПеревірка на нових сигналах (ізольована демо-модель, НЕ впливає на pipeline.py):")
    good_signals = {"v_type_ok": True, "v_schema_ok": True, "v_range_ok": True, "v_cross_table": True, "v_semantic_ok": True}
    bad_signals = {"v_type_ok": True, "v_schema_ok": True, "v_range_ok": False, "v_cross_table": None, "v_semantic_ok": False}
    print("  усі сигнали 'добре'  -> φ =", round(calibration.confidence(good_signals, path=DEMO_PATH), 3))
    print("  сигнали 'погано'     -> φ =", round(calibration.confidence(bad_signals, path=DEMO_PATH), 3))
