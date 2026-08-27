"""Provenance tracking: a structured audit trail for every answer."""
import json


def build_record(query: str, status: str, trace: list, answer=None, confidence=None,
                 source_file: str = None, retrieved_names=None, all_table_names=None,
                 scope_gate=None, query_type=None):
    """Build the record.

    `retrieved_names` and `all_table_names` capture the retrieval decision, so
    an auditor sees not only how a value was computed but from which tables.
    """
    execution_steps = [
        {k: v for k, v in step.items() if k not in ("signals",)}
        for step in trace
    ]
    record = {
        "query": query,
        "status": status,
        "answer": answer,
        "confidence": confidence,
        "source": {
            "file": source_file,
            "tables_available": all_table_names,
            "tables_retrieved": retrieved_names,
        },
        # Pre-generation decisions: scope gate and query type.
        "pre_generation": {"scope_gate": scope_gate, "query_type": query_type},
        "execution": {
            "iterations": len(trace),
            "steps": execution_steps,
            "final_signals": trace[-1].get("signals") if trace else None,
        },
    }
    return record


def to_json(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False, indent=2, default=str)
