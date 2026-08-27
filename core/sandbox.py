"""Restricted execution wrapper for generated pandas code.

This is demo-level isolation (restricted exec plus a module allowlist), not
a production sandbox; docker or a resource-limited subprocess is the intended
replacement.

Note: `__import__` is restricted rather than removed. pandas lazily imports
accelerators inside numeric operations such as `.mean()`, so removing it
entirely breaks legitimate calls with an opaque KeyError.
"""
import builtins

import numpy as np
import pandas as pd

ALLOWED_MODULES = {"pandas", "numpy", "math", "statistics", "itertools", "collections", "re", "decimal"}
BLOCKED_NAMES = {"open", "exec", "eval", "compile", "input", "breakpoint", "__loader__"}


def _restricted_import(name, *args, **kwargs):
    top_level = name.split(".")[0]
    if top_level not in ALLOWED_MODULES:
        raise ImportError(f"Імпорт '{name}' заборонено у sandbox (дозволено лише: {sorted(ALLOWED_MODULES)})")
    return builtins.__import__(name, *args, **kwargs)


def _make_safe_builtins():
    safe = {k: v for k, v in vars(builtins).items() if k not in BLOCKED_NAMES}
    safe["__import__"] = _restricted_import
    return safe


SAFE_BUILTINS = _make_safe_builtins()


def execute(code: str, tables: dict):
    """Execute code in a restricted namespace.

    Each table is exposed under its own name; the first one is also aliased
    as `df`. The code must assign its answer to `result`.
    Returns (result, None) on success or (None, Exception) on failure.
    """
    local_ns = {**tables, "pd": pd, "np": np}
    if tables:
        local_ns.setdefault("df", next(iter(tables.values())))
    global_ns = {"__builtins__": SAFE_BUILTINS}
    try:
        exec(code, global_ns, local_ns)
    except Exception as e:  # broad on purpose: exceptions drive the repair loop
        return None, e
    if "result" not in local_ns:
        return None, NameError("Код не визначив змінну `result`")
    return local_ns["result"], None
