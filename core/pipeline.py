"""Main VGGR loop: generate code, execute, verify, repair or abstain."""
import ast
import re
import os
import pandas as pd

from core.sandbox import execute
from core.exceptions_taxonomy import (classify_exception, deterministic_repair,
                                     DETERMINISTIC_FIXABLE, TRUNCATION)
from core.verifier import verify
from core.calibration import confidence as confidence_fn
from core.escalation import escalation_policy, L0, LB, ACCEPT, ABSTAIN
from core import lang
from core import scope_gate
from generation import query_classifier
from generation import llm_client
from generation.llm_client import CODEGEN_MAX_TOKENS, call_llm
from core import provenance


# Escalating room for code generation. A cut reply is answered with more
# space, not with another attempt at the same size.
TOKEN_BUDGETS = (CODEGEN_MAX_TOKENS, 2400, 4000)

# Longest row label kept in full once the schema itself is the problem.
COMPACT_LABEL = 28

# Factors a program may introduce without reading them: the percent convention
# and the thousands/millions caption. Any other numeric literal in generated
# code that appears nowhere in the tables is recorded, because the caption is
# supplied as metadata and a number copied out of it would make it data.
UNIT_CONSTANTS = (0.0, 1.0, 2.0, 100.0, 1000.0, 1e6, 1e9)


def foreign_literals(code: str, tables: dict) -> list:
    """Numeric literals in `code` that are in no table and are not unit factors.

    A non-empty list does not mean the answer is wrong; it means the program
    introduced a number from somewhere other than the data, which is exactly
    what the caption option must not be allowed to cause unnoticed.
    """
    import ast as _ast

    try:
        tree = _ast.parse(code or "")
    except SyntaxError:
        return []

    known = set()
    for df in tables.values():
        for col in df.columns[1:]:
            for v in pd.to_numeric(df[col], errors="coerce").dropna().tolist():
                known.add(round(float(v), 6))

    out = []
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.Constant):
            continue
        v = node.value
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        f = round(float(v), 6)
        if f in known or f in UNIT_CONSTANTS or abs(f) < 1e-9:
            continue
        out.append(float(v))
    return sorted(set(out))


# Whether the code generator is shown the table's VALUES or only its labels.
#
# Off by default, so the frozen regressions stay byte-identical and the
# confirmed run (5.26) remains reproducible as it was measured.
#
# Why it exists: measured in 5.30.1, the schema-only prompt gives the generator
# column names and first-column labels and NO numbers, while the direct-answer
# route is handed the whole table. The two routes were therefore compared on
# different information, and the code route had less. 28% of its wrong answers
# had the right magnitude with only the sign or the scale wrong -- exactly what
# a model that cannot see `-14.9` would produce.
VALUES_ENV = "VKSK_CODE_SEES_VALUES"

# A cap, because a long filing table would otherwise eat the generation budget
# it is meant to help. Beyond this the labels alone are shown for the tail.
VALUES_MAX_ROWS = int(os.environ.get("VKSK_CODE_VALUES_MAX_ROWS", "60"))


def _cell_repr(v) -> str:
    """A cell the way the code must address it: quoted when it is text."""
    if pd.isna(v):
        return ""
    if isinstance(v, str):
        return repr(v)
    return str(v)


def code_sees_values() -> bool:
    return os.environ.get(VALUES_ENV, "0").strip() not in ("", "0", "false", "no")


def _schema_description(tables: dict, compact: bool = False,
                        values: bool = None) -> str:
    prof = lang.get()
    values = code_sees_values() if values is None else values
    # Compacting is the last resort against a truncated reply, so it drops the
    # values again rather than fighting the budget it is trying to save.
    if compact:
        values = False
    parts = []
    for name, df in tables.items():
        cols = list(df.columns)
        rows = df.iloc[:, 0].astype(str).tolist() if df.shape[1] else []
        if compact:
            # Last resort: the labels are what make the prompt long, and a
            # long label is also what the model then copies into a string
            # literal and runs out of room inside.
            rows = [r[:COMPACT_LABEL] for r in rows]
        if not values:
            parts.append(prof.SCHEMA_LINE.format(name=name, cols=cols, rows=rows))
            continue
        body = [prof.SCHEMA_VALUES_HEAD.format(name=name, cols=cols)]
        shown = df.head(VALUES_MAX_ROWS)
        for _, row in shown.iterrows():
            # Types, not just text. Serialised as `2002 | 2672.0` the label
            # reads as a number, the model writes `df['line item'] == 2002`
            # and the string index raises KeyError -- measured 33 of them
            # against 9 on the schema-only prompt (defect, 5.34.3). Quoting
            # the strings restores what the schema list used to convey.
            cells = " | ".join(_cell_repr(v) for v in row.tolist())
            body.append(prof.SCHEMA_VALUES_ROW.format(cells=cells))
        if len(df) > len(shown):
            body.append(prof.SCHEMA_VALUES_TRUNCATED.format(n=len(df) - len(shown)))
        parts.append("\n".join(body))
    return "\n".join(parts)


def _build_prompt(query, tables, feedback=None, instructions=None, compact=False,
                  caption=None):
    prof = lang.get()
    prompt = prof.PROMPT_HEAD.format(
        schema=_schema_description(tables, compact=compact), query=query)
    if caption:
        # Description, not data. The wording says so to the model, and
        # `foreign_literals` checks afterwards that no number came out of it.
        prompt += (f'\nTable caption (describes the table; do NOT take any '
                   f'number from it): "{caption}"\n')
    if instructions:
        # Dataset conventions, not method. A corpus states its own rules about
        # units and answer form, and a system evaluated on it has to follow
        # them; leaving them out measures prompt hygiene, not verification.
        prompt += "\n" + instructions + "\n"
    if feedback:
        prompt += prof.PROMPT_FEEDBACK.format(feedback=feedback)
    prompt += prof.PROMPT_TAIL
    return prompt


# Some instruction-tuned models answer a "write code" prompt with their
# reasoning in prose and the code appended, sometimes glued to the last
# sentence without a newline:
#
#   Change = 68.0 - 74.0 = -6.0.vehicles_row = df[df['line item'] == 'vehicles']
#
# The code itself is correct; only the reply around it is not code. Groq's
# gpt-oss never does this, so the salvage is off by default: turning it on
# globally would change what the confirmed runs executed. It is a
# provider-specific adaptation and has to be reported as one, because a model
# that needs it is being helped in a way the others were not.
SALVAGE_ENV = "VKSK_CODE_SALVAGE"

_ASSIGN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\s*=(?!=)")


def salvage_enabled() -> bool:
    return os.environ.get(SALVAGE_ENV, "0").strip() not in ("", "0", "false", "no")


_SALVAGE = {"calls": 0, "salvaged": 0, "failed": 0}


def salvage_reset():
    _SALVAGE.update({"calls": 0, "salvaged": 0, "failed": 0})


def salvage_stats() -> dict:
    return dict(_SALVAGE)


def salvage_code(text: str) -> str:
    """Return the longest tail of `text` that parses as Python and sets `result`.

    Tried only when the reply as a whole does not parse, so a well-formed reply
    is returned byte-identical and no existing behaviour moves.
    """
    src = str(text or "")
    _SALVAGE["calls"] += 1
    try:
        ast.parse(src)
        return src
    except SyntaxError:
        pass
    if not salvage_enabled():
        _SALVAGE["failed"] += 1
        return src
    starts = sorted({m.start() for m in _ASSIGN.finditer(src)})
    for i in starts:
        cand = src[i:]
        try:
            ast.parse(cand)
        except SyntaxError:
            continue
        if "result" in cand:
            _SALVAGE["salvaged"] += 1
            return cand
    _SALVAGE["failed"] += 1
    return src


def generate_code(query, tables, feedback=None, instructions=None,
                  max_tokens=None, compact=False, caption=None):
    """-> (code, truncated). `truncated` comes from the provider, not a guess."""
    code = call_llm(_build_prompt(query, tables, feedback, instructions, compact,
                                  caption),
                    system=lang.get().CODEGEN_SYSTEM, tag="codegen",
                    max_tokens=max_tokens or CODEGEN_MAX_TOKENS)
    return salvage_code(code), llm_client.was_truncated()


def generate_within_budget(query, tables, feedback, instructions, trace, i,
                           caption=None):
    """Generate, giving the model more room while its reply comes back cut.

    The budget grows, then the schema is compacted. Each attempt is recorded:
    it spent a call, and a trace that hid that would understate the cost.
    """
    code = ""
    for attempt, budget in enumerate(TOKEN_BUDGETS):
        compact = attempt == len(TOKEN_BUDGETS) - 1
        code, truncated = generate_code(query, tables, feedback, instructions,
                                        max_tokens=budget, compact=compact,
                                        caption=caption)
        if not truncated:
            return code
        trace.append({"iter": i, "level": LB, "exception_type": TRUNCATION,
                      "error": f"reply cut at max_tokens={budget}",
                      "code": code, "compact_schema": compact})
    return code


def run(query: str, tables: dict, retrieved_names: list = None, max_iter: int = 5,
        threshold: float = 0.75, verbose: bool = True, source_file: str = None,
        use_scope_gate: bool = True, instructions: str = None,
        caption: str = None):
    """Run the correction loop for one query.

    `tables` holds every table in the document. `retrieved_names` is the
    subset chosen by retrieval; None keeps the generator's view unrestricted.

    Generation and verification deliberately see different table sets:
    `scope` narrows the generator's prompt, while the verifier keeps the full
    set so the cross-table signal still has something to compare against.

    Returns {"status", "answer"?, "confidence"?, "trace", "provenance"}.
    """
    trace = []
    scope = {k: tables[k] for k in retrieved_names if k in tables} if retrieved_names else tables
    if not scope:
        scope = tables
    prov_ctx = {"retrieved_names": list(scope.keys()), "all_table_names": list(tables.keys())}

    # The gate inspects the whole document, not just `scope`: whether the data
    # exists at all is a property of the document, not of what retrieval picked.
    gate = scope_gate.check(query, tables, caption) if use_scope_gate else None
    prov_ctx["scope_gate"] = gate

    qtype, qconf = query_classifier.classify_with_confidence(query)
    prov_ctx["query_type"] = qtype
    if verbose and qtype:
        print(f"[М3a] тип запиту: {qtype}" + (f" (впевненість {qconf:.2f})" if qconf else ""))

    if gate is not None and gate["status"] == scope_gate.OUT_OF_SCOPE:
        if verbose:
            print(f"[гейт] OUT_OF_SCOPE — жоден предметний термін не знайдено "
                  f"в схемі документа: {gate['unmatched']}")
        # Empty trace: the refusal happens before the first iteration, so no
        # iterations were spent.
        rec = provenance.build_record(query, "ABSTAIN", trace, source_file=source_file, **prov_ctx)
        return {"status": "ABSTAIN", "abstain_reason": "OUT_OF_SCOPE",
                "trace": trace, "provenance": rec}

    code = generate_within_budget(query, scope, None, instructions, trace, 0,
                                  caption=caption)

    for i in range(max_iter + 1):
        result, err = execute(code, scope)

        if err is not None:
            exc_type = classify_exception(err)
            step = {"iter": i, "exception_type": exc_type, "error": str(err),
                    "code": code}

            if exc_type in DETERMINISTIC_FIXABLE:
                # Repair against `scope` only, so a fix cannot point at a table
                # the sandboxed code cannot see.
                fixed = deterministic_repair(code, err, scope)
                if fixed:
                    step["level"] = L0
                    trace.append(step)
                    code = fixed
                    if verbose:
                        print(f"[i={i}] {exc_type} -> L0 детерміністичний фікс (без LLM)")
                    continue

            level = escalation_policy(None, threshold, exc_type, trace, i, max_iter)
            step["level"] = level
            trace.append(step)
            if level == ABSTAIN:
                if verbose:
                    print(f"[i={i}] {exc_type} -> ABSTAIN (ліміт ітерацій)")
                rec = provenance.build_record(query, "ABSTAIN", trace, source_file=source_file, **prov_ctx)
                return {"status": "ABSTAIN", "abstain_reason": "EXCEPTION",
                        "trace": trace, "provenance": rec}
            if verbose:
                print(f"[i={i}] {exc_type} -> {level} (звернення до LLM)")
            code = generate_within_budget(query, scope, f"{exc_type}: {err}",
                                          instructions, trace, i, caption=caption)
            continue

        signals = verify(code, result, query, tables)
        conf = confidence_fn(signals)
        step = {"iter": i, "result": result, "signals": signals,
                "confidence": conf, "code": code,
                "foreign_literals": foreign_literals(code, scope)}

        level = escalation_policy(conf, threshold, None, trace, i, max_iter)
        step["level"] = level
        trace.append(step)

        if level == ACCEPT:
            if verbose:
                print(f"[i={i}] прийнято, confidence={conf:.2f}")
            rec = provenance.build_record(query, "ACCEPT", trace, answer=result, confidence=conf,
                                          source_file=source_file, **prov_ctx)
            return {"status": "ACCEPT", "answer": result, "confidence": conf,
                    "trace": trace, "provenance": rec}
        if level == ABSTAIN:
            if verbose:
                print(f"[i={i}] ABSTAIN, confidence={conf:.2f} < {threshold}")
            rec = provenance.build_record(query, "ABSTAIN", trace, source_file=source_file, **prov_ctx)
            return {"status": "ABSTAIN", "abstain_reason": "LOW_CONFIDENCE",
                    "trace": trace, "provenance": rec}

        if verbose:
            print(f"[i={i}] confidence={conf:.2f} < {threshold} -> {level}")
        code = generate_within_budget(
            query, scope,
            lang.get().FEEDBACK_LOW_CONFIDENCE.format(result=result, signals=signals),
            instructions, trace, i, caption=caption)

    rec = provenance.build_record(query, "ABSTAIN", trace, source_file=source_file, **prov_ctx)
    return {"status": "ABSTAIN", "abstain_reason": "MAX_ITER",
            "trace": trace, "provenance": rec}
