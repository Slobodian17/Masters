"""Main VGGR loop: generate code, execute, verify, repair or abstain."""
from core.sandbox import execute
from core.exceptions_taxonomy import classify_exception, deterministic_repair, DETERMINISTIC_FIXABLE
from core.verifier import verify
from core.calibration import confidence as confidence_fn
from core.escalation import escalation_policy, L0, ACCEPT, ABSTAIN
from core import scope_gate
from generation import query_classifier
from generation.llm_client import call_llm
from core import provenance

CODE_GEN_SYSTEM = (
    "Ти — аналітик, що пише короткий Python/Pandas код. Таблиці доступні "
    "під іменами, вказаними у схемі нижче (перша таблиця також доступна як "
    "`df`). Завжди записуй фінальну відповідь у змінну `result`. "
    "Виводь ЛИШЕ код, без пояснень і без markdown-огорож."
)


def _schema_description(tables: dict) -> str:
    parts = []
    for name, df in tables.items():
        cols = list(df.columns)
        rows = df.iloc[:, 0].astype(str).tolist() if df.shape[1] else []
        parts.append(f"- {name}: стовпці={cols}; значення першого стовпця (показники)={rows}")
    return "\n".join(parts)


def _build_prompt(query, tables, feedback=None):
    prompt = f"Схема таблиць:\n{_schema_description(tables)}\n\nЗапит: {query}\n"
    if feedback:
        prompt += f"\nПопередня спроба мала проблему: {feedback}\nВрахуй це і напиши виправлений код.\n"
    prompt += "Напиши Pandas-код."
    return prompt


def generate_code(query, tables, feedback=None):
    return call_llm(_build_prompt(query, tables, feedback), system=CODE_GEN_SYSTEM)


def run(query: str, tables: dict, retrieved_names: list = None, max_iter: int = 5,
        threshold: float = 0.75, verbose: bool = True, source_file: str = None,
        use_scope_gate: bool = True):
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
    gate = scope_gate.check(query, tables) if use_scope_gate else None
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

    code = generate_code(query, scope)

    for i in range(max_iter + 1):
        result, err = execute(code, scope)

        if err is not None:
            exc_type = classify_exception(err)
            step = {"iter": i, "exception_type": exc_type, "error": str(err)}

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
                return {"status": "ABSTAIN", "trace": trace, "provenance": rec}
            if verbose:
                print(f"[i={i}] {exc_type} -> {level} (звернення до LLM)")
            code = generate_code(query, scope, feedback=f"{exc_type}: {err}")
            continue

        signals = verify(code, result, query, tables)
        conf = confidence_fn(signals)
        step = {"iter": i, "result": result, "signals": signals, "confidence": conf}

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
            return {"status": "ABSTAIN", "trace": trace, "provenance": rec}

        if verbose:
            print(f"[i={i}] confidence={conf:.2f} < {threshold} -> {level}")
        code = generate_code(
            query, scope,
            feedback=f"Результат {result!r} має низьку довіру (сигнали: {signals}). Перевір формулу й одиниці виміру.",
        )

    rec = provenance.build_record(query, "ABSTAIN", trace, source_file=source_file, **prov_ctx)
    return {"status": "ABSTAIN", "trace": trace, "provenance": rec}
