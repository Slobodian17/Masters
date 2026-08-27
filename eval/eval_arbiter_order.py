"""Controlled check: does the ORDER of examples affect the arbiter?

Paraphrases leave the generator grouped by class, and arbitration batches them,
so every batch held a single class. The arbiter never sees gold labels, but a
homogeneous batch nudges it toward one label for all of them, which would make
a low disagreement rate an artefact of ordering rather than a sign of quality.

Same paraphrases, same arbiter, same batch size; only the order changes.
"""
import json
import os

from generation import llm_client
from generation.dataset_paraphrase import DATA_DIR, OUT_PATH, PENDING_PATH, arbitrate

GROUPED_RATE = 0.008  # measured on the grouped order


def load_paraphrases() -> list:
    """All paraphrases from the first run: accepted plus rejected."""
    with open(OUT_PATH, encoding="utf-8") as f:
        dataset = json.load(f)
    with open(PENDING_PATH, encoding="utf-8") as f:
        pending = json.load(f)

    rows = [d for d in dataset if d.get("source") == "llm_paraphrase"]
    rows += [d for d in pending if d.get("source") == "llm_paraphrase"]
    # drop the previous run's verdict
    return [{k: v for k, v in r.items() if k != "arbiter"} for r in rows]


def main():
    if llm_client.active_provider() == "mock":
        print("Потрібен GROQ_API_KEY.")
        return

    rows = load_paraphrases()
    print("=" * 78)
    print("КОНТРОЛЬНА ПЕРЕВІРКА: ЧИ ВПЛИВАЄ ПОРЯДОК НА АРБІТРА")
    print("=" * 78)
    print(f"Перефразувань: {len(rows)}")
    print(f"Згрупований порядок (виміряно раніше): {GROUPED_RATE:.1%} розбіжностей\n")
    print("Перезапуск арбітражу з ПЕРЕМІШАНИМ порядком (та сама вибірка,")
    print("той самий арбітр, той самий розмір батча):\n")

    llm_client.reset_stats()
    accepted, disagreements, failed = arbitrate(rows)
    stats = llm_client.get_stats()

    judged = len(accepted) + len(disagreements)
    rate = len(disagreements) / judged if judged else 0.0
    print()
    print(f'  оцінено успішно: {judged} з {len(rows)}')
    print(f'  прийнято: {len(accepted)}')
    print(f'  розбіжностей: {len(disagreements)} ({rate:.1%} від оцінених)')
    print(f'  ЗБОЇВ виклику (виключено з частки): {len(failed)}')
    print(f"  викликів: {stats['calls']}, ретраїв: {stats['retries']}")
    if len(failed) > len(rows) * 0.2:
        print()
        print('  УВАГА: збоїв понад 20% — результат НЕНАДІЙНИЙ, не інтерпретувати.')

    print("\n" + "-" * 78)
    print("ВИСНОВОК")
    print("-" * 78)
    print(f"  згрупований порядок:  {GROUPED_RATE:.1%}")
    print(f"  перемішаний порядок:  {rate:.1%}")

    ratio = (rate / GROUPED_RATE) if GROUPED_RATE else float("inf")
    if rate >= GROUPED_RATE * 2:
        print(f"\n  ПІДОЗРА ПІДТВЕРДИЛАСЬ: розбіжностей у {ratio:.1f}x більше.")
        print("  Згрупований батч штучно завищував згоду арбітра —")
        print("  результат першого прогону недійсний, чинним є перемішаний.")
    elif rate <= GROUPED_RATE * 0.5:
        print("\n  Зворотний ефект: перемішування ЗМЕНШИЛО розбіжності.")
        print("  Потребує окремого пояснення, перш ніж приймати будь-яке число.")
    else:
        print("\n  Підозра НЕ підтвердилась: частки одного порядку.")
        print("  Низька розбіжність відображає якість перефразувань,")
        print("  а не артефакт групування.")

    out = os.path.join(DATA_DIR, "arbiter_order_check.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"n": len(rows), "grouped_rate": GROUPED_RATE,
                   "shuffled_rate": rate,
                   "n_judged": judged, "n_failed": len(failed),
                   "disagreements": disagreements}, f, ensure_ascii=False, indent=1)
    print(f"\nЗбережено: {out}")
    return rate


if __name__ == "__main__":
    main()
