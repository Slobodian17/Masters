"""Export the exact prompts to docs/thesis_prompts.md. Zero API calls.

The texts are IMPORTED from the modules that send them, never retyped. A
prompt quoted in a thesis and a prompt actually sent must be the same string,
and the only way to guarantee that is to read it from the code.

Run:  python -m eval.export_prompts
"""
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "docs", "thesis_prompts.md")


def fence(text):
    return "```text\n" + str(text).rstrip() + "\n```"


def main():
    os.environ.setdefault("VKSK_LANG", "en")
    from core import lang
    from eval import databench_run as dr
    from eval import finqa_baseline as fb
    from eval import finqa_run as fr
    from eval import finqa_repair as frp
    from generation import llm_client as lc
    prof = lang.get()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    w = []
    a = w.append
    a("# Промпти, використані в роботі\n")
    a("Згенеровано з коду: `python -m eval.export_prompts`. Тексти не "
      "передруковані вручну — вони імпортовані з тих самих модулів, що "
      "надсилають запити, тож документ не може розійтися з прогонами.\n")
    a(f"Моделі: `openai/gpt-oss-120b` (FinQA), "
      f"`gemini-flash-lite-latest` (DataBench). "
      f"Ліміт відповіді генерації коду: {lc.CODEGEN_MAX_TOKENS} токенів, "
      f"решта викликів — {lc.MAX_TOKENS}.\n")
    a("---\n")

    a("## 1. R1 — пряма відповідь (FinQA)\n")
    a("Системний промпт. Варіант обирається змінною `VKSK_ANSWER_MARKER`; "
      "у головних прогонах діяв `marker`, бо без нього gpt-oss витрачав "
      "бюджет на міркування й повертав порожнечу.\n")
    a("**Без маркера (`VKSK_ANSWER_MARKER` не задано):**\n")
    a(fence(fb.SYSTEM_PLAIN) + "\n")
    a("**З маркером (використано в головних прогонах):**\n")
    a(fence(fb.SYSTEM_MARKED) + "\n")
    a("**Суворий варіант (використовувався лише для Gemma):**\n")
    a(fence(fb.SYSTEM_STRICT) + "\n")
    a("Тіло запиту:\n")
    a(fence("Table:\n{таблиця як 'col | col | col' та рядки}\n\n"
            + fr.FINQA_INSTRUCTIONS
            + "\n\nQuestion: {питання}\nAnswer with one number only.") + "\n")
    a("Таблиця серіалізується як рядки, розділені `|`; порожня клітинка — "
      "порожній рядок. У режимі `schema` замість значень подаються лише "
      "`columns:` та `row labels:`.\n")
    a("Підпис таблиці додається **обом** маршрутам або жодному — інакше "
      "порівняння зарахувало б методу інформацію, а не метод:\n")
    a(fence('Table caption (describes the table; do NOT take any number '
            'from it): "{підпис}"') + "\n")

    a("---\n")
    a("## 2. R2 — генерація коду (FinQA)\n")
    a("Системний промпт:\n")
    a(fence(prof.CODEGEN_SYSTEM) + "\n")
    a("Тіло запиту складається з трьох частин:\n")
    a(fence(prof.PROMPT_HEAD) + "\n")
    a("де кожен рядок схеми має вигляд\n")
    a(fence(prof.SCHEMA_LINE) + "\n")
    a("далі — умовності датасету (той самий текст, що й у R1):\n")
    a(fence(fr.FINQA_INSTRUCTIONS) + "\n")
    a("і завершення:\n")
    a(fence(prof.PROMPT_TAIL) + "\n")
    a("**Генератор не бачить значень таблиці за замовчуванням** — лише назви "
      "стовпців і мітки рядків. Режим зі значеннями вмикається окремо "
      "(`VKSK_CODE_SEES_VALUES`) і в головних прогонах був вимкнений.\n")

    a("---\n")
    a("## 3. Ремонт L1–L3 (FinQA)\n")
    a("Ремонт не має власного системного промпта: це той самий виклик "
      "генерації з доданим блоком зворотного звʼязку перед завершенням.\n")
    a(fence(prof.PROMPT_FEEDBACK) + "\n")
    a("Рівень визначає `core/escalation.py` за класом помилки:\n")
    a("| рівень | коли | що змінюється |\n|---|---|---|\n"
      "| **L0** | відома помилка назви стовпця | виправляється локально, "
      "**без виклику моделі** |\n"
      "| **LB** | відповідь обрізано лімітом | більший ліміт або скорочена "
      "схема, той самий промпт |\n"
      "| **L1** | виняток уперше | перегенерація зі `{feedback}` = текст "
      "винятку |\n"
      "| **L2** | той самий виняток повторно, або провал перевірок | "
      "перегенерація зі `{feedback}` = опис сигналу, що впав |\n"
      "| **L3** | ліміт ітерацій вичерпано | відмова або тригер повторного "
      "пошуку |\n")
    a("Межа ітерацій у прогонах FinQA — "
      f"{fr.MAX_ITER}.\n")

    a("---\n")
    a("## 4. Генерація коду (DataBench)\n")
    a("Системний промпт:\n")
    a(fence(dr.SYS_CODE) + "\n")
    a("Тіло запиту:\n")
    a(fence(dr.P_CODE) + "\n")
    a("Ремонт після падіння:\n")
    a(fence(dr.P_REPAIR) + "\n")
    a(f"`{{nhead}}` дорівнює {dr.HEAD_ROWS}: п'ять рядків подаються **лише "
      f"для формату**, а код працює на всій таблиці через `df`.\n")

    a("---\n")
    a("## 5. Пряма відповідь (DataBench)\n")
    a("Системний промпт:\n")
    a(fence(dr.SYS_DIRECT) + "\n")
    a("Тіло запиту:\n")
    a(fence(dr.P_DIRECT) + "\n")
    a(f"`{{window_note}}` називає, скільки рядків віддав пошук і скільки їх у "
      f"таблиці. Вікно обмежене {dr.WINDOW_CHARS} символами, а довгі клітинки "
      f"урізано до {dr.CELL_CHARS}: без цього одне питання на `037_Ted` дає "
      f"промпт на 459 594 токени, що саме по собі перевищує ліміт.\n")

    a("---\n")
    a("## 6. Ремонт формули над знайденими операндами (FinQA, розвідувальне)\n")
    a("Використовувався в досліді про адресний ремонт (розділ 8 RESULTS.md); "
      "у фінальну конфігурацію не ввійшов, бо гілку закрито за арифметикою "
      "+19/−16.\n")
    a("Системний промпт:\n")
    a(fence(frp.SYSTEM) + "\n")
    a("Тіло запиту:\n")
    a(fence(frp.TEMPLATE) + "\n")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(w))
    print("Записано:", OUT)


if __name__ == "__main__":
    main()
