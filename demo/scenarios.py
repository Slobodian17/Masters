"""The five demo scenarios, and the rule that picked each one.

Every scenario is a REAL question from the confirmatory run (5.26), taken from
`confirm300` and `baseline_table_confirm300` by a stated rule rather than by
eye. The rules are here so the selection is checkable: a demo that shows only
what flatters the method is an advertisement, not a defence.

Scenario 5 is the one that matters most: both routes agreeing on a wrong
answer is exactly what agreement cannot catch, and it is shown deliberately.
"""

DEMO_MODEL = "openai/gpt-oss-120b (Groq)"
DEMO_SOURCE = "кеш confirm300 + baseline_table_confirm300 (прогін 5.26, n = 300)"

SCENARIOS = [
    {
        "key": "1",
        "id": "AMAT/2018/page_33.pdf-2",
        "title": "Маршрути погодились, відповідь правильна",
        "rule": "перший за порядком кеша випадок: ACCEPT за правилом згоди, "
                "відповідь правильна, таблиця не більша за 6 рядків",
        "takeaway": "Типовий робочий випадок: два незалежні прочитання дали "
                    "те саме число, і воно збіглося з еталоном.",
    },
    {
        "key": "2",
        "id": "JKHY/2017/page_26.pdf-2",
        "title": "Маршрути розійшлись — система відмовляється",
        "rule": "перший випадок ABSTAIN за розбіжністю, де код дав число і "
                "рівно один із маршрутів правий",
        "takeaway": "Розбіжність спрацювала як детектор: прямий маршрут правий, "
                    "код помилився з базою відношення. Система не видає числа, "
                    "а просить перевірити.",
    },
    {
        "key": "3",
        "id": "BLL/2006/page_67.pdf-4",
        "title": "Ітеративне виправлення коду після винятку",
        "rule": "перший із 8 випадків, де ПЕРША ітерація впала з винятком, "
                "фінальна відповідь правильна, рівно 2 кроки (падіння -> "
                "виправлення) і таблиця <= 8 рядків",
        "takeaway": "Саме цей механізм дає +10 питань зі 300. Перша спроба "
                    "звертається до мітки, якої в таблиці немає, і падає з "
                    "KeyError; друга бере правильний рядок.",
    },
    {
        "key": "4a",
        "id": "BLL/2006/page_67.pdf-4",
        "title": "Гейт у своєму призначенні: даних немає в таблиці",
        "live_gate": "what was the ceo total compensation in 2006?",
        "rule": "та сама таблиця, що в сценарії 3, і питання про величину, "
                "якої в ній немає; гейт детермінований, тому рахується ЖИВЦЕМ",
        "takeaway": "Жоден токен питання не знаходить відповідника в схемі, "
                    "тож код не запускається взагалі. Це і є призначення "
                    "гейта — не витрачати виклик на питання не за адресою.",
    },
    {
        "key": "4b",
        "id": "AES/2001/page_85.pdf-3",
        "title": "Гейт помиляється: ХИБНА відмова",
        "rule": "перший випадок abstain_reason = OUT_OF_SCOPE у кеші",
        "takeaway": "Питання нормальне й про цю саму таблицю, але гейт "
                    "відмовив, а пряма відповідь на нього правильна. "
                    "Зміряно (5.34.2): 23 такі відмови з 300. Через це гейт "
                    "для FinQA ВИМКНЕНО у кроці гігієни.",
    },
    {
        "key": "5",
        "id": "CME/2010/page_69.pdf-4",
        "title": "ОБМЕЖЕННЯ: обидва маршрути погодились на хибному",
        "rule": "випадок із таксономії спільних хибних, категорія BASE "
                "(хибний знаменник), таблиця не більша за 8 рядків",
        "takeaway": "Згода тут безсила: обидва прочитання дали те саме хибне "
                    "число, вибирати нема з чого. Таких 53 зі 300, і жоден "
                    "арбітраж їх не виправить.",
    },
]

# TAT-DQA page for the extraction demo: a page where extraction succeeded and
# the answer came out right, so the first screen shows the chain working.
TATDQA_PAGE = "8c4cc201460e6703dc58f5d07900f34b"
TATDQA_NOTE = ("TAT-DQA (Zhu та ін., 2022), ліцензія CC-BY-4.0. "
               "Сторінка реального фінансового звіту.")
