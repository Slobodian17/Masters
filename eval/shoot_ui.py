"""Screenshots of the demo UI for the thesis. Zero API calls.

The UI is replay-only, so a screenshot is reproducible: the same cache gives
the same screen. Each shot is taken on a question chosen by a stated rule
(`demo/databench_scenarios.py`), not by whichever one looked best.

Run:  python -m eval.shoot_ui
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "eval", "figures", "thesis")
PORT = 8599
BASE = f"http://localhost:{PORT}"

SHOTS = [
    ("fig3_7_ui_demo.png", f"{BASE}/?page=DataBench%3A+%D0%BF%D0%BE%D1%88%D1%83"
                           f"%D0%BA+%D1%83+%D0%BA%D0%BE%D1%80%D0%BF%D1%83%D1%81"
                           f"%D1%96&case=1", 2400),
    ("figG_1_ui_main.png", f"{BASE}/?page=DataBench%3A+%D0%BF%D0%BE%D1%88%D1%83"
                           f"%D0%BA+%D1%83+%D0%BA%D0%BE%D1%80%D0%BF%D1%83%D1%81"
                           f"%D1%96&case=2", 2400),
    ("figG_2_ui_steps.png", f"{BASE}/?page=DataBench%3A+%D0%BF%D0%BE%D1%88%D1%83"
                            f"%D0%BA+%D1%83+%D0%BA%D0%BE%D1%80%D0%BF%D1%83%D1%81"
                            f"%D1%96&case=3", 2400),
]


def crop_blank(path, keep=40):
    """Trim the uniform tail Streamlit leaves below the content.

    A tall viewport is the only way to capture a long Streamlit page -- its
    scroll container ignores `full_page` -- so the shot always ends in dead
    space. Cutting it here beats shipping a figure that is half margin.
    """
    from PIL import Image
    im = Image.open(path).convert("RGB")
    w, h = im.size
    px = im.load()
    # the content column starts right of the sidebar; sample a few x positions
    xs = [int(w * f) for f in (0.35, 0.5, 0.65, 0.8)]
    last = 0
    for y in range(h - 1, -1, -1):
        row = [px[x, y] for x in xs]
        if any(abs(c[0] - 255) + abs(c[1] - 255) + abs(c[2] - 255) > 18
               for c in row):
            last = y
            break
    bottom = min(h, last + keep)
    if bottom < h - 10:
        im.crop((0, 0, w, bottom)).save(path)
    return bottom


def main():
    env = dict(os.environ, PYTHONUTF8="1", VKSK_LANG="en")
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.port", str(PORT), "--server.headless", "true",
         "--browser.gatherUsageStats", "false"],
        cwd=HERE, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        from playwright.sync_api import sync_playwright
        os.makedirs(OUT, exist_ok=True)
        with sync_playwright() as pw:
            br = pw.chromium.launch()
            pg = br.new_page(viewport={"width": 1480, "height": 1000},
                             device_scale_factor=2)
            # Streamlit needs a moment to bind the port; retry rather than
            # guess a sleep long enough for the slowest case.
            for attempt in range(40):
                try:
                    pg.goto(BASE, timeout=4000)
                    break
                except Exception:
                    time.sleep(1.5)
            for name, url, height in SHOTS:
                pg.set_viewport_size({"width": 1480, "height": height})
                pg.goto(url, timeout=60000)
                # the page renders tables and re-executes cached code
                pg.wait_for_timeout(9000)
                try:
                    pg.wait_for_selector("text=Крок 5", timeout=20000)
                except Exception:
                    pass
                path = os.path.join(OUT, name)
                pg.screenshot(path=path, full_page=True)
                h = crop_blank(path)
                print(f"  {name}  (обрізано до {h} px)")
            br.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
    print("Знімки у", OUT)


if __name__ == "__main__":
    main()
