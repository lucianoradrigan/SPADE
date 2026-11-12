"""M6.1 (2026-11-12): coherence check of the 18 pages in a real browser, read-only.
Per page: open, run its primary action if it has one, then collect the visible text and the labels of numeric
controls (sliders, number inputs). Checks: SIMULATED provenance label, AI SUGGESTION label where a prediction
is shown, a unit on numeric controls, no Spanish, no 'live' / 'production ready' / 'hardware verified'.
Screenshots at desktop (1440 px) and narrow (390 px) widths."""
import json, re, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE, OUT = sys.argv[1], Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, "src")
from driveflow.viz.app import PAGES  # noqa: E402

FORBIDDEN = {"live": r"\blive\b", "production ready": r"production[- ]ready", "hardware verified": r"hardware[- ]verified"}
SPANISH = r"\b(el|los|las|una|para|con|sin|simulación|motor de|velocidad|corriente|resultado[s]?|configuración|ejecutar|datos)\b"
UNIT = re.compile(r"\((?:[^)]*[A-Za-zΩ°%µ·/])[^)]*\)|\b(?:rad/s|N·m|Nm|V|A|Hz|kHz|s|ms|%|Ω|H|Wb|kg|W|F)\b")
DIMENSIONLESS = re.compile(r"seed|steps|density|count|number of|pole pairs|epochs|batch|samples|windows|runs|order|index|top-?k|cap|severity \(module|trials|particles|iterations|episodes|n_|population", re.I)

def settle(page, timeout=240):
    t0 = time.time(); page.wait_for_selector("[data-testid='stApp']", timeout=60000); page.wait_for_timeout(1000); quiet = 0.0
    while time.time() - t0 < timeout:
        busy = page.locator("[data-testid='stStatusWidget']").count() > 0 or page.locator("[data-testid='stSpinner']").count() > 0
        quiet = 0.0 if busy else quiet + 0.5
        if quiet >= 2.0: break
        page.wait_for_timeout(500)

res = []
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
    wide = b.new_context(viewport={"width": 1440, "height": 1600}, accept_downloads=True).new_page()
    narrow = b.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=1).new_page()
    wide.goto(BASE + "/", wait_until="domcontentloaded"); settle(wide)
    i = 0
    for section, items in PAGES.items():
        for file, label, _ in items:
            i += 1; name = Path(file).stem; url = f"{BASE}/" + ("" if name == "overview" else name)
            wide.goto(url, wait_until="domcontentloaded"); settle(wide)
            primary = [x for x in wide.locator("[data-testid='stBaseButton-primary']").all() if x.is_visible() and x.is_enabled()]
            action = None
            if primary and name != "assistant":
                action = primary[0].inner_text().strip().split("\n")[-1]; primary[0].click(); settle(wide, 600)
            # open every expander so their controls are inventoried too
            for e in wide.locator("[data-testid='stExpander'] summary").all():
                try:
                    if e.is_visible(): e.click(); wide.wait_for_timeout(150)
                except Exception: pass
            main = wide.locator("[data-testid='stMain']").first
            text = main.inner_text()
            labels = []
            for sel in ("stNumberInput", "stSlider"):
                for w in wide.locator(f"[data-testid='{sel}']").all():
                    lab = w.locator("[data-testid='stWidgetLabel']")
                    if lab.count(): labels.append({"kind": sel, "label": lab.first.inner_text().strip()})
            no_unit = [l["label"] for l in labels if not UNIT.search(l["label"]) and not DIMENSIONLESS.search(l["label"])]
            rec = {"n": i, "section": section, "page": label, "file": file, "action": action,
                   "simulated_label": bool(re.search(r"SIMULATED", text)), "ai_suggestion_label": "AI SUGGESTION — REQUIRES VERIFICATION" in text,
                   "shows_prediction": bool(re.search(r"\b(predicted|prediction|classifier says|suggested class|confidence)\b", text, re.I)),
                   "forbidden": {k: re.findall(r".{0,40}" + v + r".{0,40}", text, re.I)[:3] for k, v in FORBIDDEN.items() if re.search(v, text, re.I)},
                   "spanish_hits": sorted(set(m.lower() for m in re.findall(SPANISH, text, re.I)))[:10],
                   "n_numeric_controls": len(labels), "numeric_without_unit": no_unit,
                   "exceptions": [e.inner_text()[:200] for e in wide.locator("[data-testid='stException']").all()]}
            wide.evaluate("() => { const m = document.querySelector(\"[data-testid='stMain']\"); if (m) m.scrollTo(0, 0); window.scrollTo(0,0); }")
            wide.screenshot(path=str(OUT / f"desktop_{i:02d}_{name}.png"), full_page=False)
            narrow.goto(url, wait_until="domcontentloaded"); settle(narrow)
            rec["narrow_horizontal_scroll"] = narrow.evaluate("() => document.documentElement.scrollWidth > window.innerWidth + 1")
            narrow.screenshot(path=str(OUT / f"narrow_{i:02d}_{name}.png"), full_page=False)
            res.append(rec)
            print(f"{i:02d} {label:20s} SIM={rec['simulated_label']!s:5} AI={rec['ai_suggestion_label']!s:5} pred={rec['shows_prediction']!s:5} forb={list(rec['forbidden'])} es={rec['spanish_hits']} nounit={no_unit[:4]} hscroll={rec['narrow_horizontal_scroll']} exc={len(rec['exceptions'])}", flush=True)
    b.close()
(OUT / "inventory_pages.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
