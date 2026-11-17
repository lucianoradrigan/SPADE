"""Release gates G1-G8 for SPADE (B1, 2026-11-16; lesson of AI-N39 and AI-N40: no gate may depend on remembering something).

Runs the gates of docs/RELEASING.md and writes docs/audit/release/<label>_gates.{md,json} with the commit, the time, the
command and the result of each gate. A gate that cannot run here is reported as "manual" (with the reason), never as
"pass". Everything SPADE produces is simulated; the gates check the software, not a machine.

    python scripts/release_gates.py --label v0.2.3                 # every gate
    python scripts/release_gates.py --label v0.2.3 --only G7,G8     # some gates
    python scripts/release_gates.py --label v0.2.3 --browser-python /path/to/python-with-playwright

Gates:
  G1  full suite (pytest, slow included): exit 0
  G2  scripts/smoke_test.py --launch: exit 0
  G3  CI (tests.yml) on the commit: every job "success" (gh CLI)
  G4  fresh clone of the pushed branch (no data/, no workspace/, Ollama unreachable): smoke --launch, then every page in
      a real browser (Playwright) without an exception
  G5  assistant with the real model: pytest --run-llm -m llm, then one real turn in a browser (Playwright, Ollama)
  G6  AI Diagnostics: every selectable classifier at speeds inside and outside its trained envelope (AppTest; the
      evaluated run must be the selected one), plus the default sample in a real browser (Playwright)
  G7  byte-for-byte regression tests unchanged since the previous tag
  G8  no absolute home paths in tracked code, except the documented ones

Standard library only. Playwright is not a SPADE dependency: the browser parts run with --browser-python (an interpreter
that can import playwright; default: this one) and are "manual" when it cannot.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATES = ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8")
#: G7: the byte-for-byte regression tests and their reference data (CLAUDE.md: not modified)
BYTE_FOR_BYTE = ("tests/test_datagen.py", "tests/test_scml_system_regression.py", "tests/test_patch3_regression.py",
                 "tests/test_r2_sensitivity_regression.py", "tests/data")
#: G8: tracked files allowed to contain '/home/' (each one is a documented, benign use)
HOME_PATH_ALLOWED = {
    "scripts/fmu/om.sh": "documents where OpenModelica is found on the author's machine (comment)",
    "tests/runs/test_store.py": "asserts that a home path is rejected",
    "tests/compat/test_smoke.py": "asserts that no home path leaks into the output",
    "scripts/release_gates.py": "this list and the G8 pattern",
}
G6_SPEEDS = (80.0, 150.0, 220.0, 300.0, 350.0)
OUTSIDE = "Outside the trained range"
CHROME = "/usr/bin/google-chrome"


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sh(cmd, cwd=ROOT, env=None, timeout=None) -> tuple[int, str]:
    """Runs a command and returns (exit code, combined output). A timeout is exit code 124."""
    try:
        p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
                           shell=isinstance(cmd, str))
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired as exc:
        return 124, f"timeout after {timeout} s: {exc}"
    except OSError as exc:  # the program does not exist or cannot run
        return 127, f"cannot run {cmd!r}: {exc}"


def gate(name, status, command, summary, **extra) -> dict:
    assert status in ("pass", "fail", "manual"), status
    return {"gate": name, "status": status, "command": command, "summary": summary, "time": now(), **extra}


def last_line(text: str) -> str:
    lines = [l for l in text.strip().splitlines() if l.strip()]
    return lines[-1][:300] if lines else ""


def py_env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(["src", "."])
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    return env


def can_import_playwright(python: str) -> bool:
    code, _ = sh([python, "-c", "import playwright.sync_api"], timeout=60)
    return code == 0


def wait_health(url: str, seconds: int = 120) -> bool:
    t0 = time.time()
    while time.time() - t0 < seconds:
        try:
            with urllib.request.urlopen(url + "/_stcore/health", timeout=5) as r:
                if b"ok" in r.read():
                    return True
        except OSError:
            pass
        time.sleep(1)
    return False


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_app(python: str, cwd: Path, port: int, env: dict, log: Path) -> subprocess.Popen:
    fh = open(log, "w", encoding="utf-8")
    return subprocess.Popen([python, "-m", "streamlit", "run", "src/driveflow/viz/app.py", "--server.headless", "true", "--server.port", str(port),
                             "--server.fileWatcherType", "none"], cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT)


def stop_app(proc: subprocess.Popen) -> None:
    proc.terminate()  # our own child only (never pkill -f)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


# ----------------------------------------------------------------------------------------------------------- gates
def g1(args) -> dict:
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rx"]
    code, out = sh(cmd, env=py_env(), timeout=4 * 3600)
    (args.work / "g1_suite.txt").write_text(out, encoding="utf-8")
    summary = last_line(out)
    new_xfail_without_id = [l for l in out.splitlines() if l.startswith("XFAIL") and not re.search(r"\b(?:S|AI)-N?\d+", l)]
    ok = code == 0 and " failed" not in summary and not new_xfail_without_id
    return gate("G1", "pass" if ok else "fail", " ".join(cmd[1:]), summary, exit_code=code, xfail_without_finding=new_xfail_without_id)


def g2(args) -> dict:
    cmd = [sys.executable, "scripts/smoke_test.py", "--launch"]
    code, out = sh(cmd, env=py_env(), timeout=900)
    (args.work / "g2_smoke.txt").write_text(out, encoding="utf-8")
    return gate("G2", "pass" if code == 0 else "fail", " ".join(cmd[1:]), last_line(out), exit_code=code)


def g3(args) -> dict:
    if not shutil.which("gh"):
        return gate("G3", "manual", "gh run list", "gh CLI not available: check tests.yml on the commit by hand")
    sha = args.commit
    cmd = ["gh", "run", "list", "--workflow", "tests.yml", "--commit", sha, "--limit", "5", "--json", "databaseId,status,conclusion"]
    code, out = sh(cmd, timeout=120)
    if code != 0:
        return gate("G3", "manual", " ".join(cmd), f"gh failed: {last_line(out)}")
    runs = json.loads(out or "[]")
    done = [r for r in runs if r["status"] == "completed"]
    if not done:
        return gate("G3", "manual", " ".join(cmd), f"no completed tests.yml run for {sha[:7]} yet (push, then re-run --only G3)", runs=runs)
    run = done[0]
    code, out = sh(["gh", "run", "view", str(run["databaseId"]), "--json", "jobs"], timeout=120)
    jobs = {j["name"]: j["conclusion"] for j in json.loads(out).get("jobs", [])} if code == 0 else {}
    ok = run["conclusion"] == "success" and jobs and all(c == "success" for c in jobs.values())
    return gate("G3", "pass" if ok else "fail", " ".join(cmd), f"run {run['databaseId']}: {sum(c == 'success' for c in jobs.values())}/{len(jobs)} jobs success",
                run=run["databaseId"], jobs=jobs)


def g4(args) -> dict:
    if not can_import_playwright(args.browser_python):
        return gate("G4", "manual", "fresh clone + pages in a browser", f"{args.browser_python} cannot import playwright")
    if not shutil.which("uv"):
        return gate("G4", "manual", "fresh clone + pages in a browser", "uv not available to build the clone's environment")
    _, url = sh(["git", "remote", "get-url", "origin"])
    url = url.strip()
    clone = Path(tempfile.mkdtemp(prefix="spade_g4_"))
    log = []
    steps = [(["git", "clone", "--quiet", "--branch", args.branch, "--single-branch", url, str(clone / "repo")], ROOT, 900)]
    for cmd, cwd, t in steps:
        code, out = sh(cmd, cwd=cwd, timeout=t)
        log.append(out)
        if code != 0:
            return gate("G4", "fail", " ".join(cmd), f"clone failed: {last_line(out)}")
    repo = clone / "repo"
    _, head = sh(["git", "rev-parse", "HEAD"], cwd=repo)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "SPADE_WORKSPACE", "VIRTUAL_ENV")}
    env.update(OLLAMA_HOST="127.0.0.1:9", TF_CPP_MIN_LOG_LEVEL="2")
    for cmd in (["uv", "venv", "--quiet", "--python", f"{sys.version_info.major}.{sys.version_info.minor}", ".venv"],
                ["uv", "pip", "install", "--quiet", "--python", ".venv/bin/python", "-e", ".[dev,viz]"]):
        code, out = sh(cmd, cwd=repo, env=env, timeout=1800)
        if code != 0:
            return gate("G4", "fail", " ".join(cmd), f"environment failed: {last_line(out)}")
    py = str(repo / ".venv" / "bin" / "python")
    fresh = {"commit": head.strip()[:7], "data_absent": not (repo / "data").exists(), "workspace_absent": not (repo / "workspace").exists()}
    code, out = sh([py, "scripts/smoke_test.py", "--launch"], cwd=repo, env=env, timeout=900)
    fresh["smoke"] = last_line(out)
    if code != 0:
        return gate("G4", "fail", "smoke_test.py --launch (fresh clone)", f"smoke failed: {fresh['smoke']}", fresh=fresh)
    port = free_port()
    app = start_app(py, repo, port, env, args.work / "g4_app.log")
    try:
        if not wait_health(f"http://127.0.0.1:{port}"):
            return gate("G4", "fail", "streamlit run (fresh clone)", "the app did not answer its health endpoint", fresh=fresh)
        pages_json = args.work / "g4_pages.json"
        code, out = sh([args.browser_python, str(Path(__file__).resolve()), "--browser-pages", f"http://127.0.0.1:{port}", str(pages_json), str(repo)],
                       env=env, timeout=3600)
    finally:
        stop_app(app)
    shutil.rmtree(clone, ignore_errors=True)
    if code != 0 or not pages_json.exists():
        return gate("G4", "fail", "browser pages (fresh clone)", f"page check failed: {last_line(out)}", fresh=fresh)
    pages = json.loads(pages_json.read_text(encoding="utf-8"))
    bad = [p["page"] for p in pages if p["exceptions"] or p["missing"]]
    return gate("G4", "pass" if not bad and fresh["data_absent"] else "fail", "fresh clone; smoke --launch; every page in Chrome (Playwright)",
                f"commit {fresh['commit']}: smoke ok; {len(pages) - len(bad)}/{len(pages)} pages without exception", fresh=fresh, failing_pages=bad)


def g5(args) -> dict:
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--run-llm", "-m", "llm", "tests/"]
    code, out = sh(cmd, env=py_env(), timeout=1800)
    (args.work / "g5_llm.txt").write_text(out, encoding="utf-8")
    llm = last_line(out)
    if "no tests ran" in llm or "skipped" in llm and "passed" not in llm:
        return gate("G5", "manual", " ".join(cmd[1:]), f"the LLM tests did not run here ({llm}): is Ollama reachable?")
    if code != 0:
        return gate("G5", "fail", " ".join(cmd[1:]), llm)
    if not can_import_playwright(args.browser_python):
        return gate("G5", "manual", " ".join(cmd[1:]) + " + browser turn", f"LLM tests: {llm}; browser turn not run: {args.browser_python} cannot import playwright")
    port = free_port()
    app = start_app(sys.executable, ROOT, port, py_env(), args.work / "g5_app.log")
    try:
        if not wait_health(f"http://127.0.0.1:{port}"):
            return gate("G5", "fail", "streamlit run", "the app did not answer its health endpoint")
        code, out = sh([args.browser_python, str(Path(__file__).resolve()), "--browser-turn", f"http://127.0.0.1:{port}", str(args.work / "g5_browser_turn.png")], timeout=600)
    finally:
        stop_app(app)
    turn = last_line(out)
    m = re.search(r"messages=(\d+) after (\d+) s", turn)
    ok = code == 0 and m and int(m.group(1)) >= 3
    return gate("G5", "pass" if ok else "fail", " ".join(cmd[1:]) + " + one real turn in Chrome (Playwright)", f"LLM tests: {llm}; browser turn: {turn}")


G6_APPTEST = r'''
import json, re, sys
from pathlib import Path
from streamlit.testing.v1 import AppTest
from driveflow.ai import envelope
from driveflow.ai.custom_training import list_classifier_runs
from driveflow.ai.registry import resolve
speeds = [float(x) for x in sys.argv[2].split(",")]
promoted = resolve("dc_motor", "pc", "classifier").resolve()
runs = [None] + [str(r["run_dir"]) for r in list_classifier_runs("dc_motor") if Path(r["run_dir"]).resolve() != promoted]
out = []
for run in runs:
    rdir = promoted if run is None else Path(run)
    for w in speeds:
        at = AppTest.from_file("src/driveflow/viz/app.py", default_timeout=300); at.run(); at.switch_page("pages/ai_diagnostics.py").run()
        if run is not None:
            [s for s in at.selectbox if s.label == "Classifier"][0].set_value(run).run()
        at.slider(key="ia_omega_ref_slider").set_value(w).run()
        [b for b in at.button if "Generate" in b.label][0].click().run()
        html = " ".join(str(getattr(h, "body", "")) for h in at.get("html"))
        boxes = [e.value for e in list(at.error) + list(at.warning)]
        out.append({"run": "/".join(rdir.parts[-2:]), "promoted": run is None, "speed": w, "expected": envelope.check(rdir, w)["status"],
                    "evaluated_is_selected": "/".join(rdir.parts[-2:]) in html, "outside_box": any("Outside the trained range" in b and "no diagnosis" in b for b in boxes),
                    "shown": [str(m.value) for m in at.metric][:2], "exception": bool(at.exception)})
        print(json.dumps(out[-1]), flush=True)
Path(sys.argv[1]).write_text(json.dumps(out, indent=1), encoding="utf-8")
'''


def g6_ok(row: dict) -> bool:
    """inside: a prediction, healthy -> normal or abstains, no box; outside: the box and no prediction; always the selected run."""
    if row["exception"] or not row["evaluated_is_selected"]:
        return False
    if row["expected"] == "inside":
        return not row["outside_box"] and bool(row["shown"]) and any(v in ("normal", "No class suggested") for v in row["shown"])
    if row["expected"] == "outside":
        return row["outside_box"] and not row["shown"]
    return False  # unknown envelope: every selectable run must declare one (D3)


def g6(args) -> dict:
    script = args.work / "g6_apptest.py"
    script.write_text(G6_APPTEST, encoding="utf-8")
    rows_file = args.work / "g6_sweep.json"
    cmd = [sys.executable, str(script), str(rows_file), ",".join(str(s) for s in G6_SPEEDS)]
    code, out = sh(cmd, env=py_env(), timeout=4 * 3600)
    if code != 0 or not rows_file.exists():
        return gate("G6", "fail", "AppTest sweep", f"sweep failed: {last_line(out)}")
    rows = json.loads(rows_file.read_text(encoding="utf-8"))
    bad = [r for r in rows if not g6_ok(r)]
    browser = "browser check not run"
    status = "pass" if not bad else "fail"
    if can_import_playwright(args.browser_python):
        port = free_port()
        app = start_app(sys.executable, ROOT, port, py_env(), args.work / "g6_app.log")
        try:
            code, out = (sh([args.browser_python, str(Path(__file__).resolve()), "--browser-g6", f"http://127.0.0.1:{port}", str(args.work)], timeout=900)
                         if wait_health(f"http://127.0.0.1:{port}") else (1, "the app did not answer"))
        finally:
            stop_app(app)
        browser = last_line(out)
        if code != 0:
            status = "fail"
    elif status == "pass":
        status, browser = "manual", f"AppTest sweep passed; browser check not run ({args.browser_python} cannot import playwright)"
    return gate("G6", status, f"AppTest: every selectable classifier x {G6_SPEEDS} rad/s (healthy, defaults) + default sample in Chrome",
                f"{len(rows) - len(bad)}/{len(rows)} cases as expected; {browser}", failing=bad)


def g7(args) -> dict:
    if not args.previous_tag:
        return gate("G7", "manual", "git diff <previous tag>", "no previous tag found: give --previous-tag")
    cmd = ["git", "diff", "--stat", args.previous_tag, args.commit, "--", *BYTE_FOR_BYTE]
    code, out = sh(cmd)
    changed = out.strip()
    return gate("G7", "pass" if code == 0 and not changed else "fail", " ".join(cmd),
                f"unchanged since {args.previous_tag}" if not changed else f"changed: {last_line(changed)}")


def g8(args) -> dict:
    cmd = ["git", "grep", "-l", "/home/", args.commit, "--", "*.py", "*.yaml", "*.toml", "*.sh"]
    _, out = sh(cmd)
    files = sorted({l.split(":", 1)[1] for l in out.splitlines() if ":" in l})
    bad = [f for f in files if f not in HOME_PATH_ALLOWED]
    return gate("G8", "pass" if not bad else "fail", " ".join(cmd), "none outside the documented files" if not bad else f"absolute home paths in: {', '.join(bad)}",
                allowed={f: HOME_PATH_ALLOWED[f] for f in files if f in HOME_PATH_ALLOWED}, offending=bad)


RUNNERS = {"G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5, "G6": g6, "G7": g7, "G8": g8}


# --------------------------------------------------------------------------------------------- browser helpers
def _settle(page, timeout=180):
    t0 = time.time()
    page.wait_for_selector("[data-testid='stApp']", timeout=60000)
    page.wait_for_timeout(1200)
    quiet = 0.0
    while time.time() - t0 < timeout:
        busy = page.locator("[data-testid='stStatusWidget']").count() > 0 or page.locator("[data-testid='stSpinner']").count() > 0
        quiet = 0.0 if busy else quiet + 0.5
        if quiet >= 2.0:
            break
        page.wait_for_timeout(500)


def _browser():
    from playwright.sync_api import sync_playwright

    pw = sync_playwright().start()
    kw = {"executable_path": CHROME} if Path(CHROME).exists() else {}
    return pw, pw.chromium.launch(args=["--no-sandbox"], **kw)


def browser_pages(base: str, out_json: str, repo: str) -> None:
    """G4: opens the root first, then every page of PAGES; records exceptions and 'page does not exist'."""
    sys.path.insert(0, str(Path(repo) / "src"))
    from driveflow.viz.app import PAGES

    pw, b = _browser()
    page = b.new_context(viewport={"width": 1440, "height": 1500}).new_page()
    page.goto(base + "/", wait_until="domcontentloaded")
    _settle(page)
    rows = []
    for items in PAGES.values():
        for file, label, _ in items:
            name = Path(file).stem
            page.goto(f"{base}/" + ("" if name == "overview" else name), wait_until="domcontentloaded")
            _settle(page)
            rows.append({"page": label, "missing": page.get_by_text("does not seem to exist").count() > 0,
                         "exceptions": [e.inner_text()[:300] for e in page.locator("[data-testid='stException']").all()]})
            print(("OK " if not rows[-1]["exceptions"] else "EXC ") + label, flush=True)
    b.close()
    pw.stop()
    Path(out_json).write_text(json.dumps(rows, indent=1), encoding="utf-8")


def browser_turn(base: str, shot: str) -> None:
    """G5: one real assistant turn in the browser (the lesson of AI-N39)."""
    pw, b = _browser()
    page = b.new_context(viewport={"width": 1440, "height": 1300}).new_page()
    page.goto(base + "/", wait_until="domcontentloaded")
    _settle(page)
    page.goto(base + "/assistant", wait_until="domcontentloaded")
    page.wait_for_selector("[data-testid='stChatInputTextArea'], textarea", timeout=90000)
    page.wait_for_timeout(3000)
    box = page.locator("[data-testid='stChatInputTextArea'], textarea").first
    box.fill("Simulate the stock DC motor with a PI controller at 100 rad/s for 0.5 s.")
    box.press("Enter")
    t0, n = time.time(), 0
    while time.time() - t0 < 180:
        n = page.locator("[data-testid='stChatMessage']").count()
        if n >= 3:
            break
        page.wait_for_timeout(1000)
    page.wait_for_timeout(2000)
    page.screenshot(path=shot)
    b.close()
    pw.stop()
    print(f"messages={n} after {time.time() - t0:.0f} s")


def browser_g6(base: str, out_dir: str) -> None:
    """G6 in a real browser: the default sample shows a prediction; at 350 rad/s the no-diagnosis box and no prediction."""
    pw, b = _browser()
    results = {}
    for speed in (None, 350):
        page = b.new_context(viewport={"width": 1440, "height": 1500}).new_page()
        page.goto(base + "/", wait_until="domcontentloaded")
        _settle(page)
        page.goto(base + "/ai_diagnostics", wait_until="domcontentloaded")
        _settle(page)
        if speed is not None:
            page.get_by_text("Custom value").first.click()
            _settle(page)
            num = page.locator("[data-testid='stNumberInput'] input").first
            num.fill(str(speed))
            num.press("Enter")
            _settle(page)
        page.get_by_role("button", name="Generate").first.click()
        _settle(page, 300)
        text = page.locator("[data-testid='stMain']").first.inner_text()
        results["default" if speed is None else str(speed)] = {"box": OUTSIDE in text and "no diagnosis" in text, "prediction": "Classifier output" in text}
        page.screenshot(path=str(Path(out_dir) / f"g6_browser_{'default' if speed is None else speed}.png"))
        page.context.close()
    b.close()
    pw.stop()
    ok = results["default"]["prediction"] and not results["default"]["box"] and results["350"]["box"] and not results["350"]["prediction"]
    print(f"browser: default prediction={results['default']['prediction']} box={results['default']['box']}; "
          f"350 rad/s box={results['350']['box']} prediction={results['350']['prediction']}")
    sys.exit(0 if ok else 1)


# ------------------------------------------------------------------------------------------------------ report
def write_report(args, results: list[dict]) -> tuple[Path, Path]:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    data = {"label": args.label, "commit": args.commit, "branch": args.branch, "previous_tag": args.previous_tag, "started": args.started,
            "finished": now(), "host": f"{sys.platform}, Python {sys.version.split()[0]}", "simulated": True,
            "command": "python scripts/release_gates.py " + " ".join(args.argv), "gates": results,
            "all_pass": all(r["status"] == "pass" for r in results) and len(results) == len(GATES)}
    jp, mp = out / f"{args.label}_gates.json", out / f"{args.label}_gates.md"
    jp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [f"# {args.label}: release gates", "", f"Commit `{args.commit[:7]}` on `{args.branch}`, {args.started} → {data['finished']} ({data['host']}). "
             "**All SPADE results are simulated.** Written by `scripts/release_gates.py`; raw: `" + jp.name + "`.", "",
             "| Gate | Status | Result | Command |", "|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['gate']} | **{r['status']}** | {r['summary']} | `{r['command'][:160]}` |")
    verdict = "all eight gates pass" if data["all_pass"] else "**not all gates pass** (a 'manual' gate is not a pass)"
    lines += ["", f"Verdict: {verdict}.", ""]
    mp.write_text("\n".join(lines), encoding="utf-8")
    return jp, mp


def previous_tag(commit: str) -> str | None:
    code, out = sh(["git", "describe", "--tags", "--abbrev=0", "--match", "v[0-9]*", f"{commit}^"])
    return out.strip() if code == 0 and out.strip() else None


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--browser-pages":
        browser_pages(*argv[1:4])
        return 0
    if argv and argv[0] == "--browser-turn":
        browser_turn(*argv[1:3])
        return 0
    if argv and argv[0] == "--browser-g6":
        browser_g6(*argv[1:3])
        return 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="e.g. v0.2.3; names the evidence files")
    ap.add_argument("--only", default=",".join(GATES), help="comma-separated gates (default: all)")
    ap.add_argument("--branch", default=None, help="the pushed branch for G4 (default: the current branch)")
    ap.add_argument("--previous-tag", default=None, help="G7 baseline (default: the last v* tag before the commit)")
    ap.add_argument("--browser-python", default=sys.executable, help="an interpreter that can import playwright")
    ap.add_argument("--out", default=str(ROOT / "docs" / "audit" / "release"))
    ap.add_argument("--work", default=None, help="raw logs (default: a temporary directory)")
    args = ap.parse_args(argv)
    args.argv = argv
    only = [g.strip().upper() for g in args.only.split(",") if g.strip()]
    unknown = [g for g in only if g not in GATES]
    if unknown:
        ap.error(f"unknown gates: {unknown}")
    _, args.commit = sh(["git", "rev-parse", "HEAD"])
    args.commit = args.commit.strip()
    if args.branch is None:
        _, b = sh(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        args.branch = b.strip()
    args.previous_tag = args.previous_tag or previous_tag(args.commit)
    args.out = Path(args.out)
    args.work = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="spade_gates_"))
    args.work.mkdir(parents=True, exist_ok=True)
    args.started = now()
    results = []
    for g in GATES:
        if g in only:
            print(f"{g} ...", flush=True)
            results.append(RUNNERS[g](args))
            print(f"{g} {results[-1]['status']}: {results[-1]['summary']}", flush=True)
    jp, mp = write_report(args, results)
    print(f"written {mp.relative_to(ROOT) if mp.is_relative_to(ROOT) else mp}")
    return 0 if all(r["status"] == "pass" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
