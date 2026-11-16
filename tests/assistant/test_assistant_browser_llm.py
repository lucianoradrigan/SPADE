"""R3 (2026-11-15), the lesson of AI-N39: the Assistant page in a REAL browser with the REAL local model.

AppTest does not execute the avatar component's JavaScript, so a page that works under AppTest can still never finish a
turn in a browser (AI-N39). This test starts the app, opens the Assistant page in Chromium (Playwright), sends one
request, and requires (1) the reply to arrive, (2) the avatar to pass through 'thinking' or 'using_tool' and to end in a
non-busy state, (3) no exception on the page.

Runs only with `pytest --run-llm` (marker llm) and when Playwright and a Chrome/Chromium are available; Playwright is not
a project dependency (pytest.importorskip). Needs Ollama with the pinned model.
"""

import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.llm
ROOT = Path(__file__).resolve().parents[2]
STATES_JS = """() => { const out = []; const walk = (root) => { root.querySelectorAll('*').forEach(el => {
  if (el.classList && el.classList.contains('rad-root') && el.dataset.state) out.push([el.classList.contains('compact') ? 'header' : 'side', el.dataset.state]);
  if (el.shadowRoot) walk(el.shadowRoot); }); }; walk(document); return out; }"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_one_real_turn_in_a_browser_with_the_avatar_states(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    chrome = next((p for p in ("/usr/bin/google-chrome", "/opt/pw-browsers/chromium", shutil.which("chromium") or "") if p and Path(p).exists()), None)
    port = _free_port()
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'src'}{os.pathsep}{ROOT}", "SPADE_WORKSPACE": str(tmp_path / "ws"), "TF_CPP_MIN_LOG_LEVEL": "3"}
    app = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "src" / "driveflow" / "viz" / "app.py"), "--server.headless", "true",
                            "--server.port", str(port), "--server.fileWatcherType", "none"], cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        import urllib.request
        for _ in range(90):
            try:
                if urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1).read() == b"ok":
                    break
            except OSError:
                time.sleep(1)
        with playwright.sync_playwright() as pw:
            b = pw.chromium.launch(**({"executable_path": chrome} if chrome else {}), args=["--no-sandbox"])
            pg = b.new_context(viewport={"width": 1440, "height": 1300}).new_page()
            pg.goto(f"http://127.0.0.1:{port}/assistant", wait_until="domcontentloaded")
            box = pg.locator("[data-testid='stChatInputTextArea'], textarea").first
            box.wait_for(timeout=90000)
            pg.wait_for_timeout(2000)
            if pg.get_by_text("Assistant unavailable").count():
                pytest.skip("the local model runtime is not running")
            box.fill("Simulate the stock DC motor with a PI controller at 100 rad/s for 0.5 s.")
            box.press("Enter")
            seen, t0 = {}, time.time()  # avatar kind ('header' = compact, 'side') -> its state sequence
            while time.time() - t0 < 180:
                found = pg.evaluate(STATES_JS)  # during a repaint an old and a new element of a kind can coexist
                states = [st for _, st in found]
                for kind, st in found:
                    seq = seen.setdefault(kind, [])
                    if st not in seq[-2:]:
                        seq.append(st)
                if pg.locator("[data-testid='stChatMessage']").count() >= 3 and states and all(s not in ("thinking", "using_tool") for s in states):
                    break
                pg.wait_for_timeout(250)
            n = pg.locator("[data-testid='stChatMessage']").count()
            print("avatar states:", seen, f"reply after {time.time() - t0:.0f} s")
            exceptions = pg.locator("[data-testid='stException']").count()
            pg.screenshot(path=str(tmp_path / "turn.png"))
            b.close()
        assert n >= 3, f"no reply after 180 s (AI-N39?); avatar states seen: {seen}"
        assert exceptions == 0
        assert set(seen) == {"header", "side"}, f"expected the side and the compact header avatar: {seen}"
        for k, seq in seen.items():  # R3: BOTH avatars follow the turn (on narrow screens only the header one is visible)
            assert {"thinking", "using_tool"} & set(seq), f"avatar {k} never showed the turn: {seq}"
            assert seq[-1] not in ("thinking", "using_tool"), (k, seq)
    finally:
        app.terminate()
        app.wait(timeout=30)
