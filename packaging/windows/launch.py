"""Starts SPADE from the Windows installer (packaging/windows): the job worker in the background and
the Streamlit app in this window, which opens the browser at http://localhost:8501. Closing this
window stops both. Run by the "SPADE" shortcut with the bundled Python.

The installed tree is a SPADE checkout (<app>/SPADE) plus an embedded Python (<app>/python) whose
python312._pth puts <app>/SPADE/src on the path, so the repository-relative paths (configs/, data/,
workspace/) resolve exactly as in a clone.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PORT = 8501


def _port_in_use(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def main() -> int:
    os.chdir(REPO)
    if _port_in_use(PORT):
        import webbrowser

        print(f"SPADE is already running: opening http://localhost:{PORT}")
        webbrowser.open(f"http://localhost:{PORT}")
        return 0
    env = dict(os.environ, PYTHONUTF8="1", TF_CPP_MIN_LOG_LEVEL="2")
    print("SPADE is starting. The browser opens at http://localhost:8501 in a few seconds.")
    print("Keep this window open while you use SPADE; close it to stop SPADE.\n")
    worker = subprocess.Popen([sys.executable, "-m", "driveflow.jobs.worker", "serve"], cwd=REPO, env=env)
    try:
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(REPO / "src" / "driveflow" / "viz" / "app.py"),
                                "--server.port", str(PORT), "--server.headless", "false", "--browser.gatherUsageStats", "false"],
                               cwd=REPO, env=env)
    except KeyboardInterrupt:
        return 0
    finally:
        worker.terminate()


if __name__ == "__main__":
    sys.exit(main())
