# Releasing SPADE

How a SPADE release is checked and published. Written 2026-11-16 (task B1). **Everything SPADE produces is simulated**: the gates check the software, not a machine.

Lesson of AI-N39 and AI-N40: a gate that depends on remembering something will be forgotten. AI-N39 passed every test and still never finished a turn in a real browser. AI-N40 was never evaluated above 220 rad/s while the GUI defaulted to 300 rad/s. Then AI-N42: a G6 that "selected" a model evaluated a different one. The gates are therefore a script, and the script says **manual** when it cannot check something, never **pass**.

## The gates

| Gate | What it checks | Automated by `scripts/release_gates.py` |
|---|---|---|
| G1 | Full suite (`pytest`, slow included): 0 failures; every xfail cites a finding | yes |
| G2 | `scripts/smoke_test.py --launch` | yes |
| G3 | CI (`tests.yml`) on the release commit: every job green | yes, with the `gh` CLI; otherwise manual |
| G4 | Fresh clone of the pushed branch, without `data/` or `workspace/`, Ollama unreachable: smoke, then every page in a real browser without an exception | yes, with Playwright and `uv`; otherwise manual |
| G5 | Assistant with the real model: `pytest --run-llm -m llm`, then **one real turn in a browser** | yes, with Ollama and Playwright; otherwise manual |
| G6 | AI Diagnostics: **every selectable classifier** at 80/150/220/300/350 rad/s. Inside its trained envelope: a prediction, healthy → normal or abstains. Outside: the no-diagnosis box and no prediction. The evaluated run must be the selected one (model badge). Plus the default sample and 350 rad/s in a real browser | AppTest sweep yes; browser part with Playwright, otherwise manual |
| G7 | Byte-for-byte regression tests and `tests/data` unchanged since the previous `v*` tag | yes |
| G8 | No absolute home path in tracked code, except the files listed in `HOME_PATH_ALLOWED` | yes |

## Running them

```bash
# from the dev worktree, with the commit to release checked out and pushed
PYTHONPATH=src:. .venv/bin/python scripts/release_gates.py --label v0.2.3 \
    --browser-python /path/to/a/python/with/playwright
# some gates only (for example after the CI finishes)
PYTHONPATH=src:. .venv/bin/python scripts/release_gates.py --label v0.2.3 --only G3
```

- Output: `docs/audit/release/<label>_gates.md` and `.json` (commit, times, command and result of each gate). Raw logs go to `--work` (a temporary directory by default).
- Exit code 0 only if every requested gate passes.
- Playwright is not a SPADE dependency. Use an interpreter that has it (`--browser-python`); system Chrome at `/usr/bin/google-chrome` is used when present. Every browser check opens the root `/` first.
- The app is stopped by the script itself (its own child process). By hand, use `fuser -k <port>/tcp`, never `pkill -f`.
- The G4 clone and its environment are built in a temporary directory and removed afterwards.

## Publishing (only after the eight gates pass, and with Luciano's OK for that release)

1. Bump the version: `pyproject.toml`, `src/driveflow/viz/components/citation.py` and the README (`tests/test_ui_contract.py` checks they agree).
2. Write `docs/RELEASE_NOTES_vX.Y.Z.md`: what is fixed, what is new, the open findings, "all results are simulated", the assistant's status.
3. Commit on `dev`, push, and wait for CI 5/5.
4. Fast-forward `main`, tag and publish, with no `--force`:

```bash
git fetch origin
git merge-base --is-ancestor origin/main <commit> && echo FF_OK
git push origin <commit>:refs/heads/main
git tag -a vX.Y.Z <commit> -m "SPADE vX.Y.Z"
git push origin vX.Y.Z
gh release create vX.Y.Z --verify-tag --title "SPADE vX.Y.Z" --notes-file docs/RELEASE_NOTES_vX.Y.Z.md
```

5. If `FF_OK` is not printed, or a step fails half way, undo nothing: record the exact output in `docs/audit/release/vX.Y.Z_published.md`.
6. A finding discovered after a release goes into that release's text as a "Known issue" line; the tag is never moved.
