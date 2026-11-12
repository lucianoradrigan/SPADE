# AI-N39: the Assistant page never finished a turn in a real browser with the real model

**Found 2026-11-12** while taking the screenshots of M6.3, after the M5 stability verdict. **Corrected** on `feat/AI-N39-avatar-rerun`. Everything is **simulated**; the model is the pinned local `qwen3.5:9b`.

## What happened

- In a real browser (Chrome through Playwright), with Ollama running, the page showed the user's message and "Thinking…" forever: **no reply after 180–300 s**, on `dev` and on the published **v0.2.0** (the page and component code are identical; `git diff v0.2.0 dev` is empty for them).
- A logging proxy between the page and Ollama (`scripts/oneoff/ollama_logging_proxy.py`) showed **one** `/api/chat` call (2.5 s) and then nothing. The loop needs about six calls for this request.
- The Streamlit debug log showed a **rerun storm**: 813 script runs and 1352 "Ignoring event from non-current ScriptRunner" in 3 minutes. Each run cancelled the previous one.
- Ollama itself answered in 2.9 s (GPU). With the page idle the app ran only its 2 s job fragment (about 5 runs in 20 s): **the storm starts with a turn**.

## Cause

- The avatar component reported `{reduced_motion, state, has_assets}` back to Python with `setStateValue` whenever its state changed (`avatar.js`). In a browser, setting a component's state **requests a rerun**.
- The Assistant page repaints the avatar on every agent event, with a new key each time, so every repaint is a new component that reports again.
- During a turn, the first state change (`thinking`) restarted the script in the middle of the turn. The new run did not resume the turn, repainted, reported, and so on.
- **Confirmation:** with that single call disabled in a throwaway clone, the same turn completed in 15 s with six model calls.
- **Why no test caught it:** `AppTest` (the page tests, the F1.10 demo and gate G5 of the M5 check) does not run the component's JavaScript. The fresh-clone checks (v0.2.0 and M5 G4) ran the page with Ollama unreachable, so no turn ever started.

## Fix (small; no default, model or existing test changed)

- `radrigan_avatar(..., report_env=False)`: the component reports only when Python asks for it.
- `avatar.js` checks `d.report_env`.
- Only the avatar demo page (`scripts/assistant_avatar_demo.py`, the FE-3 spike that displays the report) asks. The Assistant page never read the report (`on_env_change` was a no-op).

## Evidence after the fix

| Check | Before (dev / v0.2.0 code) | After (this branch) |
|---|---|---|
| One turn in Chrome, real model: "Simulate the stock DC motor with a PI controller at 100 rad/s for 0.5 s." | no reply in 180 s; 1 model call | **reply in 13 s**; 6 model calls |
| Script runs in the session | 813 (3 min) | 4 |
| "non-current ScriptRunner" events | 1352 | 0 |
| Screenshot | `ai_n39/before_fix_stuck.png` | `ai_n39/after_fix_reply.png` |

**Commands:**

```bash
PROXY_LOG=proxy.log python scripts/oneoff/ollama_logging_proxy.py &
OLLAMA_HOST=127.0.0.1:11500 streamlit run src/driveflow/viz/app.py --server.port 8777 --logger.level debug
python scripts/oneoff/assistant_browser_turn.py http://127.0.0.1:8777 shot.png
```

The proxy listens on 11500 and forwards to 11434. The page's own guard accepts it because it is localhost.

**Test:** `tests/assistant/test_ai_n39_avatar_rerun.py` (4) checks the contract: the page never asks for the report, and the JS reports only when asked. The browser check above is the behavioural evidence; Playwright is not a project dependency.

**v0.2.0 is affected.** The release notes say the assistant is experimental, but a user with Ollama running cannot get a reply in the browser; the CLI (`python -m driveflow.assistant`) and the MCP server are not affected (no component). Whether to annotate the published v0.2.0 release is Luciano's decision.
