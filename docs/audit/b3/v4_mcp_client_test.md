# V4: the MCP server with a real client (round C, T4, 2026-11-09)

Raw values: [`v4_mcp_client_test.json`](v4_mcp_client_test.json). **Simulated;** nothing was launched.

## How it was run

- **Client:** Claude Code 2.1.319, in a separate, non-interactive session (`claude -p`) started from a scratch directory, with permission for the `mcp__spade__*` tools only.
- **Registration:** `claude mcp add -s local spade -e PYTHONPATH=<repo>/src -e SPADE_WORKSPACE=<scratch>/ws -- <repo>/.venv/bin/python <wrapper>`.
  - The wrapper runs `python -m driveflow.assistant.mcp_server` and logs every JSON-RPC line, which is the evidence.
  - The registration was removed afterwards (`claude mcp remove spade -s local`).
- `claude mcp list` showed `spade … ✔ Connected`.

## What happened

- **`initialize`:** the client offered protocol `2025-11-25` with the capabilities `roots` and **`elicitation` (`form`, `url`)**. The server answered with its own revision, `2025-06-18`, and the client went on.
- **Discovery probe:** before `initialize`, the client sent `server/discover`; the server answered "method not found" (−32601), and the client continued.
- **`tools/list`:** the client saw the 10 tools.

| Step | Tool | Result |
|---|---|---|
| 1 | `describe_plant` (`dc_perm_ex_v2`) | ok: the stock motor and the PI/MPC design check |
| 2 | `update_draft` (plant, PI, ω_ref = 100 rad/s with `user_message`) | ok: three operations applied; 100 rad/s marked as coming from the user; draft ready |
| 3 | `validate_draft` | ok: nothing missing or invalid; five important fields left at their default |
| 4 | `estimate_cost` | ok: 1 experiment, 1 simulated second, about 3.16 CPU s |
| 5 | `propose_experiments` | ok: one experiment, with the draft hash `d28886c7…` |
| 6 | `run_experiments` | the server sent **`elicitation/create`** with the plant, the number of experiments and the hash. The client answered **`cancel`** (a `-p` session has no user to ask), and the server returned `isError: true`: "the user did not confirm (action 'cancel'); nothing was launched" |

- **Jobs in the workspace afterwards:** 0. The session's final line: `LAUNCHED: no`. The session cost USD 0.22.

## Reading

- **Claude Code supports elicitation,** so `run_experiments` is reachable from it, but only through the user's own answer.
- **Without a person at the terminal,** the request is cancelled and nothing runs, which is the intended behaviour.
- **What the client received from SPADE:** the plant description, the draft and its status, the cost estimate and the proposal. These are configurations and SPADE's own data. No user file was read (see `tests/assistant/test_mcp_egress.py`).

## Not executed

- **Accepting the confirmation in an interactive session:** not executed, because it needs a person at the terminal. Steps for Luciano: `docs/assistant/MCP.md`, "Try it in Claude Desktop".
- **Claude Desktop itself:** not executed, because it is not installed on this machine.
