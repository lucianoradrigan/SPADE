# SPADE assistant tools over MCP (local)

`python -m driveflow.assistant.mcp_server` exposes the assistant's tools to an MCP client, over **stdio
only**. It opens no socket, calls no LLM and adds no dependency: it is JSON-RPC 2.0 written with the
standard library (`src/driveflow/assistant/mcp_server.py`). Protocol revision: `2025-06-18`.

**Everything it launches is simulated** and labelled `SIMULATED · AI-PLANNED`, like the assistant's runs.

## Tools

These are the same tool functions as the Assistant page, applied to one session per server process
(configuration mode):

- `describe_plant`
- `update_draft`
- `validate_draft`
- `estimate_cost`
- `propose_experiments`
- `run_experiments`
- `get_job_status`
- `summarize_results`
- `explain_spade`
- `search_docs`

## The same policies as the assistant

- **Tools by state.** A call that the assistant would refuse now is refused with the same message (`policy.check_call`). For example:
  - `run_experiments` before `propose_experiments`;
  - `summarize_results` before the job has finished.
- **Every number needs its source.** `update_draft` checks each value against the evidence ledger.
  - Pass the user's message verbatim as `user_message`: a value with origin `user` must appear in it.
  - This server never sees the user, so the ledger records those messages as relayed by the client.
  - A change with a number that has no source is refused all-or-none (`isError: true`).
- **Only the user confirms a launch.** `run_experiments` sends an MCP `elicitation/create` request to the client.
  - The request names the plant, the number of experiments and the proposal's draft hash.
  - Only an `accept` with `confirm: true` sets the confirmation mark, exactly as the Run button does, and it is tied to that draft hash.
  - Declining or cancelling launches nothing.
  - One confirmation is one launch.
  - If the draft changes after the proposal, a new proposal is needed.
  - A client without the `elicitation` capability **cannot launch**. There is no tool, argument or token that confirms.
- **A SPADE worker runs the job:** `python -m driveflow.jobs.worker serve`, or the Experiment Runs page.

## Client configuration (example)

Every MCP client has its own configuration file. The usual form is:

```json
{
  "mcpServers": {
    "spade": {
      "command": "/path/to/SPADE/.venv/bin/python",
      "args": ["-m", "driveflow.assistant.mcp_server"],
      "env": {"PYTHONPATH": "/path/to/SPADE/src", "TF_CPP_MIN_LOG_LEVEL": "3"}
    }
  }
}
```

`SPADE_WORKSPACE` selects the workspace (default: the repository's).

## What leaves the machine with a cloud client (decision C7, 2026-11-09)

Cloud MCP clients are allowed. The server itself stays local: stdio, no socket, no LLM. A cloud client
sends to its model provider what the tools return, and only that:

| Leaves the machine | Comes from |
|---|---|
| **Configurations:** plant descriptions (fields, units, ranges, defaults, the stock motor), the draft and its status, cost estimates, the proposal table | `describe_plant`, `update_draft`, `validate_draft`, `estimate_cost`, `propose_experiments` |
| **Simulated results:** job state and KPIs of finished SPADE jobs, all labelled SIMULATED | `get_job_status`, `summarize_results` |
| **SPADE's documentation:** catalog entries and snippets of `docs/` and the README, with file and section | `explain_spade`, `search_docs` |
| What the client itself sends: the user's messages (`user_message`) and the tool arguments | the client |

- **No tool reads the user's files.** No tool takes a file name, path or URL. The tools read only SPADE's own code, configuration and documentation, the SPADE workspace (job results) and Python's libraries.
- **This is checked by `tests/assistant/test_mcp_egress.py`.** It calls every exposed tool, also with path-like arguments such as `/etc/passwd`, `../../.ssh/id_rsa` and a temporary "user" file, under a Python audit hook that records every file opened.
  - It fails if any file outside SPADE, its workspace and Python is opened.
  - It fails if a tool argument names a file or a path.
  - It fails if a file's content comes back.
- **With a local client,** whose model runs on this machine, nothing leaves it.

## Try it in Claude Desktop

Claude Desktop starts the server itself over stdio, so it must run on the machine where SPADE is installed. Add this to `claude_desktop_config.json`:
- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
- **In the app:** Settings → Developer → Edit Config.

The paths below are those of the reference machine: change them to your checkout.

```json
{
  "mcpServers": {
    "spade": {
      "command": "/home/seven/post_doc/plataform_proposed/SPADE-dev/.venv/bin/python",
      "args": ["-m", "driveflow.assistant.mcp_server"],
      "env": {
        "PYTHONPATH": "/home/seven/post_doc/plataform_proposed/SPADE-dev/src",
        "TF_CPP_MIN_LOG_LEVEL": "3"
      }
    }
  }
}
```

Restart Claude Desktop. The tools icon should list **spade** with 10 tools.

**Three 5-minute checks**

| # | Ask Claude | What should happen |
|---|---|---|
| 1 | "With the spade tools, describe the plant dc_perm_ex_v2 and explain what S-21 is." | It calls `describe_plant` and `explain_spade` or `search_docs`. The answer quotes the stock motor (r_a 0.016 Ω, l_a 19 µH, ψ_e 0.165 V·s/rad, 65 V supply) and the S-21 entry, with the file it came from |
| 2 | "Set up a simulation of the stock DC motor with PI at 100 rad/s and tell me the cost; do not launch it." | `update_draft` passes your sentence as `user_message`, then come `validate_draft` and `estimate_cost`: 1 experiment, about 3 CPU s. If it tries a number you did not write, `update_draft` refuses it ("the user did not write this quantity") |
| 3 | "Propose it and launch it." | `propose_experiments`, then `run_experiments`. **Claude Desktop must ask *you***, in a dialog with the plant, the number of experiments and the draft hash. **Decline:** the result says nothing was launched. **Accept:** a job is queued; start a worker (`python -m driveflow.jobs.worker serve`) or open Experiment Runs to run it. If no dialog appears and the result says the client "cannot ask its user for confirmation", this Claude Desktop version has no elicitation, and nothing was launched |

## Tests

`tests/assistant/test_mcp_server.py` covers:
- initialization and the tool list;
- an evidence refusal;
- no launch before a proposal, without elicitation, on decline or cancel, or after a draft change;
- one launch per confirmation, tied to the hash;
- results only after the job finished;
- local only;
- a stdio round trip.

`tests/assistant/test_mcp_egress.py` covers what may leave the machine (see above). The real-client
test with Claude Code is in [`docs/audit/b3/v4_mcp_client_test.md`](../audit/b3/v4_mcp_client_test.md).

The tests use no network and no LLM.

## Not executed

- A launch accepted through an interactive client.
- Claude Desktop.

The non-interactive session with Claude Code is documented in `docs/audit/b3/v4_mcp_client_test.md`.
