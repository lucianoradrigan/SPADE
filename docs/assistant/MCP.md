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

> **Data leave the machine if the client's model is remote.** The server is local, but an MCP client
> sends the tool results (plant descriptions, drafts, KPIs) to the model it uses. SPADE's rule that no
> data leave the machine holds only with a client whose model runs locally. Choosing a cloud client is a
> decision for the person running it.

## Tests

`tests/assistant/test_mcp_server.py` covers:
- initialization and the tool list;
- an evidence refusal;
- no launch before a proposal, without elicitation, on decline or cancel, or after a draft change;
- one launch per confirmation, tied to the hash;
- results only after the job finished;
- local only;
- a stdio round trip.

The tests use no network and no LLM.

## Not executed

A session with a real MCP client (no client is installed here). The protocol was checked against the
2025-06-18 message shapes by the tests only.
