# F1.3 — bounded tool loop, confirmation policy, tools by state

**No language model involved:** the model is a scripted `ReplayClient`; tools are stand-ins except `update_draft` (real `Draft.patch`).

- **Command:** `python experiments/report_f1_3_loop_policy.py`. Raw values: [`f1_3_loop_policy.json`](f1_3_loop_policy.json). Tests: `tests/assistant/test_assistant_loop.py`.
- **Code:** `src/driveflow/assistant/core/loop.py` (`Agent`, `Tool`, `AgentTrace`, limits), `src/driveflow/assistant/core/policy.py` (`SessionState`, `visible_tools`, `check_call`, `confirm`, `is_affirmative`).
- **Limits** (configs/assistant.yaml): tool calls, model calls and tokens per turn; three failed tool calls in a row end the turn with a message. Errors (invalid JSON, refused or failing tools) always go back to the model.

## Tools the model sees, by state (C2)

| State | Visible tools |
|---|---|
| undetermined, empty draft | `set_mode`, `explain_spade`, `search_docs`, `recommend_plant`, `describe_plant` |
| question | `set_mode`, `explain_spade`, `search_docs` |
| configuration, draft not ready | `set_mode`, `explain_spade`, `search_docs`, `recommend_plant`, `describe_plant`, `suggest_value`, `update_draft`, `validate_draft`, `estimate_cost` |
| configuration, draft ready | `set_mode`, `explain_spade`, `search_docs`, `recommend_plant`, `describe_plant`, `suggest_value`, `update_draft`, `validate_draft`, `estimate_cost`, `propose_experiments` |
| configuration, launch proposed (same draft) | `set_mode`, `explain_spade`, `search_docs`, `recommend_plant`, `describe_plant`, `suggest_value`, `update_draft`, `validate_draft`, `estimate_cost`, `propose_experiments`, `run_experiments` |
| configuration, a job exists | `set_mode`, `explain_spade`, `search_docs`, `recommend_plant`, `describe_plant`, `suggest_value`, `update_draft`, `validate_draft`, `estimate_cost`, `propose_experiments`, `run_experiments`, `get_job_status`, `summarize_results` |
| objective, no objective yet | `set_mode`, `explain_spade`, `search_docs`, `recommend_plant`, `describe_plant`, `suggest_value`, `update_draft`, `describe_tunable`, `update_objective`, `validate_draft` |

Tools named here but built in later tasks (`explain_spade`, `search_docs` in F1.7; the configuration and objective tools in F1.5–F1.6) are only sent to the model once they exist in the registry.

## Confirmation scenarios

| Scenario | User's reply | Interface mark | Launched | Refusal sent back to the model |
|---|---|---|---|---|
| user says yes, interface marks it | sí | typed: sí | **yes** | — |
| user asks something else, model tries to launch | ¿cuánto tarda? | — | **no** | the user has not confirmed this proposal; wait for their answer (the interface records it) |
| model claims the user confirmed | perfecto | — | **no** | the user has not confirmed this proposal; wait for their answer (the interface records it) |
| draft changed in the same turn as the yes | sí, y 2 s | typed | **no** | tool 'run_experiments' is not available now (mode configuration); available: ['set_mode', 'explain_spade', 'search_docs', 'recommend_plant', 'describe_plant', 'suggest_value', 'update_draft', 'validate_draft', 'estimate_cost', 'propose_experiments'] |

Only the interface layer sets the confirmation mark (`SessionState.confirm`, called by the CLI / page when the user presses Run or types a bare yes right after a proposal); no tool can set it, and the mark is bound to the user's latest message and to the hash of the proposed draft.

## Not executed

- The loop with a real local model: no ejecutado (model chosen in F1.1).
- F1.1's comparison of 'all tools visible' vs 'tools by state': no ejecutado yet (F1.1).
