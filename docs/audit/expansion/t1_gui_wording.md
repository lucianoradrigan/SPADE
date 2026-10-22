# T1 (round 2026-10-22): "live" removed from visible GUI text

Command for the inventory: `grep -rniE "\blive\b|production[ -]ready|hardware[ -]verified" src/driveflow/viz --include=*.py`.
Regression test: `pytest tests/test_gui_wording.py`. The test scans every non-docstring string literal under `src/driveflow/viz/`. Run against the pre-fix code, it reports the two strings changed below.

## Changed (visible text)

| File | Before | After |
|---|---|---|
| `components/motor_status.py` (note under the motor, Overview / DC Motor / Monitoring) | "no valid positive speed from a live source" | "no valid positive speed from a simulation run or monitoring pass" |
| `views/system_diagram.py` (Model registry node) | "a training run existing doesn't mean it's live." | "a training run existing doesn't mean it is the one in use." |
| `components/motor_status.py` (module docstring, not rendered) | "from a live (non-offline) source" | "from an available (non-offline) source" |

**Why "simulation run or monitoring pass".** The component's source modes are `simulated`, `connected`, `demo` and `unavailable` (`monitoring/motor_status.py`). `connected` is not implemented in this repository, so in practice the speed comes from a simulation run (DC Motor page) or a simulated monitoring pass (Monitoring page).

## Kept, with the reason

| File | Text | Reason |
|---|---|---|
| `components/state_panel.py` | `aria-live="polite"` | HTML accessibility attribute, not visible text |
| `views/documentation.py` | "CONNECTED / LIVE … Reserved for real telemetry with heartbeat — not implemented" | States explicitly that LIVE is reserved |
| `adapters/monitoring_adapter.py`, `views/monitoring.py`, `components/connection_status.py`, `theme/tokens.py`, `dashboard.py`, `adapters/ai_adapter.py` | docstrings or comments ("LIVE / CONNECTED is reserved…", "live channel" = a channel that varies) | Not rendered |

## "production ready" / "hardware verified" (reported, not changed)

| File | Text | Assessment |
|---|---|---|
| `views/model_registry.py` | KPI "Exported (.tflite)" with "export only — not hardware-verified"; KPI "Hardware verified" with "requires deployment evidence" (count is 0) | Correct: it states the absence of hardware evidence |
| `views/documentation.py` | "no run is shown as hardware-verified" | Correct |
| `adapters/registry_adapter.py` | docstring: "Hardware verified -- only with a deployment_evidence.json" | Not rendered |

No visible "production ready" text exists.
