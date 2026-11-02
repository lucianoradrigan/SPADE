"""Phase 1, F1.6: objective mode (a) end to end, with the objective tools called as a model would (no LLM).

Two requests, each tuned with SPADE's fixed objective J (PSO, Tuning Studio's default budget of 48
evaluations, seed 0) on a real worker in a temporary workspace, then every user threshold checked
on the EVALUATION grid for the baseline and the candidate:
  A. stock DC motor: overshoot < 5 % and settling < 50 ms;
  B. dc_pmg132_v1 (passed as its datasheet values): overshoot < 2 % and peak current <= 150 A.

    python experiments/report_f1_6_objective.py
Writes docs/audit/assistant/f1_6_objective.{md,json}.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "audit" / "assistant" / "f1_6_objective"
CASES = [
    ("A", "dc_perm_ex_v1", "Busca un PI para el motor DC de stock con sobrepaso menor a 5 % y asentamiento menor a 50 ms.",
     [{"metric": "overshoot_pct", "op": "<", "value": 5, "unit": "%"}, {"metric": "settling_time_s", "op": "<", "value": 50, "unit": "ms"}]),
    ("B", "dc_pmg132_v1", "Para el PMG 132 quiero sobrepaso menor a 2 % y corriente pico de 150 A como máximo.",
     [{"metric": "overshoot_pct", "op": "<", "value": 2, "unit": "%"}, {"metric": "peak_current_a", "op": "<=", "value": 150, "unit": "A"}]),
]


def main() -> None:
    os.environ["SPADE_WORKSPACE"] = tempfile.mkdtemp(prefix="spade_f1_6_")
    from driveflow.assistant import objective as O
    from driveflow.assistant.core.policy import SessionState
    from driveflow.jobs.worker import Worker
    from driveflow.runs.store import workspace_root

    results = []
    for label, plant, text, reqs in CASES:
        s = SessionState(mode="objective")
        s.last_user_seq = s.next_seq()
        s.ledger.add_user_message(1, text)
        s.draft.patch([{"op": "set_plant", "plant": plant}, {"op": "set_controller", "controller_type": "PI"}], s.ledger)
        O.update_objective(s, requirements=reqs, use_defaults=True)
        proposal = O.propose_tuning(s)
        s.last_user_seq = s.next_seq()
        s.ledger.add_user_message(s.last_user_seq, "sí")
        assert s.confirm(s.last_user_seq, "typed: sí")
        job = O.run_tuning(s)
        t0 = time.time()
        Worker(workspace_root()).serve(once=True, poll_s=0.05)
        wall = time.time() - t0
        check = O.check_requirements(s)
        results.append({"case": label, "plant": plant, "request": text, "proposal": proposal, "job": job, "wall_s": wall, "check": check})
    result = {"command": "python experiments/report_f1_6_objective.py", "llm": "none (tool calls written by hand)", "cases": results}
    OUT.with_suffix(".json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n")
    lines = [
        "# F1.6 — objective mode (a)",
        "",
        "**Simulated results; no language model.** Tuning with SPADE's fixed objective `J` (PSO, 48 evaluations = Tuning Studio's default, seed 0) on a real worker; "
        "each user threshold is then checked on the **evaluation grid** (separate from the tuning grid). Results are candidates: no default changes.",
        "",
        f"- **Command:** `{result['command']}`. Raw values: [`f1_6_objective.json`](f1_6_objective.json). Tests: `tests/assistant/test_assistant_objective.py`.",
        "- **Code:** `src/driveflow/assistant/objective.py` (`OBJECTIVE_TOOLS`: describe_tunable, update_objective, propose_tuning, run_tuning, check_requirements). "
        "Tuning runs planned by the assistant are labelled `SIMULATED · AI-PLANNED` (`jobs.handlers._origin`).",
        "",
    ]
    for r in results:
        c = r["check"]
        lines += [f"## Case {r['case']}: `{r['plant']}`", "", f"> {r['request']}", "",
                  f"Run `{c['run_id']}` ({c['origin']}), {r['wall_s']:.0f} s wall clock. Evaluation grid: {', '.join(c['evaluation_grid'] or [])}.", "",
                  "| Controller | Parameters | " + " | ".join(f"{q['metric']} {q['op']} {q['value']} {q['unit']}" for q in r["proposal"]["requirements"]) + " |",
                  "|---|---|" + "---|" * len(r["proposal"]["requirements"])]
        for row in c["results"]:
            p = row["params"]
            cells = []
            for ch in row["checks"]:
                w = "never settles" if ch.get("worst_case") is None else f"{ch['worst_case']:.4g} {ch['unit']}"
                cells.append(f"{'**met**' if ch['met'] else 'not met'}: worst {w}" + (f", margin {ch['margin']:+.3g}" if ch.get("margin") is not None else ""))
            lines.append(f"| {row['label']} | ω_bw {p['speed_bandwidth_rad_s']:.1f}, ζ {p['speed_damping']:.3f}, b {p['setpoint_weight']:.3f} | " + " | ".join(cells) + " |")
        lines += ["", f"**Verdict:** {c['verdict']}.", ""]
    lines += [
        "Worst case = the largest value over the evaluation-grid conditions that measure the metric (step conditions for overshoot and settling, every condition for peak current); "
        "a step that never settles fails a settling threshold. A negative verdict is a deliverable, not an error.",
        "",
        "**Case B is affected by finding S-21** (`s21_one_quadrant.md`): on a motor with the 0b-D1 friction the one-quadrant drive does not coast down after an overshoot but speeds up, "
        "so the baseline's \"overshoot\" of thousands of % on the scaled grid is that model effect, not the PI. The tools report it faithfully; the numbers of case B are not a valid tuning result until S-21 is resolved.",
        "",
        "## Scope decisions",
        "",
        "- Only the DC-motor speed PI is tuned from the chat (`dc_perm_ex_v1` with any motor; `dc_pmg132_v1` passed as its datasheet values, identical simulation checked by a test). "
        "The buck PI stays in Tuning Studio; MPC, DPC and FOC are not tunable and the assistant says so.",
        "- Thresholds must be quantities the user wrote (same no-invention rule as the draft); the budget comes from the user or from Tuning Studio's default.",
        "",
        "## Not executed",
        "",
        "- `J_spec` with the user's constraints (phase 3).",
        "- The same conversation with a real local model: no ejecutado (F1.1).",
    ]
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
