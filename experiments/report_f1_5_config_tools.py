"""Phase 1, F1.5: the configuration-mode tools, end to end with a scripted model (no LLM).

A scripted conversation (the tool calls a model would make) recommends the plant, builds the draft,
proposes, is confirmed by the interface, launches one scenario_sweep job that a real SPADE worker
runs in a temporary workspace, and summarises the results with SPADE's KPIs. Plus the
recommend_plant table and the suggestions available from internal sources (P5).

    python experiments/report_f1_5_config_tools.py
Writes docs/audit/assistant/f1_5_config_tools.{md,json}.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "audit" / "assistant" / "f1_5_config_tools"
REQUEST = "Simula el motor DC de stock con PI a 50 y 100 rad/s, 0.5 s cada una, sin carga extra."
OPS = [{"op": "set_plant", "plant": "dc_perm_ex_v1"}, {"op": "set_controller", "controller_type": "PI"},
       {"op": "vary", "field": "omega_ref_rad_s", "values": [50, 100], "unit": "rad/s"}, {"op": "set", "field": "duration_s", "value": 0.5, "unit": "s"}]
RECOMMEND = ["motor DC de imanes permanentes de 220 V", "PMG 132", "servo brushless PMSM", "inversor trifásico con filtro LC", "buck de 48 V a 12 V",
             "motor de inducción de 4 polos", "convertidor boost", "un sistema de bombeo"]


def call(name, **a):
    return {"function": {"name": name, "arguments": a}}


def reply(content="", *calls):
    msg = {"role": "assistant", "content": content, **({"tool_calls": list(calls)} if calls else {})}
    return {"request": {"messages": [], "tools": [], "seed": 0}, "reply": {"message": msg}}


def main() -> None:
    os.environ["SPADE_WORKSPACE"] = tempfile.mkdtemp(prefix="spade_f1_5_")
    from driveflow.assistant import tools as T
    from driveflow.assistant.core import llm
    from driveflow.assistant.core.loop import Agent
    from driveflow.assistant.core.policy import SessionState
    from driveflow.assistant.draft import audit_trace
    from driveflow.jobs.worker import Worker
    from driveflow.runs.store import workspace_root

    script = [reply("", call("set_mode", mode="configuration"), call("recommend_plant", description="motor DC de stock")),
              reply("", call("update_draft", ops=OPS)), reply("", call("propose_experiments")), reply("¿Lanzo los 2 experimentos?"),
              reply("", call("run_experiments")), reply("Lanzado."), reply("", call("get_job_status"), call("summarize_results")), reply("Listo.")]
    a = Agent(llm.ReplayClient({"format": 1, "exchanges": script}, strict=False), T.CONFIG_TOOLS)
    a.user_turn(REQUEST)
    a.user_turn("sí", confirmed_how="typed: sí")
    Worker(workspace_root()).serve(once=True, poll_s=0.05)
    a.user_turn("¿resultados?")
    res = {e["name"]: e["result"] for e in a.trace.events if e["kind"] == "tool_result"}
    summary = res["summarize_results"]
    rec_table = [{"description": d, **{k: v for k, v in T.recommend_plant(SessionState(), d).items() if k in ("matches", "not_simulable")}} for d in RECOMMEND]
    s = SessionState(mode="configuration")
    s.ledger.add_user_message(1, "motor DC")
    s.draft.patch([{"op": "set_plant", "plant": "dc_perm_ex_v1"}], s.ledger)
    sugg = {f: T.suggest_value(s, f) for f in ("omega_ref_rad_s", "load_torque_nm", "duration_s", "dc_supply_v", "motor_parameter_overrides.l_a")}
    result = {"command": "python experiments/report_f1_5_config_tools.py", "llm": "none (scripted tool calls)", "request": REQUEST, "proposal": res["propose_experiments"],
              "job": res["run_experiments"], "status": res["get_job_status"], "summary": summary, "audit_trace_problems": audit_trace(a.trace.draft_events),
              "recommend_plant": rec_table, "suggestions": sugg}
    OUT.with_suffix(".json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n")
    lines = [
        "# F1.5 — configuration-mode tools",
        "",
        "**Simulated results; no language model** (the tool calls are scripted). The job ran on a real SPADE worker in a temporary workspace.",
        "",
        f"- **Command:** `{result['command']}`. Raw values: [`f1_5_config_tools.json`](f1_5_config_tools.json). Tests: `tests/assistant/test_assistant_tools.py`.",
        "- **Code:** `src/driveflow/assistant/tools.py` (`CONFIG_TOOLS`): recommend_plant, describe_plant, suggest_value, update_draft, validate_draft, estimate_cost, "
        "propose_experiments, run_experiments, get_job_status, summarize_results.",
        "",
        "## End-to-end conversation",
        "",
        f"> {REQUEST}",
        "",
        f"Proposal shown to the user: {len(res['propose_experiments']['experiments'])} experiments, estimated {res['propose_experiments']['cost']['est_cpu_s']:.1f} CPU s. "
        f"Launched only after the interface recorded the user's \"sí\": job `{res['run_experiments']['job_id']}` → run `{summary['run_id']}`, status **{summary['run_status']}**, origin `{summary['origin']}`.",
        "",
        "| # | ω_ref (rad/s) | Final speed (rad/s) | Rise time (s) | Settling time (s) | Overshoot (%) | Steady current (A) | Peak current (A) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in summary["rows"]:
        k = r["kpis"]
        st = k["speed_step"]
        lines.append(f"| {r['index']} | {k['speed_reference_rad_s']:g} | {k['final_speed_rad_s']:.3f} | {st['rise_time_s']} | {st['settling_time_s']} | {st['overshoot_pct']:.3f} | "
                     f"{k['steady_current_a']:.3f} | {k['peak_current_a']:.2f} |")
    lines += [
        "",
        f"No-invention replay of the session (`audit_trace`): {len(result['audit_trace_problems'])} problems. "
        "KPIs come from the run's parquet files through `validation.step_metrics` (rise 10–90 %, settling ±2 % band, overshoot % of the step).",
        "",
        "## recommend_plant",
        "",
        "| User's description | Recommended | Not simulable (closest substitute) |",
        "|---|---|---|",
    ] + [f"| {x['description']} | {', '.join('`' + m['plant'] + '`' for m in x['matches']) or '—'} | "
         f"{(x['not_simulable']['reason'] + ' → ' + ', '.join('`' + p['plant'] + '`' for p in x['not_simulable']['closest_substitutes'])) if 'not_simulable' in x else '—'} |" for x in rec_table] + [
        "",
        "## suggest_value: internal sources only (decision P5)",
        "",
        "| Field (DC stock motor) | Suggested | Rule / why not |",
        "|---|---|---|",
    ] + [f"| `{f}` | {v['value'] if v['value'] is not None else '— (ask)'} {v['unit'] or '' if v['value'] is not None else ''} | {v.get('rule') or v.get('why')} |" for f, v in sugg.items()] + [
        "",
        "## Observations",
        "",
        "- `dc_pmg132_v1` has no native plant profile (`runs.profiles.NATIVE_PROFILES`) and no per-condition metric in `jobs.handlers._condition_metrics`; "
        "`run_experiments` launches it with `plant_profile_id = None` and `summarize_results` computes its KPIs from the parquet, so the assistant is not affected. Left as is (Phase 0b scope).",
        "",
        "## Not executed",
        "",
        "- The same conversation with a real local model: no ejecutado (F1.1).",
    ]
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
