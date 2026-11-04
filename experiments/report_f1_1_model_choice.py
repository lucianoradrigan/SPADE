"""Phase 1, F1.1: comparison of local models (Ollama) on the assistant's real tools and policy.

Bench: docs/audit/assistant/f1_1_prompts.json (12 one-turn requests and the system prompt, committed
before any model ran; distinct from the F1.11 benchmark). Each prompt runs in a fresh session, 3 times
(seed 0, 1, 2), per model and per tool visibility:
  (a) "all": every tool schema is shown to the model (the policy still refuses calls not allowed now);
  (b) "state": only the tools policy.visible_tools allows in the current mode and draft state (F1.3).

Metrics per model x configuration:
  - valid calls: tool calls whose arguments parse as JSON and validate against the tool's schema (%);
  - correct tool: the turn ends in the expected mode, calls one of the expected tools, attempts no
    forbidden one (%);
  - launch attempts without confirmation (run_experiments / run_tuning calls; no turn confirms, so all
    are unconfirmed) and launches executed (must be 0);
  - invented numbers: update_draft patches refused because a number has no evidence, and accepted
    patches the no-invention replay (draft.audit_trace) refuses (must be 0);
  - latency per turn (median, p95) and generation tokens/s.

    python experiments/report_f1_1_model_choice.py --models qwen3:14b ornith:9b [--seeds 0 1 2]
Writes docs/audit/assistant/f1_1_model_choice.{md,json}; raw traces to --traces (not committed).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "audit" / "assistant" / "f1_1_model_choice"
PROMPTS = ROOT / "docs" / "audit" / "assistant" / "f1_1_prompts.json"
LAUNCH = ("run_experiments", "run_tuning")
NO_EVIDENCE = ("the user did not write this quantity", "no suggest_value result", "no tool result contains this number")
LICENSES = {"qwen3": "Apache-2.0", "ornith": "MIT"}  # verified on ollama.com / Hugging Face model cards, 2026-11-04


def _nvidia_used_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def _model_info(client, tag: str) -> dict:
    m = next((x for x in client.installed_models() if x.get("name") == tag or x.get("model") == tag), {})
    d = m.get("details") or {}
    return {"tag": tag, "digest": m.get("digest"), "size_bytes": m.get("size"), "parameter_size": d.get("parameter_size"),
            "quantization": d.get("quantization_level"), "family": d.get("family"), "license": LICENSES.get(tag.split(":")[0], "unknown")}


def _loaded(client) -> list:
    try:
        return client._request("/api/ps", timeout=10).get("models", [])
    except Exception:  # noqa: BLE001
        return []


def run_one(tag: str, digest: str, prompt: dict, system_prompt: str, seed: int, visibility: str) -> dict:
    import jsonschema

    from driveflow.assistant import catalog, objective, tools
    from driveflow.assistant.config import load_config
    from driveflow.assistant.core import llm, loop
    from driveflow.assistant.core.loop import Agent
    from driveflow.assistant.draft import audit_trace

    cfg = load_config()
    cfg = {**cfg, "model": {"tag": tag, "digest": digest}}
    all_tools = tools.CONFIG_TOOLS + objective.OBJECTIVE_TOOLS + catalog.CATALOG_TOOLS
    agent = Agent(llm.OllamaClient(cfg), all_tools, system_prompt=system_prompt, seed=seed, limits=cfg.get("limits"))
    schemas = {t.name: t.parameters for t in [loop.SET_MODE, *all_tools]}
    saved = loop.visible_tools
    if visibility == "all":
        loop.visible_tools = lambda state: list(agent.tools)  # shown to the model only; check_call keeps the policy
    t0 = time.perf_counter()
    error = None
    try:
        res = agent.user_turn(prompt["text"])
        stopped, text = res.stopped, res.text
    except Exception as exc:  # noqa: BLE001
        stopped, text, error = "exception", "", f"{type(exc).__name__}: {exc}"
    finally:
        loop.visible_tools = saved
    wall = time.perf_counter() - t0
    calls, valid, gen_tokens, gen_time = [], 0, 0, 0.0
    for e in agent.trace.events:
        if e["kind"] == "model":
            gen_tokens += e.get("completion_tokens") or 0
            gen_time += e.get("latency_s") or 0.0
            for c in e["tool_calls"]:
                ok = c["error"] is None and c["name"] in schemas
                if ok:
                    try:
                        jsonschema.validate(c["arguments"], schemas[c["name"]])
                    except jsonschema.ValidationError:
                        ok = False
                valid += ok
                calls.append({"name": c["name"], "valid": ok, "arguments": c["arguments"]})
    names = [c["name"] for c in calls if c["name"] != "set_mode"]
    refused_invented = [e for e in agent.trace.events if e["kind"] == "tool_error" and e.get("name") == "update_draft" and any(s in str(e.get("error")) for s in NO_EVIDENCE)]
    executed = [e["name"] for e in agent.trace.events if e["kind"] == "tool_result"]
    correct = (agent.state.mode == prompt["expected_mode"] and any(n in prompt["expect_any"] for n in names) and not any(n in prompt["forbidden"] for n in names))
    return {"prompt": prompt["id"], "seed": seed, "visibility": visibility, "mode": agent.state.mode, "tools_called": [c["name"] for c in calls],
            "n_calls": len(calls), "n_valid": valid, "correct_tool": bool(correct),
            "launch_attempts": sum(n in LAUNCH for n in names), "launches_executed": sum(n in LAUNCH for n in executed),
            "invented_numbers_refused": len(refused_invented), "audit_trace_problems": len(audit_trace(agent.trace.draft_events)),
            "wall_s": wall, "gen_tokens": gen_tokens, "gen_s": gen_time, "stopped": stopped, "error": error, "answer": text[:600],
            "trace": agent.trace.events}


def summarise(rows: list) -> dict:
    lat = [r["wall_s"] for r in rows]
    n_calls = sum(r["n_calls"] for r in rows)
    return {"turns": len(rows), "errors": sum(r["error"] is not None for r in rows),
            "valid_calls_pct": 100.0 * sum(r["n_valid"] for r in rows) / n_calls if n_calls else None, "tool_calls": n_calls,
            "correct_tool_pct": 100.0 * np.mean([r["correct_tool"] for r in rows]),
            "launch_attempts_unconfirmed": int(sum(r["launch_attempts"] for r in rows)), "launches_executed": int(sum(r["launches_executed"] for r in rows)),
            "invented_numbers_refused": int(sum(r["invented_numbers_refused"] for r in rows)), "audit_trace_problems": int(sum(r["audit_trace_problems"] for r in rows)),
            "latency_median_s": float(np.median(lat)), "latency_p95_s": float(np.percentile(lat, 95)),
            "tokens_per_s": float(sum(r["gen_tokens"] for r in rows) / max(sum(r["gen_s"] for r in rows), 1e-9)),
            "stopped_by_limit": int(sum(r["stopped"] not in (None, "exception") for r in rows)),
            "correct_by_kind": {k: 100.0 * np.mean([r["correct_tool"] for r in rows if r["prompt"][0] == k]) for k in ("c", "o", "q")}}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--traces", type=Path, default=None)
    args = ap.parse_args(argv)
    os.environ.setdefault("SPADE_WORKSPACE", tempfile.mkdtemp(prefix="spade_f1_1_"))
    from driveflow.assistant.config import load_config
    from driveflow.assistant.core import llm

    bench = json.loads(PROMPTS.read_text())
    client = llm.OllamaClient({**load_config(), "model": {"tag": "probe", "digest": None}})
    res = {"command": "python experiments/report_f1_1_model_choice.py " + " ".join(argv if argv is not None else sys.argv[1:]),
           "host": platform.platform(), "runtime_version": client.version(), "gpu_vram_idle_mib": _nvidia_used_mib(),
           "config": {k: load_config().get(k) for k in ("temperature", "context_tokens", "think", "limits")}, "models": {}}
    for tag in args.models:
        info = _model_info(client, tag)
        rows = []
        t0 = time.time()
        for vis in ("all", "state"):
            for p in bench["prompts"]:
                for seed in args.seeds:
                    r = run_one(tag, info["digest"], p, bench["system_prompt"], seed, vis)
                    rows.append(r)
                    print(tag, vis, p["id"], seed, r["mode"], r["tools_called"], f"{r['wall_s']:.1f}s", r["error"] or "", flush=True)
        info["vram_loaded_mib"] = _nvidia_used_mib()
        info["ollama_ps"] = [{k: m.get(k) for k in ("name", "size", "size_vram", "context_length")} for m in _loaded(client)]
        info["elapsed_s"] = time.time() - t0
        info["by_visibility"] = {vis: summarise([r for r in rows if r["visibility"] == vis]) for vis in ("all", "state")}
        info["per_prompt"] = {vis: {p["id"]: {"correct": int(sum(r["correct_tool"] for r in rows if r["visibility"] == vis and r["prompt"] == p["id"])),
                                              "tools": [r["tools_called"] for r in rows if r["visibility"] == vis and r["prompt"] == p["id"]]}
                                    for p in bench["prompts"]} for vis in ("all", "state")}
        res["models"][tag] = info
        if args.traces:
            args.traces.mkdir(parents=True, exist_ok=True)
            (args.traces / f"{tag.replace(':', '_')}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False, default=str))
        subprocess.run(["ollama", "stop", tag], capture_output=True)
    OUT.with_suffix(".json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str) + "\n")
    OUT.with_suffix(".md").write_text(_md(res))
    print(_md(res))


def _md(r: dict) -> str:
    L = ["# F1.1 — comparison of local models (measured)", "", f"Command: `{r['command']}`. Host: {r['host']}; Ollama {r['runtime_version']}. "
         f"Settings: {r['config']}. Bench: [`f1_1_prompts.json`](f1_1_prompts.json) (12 prompts × 3 seeds × 2 visibilities, written before measuring).", "",
         "| Model | Size | Quant. | License | VRAM loaded | Visibility | Valid calls | Correct tool (conf. / obj. / question) | Unconfirmed launch attempts / executed | Invented numbers refused / accepted | Latency median / p95 | tok/s | Errors / limit stops |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for tag, m in r["models"].items():
        for vis, s in m["by_visibility"].items():
            ps = (m.get("ollama_ps") or [{}])[0]
            L.append(f"| `{tag}` | {m.get('parameter_size')} ({(m.get('size_bytes') or 0) / 1e9:.1f} GB) | {m.get('quantization')} | {m['license']} | "
                     f"{(ps.get('size_vram') or 0) / 2**30:.1f} GiB | {vis} | {s['valid_calls_pct']:.1f} % ({s['tool_calls']}) | "
                     f"{s['correct_tool_pct']:.1f} % ({s['correct_by_kind']['c']:.0f} / {s['correct_by_kind']['o']:.0f} / {s['correct_by_kind']['q']:.0f}) | "
                     f"{s['launch_attempts_unconfirmed']} / {s['launches_executed']} | {s['invented_numbers_refused']} / {s['audit_trace_problems']} | "
                     f"{s['latency_median_s']:.1f} / {s['latency_p95_s']:.1f} s | {s['tokens_per_s']:.0f} | {s['errors']} / {s['stopped_by_limit']} |")
    L += ["", "Digests: " + "; ".join(f"`{t}` {m['digest']}" for t, m in r["models"].items()) + ".", ""]
    for tag, m in r["models"].items():
        L += [f"## `{tag}` per prompt (correct of 3)", "", "| Prompt | all | state |", "|---|---|---|"]
        for pid in m["per_prompt"]["state"]:
            L.append(f"| {pid} | {m['per_prompt']['all'][pid]['correct']} | {m['per_prompt']['state'][pid]['correct']} |")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
