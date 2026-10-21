"""GUI boundary to RL training and evaluation (expansion task E8; spec Sec. 5)."""

from __future__ import annotations

import json

import gymnasium as gym
import numpy as np
import pandas as pd

import driveflow.envs  # noqa: F401 -- registers SPADE/*
from driveflow.control.rl import benchmark as bm
from driveflow.envs import registry
from driveflow.jobs.handlers import RL_ENVS
from driveflow.jobs.schemas import JobState
from driveflow.jobs.service import JobService
from driveflow.runs.schemas import ApprovalRecord, ExperimentSpec, Run, SpadeError, new_id, utc_now
from driveflow.runs.store import RunStore

NOT_AUTHORIZED = "Using an RL policy as a default is not authorized (decision D4, 2026-10-21): the policy stays a candidate."


def env_ids() -> list[str]:
    """Environments the RL job supports AND that are registered in this build."""
    return [e for e in RL_ENVS if registry.is_registered(e)]


def baseline_name(env_id: str) -> str:
    return "PI (BuckVoltageController)" if RL_ENVS[env_id] == "buck" else "DPC v4"


def preview_baseline(env_id: str, terms: list[dict]) -> dict:
    """One executed episode of the plant's classical controller under these reward terms (the
    'Preview on baseline episode'): total reward and the per-term sums."""
    env = gym.make(env_id, reward_terms=terms)
    make = bm.buck_pi if RL_ENVS[env_id] == "buck" else (lambda e: bm.vsc_dpc(e, "v4"))
    obs, _ = env.reset(seed=0)
    act = make(env.unwrapped)
    total, sums, n, reason = 0.0, {}, 0, None
    while True:
        obs, r, term, trunc, info = env.step(act(obs))
        total += float(r)
        n += 1
        for k, v in info["reward_terms"].items():
            sums[k] = sums.get(k, 0.0) + float(v)
        if term or trunc:
            reason = info.get("termination") if term else None
            break
    env.close()
    return {"total": total, "breakdown": sums, "steps": n, "termination": reason, "controller": baseline_name(env_id)}


def submit(env_id: str, algorithm: str, episodes: int, seed: int, terms: list[dict], discrete_levels: int | None, eval_every: int):
    params = {"env_id": env_id, "episodes": int(episodes), "eval_every": int(eval_every), "reward_terms": terms}
    if algorithm == "dqn":
        params["discrete_levels"] = int(discrete_levels)
    spec = ExperimentSpec.create("rl_training", algorithm=algorithm, seed=int(seed), params=params, budget={"episodes": int(episodes)},
                                 objectives=("episode reward",), plant_profile_id=registry.ENVIRONMENTS[env_id].plant_profile_id)
    return JobService().submit(spec)


def active_jobs() -> list[dict]:
    s = JobService()
    return [{"job_id": j.job_id, "state": j.state.value, "done": j.progress_done, "total": j.progress_total}
            for j in s.list([JobState.QUEUED, JobState.PREPARING, JobState.RUNNING, JobState.CANCELLING]) if j.kind == "rl_training"]


def runs(limit: int = 50) -> list[Run]:
    return [r for r in RunStore().list_runs(500) if r.kind == "rl_training"][:limit]


def _json(run: Run, suffix: str):
    a = next((a for a in run.artifacts if a["path"].endswith(suffix)), None)
    if a is None:
        return None
    p = RunStore().artifact_path(a)
    return json.loads(p.read_text()) if p.exists() else None


def history(run: Run) -> pd.DataFrame | None:
    h = _json(run, "history.json")
    if h is None:
        return None
    df = pd.DataFrame({"episode": np.arange(1, len(h["episode_reward"]) + 1), "reward": h["episode_reward"], "loss": h["loss"],
                       "violation": [v is not None for v in h["episode_violation"]]})
    if h.get("actor_loss"):
        df["actor_loss"] = h["actor_loss"]
    df["violation_rate_10"] = df["violation"].rolling(10, min_periods=1).mean()
    ev = pd.DataFrame({"episode": h["eval_episode"], "eval_reward": h["eval_reward"]})
    return df.merge(ev, on="episode", how="left")


def playback_episodes(run: Run) -> list[int]:
    return sorted(int(a["path"].rsplit("eval_ep", 1)[1].split(".")[0]) for a in run.artifacts if "playback/eval_ep" in a["path"])


def playback(run: Run, episode: int) -> dict | None:
    return _json(run, f"playback/eval_ep{episode:04d}.json")


def benchmark(run: Run) -> dict | None:
    return _json(run, "benchmark.json")


def benchmark_table(b: dict) -> pd.DataFrame:
    rows = []
    for ctrl, res in b["results"].items():
        for opt, r in zip(b["grid"], res):
            rows.append({"Controller": ctrl, "Condition": ", ".join(f"{k}={v:g}" for k, v in opt.items() if k != "phase0_rad"), "Episode reward": r["reward"],
                         "Tracking RMSE, 2nd half (V)": r["tracking_rmse_second_half"], "Peak current (A)": r["peak_current"], "Outcome": r["termination"] or "completed"})
    return pd.DataFrame(rows)


def gate_report(run: Run) -> dict:
    b = benchmark(run) or {}
    blocking = [NOT_AUTHORIZED]
    s = b.get("summary", {})
    rl = next((k for k in s if k.startswith("RL")), None)
    if rl and any(v["mean_reward"] > s[rl]["mean_reward"] for k, v in s.items() if k != rl):
        blocking.append("The policy loses to a classical controller on the benchmark grid.")
    return {"blocking": blocking, "summary": s}


def save_candidate(run: Run) -> dict:
    data = {"run_id": run.run_id, "saved_utc": utc_now(), "status": "policy candidate -- not a default", "policy_artifact": "policy.json"}
    return RunStore().write_artifact(run.run_id, "saved_policy_candidate.json", json.dumps(data, indent=2))


def record_decision(run: Run, gate: dict, approver: str, decision: str) -> dict:
    rec = ApprovalRecord(approval_id=new_id("appr"), subject=f"RL policy for {run.metrics.get('env_id')}", baseline_run=None, candidate_run=run.run_id,
                         gate_report=gate, approver=approver.strip(), user_confirmed_at=utc_now(), decision=decision)
    try:
        return {"record": RunStore().record_approval(rec)}
    except SpadeError as exc:
        return {"error": str(exc)}
