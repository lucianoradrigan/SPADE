"""Job handlers: what a job kind actually computes -- expansion task E0.

A handler is ``fn(ctx) -> HandlerResult``. It reports progress and log lines through ``ctx``,
checks ``ctx.cancel_requested()`` between units of work (never inside a solver call), writes its
artifacts with ``ctx.write_artifact`` and returns one entry per condition. The worker turns the
condition outcomes into the job's final state:

- every condition ok -> completed; some ok and some failed -> partially_completed;
- none ok -> failed; stopped on request -> cancelled (the remaining conditions are "skipped").

Kinds:
- ``scenario_sweep``: runs datagen.runner.run_scenario once per condition (each condition is a
  dict of Scenario fields) and saves one Parquet file per successful condition.
- ``plecs_sweep`` (task E2): one PLECS RPC simulation per condition (a dict of ModelVars) on a
  ``plecs_rpc_batch`` profile; engine "plecs_rpc", so the worker runs one at a time. Failed
  conditions keep their error and are never interpolated. The dataset (one Parquet per completed
  condition, each row carrying its own ModelVars, plus manifest.json and conditions.csv) is written
  to data/plecs/<profile_id>/<run_id>/ and to the run's artifacts.
- ``controller_tuning`` (task E7): runs an optimiser (``pso``, ``bayes`` or ``nsga2``,
  driveflow.optim) on optim.objectives over the TUNING grid, then simulates the baseline and the
  candidate(s) on the separate EVALUATION grid. One condition per objective evaluation. The result
  is a candidate: nothing is changed (``Use as default`` goes through ApprovalGate).
- ``rl_training`` (task E8): trains a DQN or DDPG agent (driveflow.control.rl) on a registered
  SPADE environment with the reward terms of the spec, saves a checkpoint and an evaluation-episode
  playback at every evaluation, then benchmarks the final policy against the plant's classical
  controllers on the same condition grid. One condition per training episode.
- ``selftest.steps``: no physics; sleeps ``step_s`` per step. Used by the tests to exercise the
  worker (cancel, kill, recover) deterministically, and by "Check worker" in the GUI.
"""

from __future__ import annotations

import dataclasses
import io
import math
import time
from typing import Callable

import numpy as np

from driveflow.runs.schemas import ErrorCode, ExperimentSpec, SpadeError


@dataclasses.dataclass
class HandlerResult:
    conditions: list
    metrics: dict = dataclasses.field(default_factory=dict)
    origin: str = "SIMULATED"


@dataclasses.dataclass
class JobContext:
    spec: ExperimentSpec
    run_id: str
    log: Callable[[str], None]
    progress: Callable[[int | None, int | None, str | None], None]
    cancel_requested: Callable[[], bool]
    write_artifact: Callable[[str, bytes], dict]


@dataclasses.dataclass(frozen=True)
class HandlerInfo:
    fn: Callable[[JobContext], HandlerResult]
    validate: Callable[[ExperimentSpec], None]
    title: str
    engine: str = "native"


# --------------------------------------------------------------------------- scenario_sweep
def _scenario_from(condition: dict, index: int):
    from driveflow.datagen.scenario import Scenario

    kwargs = dict(condition)
    kwargs.setdefault("scenario_id", f"cond{index:03d}")
    return Scenario(**kwargs)


def _validate_scenario_sweep(spec: ExperimentSpec) -> None:
    if not spec.conditions:
        raise SpadeError(ErrorCode.INVALID_SPEC, "scenario_sweep needs at least one condition (a dict of Scenario fields)")
    for i, cond in enumerate(spec.conditions):
        if not isinstance(cond, dict):
            raise SpadeError(ErrorCode.INVALID_SPEC, f"condition {i} must be a dict of Scenario fields")
        try:
            _scenario_from(cond, i)
        except (TypeError, ValueError, NotImplementedError) as exc:
            raise SpadeError(ErrorCode.INVALID_SPEC, f"condition {i}: {exc}") from None


def _condition_metrics(df) -> dict:
    """A few plant-specific numbers, each named with its unit and window. NaN -> None."""
    out = {"n_samples": int(len(df)), "t_end_s": float(df["timestamp_s"].iloc[-1]) if len(df) else None}
    plant = df["plant_config_id"].iloc[0] if len(df) else None
    half = df.iloc[len(df) // 2:]
    if plant == "vsc_dpc_v1":
        err = ((half["vc_real"] - half["v_ref_real"]) ** 2 + (half["vc_imag"] - half["v_ref_imag"]) ** 2) ** 0.5
        out["rms_voltage_error_v_second_half"] = float((err ** 2).mean() ** 0.5)
    elif plant == "dc_perm_ex_v1":
        out["mean_rpm_second_half"] = float(half["rpm"].mean())
    elif plant == "buck_v1":
        out["rms_voltage_error_v_second_half"] = float(((half["buck_v_out"] - half["buck_v_ref"]) ** 2).mean() ** 0.5)
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in out.items()}


def scenario_sweep(ctx: JobContext) -> HandlerResult:
    from driveflow.datagen.export_parquet import records_to_dataframe
    from driveflow.datagen.runner import run_scenario

    conditions, n = [], len(ctx.spec.conditions)
    for i, cond in enumerate(ctx.spec.conditions):
        if ctx.cancel_requested():
            conditions += [{"index": j, "condition": c, "status": "skipped", "error": "Cancelled before this condition started."} for j, c in enumerate(ctx.spec.conditions[i:], start=i)]
            break
        ctx.progress(i, n, f"condition {i + 1}/{n}")
        t0 = time.perf_counter()
        try:
            scenario = _scenario_from(cond, i)
            df = records_to_dataframe(run_scenario(scenario))
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            art = ctx.write_artifact(f"condition_{i:03d}.parquet", buf.getvalue())
            metrics = _condition_metrics(df)
            conditions.append({"index": i, "condition": cond, "status": "ok", "metrics": metrics, "artifact": art["path"], "seconds": round(time.perf_counter() - t0, 3)})
            ctx.log(f"condition {i + 1}/{n} ok in {time.perf_counter() - t0:.2f} s ({metrics['n_samples']} samples)")
        except Exception as exc:  # one failed condition never hides the others
            conditions.append({"index": i, "condition": cond, "status": "failed", "error": f"{type(exc).__name__}: {exc}", "seconds": round(time.perf_counter() - t0, 3)})
            ctx.log(f"condition {i + 1}/{n} failed: {type(exc).__name__}: {exc}")
    ctx.progress(sum(c["status"] != "skipped" for c in conditions), n, "finished")
    return HandlerResult(conditions=conditions, metrics={"n_conditions": n, "n_ok": sum(c["status"] == "ok" for c in conditions)})


# --------------------------------------------------------------------------- plecs_sweep
def _plecs_profile(spec: ExperimentSpec):
    from driveflow.integrations.plecs import batch
    from driveflow.runs.store import RunStore

    if not spec.plant_profile_id:
        raise SpadeError(ErrorCode.INVALID_SPEC, "plecs_sweep needs plant_profile_id")
    profile = RunStore().get_profile(spec.plant_profile_id, spec.params.get("profile_version"))
    if profile.kind != batch.KIND:
        raise SpadeError(ErrorCode.INVALID_SPEC, f"profile {profile.profile_id} is a {profile.kind} profile, not {batch.KIND}")
    return profile


def _validate_plecs_sweep(spec: ExperimentSpec) -> None:
    from driveflow.integrations.plecs import batch, manifest as mf

    profile = _plecs_profile(spec)
    man = batch.profile_manifest(profile)
    if not spec.conditions:
        raise SpadeError(ErrorCode.INVALID_SPEC, "plecs_sweep needs at least one condition (a dict of ModelVars)")
    for i, cond in enumerate(spec.conditions):
        try:
            man.check_values(dict(cond))
        except (mf.ManifestError, AttributeError, TypeError) as exc:
            raise SpadeError(ErrorCode.INVALID_SPEC, f"condition {i}: {exc}") from None
    try:
        batch.solver_options(float(spec.params.get("time_span_s", float("nan"))), spec.params.get("n_output_points"))
    except (TypeError, ValueError) as exc:
        raise SpadeError(ErrorCode.INVALID_SPEC, str(exc)) from None
    limit = spec.params.get("time_limit_s", 60.0)
    if not (isinstance(limit, (int, float)) and 0 < limit <= 3600):
        raise SpadeError(ErrorCode.INVALID_SPEC, "time_limit_s must be in (0, 3600] s")


def plecs_sweep(ctx: JobContext) -> HandlerResult:
    import json

    import pandas as pd

    from driveflow.integrations.plecs import batch
    from driveflow.integrations.plecs.rpc_client import PlecsRpcClient, PlecsRpcError
    from driveflow.runs.profiles import is_stale

    profile = _plecs_profile(ctx.spec)
    man = batch.profile_manifest(profile)
    p = profile.parameters
    time_span, limit, n_pts = float(ctx.spec.params["time_span_s"]), float(ctx.spec.params.get("time_limit_s", 60.0)), ctx.spec.params.get("n_output_points")
    conds = [dict(c) for c in ctx.spec.conditions]
    n = len(conds)
    out_dir = batch.data_root() / profile.profile_id / ctx.run_id
    manifest_doc = {"profile_id": profile.profile_id, "profile_version": profile.version, "circuit_name": p["circuit_name"], "circuit_sha256": profile.file_hash,
                    "model_name": p["model_name"], "origin": batch.ORIGIN, "interface": man.to_dict(), "time_span_s": time_span, "time_limit_s": limit,
                    "n_output_points": n_pts, "method": ctx.spec.algorithm, "seed": ctx.spec.seed, "run_id": ctx.run_id, "note": batch.NOT_GYM}

    def finish(conditions, metrics):
        rows = [{"condition_index": c["index"], "status": c["status"], **c["condition"], "seconds": c.get("seconds"), "error": c.get("error"), "file": c.get("file")} for c in conditions]
        table = pd.DataFrame(rows).to_csv(index=False)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "conditions.csv").write_text(table)
        (out_dir / "manifest.json").write_text(json.dumps({**manifest_doc, "n_conditions": n, "n_ok": metrics.get("n_ok", 0)}, indent=2))
        ctx.write_artifact("conditions.csv", table.encode())
        ctx.write_artifact("manifest.json", json.dumps(manifest_doc, indent=2).encode())
        metrics["dataset_dir"] = str(out_dir.relative_to(batch.data_root().parent)) if out_dir.is_relative_to(batch.data_root().parent) else out_dir.name
        return HandlerResult(conditions=conditions, metrics=metrics, origin=batch.ORIGIN)

    if is_stale(profile, p["circuit_path"]):
        ctx.log("circuit file changed or is missing since the profile was created: profile is stale")
        reason = "The circuit or mapping changed. Re-run verification. (circuit SHA-256 no longer matches the profile)"
        return finish([{"index": i, "condition": c, "status": "failed", "error": reason} for i, c in enumerate(conds)], {"n_conditions": n, "n_ok": 0, "stale": True})

    client = PlecsRpcClient(p["rpc_url"])
    conditions = []
    try:
        model = client.load(p["circuit_path"])
        ctx.log(f"loaded {p['circuit_name']} (sha256 {profile.file_hash[:12]}) as model {model!r}")
    except PlecsRpcError as exc:
        ctx.log(f"could not load the circuit: {exc}")
        return finish([{"index": i, "condition": c, "status": "failed", "error": f"[{exc.code.value}] {exc}"} for i, c in enumerate(conds)], {"n_conditions": n, "n_ok": 0})
    try:
        for i, cond in enumerate(conds):
            if ctx.cancel_requested():
                conditions += [{"index": j, "condition": c, "status": "skipped", "error": "Cancelled before this condition started."} for j, c in enumerate(conds[i:], start=i)]
                break
            ctx.progress(i, n, f"condition {i + 1}/{n}")
            try:
                df, wall = batch.run_condition(client, model, man, cond, time_span, limit, n_pts)
                df.insert(0, "condition_index", i)
                df["profile_id"], df["run_id"] = profile.profile_id, ctx.run_id
                buf = io.BytesIO()
                df.to_parquet(buf, index=False)
                name = f"condition_{i:04d}.parquet"
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / name).write_bytes(buf.getvalue())
                art = ctx.write_artifact(name, buf.getvalue())
                summary = {f"{o}_final": float(df[o].iloc[-1]) for o in man.output_names}
                conditions.append({"index": i, "condition": cond, "status": "ok", "metrics": {"n_samples": int(len(df)), **summary}, "artifact": art["path"], "file": name, "seconds": round(wall, 3)})
                ctx.log(f"condition {i + 1}/{n} ok in {wall:.2f} s ({len(df)} samples)")
            except PlecsRpcError as exc:
                conditions.append({"index": i, "condition": cond, "status": "failed", "error": f"[{exc.code.value}] {exc}"})
                ctx.log(f"condition {i + 1}/{n} failed: [{exc.code.value}] {exc}")
            except Exception as exc:  # one failed condition never hides the others
                conditions.append({"index": i, "condition": cond, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
                ctx.log(f"condition {i + 1}/{n} failed: {type(exc).__name__}: {exc}")
    finally:
        try:
            client.close(model)
        except PlecsRpcError as exc:
            ctx.log(f"closing the model failed: {exc}")
    ctx.progress(sum(c["status"] != "skipped" for c in conditions), n, "finished")
    walls = [c["seconds"] for c in conditions if c["status"] == "ok"]
    n_ok = sum(c["status"] == "ok" for c in conditions)
    if n_ok:
        from driveflow.runs.store import RunStore

        RunStore().save_profile(batch.mark_batch_shown(profile, ctx.run_id, n_ok, n))
        ctx.log(f"profile {profile.profile_id}: batch simulation shown by this run (new version {profile.version + 1})")
    return finish(conditions, {"n_conditions": n, "n_ok": sum(c["status"] == "ok" for c in conditions), "wall_s_median": float(pd.Series(walls).median()) if walls else None})


# --------------------------------------------------------------------------- controller_tuning
TUNING_METHODS = ("pso", "bayes", "nsga2")


class _Cancelled(Exception):
    pass


def _validate_tuning(spec: ExperimentSpec) -> None:
    from driveflow.optim import objectives as ob

    if spec.algorithm not in TUNING_METHODS:
        raise SpadeError(ErrorCode.INVALID_SPEC, f"controller_tuning needs algorithm in {TUNING_METHODS}, got {spec.algorithm!r}")
    if spec.seed is None:
        raise SpadeError(ErrorCode.INVALID_SPEC, "controller_tuning needs a seed")
    bounds = spec.params.get("bounds")
    try:
        b = [[float(lo), float(hi)] for lo, hi in bounds]
    except (TypeError, ValueError):
        raise SpadeError(ErrorCode.INVALID_SPEC, f"bounds must be {len(ob.PARAMS)} [low, high] pairs") from None
    if len(b) != len(ob.PARAMS):
        raise SpadeError(ErrorCode.INVALID_SPEC, f"bounds must be {len(ob.PARAMS)} [low, high] pairs for {ob.PARAMS}")
    for (lo, hi), (alo, ahi), name in zip(b, ob.BOUNDS, ob.PARAMS):
        if not (alo <= lo < hi <= ahi):
            raise SpadeError(ErrorCode.INVALID_SPEC, f"{name}: [{lo}, {hi}] must lie inside [{alo}, {ahi}] with low < high")
    budget = (spec.budget or {}).get("evaluations")
    if not (isinstance(budget, int) and 4 <= budget <= 2000):
        raise SpadeError(ErrorCode.INVALID_SPEC, "budget.evaluations must be an int in [4, 2000]")


def controller_tuning(ctx: JobContext) -> HandlerResult:
    import json

    from driveflow.optim import objectives as ob
    from driveflow.optim.bayes import bayes_opt
    from driveflow.optim.multiobjective import nsga2
    from driveflow.optim.pso import pso

    spec = ctx.spec
    bounds = [[float(lo), float(hi)] for lo, hi in spec.params["bounds"]]
    n_eval = int(spec.budget["evaluations"])
    seed = int(spec.seed)
    conditions = []

    def callback(i, x, fx):
        f = np.atleast_1d(np.asarray(fx, dtype=float))
        conditions.append({"index": i - 1, "condition": ob.as_kwargs(x), "status": "ok" if np.all(np.isfinite(f)) else "failed",
                           "metrics": {"objective": [float(v) for v in f]}, **({} if np.all(np.isfinite(f)) else {"error": "non-finite simulation"})})
        ctx.progress(i, n_eval, f"evaluation {i}/{n_eval}")
        if ctx.cancel_requested():
            raise _Cancelled

    method = spec.algorithm
    try:
        if method == "pso":
            n_part = max(4, min(16, n_eval // 4))
            res = pso(ob.scalar_objective, bounds, n_particles=n_part, iterations=max(1, n_eval // n_part), seed=seed, callback=callback)
            candidates = [{"label": "best", "params": ob.as_kwargs(res.x_best)}]
        elif method == "bayes":
            n_init = max(4, n_eval // 4)
            res = bayes_opt(ob.scalar_objective, bounds, n_init=n_init, iterations=n_eval - n_init, seed=seed, callback=callback)
            candidates = [{"label": "best", "params": ob.as_kwargs(res.x_best)}]
        else:
            pop = max(4, min(24, (n_eval // 4) // 2 * 2))
            res = nsga2(ob.vector_objective, bounds, population=pop, generations=max(1, n_eval // pop), seed=seed, callback=callback)
            front = sorted(res.front_idx.tolist(), key=lambda k: float(res.F[k][0]))
            candidates = [{"label": f"pareto_{j}", "params": ob.as_kwargs(res.X[k]), "objectives": [float(v) for v in res.F[k]]} for j, k in enumerate(front)]
    except _Cancelled:
        conditions.append({"index": len(conditions), "condition": {}, "status": "skipped", "error": "Cancelled before the remaining evaluations."})
        return HandlerResult(conditions=conditions, metrics={"method": method, "n_evaluations": len(conditions) - 1})

    ctx.log(f"{method}: {len(conditions)} evaluations; evaluating baseline and {len(candidates)} candidate(s) on the evaluation grid")
    base_sum, base_rows = ob.evaluate(ob.BASELINE, ob.EVALUATION_GRID)
    base_tune, _ = ob.evaluate(ob.BASELINE, ob.TUNING_GRID)
    evaluated = []
    for c in candidates[:12]:
        es, er = ob.evaluate(c["params"], ob.EVALUATION_GRID)
        ts, _ = ob.evaluate(c["params"], ob.TUNING_GRID)
        evaluated.append({**c, "tuning": ts, "evaluation": es, "evaluation_rows": er})
    history = {"method": method, "seed": seed, "bounds": bounds, "params": list(ob.PARAMS),
               "X": [list(map(float, c["condition"].values())) for c in conditions if c["status"] != "skipped"],
               "objective": [c["metrics"]["objective"] for c in conditions if c["status"] != "skipped"]}
    ctx.write_artifact("tuning_history.json", json.dumps(history).encode())
    report = {"baseline": {"params": ob.BASELINE, "tuning": base_tune, "evaluation": base_sum, "evaluation_rows": base_rows}, "candidates": evaluated,
              "tuning_grid": [c.name for c in ob.TUNING_GRID], "evaluation_grid": [c.name for c in ob.EVALUATION_GRID],
              "evaluation_reused_tuning_conditions": ob.grids_overlap(), "objective": "J = mean overshoot % + 0.1 x mean settling ms + 2 x mean load dip rad/s",
              "status": "candidate -- no default changed"}
    ctx.write_artifact("tuning_report.json", json.dumps(report, default=float).encode())
    best = min(evaluated, key=lambda c: c["tuning"]["J"]) if evaluated else None
    metrics = {"method": method, "n_evaluations": len(conditions), "baseline_J_tuning": base_tune["J"], "baseline_J_evaluation": base_sum["J"],
               "best_J_tuning": best["tuning"]["J"] if best else None, "best_J_evaluation": best["evaluation"]["J"] if best else None,
               "best_params": best["params"] if best else None, "n_candidates": len(evaluated)}
    return HandlerResult(conditions=conditions, metrics=metrics)


# --------------------------------------------------------------------------- rl_training
RL_ENVS = {"SPADE/Buck-v0": "buck", "SPADE/BuckSurrogate-v0": "buck", "SPADE/Vsc-v0": "vsc"}


def _validate_rl(spec: ExperimentSpec) -> None:
    from driveflow.envs import reward as rw

    p = spec.params
    if p.get("env_id") not in RL_ENVS:
        raise SpadeError(ErrorCode.INVALID_SPEC, f"rl_training supports {sorted(RL_ENVS)}, got {p.get('env_id')!r}")
    if spec.algorithm not in ("dqn", "ddpg"):
        raise SpadeError(ErrorCode.INVALID_SPEC, f"rl_training needs algorithm 'dqn' or 'ddpg', got {spec.algorithm!r}")
    if spec.algorithm == "dqn":
        n = p.get("discrete_levels")
        if not (isinstance(n, int) and 2 <= n <= 41):
            raise SpadeError(ErrorCode.INVALID_SPEC, "DQN on a continuous action space needs discrete_levels (2..41 per action dimension)")
        if RL_ENVS[p["env_id"]] == "vsc" and n > 11:
            raise SpadeError(ErrorCode.INVALID_SPEC, "the VSC action is 2-D: at most 11 levels per axis (121 actions)")
    if spec.seed is None:
        raise SpadeError(ErrorCode.INVALID_SPEC, "rl_training needs a seed")
    eps = p.get("episodes")
    if not (isinstance(eps, int) and 1 <= eps <= 2000):
        raise SpadeError(ErrorCode.INVALID_SPEC, "episodes must be an int in [1, 2000]")
    try:
        rw.parse_terms(p.get("reward_terms"))
    except (TypeError, ValueError) as exc:
        raise SpadeError(ErrorCode.INVALID_SPEC, f"reward_terms: {exc}") from None


def rl_training(ctx: JobContext) -> HandlerResult:
    import json

    import gymnasium as gym

    import driveflow.envs  # noqa: F401 -- registers SPADE/*
    from driveflow.control.rl import benchmark as bm
    from driveflow.control.rl.common import ActionDiscretizer, run_episode
    from driveflow.control.rl.ddpg import DDPGAgent, DDPGConfig
    from driveflow.control.rl.dqn import DQNAgent, DQNConfig
    from driveflow.control.rl.policy_controller import PolicyController

    spec, p = ctx.spec, ctx.spec.params
    env_kwargs = {**(p.get("env_kwargs") or {}), "reward_terms": p.get("reward_terms")}
    env = gym.make(p["env_id"], **env_kwargs)
    u = env.unwrapped
    obs_dim = int(np.prod(env.observation_space.shape))
    eval_every = int(p.get("eval_every", 10))
    seed = int(spec.seed)
    if spec.algorithm == "dqn":
        disc = ActionDiscretizer.grid(env.action_space.low, env.action_space.high, int(p["discrete_levels"]))
        agent = DQNAgent(obs_dim, disc, DQNConfig(episodes=p["episodes"], eval_every=eval_every), seed=seed)
    else:
        agent = DDPGAgent(obs_dim, env.action_space.low, env.action_space.high, DDPGConfig(episodes=p["episodes"], eval_every=eval_every), seed=seed)
    conditions = []

    class _Stop(Exception):
        pass

    def callback(ep, h):
        conditions.append({"index": ep - 1, "condition": {"episode": ep}, "status": "ok",
                           "metrics": {"reward": h.episode_reward[-1], "length": h.episode_length[-1], "termination": h.episode_violation[-1], "loss": h.loss[-1]}})
        ctx.progress(ep, p["episodes"], f"episode {ep}/{p['episodes']}")
        if ctx.cancel_requested():
            raise _Stop

    def checkpoint(ep, ag):
        ctx.write_artifact(f"checkpoints/policy_ep{ep:04d}.json", json.dumps(ag.policy_dict()).encode())
        trace = run_episode(env, ag.act, seed=1001, record=True)
        stride = max(1, len(trace["trace"]) // 400)
        ctx.write_artifact(f"playback/eval_ep{ep:04d}.json", json.dumps({"episode": ep, "seed": 1001, "reward": trace["reward"], "termination": trace["termination"],
                                                                          "stride": stride, "trace": trace["trace"][::stride]}).encode())

    try:
        h = agent.train(env, callback=callback, checkpoint=checkpoint)
    except _Stop:
        conditions.append({"index": len(conditions), "condition": {"episode": len(conditions) + 1}, "status": "skipped", "error": "Cancelled before the remaining episodes."})
        return HandlerResult(conditions=conditions, metrics={"algorithm": spec.algorithm, "env_id": p["env_id"], "episodes_done": len(conditions) - 1})
    policy = agent.policy_dict()
    ctx.write_artifact("policy.json", json.dumps(policy).encode())
    ctx.write_artifact("history.json", h.to_json().encode())
    # benchmark on the plant's grid, same env (limits, timestep, reward terms) for every controller
    pc = PolicyController(policy)
    kind = RL_ENVS[p["env_id"]]
    controllers = {f"RL ({spec.algorithm.upper()})": (lambda e: pc.control)}
    if kind == "buck":
        controllers["PI (BuckVoltageController)"] = bm.buck_pi
        grid = bm.BUCK_GRID
    else:
        controllers["DPC v3"] = lambda e: bm.vsc_dpc(e, "v3")
        controllers["DPC v4"] = lambda e: bm.vsc_dpc(e, "v4")
        grid = bm.VSC_GRID
    results = bm.run_grid(env, controllers, grid)
    bench = {"env_id": p["env_id"], "grid": [dict(g) for g in grid], "results": results, "summary": bm.summarize(results), "reward_terms": p.get("reward_terms"),
             "status": "policy candidate -- not a default"}
    ctx.write_artifact("benchmark.json", json.dumps(bench, default=float).encode())
    s = bench["summary"]
    rl_name = f"RL ({spec.algorithm.upper()})"
    metrics = {"algorithm": spec.algorithm, "env_id": p["env_id"], "episodes": p["episodes"], "final_eval_reward": h.eval_reward[-1] if h.eval_reward else None,
               "rl_mean_reward_grid": s[rl_name]["mean_reward"], "rl_median_tracking_rmse": s[rl_name]["median_tracking_rmse"], "rl_terminated": s[rl_name]["n_terminated"],
               "baselines": {k: v for k, v in s.items() if k != rl_name}}
    env.close()
    return HandlerResult(conditions=conditions, metrics=metrics)


# --------------------------------------------------------------------------- selftest.steps
def _validate_selftest(spec: ExperimentSpec) -> None:
    steps, step_s = spec.params.get("steps", 3), spec.params.get("step_s", 0.1)
    if not (isinstance(steps, int) and 1 <= steps <= 10_000) or not (isinstance(step_s, (int, float)) and 0 <= step_s <= 60):
        raise SpadeError(ErrorCode.INVALID_SPEC, "selftest.steps needs 1 <= steps <= 10000 and 0 <= step_s <= 60")


def selftest_steps(ctx: JobContext) -> HandlerResult:
    steps, step_s = ctx.spec.params.get("steps", 3), ctx.spec.params.get("step_s", 0.1)
    fail_at = set(ctx.spec.params.get("fail_steps", ()))
    conditions = []
    for i in range(steps):
        if ctx.cancel_requested():
            conditions += [{"index": j, "condition": {"step": j}, "status": "skipped", "error": "Cancelled before this step started."} for j in range(i, steps)]
            break
        ctx.progress(i, steps, f"step {i + 1}/{steps}")
        time.sleep(step_s)
        if i in fail_at:
            conditions.append({"index": i, "condition": {"step": i}, "status": "failed", "error": "requested failure (selftest)"})
        else:
            conditions.append({"index": i, "condition": {"step": i}, "status": "ok", "metrics": {"step": i}})
    ctx.progress(sum(c["status"] != "skipped" for c in conditions), steps, "finished")
    ctx.write_artifact("selftest.txt", f"{len(conditions)} steps recorded\n".encode())
    return HandlerResult(conditions=conditions, metrics={"steps": steps})


HANDLERS = {
    "scenario_sweep": HandlerInfo(scenario_sweep, _validate_scenario_sweep, "Scenario sweep (datagen.runner)"),
    "plecs_sweep": HandlerInfo(plecs_sweep, _validate_plecs_sweep, "PLECS RPC sweep (dataset)", engine="plecs_rpc"),
    "controller_tuning": HandlerInfo(controller_tuning, _validate_tuning, "Controller tuning (PSO / BO / NSGA-II)"),
    "rl_training": HandlerInfo(rl_training, _validate_rl, "RL training + benchmark (DQN / DDPG)"),
    "selftest.steps": HandlerInfo(selftest_steps, _validate_selftest, "Worker self-test (no physics)", engine="selftest"),
}


def handler_for(kind: str) -> HandlerInfo:
    try:
        return HANDLERS[kind]
    except KeyError:
        raise SpadeError(ErrorCode.UNKNOWN_JOB_KIND, f"unknown job kind {kind!r}; known: {sorted(HANDLERS)}") from None
