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
- ``selftest.steps``: no physics; sleeps ``step_s`` per step. Used by the tests to exercise the
  worker (cancel, kill, recover) deterministically, and by "Check worker" in the GUI.
"""

from __future__ import annotations

import dataclasses
import io
import math
import time
from typing import Callable

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
    "selftest.steps": HandlerInfo(selftest_steps, _validate_selftest, "Worker self-test (no physics)", engine="selftest"),
}


def handler_for(kind: str) -> HandlerInfo:
    try:
        return HANDLERS[kind]
    except KeyError:
        raise SpadeError(ErrorCode.UNKNOWN_JOB_KIND, f"unknown job kind {kind!r}; known: {sorted(HANDLERS)}") from None
