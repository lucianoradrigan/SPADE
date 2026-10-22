"""Custom classifier training from the GUI (AI Diagnostics → Train a model).

The promoted classifiers were trained by scripts on fixed grids
(experiments/generate_diagnosis_grid_dataset.py). This module lets a user define their own grid
with the same options the simulation pages expose, simulate it and train a classifier on it,
without touching the registry:

    request (a JSON-safe dict, see validate_request)
      -> build_conditions()   one Scenario per run: classes x controllers x setpoints x severities
                              x runs per cell, with per-run noise and plant drift drawn from one RNG
      -> simulate()           run_scenario per condition (progress + cancel hooks)
      -> train()              models.training._train_classifier on the chosen channels, with
                              calibration and an OOD detector; artifacts under
                              configs/classifiers/custom/<name>/<date>_runNN/

A trained model is a CANDIDATE: it can be evaluated in AI Diagnostics and used as the starting
point of Transfer Learning, but it is never promoted here (promotion stays a separate, confirmed
step: experiments/promote_run.py or the Model Registry page).

Domains: dc_motor (bearing faults, built-in and user-defined fault orders) and vsc_dpc (synthetic
filter-capacitor degradation). The PMSM and DC-DC plants have no fault model -- every record is
labelled "normal" -- so there is nothing to classify there.
"""

from __future__ import annotations

import copy
import datetime as dt
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from driveflow.ai.audit_findings import ensure_synthetic_class_finding
from driveflow.ai.registry import REPO_ROOT
from driveflow.datagen.scenario import CALIBRATED_MECHANICAL_SEVERITY
from driveflow.sim.vsc_system import MIN_STABLE_LOAD_RESISTANCE_OHM

HEALTHY = "normal"
DC_BUILTIN_FAULTS = ("outer_race", "inner_race", "ball", "cage")
VSC_DEGRADED = "capacitor_degraded"
DOMAINS = ("dc_motor", "vsc_dpc")
#: Upper bound on simulated runs per request (about 10-20 min of simulation on a laptop at 1 s per run).
MAX_RUNS = 600
#: Each class needs runs in train, validation and test (grouped split by run).
MIN_RUNS_PER_CLASS = 3
CUSTOM_ROOT = REPO_ROOT / "configs" / "classifiers" / "custom"
DATASET_ROOT = REPO_ROOT / "data" / "custom"
PRESETS = {
    "pc_server_5class_w2048": "PC server CNN-SE, 2048-sample window (the promoted dc_motor architecture)",
    "pc_server": "PC server CNN-SE, 512-sample window",
    "rpi5_resnet1d_se_5class_w2048": "Raspberry Pi 5 ResNet1D-SE, 2048-sample window",
    "esp32_dscnn": "ESP32 depthwise-separable CNN, 512-sample window",
}
DC_CHANNELS = ("acc_x", "acc_y", "acc_z", "current_r", "rpm", "torque_nm")
VSC_CHANNELS = ("vc_real", "vc_imag", "i_f_real", "i_f_imag")
NOMINAL_ELECTRICAL_SEVERITY = 8.0
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


class RequestError(ValueError):
    """The request cannot be run as given; the message says what to change."""


# --------------------------------------------------------------------------------- request
def dc_class(label: str, fault_type: str | None = None, fault_order_override: float | None = None, extra_faults: list | None = None,
             mechanical_severity: float | None = None) -> dict:
    """One class of a dc_motor request. Built-in faults use the calibrated mechanical severity
    unless one is given; a user-defined fault (an order override) must give one."""
    if fault_type is None:
        return {"label": HEALTHY, "fault_type": None}
    mech = mechanical_severity if mechanical_severity is not None else CALIBRATED_MECHANICAL_SEVERITY.get(fault_type)
    return {"label": label, "fault_type": fault_type, "fault_order_override": fault_order_override,
            "extra_faults": [list(e) for e in (extra_faults or [])], "mechanical_severity": mech}


def default_request(domain: str) -> dict:
    """A small, valid request: the shape the GUI fills in."""
    if domain == "dc_motor":
        return {
            "domain": "dc_motor", "name": "my_dc_classifier",
            "classes": [dc_class(HEALTHY), dc_class("outer_race", "outer_race"), dc_class("inner_race", "inner_race")],
            "operating": {"controllers": ["PI"], "control_mode": "speed", "setpoints": [100.0, 150.0, 200.0], "severity_scales": [0.5, 1.0, 2.0],
                          "electrical_severity_nm": NOMINAL_ELECTRICAL_SEVERITY, "noise_max_pct": 50.0, "noise_free_share": 1 / 3,
                          "motor": None, "drift": {"psi_e": [0.9, 1.1], "j_rotor": [0.8, 1.5]}, "pi_setpoint_weight": None,
                          "duration_s": 1.5, "skip_start_s": 0.5, "runs_per_cell": 1, "seed": 0},
            "model": {"preset": "pc_server_5class_w2048", "input_window": 2048, "channels": list(DC_CHANNELS), "epochs": 15, "batch_size": 32,
                      "cap_per_class": 1500, "keep_rpm_level": True, "fit_ood": True, "val_frac": 0.2, "test_frac": 0.2, "seed": 42},
        }
    if domain == "vsc_dpc":
        return {
            "domain": "vsc_dpc", "name": "my_vsc_classifier",
            "classes": [{"label": HEALTHY}, {"label": VSC_DEGRADED, "capacitor_scale": [0.5, 0.8], "esr_ohm": [0.0, 0.2]}],
            "operating": {"load_resistance_ohm": [6.0, 8.0064, 12.0], "reference_magnitude_v": [50.0], "reference_omega_rad_s": [2 * np.pi * 50],
                          "voltage_limit": False, "duration_s": 0.3, "skip_start_s": 0.05, "runs_per_cell": 2, "seed": 0},
            "model": {"preset": "pc_server", "input_window": 512, "channels": list(VSC_CHANNELS), "epochs": 15, "batch_size": 32,
                      "cap_per_class": 1500, "keep_rpm_level": False, "fit_ood": True, "val_frac": 0.2, "test_frac": 0.2, "seed": 42},
        }
    raise RequestError(f"domain must be one of {DOMAINS}, got {domain!r}")


def _floats(name: str, values, lo: float, hi: float) -> list:
    try:
        out = [float(v) for v in values]
    except (TypeError, ValueError):
        raise RequestError(f"{name} must be a list of numbers") from None
    if not out:
        raise RequestError(f"{name} needs at least one value")
    bad = [v for v in out if not (lo <= v <= hi)]
    if bad:
        raise RequestError(f"{name}: {bad} outside [{lo:g}, {hi:g}]")
    return sorted(set(out))


def _range(name: str, pair, lo: float, hi: float) -> list:
    if pair is None:
        return None
    a, b = _floats(name, pair, lo, hi) * (2 if len(set(pair)) == 1 else 1)
    return [a, b]


def validate_request(req: dict) -> dict:
    """Checks a request and returns a normalised copy. Raises RequestError with a message the GUI
    can show as is."""
    req = copy.deepcopy(req)
    domain = req.get("domain")
    if domain not in DOMAINS:
        raise RequestError(f"domain must be one of {DOMAINS}, got {domain!r}")
    if not _NAME_RE.match(str(req.get("name", ""))):
        raise RequestError("name: 1-40 characters, letters, digits, '_' or '-'")
    classes = req.get("classes") or []
    labels = [c.get("label") for c in classes]
    if len(classes) < 2:
        raise RequestError("select at least two classes (a classifier needs something to tell apart)")
    if len(set(labels)) != len(labels):
        raise RequestError(f"class labels must be unique, got {labels}")
    op, m = req.setdefault("operating", {}), req.setdefault("model", {})
    op["duration_s"] = float(op.get("duration_s", 1.0))
    op["skip_start_s"] = float(op.get("skip_start_s", 0.0))
    if not (0.05 <= op["duration_s"] <= 10.0):
        raise RequestError("duration_s must lie in [0.05, 10] s")
    if not (0.0 <= op["skip_start_s"] < op["duration_s"]):
        raise RequestError("skip_start_s must lie in [0, duration_s)")
    op["runs_per_cell"] = int(op.get("runs_per_cell", 1))
    if not (1 <= op["runs_per_cell"] <= 50):
        raise RequestError("runs_per_cell must lie in [1, 50]")
    op["seed"] = int(op.get("seed", 0))

    if domain == "dc_motor":
        for c in classes:
            if c.get("fault_type") is None:
                if c["label"] != HEALTHY:
                    raise RequestError(f"class {c['label']!r} has no fault_type; only {HEALTHY!r} is fault-free")
            elif c["fault_type"] not in DC_BUILTIN_FAULTS and c.get("fault_order_override") is None:
                raise RequestError(f"class {c['label']!r}: a user-defined fault needs its characteristic order (fault_order_override)")
            elif c.get("mechanical_severity") is None:
                raise RequestError(f"class {c['label']!r} needs a mechanical severity")
        op["controllers"] = list(dict.fromkeys(op.get("controllers") or ["PI"]))
        if not set(op["controllers"]) <= {"PI", "MPC"}:
            raise RequestError("controllers must be PI and/or MPC")
        if op.get("control_mode", "speed") not in ("speed", "torque"):
            raise RequestError("control_mode must be 'speed' or 'torque'")
        op["control_mode"] = op.get("control_mode", "speed")
        op["setpoints"] = _floats("setpoints", op.get("setpoints"), *((10.0, 350.0) if op["control_mode"] == "speed" else (0.0, 38.0)))
        op["severity_scales"] = _floats("severity_scales", op.get("severity_scales") or [1.0], 0.05, 5.0)
        op["electrical_severity_nm"] = float(op.get("electrical_severity_nm", NOMINAL_ELECTRICAL_SEVERITY))
        if not (0.0 <= op["electrical_severity_nm"] <= 20.0):
            raise RequestError("electrical_severity_nm must lie in [0, 20] N·m")
        op["noise_max_pct"] = float(op.get("noise_max_pct", 0.0))
        op["noise_free_share"] = float(op.get("noise_free_share", 1.0))
        if not (0.0 <= op["noise_max_pct"] <= 100.0 and 0.0 <= op["noise_free_share"] <= 1.0):
            raise RequestError("noise_max_pct must lie in [0, 100] and noise_free_share in [0, 1]")
        drift = op.get("drift") or {}
        op["drift"] = {k: _range(f"drift.{k}", v, 0.5, 2.0) for k, v in drift.items() if v is not None}
        if not set(op["drift"]) <= {"psi_e", "j_rotor", "r_a"}:
            raise RequestError("drift may vary psi_e, j_rotor and r_a only")
        default_channels, allowed_channels = DC_CHANNELS, set(DC_CHANNELS) | {"current_s", "current_t"}
    else:
        for c in classes:
            if c["label"] == HEALTHY:
                continue
            if c["label"] != VSC_DEGRADED:
                raise RequestError(f"vsc_dpc classes are {HEALTHY!r} and {VSC_DEGRADED!r}, got {c['label']!r}")
            c["capacitor_scale"] = _range("capacitor_scale", c.get("capacitor_scale") or [1.0, 1.0], 0.05, 1.0)
            c["esr_ohm"] = _range("esr_ohm", c.get("esr_ohm") or [0.0, 0.0], 0.0, 2.0)
            if c["capacitor_scale"] == [1.0, 1.0] and c["esr_ohm"] == [0.0, 0.0]:
                raise RequestError("capacitor_degraded needs a capacitor scale below 1 or a non-zero ESR")
        op["load_resistance_ohm"] = _floats("load_resistance_ohm", op.get("load_resistance_ohm"), MIN_STABLE_LOAD_RESISTANCE_OHM, 50.0)
        op["reference_magnitude_v"] = _floats("reference_magnitude_v", op.get("reference_magnitude_v") or [50.0], 10.0, 150.0)
        op["reference_omega_rad_s"] = _floats("reference_omega_rad_s", op.get("reference_omega_rad_s") or [2 * np.pi * 50], 50.0, 700.0)
        op["voltage_limit"] = bool(op.get("voltage_limit", False))
        default_channels, allowed_channels = VSC_CHANNELS, set(VSC_CHANNELS)

    if m.get("preset", "pc_server") not in PRESETS:
        raise RequestError(f"model.preset must be one of {sorted(PRESETS)}")
    m["preset"] = m.get("preset", "pc_server")
    m["input_window"] = int(m.get("input_window", 512))
    if not (64 <= m["input_window"] <= 8192):
        raise RequestError("model.input_window must lie in [64, 8192] samples")
    usable_s = op["duration_s"] - op["skip_start_s"]
    if m["input_window"] * 1e-4 > usable_s:
        raise RequestError(f"one window is {m['input_window'] * 1e-4:.3f} s but each run keeps only {usable_s:.3f} s after skip_start_s -- lengthen the runs or shorten the window")
    m["channels"] = list(dict.fromkeys(m.get("channels") or default_channels))
    bad = sorted(set(m["channels"]) - allowed_channels)
    if bad:
        raise RequestError(f"channels {bad} are not available for {domain}; choose from {sorted(allowed_channels)}")
    for k, lo, hi, default in (("epochs", 1, 200, 15), ("batch_size", 4, 256, 32), ("cap_per_class", 50, 20000, 1500), ("seed", 0, 2**31 - 1, 42)):
        m[k] = int(m.get(k, default))
        if not (lo <= m[k] <= hi):
            raise RequestError(f"model.{k} must lie in [{lo}, {hi}]")
    m["val_frac"], m["test_frac"] = float(m.get("val_frac", 0.2)), float(m.get("test_frac", 0.2))
    if not (0.05 <= m["val_frac"] <= 0.4 and 0.05 <= m["test_frac"] <= 0.4):
        raise RequestError("val_frac and test_frac must lie in [0.05, 0.4]")
    m["keep_rpm_level"] = bool(m.get("keep_rpm_level", False)) and "rpm" in m["channels"]
    m["fit_ood"] = bool(m.get("fit_ood", True))

    n = len(build_conditions(req, validated=True))
    per_class = n // len(classes)
    if n > MAX_RUNS:
        raise RequestError(f"{n} runs requested; the limit is {MAX_RUNS}. Reduce setpoints, severities, controllers or runs per cell")
    if per_class < MIN_RUNS_PER_CLASS:
        raise RequestError(f"{per_class} run(s) per class; each class needs at least {MIN_RUNS_PER_CLASS} (train, validation and test are split by run)")
    return req


# ------------------------------------------------------------------------------ conditions
def build_conditions(req: dict, validated: bool = False) -> list:
    """One dict per simulated run: {"label", "scenario": Scenario kwargs, "grid": what varied}.
    Every class gets the same operating cells, so the classes are balanced by construction;
    noise and plant drift are drawn per run from one RNG seeded with operating.seed."""
    if not validated:
        req = validate_request(req)
    op = req["operating"]
    rng = np.random.default_rng(op["seed"])
    out = []
    if req["domain"] == "dc_motor":
        cells = itertools.product(op["controllers"], op["setpoints"], op["severity_scales"], range(op["runs_per_cell"]))
        for controller, setpoint, scale, rep in cells:
            for c in req["classes"]:
                i = len(out)
                noisy = rng.random() >= op["noise_free_share"]
                e_noise, m_noise = (float(rng.uniform(0, op["noise_max_pct"])), float(rng.uniform(0, op["noise_max_pct"]))) if noisy else (0.0, 0.0)
                motor = dict(op.get("motor") or {})
                for p, (lo, hi) in op["drift"].items():
                    base = motor.get(p, _NOMINAL_MOTOR[p])
                    motor[p] = base * float(rng.uniform(lo, hi))
                kw = {"scenario_id": f"custom_{c['label']}_{i:04d}", "controller_type": controller, "duration_s": op["duration_s"], "seed": 1000 + i,
                      "electrical_noise_pct": e_noise, "mechanical_noise_pct": m_noise, "motor_parameter_overrides": motor or None}
                if op["control_mode"] == "speed":
                    kw["omega_ref_rad_s"] = setpoint
                else:
                    kw["torque_ref_nm"] = setpoint
                if controller == "PI" and op.get("pi_setpoint_weight") is not None:
                    kw["pi_setpoint_weight"] = float(op["pi_setpoint_weight"])
                if c.get("fault_type") is not None:
                    kw.update(fault_type=c["fault_type"], electrical_severity=op["electrical_severity_nm"] * scale,
                              mechanical_severity=float(c["mechanical_severity"]) * scale)
                    if c.get("fault_order_override") is not None:
                        kw["fault_order_override"] = float(c["fault_order_override"])
                    if c.get("extra_faults"):
                        kw["extra_faults"] = [(str(n), None if o is None else float(o)) for n, o in c["extra_faults"]]
                out.append({"label": c["label"], "scenario": kw,
                            "grid": {"controller": controller, "setpoint": setpoint, "severity_scale": scale if c.get("fault_type") else 0.0,
                                     "electrical_noise_pct": e_noise, "mechanical_noise_pct": m_noise, **{f"motor_{k}": v for k, v in motor.items()}}})
    else:
        cells = itertools.product(op["load_resistance_ohm"], op["reference_magnitude_v"], op["reference_omega_rad_s"], range(op["runs_per_cell"]))
        for r_ohm, mag, omega, rep in cells:
            for c in req["classes"]:
                i = len(out)
                kw = {"scenario_id": f"custom_{c['label']}_{i:04d}", "controller_type": "DPC", "plant_config_id": "vsc_dpc_v1", "duration_s": op["duration_s"],
                      "seed": 1000 + i, "load_resistance_ohm": r_ohm, "reference_magnitude_v": mag, "reference_omega_rad_s": omega}
                if op["voltage_limit"]:
                    from driveflow.sim.vsc_params import UPSTREAM_CONVERTER_LIMIT_V

                    kw["vsc_voltage_limit_v"] = float(UPSTREAM_CONVERTER_LIMIT_V)
                grid = {"load_resistance_ohm": r_ohm, "reference_magnitude_v": mag, "reference_omega_rad_s": omega}
                if c["label"] == VSC_DEGRADED:
                    kw["vsc_capacitor_scale"] = float(rng.uniform(*c["capacitor_scale"]))
                    kw["vsc_esr_ohm"] = float(rng.uniform(*c["esr_ohm"]))
                    if kw["vsc_capacitor_scale"] == 1.0 and kw["vsc_esr_ohm"] == 0.0:
                        kw["vsc_esr_ohm"] = 1e-3  # never an unlabelled-healthy draw inside the degraded class
                    grid.update(capacitor_scale=kw["vsc_capacitor_scale"], esr_ohm=kw["vsc_esr_ohm"])
                out.append({"label": c["label"], "scenario": kw, "grid": grid})
    return out


_NOMINAL_MOTOR = {"psi_e": 0.165, "j_rotor": 0.025, "r_a": 0.016}


def summary(req: dict) -> dict:
    """What the GUI shows before submitting: run count, simulated seconds, runs per class."""
    req = validate_request(req)
    conds = build_conditions(req, validated=True)
    per_class = pd.Series([c["label"] for c in conds]).value_counts().to_dict()
    sim_s = len(conds) * req["operating"]["duration_s"]
    return {"n_runs": len(conds), "simulated_s": sim_s, "runs_per_class": per_class, "classes": [c["label"] for c in req["classes"]],
            "estimated_single_process_s": sim_s * SECONDS_PER_SIMULATED_S[req["domain"]]}


# ------------------------------------------------------------------------------- simulate
def simulate_one(condition: dict) -> pd.DataFrame:
    """One run; its label is set from the request's class (a user-defined fault keeps the label
    the user gave it, whatever Scenario.label would derive)."""
    from driveflow.datagen import Scenario, run_scenario
    from driveflow.datagen.export_parquet import records_to_dataframe

    df = records_to_dataframe(run_scenario(Scenario(**condition["scenario"])))
    df["label"] = condition["label"]
    for k, v in condition["grid"].items():
        df[f"grid_{k}"] = v
    return df


def _outcome(i: int, cond: dict, status: str, error: str | None = None) -> dict:
    out = {"index": i, "condition": {**cond["grid"], "label": cond["label"]}, "status": status}
    if error:
        out["error"] = error
    return out


def simulate(conditions: list, progress=None, cancelled=None, workers: int = 1) -> tuple[pd.DataFrame | None, list]:
    """Runs every condition; returns (dataset, per-condition outcomes, in condition order). A failed
    run is kept in the outcomes with its error and left out of the dataset; nothing is
    interpolated. workers > 1 runs the simulations in that many processes (spawned, so the parent's
    TensorFlow state is not forked); cancellation then stops submitting new runs."""
    results: dict = {}
    n = len(conditions)

    def record(i, df=None, error=None):
        results[i] = (df, _outcome(i, conditions[i], "ok" if error is None else "failed", error))
        if progress is not None:
            progress(len(results), n, f"simulated {len(results)}/{n} runs")

    stopped_at = None
    if workers <= 1:
        for i, cond in enumerate(conditions):
            if cancelled is not None and cancelled():
                stopped_at = i
                break
            try:
                record(i, simulate_one(cond))
            except Exception as exc:  # one failed run never hides the others
                record(i, error=f"{type(exc).__name__}: {exc}")
    else:
        import concurrent.futures as cf
        import multiprocessing as mp

        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            pending, next_i = {}, 0
            while next_i < n or pending:
                while next_i < n and len(pending) < 2 * workers and stopped_at is None:
                    if cancelled is not None and cancelled():
                        stopped_at = next_i
                        break
                    pending[pool.submit(simulate_one, conditions[next_i])] = next_i
                    next_i += 1
                if not pending:
                    break
                done, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                for fut in done:
                    i = pending.pop(fut)
                    try:
                        record(i, fut.result())
                    except Exception as exc:
                        record(i, error=f"{type(exc).__name__}: {exc}")
    if stopped_at is not None:
        for j in range(n):
            if j not in results:
                results[j] = (None, _outcome(j, conditions[j], "skipped", "Cancelled before this run started."))
    frames = [results[i][0] for i in sorted(results) if results[i][0] is not None]
    return (pd.concat(frames, ignore_index=True) if frames else None), [results[i][1] for i in sorted(results)]


#: Wall-clock seconds per simulated second, measured on the development laptop (12 cores), 2026-10-21:
#: dc_motor 3.7 (72 runs of 1.5 s in 398 s with 4 processes, docs/audit/ai/custom_training_tl.md;
#: 9 runs of 0.6 s took 16 s in one process, i.e. ~3 -- extra processes gave no measurable speed-up);
#: vsc_dpc 14 (6 runs of 0.2 s in 17 s, one process). An order of magnitude only.
SECONDS_PER_SIMULATED_S = {"dc_motor": 3.7, "vsc_dpc": 14.0}


# ---------------------------------------------------------------------------------- train
def _model_config(req: dict) -> dict:
    preset = REPO_ROOT / "configs" / "classifiers" / f"{req['model']['preset']}.yaml"
    raw = yaml.safe_load(preset.read_text())
    raw.update(domain=req["domain"], tier="pc", input_window=req["model"]["input_window"], num_classes=len(req["classes"]))
    return raw


def next_run_dir(name: str, root: Path | None = None) -> Path:
    root = Path(root or CUSTOM_ROOT)
    date_str = dt.date.today().isoformat()
    n = 1
    while (root / name / f"{date_str}_run{n:02d}").exists():
        n += 1
    return root / name / f"{date_str}_run{n:02d}"


def train(req: dict, df: pd.DataFrame, root: Path | None = None, dataset_root: Path | None = None, on_epoch_end=None) -> Path:
    """Trains the requested classifier on df (simulate()'s output) and writes the run directory:
    config.yaml, model.weights.h5, metrics.json, run_metadata.json, training_request.json. The
    dataset is saved as Parquet under data/custom/ (gitignored) so it can be reused."""
    from driveflow.ai.provenance import file_sha256, run_metadata, set_global_seed
    from driveflow.models.classifiers.schemas import load_classifier_config
    from driveflow.models.common import RPM_LEVEL_SCALING
    from driveflow.models.training import _train_classifier

    req = validate_request(req)
    root, dataset_root = Path(root or CUSTOM_ROOT), Path(dataset_root or DATASET_ROOT)
    m = req["model"]
    skip = req["operating"]["skip_start_s"]
    if skip > 0:
        df = df[df["timestamp_s"] >= skip].reset_index(drop=True)
    run_dir = next_run_dir(req["name"], root)
    run_dir.mkdir(parents=True)
    config_path = run_dir / "config.yaml"
    config_path.write_text("# Generated by AI Diagnostics -> Train a model (driveflow.ai.custom_training).\n"
                           f"# Architecture from configs/classifiers/{m['preset']}.yaml; window and classes from the request.\n"
                           + yaml.safe_dump(_model_config(req), sort_keys=False))
    config = load_classifier_config(config_path)
    seed_info = set_global_seed(m["seed"])
    try:
        model, metrics = _train_classifier(
            config, df, m["epochs"], m["batch_size"], m["seed"], level_scaling=RPM_LEVEL_SCALING if m["keep_rpm_level"] else None,
            cap_per_class=m["cap_per_class"], channels=m["channels"], val_frac=m["val_frac"], test_frac=m["test_frac"],
            fit_ood_detector=m["fit_ood"], on_epoch_end=on_epoch_end,
        )
    except Exception:
        for p in run_dir.iterdir():
            p.unlink()
        run_dir.rmdir()
        raise
    dataset_root.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_root / f"{req['name']}_{run_dir.name}.parquet"
    df.to_parquet(dataset_path, index=False)
    metrics.update({
        "origin": "SIMULATED", "status": "candidate (not promoted)", "trained_from": "ai_diagnostics.custom_training",
        "class_definitions": req["classes"], "operating_grid": req["operating"],
        "synthetic_only_classes": [c["label"] for c in req["classes"] if c["label"] in ("ball", "cage", VSC_DEGRADED) or (c.get("fault_type") and c["fault_type"] not in DC_BUILTIN_FAULTS)],
    })
    model.save_weights(run_dir / "model.weights.h5")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    ensure_synthetic_class_finding(run_dir, metrics["classes"])
    (run_dir / "training_request.json").write_text(json.dumps(req, indent=2, default=float))
    (run_dir / "run_metadata.json").write_text(json.dumps({**run_metadata(), "seed": seed_info, "epochs": m["epochs"], "batch_size": m["batch_size"],
                                                           "dataset_path": str(dataset_path), "dataset_sha256": file_sha256(dataset_path),
                                                           "n_runs": int(df["source_file"].nunique())}, indent=2, default=str))
    return run_dir


# ------------------------------------------------------------------------------- catalogue
def list_model_runs(domain: str, kind: str = "classifier", root: Path | None = None) -> list[dict]:
    """Every run directory of `kind` ("classifier" or "forecaster") for `domain` on this machine
    (promoted, historical, custom-trained, imported, transfer-learning candidates):
    [{"run_dir", "label", "classes", "channels", "macro_f1", "origin"}], newest first."""
    root = Path(root) if root is not None else REPO_ROOT / "configs" / ("classifiers" if kind == "classifier" else "regressors")
    out = []
    for metrics_path in root.glob("**/metrics.json"):
        run_dir = metrics_path.parent
        if not ((run_dir / "model.weights.h5").exists() or (run_dir / "model.keras").exists()) or not (run_dir / "config.yaml").exists():
            continue
        try:
            mt = json.loads(metrics_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if mt.get("kind") != kind or mt.get("domain") != domain or mt.get("tier", "pc") != "pc":
            continue
        parts = run_dir.relative_to(root).parts
        origin = ("transfer learning" if any(p.endswith("_tl") for p in parts) else "custom" if parts[0] == "custom"
                  else "imported" if parts[0] == "external" else "script")
        out.append({"run_dir": run_dir, "label": "/".join(parts), "classes": mt.get("classes"), "channels": mt.get("channels"),
                    "macro_f1": mt.get("test_macro_f1"), "rmse": mt.get("test_rmse"), "origin": origin, "mtime": metrics_path.stat().st_mtime})
    return sorted(out, key=lambda r: -r["mtime"])


def list_classifier_runs(domain: str) -> list[dict]:
    return list_model_runs(domain, "classifier")
