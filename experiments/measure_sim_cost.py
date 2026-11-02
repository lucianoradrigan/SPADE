"""Round Assistant, task F0.5: CPU seconds per simulated second for each native plant.

For DC + PI, DC + MPC, PMSM (FOC_MTPA), VSC + DPC and the buck: REPEATS runs of DURATION_S
simulated seconds each through datagen.run_scenario, timed with time.process_time() (CPU time of
this process, all threads), after one untimed warm-up run per plant (imports, TensorFlow graph,
caches). Reports median and range.

A measurement of THIS machine, not a guarantee: another CPU, other load or other library versions
give other numbers. Run it on an otherwise idle machine.

    python experiments/measure_sim_cost.py
Writes configs/sim_cost.yaml (read by the future estimate_cost) and
docs/audit/assistant/f0_5_sim_cost.{md,json}.
"""

from __future__ import annotations

import datetime
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import yaml

from driveflow.datagen import Scenario, run_scenario

REPO = Path(__file__).resolve().parents[1]
OUT_YAML = REPO / "configs" / "sim_cost.yaml"
OUT = REPO / "docs" / "audit" / "assistant" / "f0_5_sim_cost"
DURATION_S = 0.5
REPEATS = 5
PLANTS = {
    "dc_perm_ex_v1/PI": dict(controller_type="PI"),
    "dc_perm_ex_v1/MPC": dict(controller_type="MPC"),
    "pmsm_v1/FOC_MTPA": dict(controller_type="FOC_MTPA", plant_config_id="pmsm_v1", pmsm_current_ref_a=150.0),
    "vsc_dpc_v1/DPC": dict(controller_type="DPC", plant_config_id="vsc_dpc_v1"),
    "buck_v1/PI": dict(controller_type="PI", plant_config_id="buck_v1"),
}


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _versions() -> dict:
    import scipy
    import tensorflow as tf

    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "tensorflow": tf.__version__}


def measure(kwargs: dict) -> list[float]:
    run_scenario(Scenario("warmup", duration_s=0.05, **kwargs))
    out = []
    for k in range(REPEATS):
        t0 = time.process_time()
        run_scenario(Scenario(f"cost_{k}", duration_s=DURATION_S, seed=k, **kwargs))
        out.append((time.process_time() - t0) / DURATION_S)
    return out


def main() -> None:
    when = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    machine = {"cpu": _cpu_model(), "logical_cpus": os.cpu_count(), "platform": platform.platform(), **_versions()}
    rows = {}
    for name, kwargs in PLANTS.items():
        samples = measure(kwargs)
        rows[name] = {"median_cpu_s_per_sim_s": statistics.median(samples), "min": min(samples), "max": max(samples), "samples": samples}
        print(f"{name}: median {rows[name]['median_cpu_s_per_sim_s']:.2f} CPU s per simulated s ({min(samples):.2f}-{max(samples):.2f})", flush=True)
    note = "Measured on the machine below; not a guarantee. Another CPU, background load or library versions change these numbers."
    data = {"measured": when, "command": "python experiments/measure_sim_cost.py", "note": note, "duration_s_per_run": DURATION_S, "repeats": REPEATS,
            "timer": "time.process_time (CPU of this process, all threads), after one untimed warm-up run", "machine": machine,
            "cpu_s_per_simulated_s": {k: {kk: round(vv, 4) for kk, vv in v.items() if kk != "samples"} for k, v in rows.items()}}
    OUT_YAML.write_text("# Simulation cost per native plant (round Assistant, task F0.5). " + note + "\n" + yaml.safe_dump(data, sort_keys=False))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.with_suffix(".json").write_text(json.dumps({**data, "samples": {k: v["samples"] for k, v in rows.items()}}, indent=2) + "\n")
    lines = [
        "# F0.5 — simulation cost per plant",
        "",
        f"**{note}**",
        "",
        f"- **Command:** `{data['command']}` ({when}). Raw values: [`f0_5_sim_cost.json`](f0_5_sim_cost.json); machine-readable copy for the future `estimate_cost`: `configs/sim_cost.yaml`.",
        f"- **Machine:** {machine['cpu']}, {machine['logical_cpus']} logical CPUs, {machine['platform']}; Python {machine['python']}, NumPy {machine['numpy']}, SciPy {machine['scipy']}, TensorFlow {machine['tensorflow']}.",
        f"- **Method:** {REPEATS} runs × {DURATION_S:g} simulated s per plant through `run_scenario`, `time.process_time()`, one untimed warm-up run first. The machine was otherwise idle (no test suite running).",
        "",
        "| Plant / controller | Median CPU s per simulated s | Range (5 runs) | CPU time for 1 simulated s |",
        "|---|---|---|---|",
    ] + [f"| `{k}` | **{v['median_cpu_s_per_sim_s']:.2f}** | {v['min']:.2f}–{v['max']:.2f} | ≈ {v['median_cpu_s_per_sim_s']:.1f} s |" for k, v in rows.items()] + [
        "",
        "Every native plant runs one control period per 100 µs (10 000 steps per simulated second) in a Python loop; the DC plant also runs the vibration model, the PMSM integrates with 10 Euler substeps, the DPC calls a Keras network per step.",
        "",
        "## Not executed",
        "",
        "- Other machines (Windows, macOS, the Raspberry Pi): no ejecutado. The table is valid for this machine only.",
        "- PLECS and FMU plants: no ejecutado (not native; their cost depends on the external tool and the model).",
    ]
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
