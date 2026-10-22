"""A1 evidence (round PLECS/OM, route a): PLECS sweep → dataset → surrogate → Gym environment, run
end to end in the worker against the FAKE PLECS server (tests/integrations/plecs/fake_plecs.py) in its
excited-buck test-double mode. This validates the MECHANICS of route a, not fidelity to any PLECS
circuit: the "circuit" is SPADE's own averaged buck.

Steps:
1. A plecs_rpc_batch profile with the "Buck, excited duty" manifest (PRBS duty, applied duty recorded).
2. Sweep: Vin {24, 36, 48} V × Rload {3, 7.5, 12} Ω × PRBS seeds {5, 77}, 20 ms per condition, one sample
   per 20 µs (the surrogate's control step), as a plecs_sweep job.
3. surrogate_fit job: both E5 families, split by operating point, default family on validation,
   publication of SPADE/<Profile>Surrogate-v0 and its contract check (check_env).
4. Per condition: open-loop rollout of the surrogate under the duty the circuit applied, against the
   test double's trajectory.

    python experiments/report_a1_surrogate_from_plecs.py
Writes docs/audit/plecs_om/a1_surrogate_from_plecs.{md,json}. Everything runs in temporary directories.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "audit" / "plecs_om"
CMD = "python experiments/report_a1_surrogate_from_plecs.py"


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="spade_a1_"))
    os.environ["SPADE_WORKSPACE"] = str(tmp / "ws")
    os.environ["SPADE_PLECS_DATA"] = str(tmp / "plecs")
    os.environ["SPADE_SURROGATE_DATA"] = str(tmp / "surrogates")
    sys.path.insert(0, str(ROOT / "tests" / "integrations" / "plecs"))
    from fake_plecs import FakePlecs

    from driveflow.integrations.plecs import batch
    from driveflow.integrations.plecs import manifest as mf
    from driveflow.jobs.service import JobService
    from driveflow.jobs.worker import Worker
    from driveflow.runs.schemas import ExperimentSpec
    from driveflow.runs.store import RunStore, workspace_root
    from driveflow.sim.buck_system import TAU

    t0 = time.perf_counter()
    man = mf.from_dict(mf.EXAMPLE_BUCK_SURROGATE)
    circuit = tmp / "buck_excited.plecs"
    circuit.write_text("Plecs { Name \"buck_excited\" }  // placeholder file for the fake server\n")
    span = 0.02
    with FakePlecs() as fake:
        RunStore().save_profile(batch.create_profile("buck_excited", circuit, man, rpc_url=fake.url))
        conds = batch.design_conditions(man, {"Vin": (24, 48, 3), "Rload": (3, 12, 3), "prbs_seed": (5, 77, 2)}, "grid",
                                        fixed={"D_min": 0.1, "D_max": 0.8, "T_hold": 0.001})
        s = JobService()
        sj = s.submit(ExperimentSpec.create("plecs_sweep", plant_profile_id="buck_excited", conditions=conds, seed=0, algorithm="grid",
                                            params={"time_span_s": span, "time_limit_s": 60.0, "n_output_points": int(round(span / TAU)) + 1}))
        Worker(workspace_root()).serve(once=True, poll_s=0.05)
    sweep = s.run_for(s.get(sj.job_id))
    fj = s.submit(ExperimentSpec.create("surrogate_fit", seed=0, params={"profile_id": "buck_excited", "sweep_run_id": sweep.run_id, "epochs": 30}))
    Worker(workspace_root()).serve(once=True, poll_s=0.05)
    fit = s.run_for(s.get(fj.job_id))
    sur = json.loads((Path(fit.metrics["surrogate_dir"]) / "surrogate.json").read_text())
    rep = {"command": CMD, "machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}", "runtime_s": round(time.perf_counter() - t0, 1),
           "source": "fake PLECS server, excited-buck TEST DOUBLE (SPADE's averaged buck) -- not PLECS",
           "sweep": {"run_id": sweep.run_id, "status": sweep.status, "n_conditions": len(sweep.conditions), "n_ok": sum(c["status"] == "ok" for c in sweep.conditions),
                     "origin": sweep.origin, "span_s": span, "sample_s": TAU},
           "surrogate_fit": {"run_id": fit.run_id, "status": fit.status, "origin": fit.origin, "env_id": fit.metrics.get("env_id"), "check_env_passed": fit.metrics.get("check_env_passed"),
                             "default_family": sur["default_family"], "zoh_identified": {k: sur["families"]["zoh"]["fit"][k] for k in ("l_h", "c_f", "r_l_ohm")},
                             "validation": fit.metrics.get("validation"), "test": fit.metrics.get("test"), "provenance": sur["provenance"], "split": sur["conditions"]},
           "rows": sur["rows"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "a1_surrogate_from_plecs.json").write_text(json.dumps(rep, indent=1, default=float))
    (OUT / "a1_surrogate_from_plecs.md").write_text(render(rep))
    print(render(rep))


def _p(x):
    return "—" if x is None else f"{x:.2%}"


def render(rep: dict) -> str:
    f, sw = rep["surrogate_fit"], rep["sweep"]
    z = f["zoh_identified"]
    L = ["# A1 — surrogate plants from PLECS sweep datasets (route a)", "",
         f"Generated by `{rep['command']}` ({rep['machine']}), {rep['runtime_s']} s. Raw values: [`a1_surrogate_from_plecs.json`](a1_surrogate_from_plecs.json).", "",
         "**Mechanics only, not fidelity to PLECS.** The \"circuit\" is the fake PLECS server in its excited-buck test-double mode: SPADE's own averaged buck. No PLECS ran on this machine. "
         "PLECS real: **no ejecutado: requiere la máquina del investigador** (task A2 kit).", "",
         "## What was run", "",
         f"- **Sweep** `{sw['run_id']}`: {sw['n_ok']}/{sw['n_conditions']} conditions ({sw['status']}), origin {sw['origin']}.",
         f"  - Vin × Rload × PRBS seed = 3 × 3 × 2 conditions, {sw['span_s'] * 1000:.0f} ms each, sampled every {sw['sample_s'] * 1e6:.0f} µs.",
         "  - The applied duty is a dataset column (outport `duty`), checked against the PRBS reference.",
         f"- **surrogate_fit** `{f['run_id']}`: {f['status']}, origin {f['origin']}.",
         f"  - Published environment: `{f['env_id']}`; `check_env` passed: {f['check_env_passed']}.",
         f"  - Default family: **{f['default_family']}**, chosen on the validation operating points.",
         f"  - Provenance: profile `{f['provenance']['plecs_profile']}`, circuit SHA-256 `{f['provenance']['circuit_sha256'][:16]}…`, sweep run `{f['provenance']['sweep_run_id']}`.",
         f"- **ZOH grey box identified:** L = {z['l_h'] * 1e6:.3f} µH, C = {z['c_f'] * 1e6:.3f} µF, r_L = {z['r_l_ohm']:.4f} Ω (the double's 220 µH, 220 µF, 0.05 Ω). Exact recovery is expected: same model, no noise.", "",
         "## Surrogate error against the test double, per condition (open-loop rollout under the applied duty)", "",
         "| Vin (V) | R (Ω) | PRBS seed | Split | Family | rel. RMSE i_L | rel. RMSE v_out |", "|---|---|---|---|---|---|---|"]
    for r in sorted(rep["rows"], key=lambda r: (r["split"], r["v_in_v"], r["load_resistance_ohm"], r["prbs_seed"] or 0, r["family"])):
        L.append(f"| {r['v_in_v']:g} | {r['load_resistance_ohm']:g} | {r['prbs_seed']:g} | {r['split']} | {r['family']} | {_p(r['rollout_rel_rmse_i_l'])} | {_p(r['rollout_rel_rmse_v_out'])} |")
    L += ["", "Median over the held-out points:", ""]
    for fam in ("zoh", "mlp"):
        v, t = f["validation"][fam], f["test"][fam]
        L.append(f"- **{fam}:** validation v_out {_p(v['rollout_rel_rmse_v_out'])} and i_L {_p(v['rollout_rel_rmse_i_l'])}; test v_out {_p(t['rollout_rel_rmse_v_out'])} and i_L {_p(t['rollout_rel_rmse_i_l'])}.")
    L += ["", "## Limits", "",
          "- Same model in the double and in the grey box, so the ZOH error is not informative about a switched PLECS circuit (ripple, dead time, losses).",
          "- The closed loop of the surrogate is not compared here: a PLECS circuit gives no closed-loop reference for SPADE's controller. Real PLECS data comes from the researcher kit (A2).", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
