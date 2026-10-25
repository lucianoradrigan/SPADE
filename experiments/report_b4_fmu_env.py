"""B4 (round PLECS/OpenModelica): evidence for SPADE/OmBuck-v0 (envs/fmu_env.py).

Measures, on the built BuckAveraged FMU: gymnasium's check_env; the trajectory under the native
buck PI's actions (recorded on SPADE/Buck-v0) against SPADE/Buck-v0 with B3's criterion 4
(RMS / native range <= 1 % on v_out and i_L); the PI closing the loop on the FMU environment;
reset reproducibility; latency per step.

Usage: python experiments/report_b4_fmu_env.py [--out docs/audit/plecs_om]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(env, policy=None, actions=None, seed=0, options=None):
    obs, info0 = env.reset(seed=seed, options=options)
    act = policy(env.unwrapped) if policy else None
    rows, used, wall = [], [], []
    n_steps = env.unwrapped.n_steps if actions is None else min(env.unwrapped.n_steps, len(actions))
    for k in range(n_steps):
        a = act(obs) if act else actions[k]
        used.append(np.asarray(a, dtype=float))
        t0 = time.perf_counter()
        obs, r, term, trunc, info = env.step(a)
        wall.append(time.perf_counter() - t0)
        rows.append((info.get("v_out_v", np.nan), info.get("i_l_a", np.nan), r))
        if term or trunc:
            break
    return used, np.array(rows), {"terminated": bool(term), "truncated": bool(trunc), "termination": info.get("termination"), "n_steps": len(rows)}, wall


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "audit" / "plecs_om")
    args = ap.parse_args(argv)
    import warnings

    warnings.simplefilter("ignore")
    from gymnasium.utils.env_checker import check_env

    from driveflow.control.rl import benchmark as bm
    from driveflow.envs import fmu_env, registry

    ok, reason = fmu_env.availability()
    if not ok:
        raise SystemExit(f"SPADE/OmBuck-v0 not available: {reason}")
    out = {"command": "python experiments/report_b4_fmu_env.py " + " ".join(argv if argv is not None else sys.argv[1:]),
           "host": platform.platform(), "python": platform.python_version()}
    env = registry.make("SPADE/OmBuck-v0")
    out["fmu"] = env.unwrapped.metadata["fmu"]
    check_env(env.unwrapped, skip_render_check=True)
    out["check_env"] = "passed"
    cases = {}
    for name, options in (("nominal 24 V / 6 ohm", None), ("12 V / 20 ohm", {"vout_ref_v": 12.0, "load_resistance_ohm": 20.0}),
                          ("36 V / 3 ohm", {"vout_ref_v": 36.0, "load_resistance_ohm": 3.0})):
        nat = registry.make("SPADE/Buck-v0")
        actions, n, n_end, _ = _run(nat, bm.buck_pi, options=options)
        nat.close()
        _, o, o_end, wall = _run(env, actions=actions, options=options)
        _, cl, cl_end, _ = _run(env, bm.buck_pi, options=options)
        cases[name] = {
            "options": options,
            "open_loop_native_pi_actions": {"nrms_v_out": float(np.sqrt(np.mean((o[:, 0] - n[:, 0]) ** 2)) / np.ptp(n[:, 0])),
                                            "nrms_i_L": float(np.sqrt(np.mean((o[:, 1] - n[:, 1]) ** 2)) / np.ptp(n[:, 1])),
                                            "criterion_4_limit": 0.01, "episode_end": o_end},
            "pi_closed_loop_on_fmu": {"final_v_out_v": float(cl[-1, 0]), "episode_end": cl_end},
            "native_final_v_out_v": float(n[-1, 0]),
            "step_wall_ms": {"p50": float(np.percentile(wall, 50) * 1e3), "p99": float(np.percentile(wall, 99) * 1e3)},
        }
    out["cases"] = cases
    env2 = registry.make("SPADE/OmBuck-v0", vout_ref_v=None, load_resistance_ohm=None)
    traces = []
    for _ in range(2):
        env2.action_space.seed(5)
        acts = [env2.action_space.sample() for _ in range(200)]
        _, t, _, _ = _run(env2, actions=acts, seed=11)
        traces.append(t)
    env2.close()
    out["reset_reproducible_bit_for_bit"] = bool(np.array_equal(traces[0], traces[1]))
    env.close()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "b4_fmu_gym_env.json").write_text(json.dumps(out, indent=2) + "\n")
    lines = ["# B4 — SPADE/OmBuck-v0 (FmuGymEnv over the BuckAveraged FMU)", "", f"Command: `{out['command']}`  ", f"Host: {out['host']}, Python {out['python']}.  ",
             f"FMU: `{out['fmu']['path']}`, SHA-256 `{out['fmu']['sha256']}` (the one B3 evaluated: `48d52b6a…`).", "",
             f"- `gymnasium.utils.env_checker.check_env`: {out['check_env']}.",
             f"- Reset reproducibility (seed 11, 200 random actions, twice): {'bit-identical' if out['reset_reproducible_bit_for_bit'] else 'NOT identical'}.", "",
             "| Case | v_out NRMS vs SPADE/Buck-v0 (native PI actions) | i_L NRMS | ≤ 1 % (B3 criterion 4) | PI on the FMU env: final v_out | Step p50 / p99 |", "|---|---|---|---|---|---|"]
    for name, c in cases.items():
        o = c["open_loop_native_pi_actions"]
        lines.append(f"| {name} | {o['nrms_v_out']:.1e} | {o['nrms_i_L']:.1e} | {'yes' if max(o['nrms_v_out'], o['nrms_i_L']) <= 0.01 else 'no'} | "
                     f"{c['pi_closed_loop_on_fmu']['final_v_out_v']:.3f} V (native {c['native_final_v_out_v']:.3f} V) | {c['step_wall_ms']['p50']:.3f} / {c['step_wall_ms']['p99']:.3f} ms |")
    lines += ["", "Scope (decisions G1/G2): only BuckAveraged is registered. It replicates the native buck: it validates the FMU integration, not other physics. "
              "The FMU's load R is a fixed parameter, so it is set at reset and SPADE/OmBuck-v0 has no load step inside an episode.",
              "Not executed: Windows and macOS (the FMU has linux64 binaries only).", ""]
    (args.out / "b4_fmu_gym_env.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
