"""T7 (2026-11-09): the local gym-electric-motor fork's Cont1QC clamp under braking, at three tau.

Not part of SPADE: it imports the FORK, from its own venv (SPADE never imports gym_electric_motor):

    PYTHONPATH=/mnt/datos/upstream/gym-electric-motor/src /mnt/datos/upstream/gem-venv/bin/python \
        experiments/report_gem_cont1qc_fork.py

Prints, per tau, omega and i at 50 ms and the minimum current. Results: docs/audit/upstream/gem_cont1qc_pr_draft.md.
"""

import json, sys, warnings
import numpy as np
warnings.simplefilter("ignore")
import gym_electric_motor as gem
from gym_electric_motor.physical_systems.converters import ContOneQuadrantConverter
from gym_electric_motor.physical_systems.mechanical_loads import PolynomialStaticLoad
T_END = 0.05
out = {}
for tau in (1e-4, 1e-5, 1e-6):
    load = PolynomialStaticLoad(load_parameter=dict(a=0.01, b=0.05, c=0.0, j_load=1e-3), load_initializer={"states": {"omega": 150.0}})
    env = gem.make("Cont-SC-PermExDc-v0", converter=ContOneQuadrantConverter(), load=load, visualization=(), constraints=(), tau=tau)
    env.reset(seed=0)
    ps = env.unwrapped.physical_system; w = ps.state_names.index("omega"); i = ps.state_names.index("i")
    imin = 0.0
    for k in range(int(round(T_END / tau))):
        duty = 0.2 if (k * tau) < 0.025 else 0.5   # braking, then 30 V > e (CCM again)
        (s, _), *_ = env.step(np.array([duty]))
        imin = min(imin, s[i] * ps.limits[i])
    out[f"{tau:g}"] = {"omega_end": float(s[w] * ps.limits[w]), "i_end": float(s[i] * ps.limits[i]), "i_min": imin}
ref = out["1e-06"]["omega_end"]
for k, v in out.items():
    v["omega_err_vs_1e-6"] = v["omega_end"] - ref
print(json.dumps(out, indent=1))
