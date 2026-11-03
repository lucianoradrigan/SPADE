# Draft issue for upstream gym-electric-motor — NOT PUBLISHED

**Status:** draft prepared 2026-11-03 at Luciano's request. It is **not** to be published without his confirmation. The repository is `upb-lea/gym-electric-motor` on GitHub.

**SPADE verification.**
- The upstream code has the same logic as SPADE's port. In the clone `../gym-electric-motor` at `v3.0.3` (= `origin/master`, `b474daf9`; see `gem_upstream_drift.md`):
  - `physical_systems/converters.py`: `ContOneQuadrantConverter._convert` returns `self._current_action[0] if i_in[0] >= 0 else 1`.
  - `physical_systems/physical_systems.py`: `SCMLSystem.simulate` evaluates `convert(i_in, t)` once, from the current at the start of the step, and holds it while the ODE solver integrates to `t + tau`.
- The reproduction below uses only the PyPI release (`pip install gym-electric-motor==3.0.3`; Python 3.14.4, numpy 2.5.4, scipy 1.18.1, gymnasium 1.4.0), with no SPADE code. Files:
  - script: `gem_cont1qc_mre.py` (the reported "u" is the commanded average voltage, not the voltage the converter applies);
  - output: `gem_cont1qc_mre_output.txt`.
- SPADE's own analysis: finding S-21 (`docs/SIMULATION_STABILITY_REPORT.md`), `docs/audit/assistant/s21_one_quadrant.md`, and the impact on models trained on GEM data, `docs/audit/assistant/ai_n27_impact.md`.

---

## Title

ContOneQuadrantConverter: negative armature current and energy injection (unloaded motor accelerates at duty 0)

## Description

A one-quadrant (buck) converter cannot carry a negative armature current in steady operation. When the back-EMF exceeds the commanded average voltage, the current falls to zero and stays there (discontinuous conduction), and the machine coasts.

`ContOneQuadrantConverter._convert` returns duty 1 (full supply voltage) when `i_in < 0`. `SCMLSystem.simulate` evaluates this once per step, from the current at the start of the step, and holds it for the whole step `tau`. So:

1. within a step the current can go far below zero, because nothing stops it at zero;
2. in the next step the full supply voltage is applied for the whole `tau`, the current overshoots back far above zero, and energy is injected.

With a small armature inductance the result is an unloaded, frictionless motor that **accelerates** while the transistor is commanded off.

## Minimal reproduction (gym-electric-motor 3.0.3 from PyPI)

```python
import numpy as np
import gym_electric_motor as gem
from gym_electric_motor.physical_systems.mechanical_loads import PolynomialStaticLoad
from gym_electric_motor.physical_systems.converters import ContOneQuadrantConverter

load = PolynomialStaticLoad(load_parameter=dict(a=0.0, b=0.0, c=0.0, j_load=1e-3),
                            load_initializer={"states": {"omega": 150.0}})
env = gem.make("Cont-SC-PermExDc-v0", converter=ContOneQuadrantConverter(), load=load,
               visualization=(), constraints=())
(state, ref), _ = env.reset(seed=0)
ps = env.unwrapped.physical_system
w, i = ps.state_names.index("omega"), ps.state_names.index("i")
min_i = 0.0
for k in range(1, 5001):                     # 0.5 s at tau = 1e-4 s
    (state, ref), *_ = env.step(np.array([0.0]))   # transistor always off
    min_i = min(min_i, state[i] * ps.limits[i])
print(state[w] * ps.limits[w], min_i)
```

Output (default PermExDc parameters: r_a = 0.016 Ω, l_a = 19 µH, psi_e = 0.165 Wb, u_sup = 60 V):

```
t = 0.0001 s  omega =   149.96 rad/s  i = -124.918 A
t = 0.0010 s  omega =   150.07 rad/s  i =  -48.804 A
t = 0.0100 s  omega =   151.94 rad/s  i =  -95.326 A
t = 0.1000 s  omega =   164.18 rad/s  i =  -67.950 A
t = 0.5000 s  omega =   181.82 rad/s  i =  -78.904 A
min armature current seen: -144.683 A
```

**Expected:** i ≥ 0 at all times; with duty 0, no load and no friction, ω stays at 150 rad/s.
**Observed:** i reaches −145 A, and ω rises from 150 to 182 rad/s in 0.5 s.

## Impact

- **Closed-loop speed control:** in braking at light load the controller commands a low duty, the current turns negative, and the converter applies full voltage. The speed then rises above the reference instead of falling.
- **Our measurement:** in a dataset of simulated speed-controlled runs (80–220 rad/s), the affected runs ended a median +35 % (up to +76 %) above the reference.

## Possible fix

Model discontinuous conduction for the one-quadrant converter. When the current reaches zero and the commanded voltage does not exceed the back-EMF (u ≤ e), keep i = 0 and advance only the mechanics. This needs re-evaluating the converter inside the step, or locating the zero crossing within it, instead of holding the start-of-step decision for the whole `tau`.

- **Our implementation:** a substepped version with linear localization of the zero crossing. It converges: NRMS < 1e-4 at 1000 substeps per `tau` against 10000.
- We can share details if useful.

## Environment

gym-electric-motor 3.0.3 (PyPI; same code on `master` at b474daf9), Python 3.14.4, numpy 2.5.4, scipy 1.18.1, gymnasium 1.4.0, Linux x86_64.
