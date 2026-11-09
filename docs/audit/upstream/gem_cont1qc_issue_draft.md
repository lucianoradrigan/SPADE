# Draft issue for upstream gym-electric-motor — FINAL DRAFT, NOT PUBLISHED

**Status:**
- **Final draft** (2026-11-08; "Possible fix" rewritten in three levels on 2026-11-09, T7, with the local fork's results). Luciano publishes it himself; Claude Code does not post it.
- Companion PR draft (local only): [`gem_cont1qc_pr_draft.md`](gem_cont1qc_pr_draft.md).
- First draft 2026-11-03. The repository is `upb-lea/gym-electric-motor` on GitHub.
- Everything between the two horizontal rules below "Title" is the text to paste.

**Duplicate check (2026-11-08, read-only, GitHub API via `gh`):**
- **Issues, all states** (`gh search issues … --repo upb-lea/gym-electric-motor`), queries "OneQuadrant", "one quadrant", "ContOneQuadrantConverter", "negative current converter", "discontinuous conduction", "buck converter negative", "DcPermEx accelerates":
  - only one hit: **#83** "Unexpected multiplication of input voltage" (closed 2020-09-23), about the discrete one-quadrant converter's action scaling in `DcSeriesDisc-v1`. **Not the same problem.**
  - The full list of issues (`gh issue list --state all`), filtered by converter / current / quadrant / negative / speed / braking / duty: nothing about negative current or discontinuous conduction.
- **Pull requests** ("quadrant", "converter current", "discontinuous"):
  - **#138** "Four Quadrant Converters on all DC Envs" (merged 2021-06-17) changed the DC environments' default converter to four-quadrant. It does not mention the one-quadrant converter's behaviour.
  - Consequence: the bug is reached only when a one-quadrant converter is chosen explicitly. The issue text says so.
- **Discussions** (7, GraphQL): none related; #248 "Simulation tau size" is about the step size.
- **Upstream head:** `master` = `b474daf9` (2025-12-19, merge for release 3.0.3), the commit the code check below was made on. There is no newer commit on master.
- **Verdict:** no duplicate found.

**Re-run of the reproduction (2026-11-08):** fresh venv with `pip install gym-electric-motor==3.0.3` (Python 3.14.4, numpy 2.5.4, scipy 1.18.1, gymnasium 1.4.0), `python gem_cont1qc_mre.py`. The output is identical to `gem_cont1qc_mre_output.txt`: min current −144.683 A, ω 150 → 181.82 rad/s.

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

---

## Description

Since #138 the DC environments use a four-quadrant converter by default, so this only shows up when `ContOneQuadrantConverter` is passed explicitly (as in the reproduction below).

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

## Possible fix (three levels)

1. **Document and warn.** State in the `ContOneQuadrantConverter` docstring and docs that the model allows a negative current and applies the full supply voltage when it is negative. Warn when the converter is built. This changes no result.
2. **Clamp the current at zero.** `_convert` returns the duty cycle for every sign, and the physical system keeps the armature current at zero once it reaches zero while the commanded voltage does not exceed the back-EMF. This is done in the ODE right-hand side, so the default adaptive solver stops at the zero crossing inside the step, and again after the step for fixed-step solvers. It is a small change.
   - **Our local prototype** (a fork of `b474daf9`, not submitted): the suite passes (1230 tests: the 1226 existing ones plus 4 new). With the reproduction above, i ≥ 0 and ω stays at 150.00 rad/s.
   - For motors with two currents (shunt, externally excited) the clamp is not applied; the prototype only warns there.
   - We can open it as a PR after this issue, if you want it.
3. **Model discontinuous conduction within the period.** Re-evaluate the converter inside the step, or solve the conduction interval in closed form.
   - **Our implementation in SPADE:** a substepped version with linear localization of the zero crossing converges (NRMS < 1e-4 at 1000 substeps per `tau` against 10000). A closed-form per-period version (linear back-EMF, Heun mechanics) reaches 3.0e-5 against 10000 substeps and is 28× cheaper than 1000 substeps.
   - This is the most accurate option, and the largest change.

The discrete `FiniteOneQuadrantConverter` has the same `i < 0 → 1` logic and might deserve the same treatment.

## Environment

gym-electric-motor 3.0.3 (PyPI; same code on `master` at b474daf9), Python 3.14.4, numpy 2.5.4, scipy 1.18.1, gymnasium 1.4.0, Linux x86_64.

---
