"""Action-excitation profiles for PLECS sweeps that feed a surrogate plant (task A1, round PLECS/OM).

Why: a sweep of CONSTANT ModelVars only gives step responses. To serve as a Gym plant, a surrogate
must learn the action → state dynamics over the whole action range, so the circuit must apply a
time-varying action during each run. The profile is parametrised by ModelVars, so a sweep can
vary it like any other variable, and the circuit implements it (a C-Script or lookup block, see
docs/guides/PLECS_RESEARCHER_KIT.md).

This module is the REFERENCE definition: ``sequence()`` gives the exact action a correct circuit
applies at each time. The dataset must also record the action the circuit actually applied, as an
outport (``action_output``), and ``consistency()`` compares the two.

Kinds:
- ``step``: D0 until t_step, then D1.
- ``prbs``: a PRBS7 (x^7 + x^6 + 1) Fibonacci LFSR seeded with ``prbs_seed`` (1..127). One bit per
  hold interval T_hold: level D_max if the bit is 1, D_min if 0. Bit k (k = floor(t / T_hold)) is
  the output of the k-th shift. Chosen because it is a few lines of C:

      unsigned s = seed;                      // 1..127, never 0
      // every T_hold seconds:
      unsigned bit = ((s >> 6) ^ (s >> 5)) & 1;
      s = ((s << 1) | bit) & 0x7f;
      duty = bit ? D_max : D_min;
"""

from __future__ import annotations

import math

import numpy as np

#: kind -> the parameter roles it needs (each mapped to a ModelVar name in the manifest)
KINDS = {"step": ("D0", "D1", "t_step"), "prbs": ("prbs_seed", "D_min", "D_max", "T_hold")}


class ExcitationError(ValueError):
    pass


def prbs7_bits(seed: int, n: int) -> np.ndarray:
    s = int(seed)
    if not 1 <= s <= 127:
        raise ExcitationError(f"prbs_seed must be an integer in 1..127, got {seed!r}")
    out = np.empty(n, dtype=np.int8)
    for k in range(n):
        bit = ((s >> 6) ^ (s >> 5)) & 1
        s = ((s << 1) | bit) & 0x7F
        out[k] = bit
    return out


def sequence(kind: str, params: dict, times) -> np.ndarray:
    """The action at each time of ``times`` (s) under the profile ``kind`` with ``params`` (role -> value)."""
    t = np.asarray(times, dtype=float)
    if kind not in KINDS:
        raise ExcitationError(f"excitation kind must be one of {sorted(KINDS)}, got {kind!r}")
    missing = [r for r in KINDS[kind] if r not in params]
    if missing:
        raise ExcitationError(f"{kind} excitation needs {missing}")
    if kind == "step":
        return np.where(t < float(params["t_step"]), float(params["D0"]), float(params["D1"]))
    hold = float(params["T_hold"])
    if not (math.isfinite(hold) and hold > 0):
        raise ExcitationError(f"T_hold must be > 0 s, got {hold!r}")
    lo, hi = float(params["D_min"]), float(params["D_max"])
    if not lo < hi:
        raise ExcitationError(f"D_min must be below D_max, got [{lo:g}, {hi:g}]")
    k = np.floor(t / hold + 1e-9).astype(int)
    bits = prbs7_bits(int(round(float(params["prbs_seed"]))), int(k.max()) + 1 if len(k) else 0)
    return np.where(bits[k] == 1, hi, lo)


def check_spec(spec: dict, model_var_names: set, output_names: set) -> dict:
    """Validates a manifest ``excitation`` block: {"kind", "action_output", "params": {role: ModelVar}}."""
    if not isinstance(spec, dict):
        raise ExcitationError("excitation must be an object with 'kind', 'action_output' and 'params'")
    kind = spec.get("kind")
    if kind not in KINDS:
        raise ExcitationError(f"excitation.kind must be one of {sorted(KINDS)}, got {kind!r}")
    params = dict(spec.get("params") or {})
    missing = [r for r in KINDS[kind] if r not in params]
    if missing:
        raise ExcitationError(f"excitation.params must map {list(KINDS[kind])} to ModelVars; missing {missing}")
    unknown = sorted(v for v in params.values() if v not in model_var_names)
    if unknown:
        raise ExcitationError(f"excitation.params refer to ModelVars that are not declared: {unknown}")
    action = spec.get("action_output")
    if action not in output_names:
        raise ExcitationError(f"excitation.action_output must be one of the declared outputs (the action the circuit applied), got {action!r}")
    return {"kind": kind, "action_output": action, "params": {r: params[r] for r in KINDS[kind]}}


def params_of(spec: dict, model_vars: dict) -> dict:
    """role -> value for one condition's ModelVars."""
    return {role: float(model_vars[var]) for role, var in spec["params"].items()}


def consistency(spec: dict, model_vars: dict, times, applied) -> dict:
    """How far the applied action (the circuit's outport) is from the reference sequence. A circuit
    that implements the profile correctly gives max_abs_error ~ 0 (except at switching instants,
    where sampling may land on either side)."""
    ref = sequence(spec["kind"], params_of(spec, model_vars), times)
    err = np.abs(np.asarray(applied, dtype=float) - ref)
    return {"max_abs_error": float(err.max()) if len(err) else None, "share_mismatched": float(np.mean(err > 1e-6)) if len(err) else None}
