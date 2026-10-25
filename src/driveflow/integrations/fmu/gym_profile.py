"""OpenModelica FMU as a Gymnasium plant: profiles, verification and publication -- task B5.

The Plant Studio route "Use as Gym plant" (spec Sec. 3.4, 3.8, 3.17) with the engine
"OpenModelica FMU". This module is the domain layer; viz/adapters/fmu_studio_adapter.py is the GUI
boundary.

- ``engine_report()``: what was checked about the engine (fmpy, platform, FMU folder).
- ``candidates()``: every FMU in data/fmu/ (or $SPADE_FMU_DIR) with its platforms, SHA-256 and
  the B3 verdict for its model (docs/audit/plecs_om/b3_om_fmu_spike.json). A NO-GO model is listed
  with its verdict and is never publishable; DcMotorAveraged is "Reference use only" (G2).
- ``new_profile()`` / ``set_mapping()`` / ``set_gym()``: a versioned PlantProfile of kind
  ``om_fmu_co_sim`` moving through DRAFT → IMPORTED → MAPPED → SPECIFIED.
- ``verify()`` runs the checks of spec Sec. 3.4 stage 6 (reusing B3's criteria 1, 2, 3, 4 and 6)
  and is executed by the ``fmu_verification`` job. Its record carries the FMU SHA-256 and the
  configuration hash it was run on; ``readiness()`` reports ``Stale`` when either changed.
- ``publish()`` registers ``SPADE/Fmu<Profile>-v0`` (envs.fmu_env.MappedFmuEnv) only with a
  current passing verification and a B3 verdict GO.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]


def fmu_dir() -> Path:
    # imported here: driveflow.envs registers environments at import and reads published() below
    from driveflow.envs.fmu_env import fmu_dir as _fmu_dir

    return _fmu_dir()

KIND = "om_fmu_co_sim"
ENGINE = "openmodelica_fmu"
ORIGIN = "SIMULATED · OPENMODELICA"
B3_REPORT = REPO_ROOT / "docs" / "audit" / "plecs_om" / "b3_om_fmu_spike.json"
B3_LINK = "docs/audit/plecs_om/b3_om_fmu_spike.md"
REFERENCE_ONLY = {"DcMotorAveraged": "Reference use only (see B3 post-hoc analysis)"}
#: B3 criterion 6 for averaged models
LATENCY_P50_LIMIT_MS = 0.5
#: B3 criterion 4
NATIVE_NRMS_LIMIT = 0.01
READINESS_ITEMS = ("Engine", "Circuit", "Action mapping", "Observation mapping", "Gym dynamics", "Validation", "Publishability")


# --------------------------------------------------------------------------------- engine, FMUs
def engine_report() -> dict:
    """Detected engine capabilities; never inferred from file names."""
    from driveflow.integrations.fmu.adapter import FmuError, current_platform

    rep = {"engine": "OpenModelica FMU (FMI 2.0 Co-Simulation via fmpy)", "fmu_dir": str(fmu_dir()), "checked": ["fmpy import", "platform tag"],
           "not_checked": ["OpenModelica installation (not needed to run a built FMU)"]}
    try:
        import fmpy

        rep.update(fmpy=True, fmpy_version=getattr(fmpy, "__version__", "?"), platform=current_platform(), message="fmpy available")
    except (ImportError, FmuError):
        rep.update(fmpy=False, fmpy_version=None, platform=None, message='fmpy is not installed: pip install "driveflow[fmu]"')
    return rep


def b3_verdicts(path: Path = B3_REPORT) -> dict:
    """{model: {"verdict", "failed_criteria", "sha256"}} from the B3 report; {} when it is missing."""
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    fmus = (d.get("fmu_manifest") or {}).get("fmus", {})
    return {name: {"verdict": m.get("verdict"), "failed_criteria": m.get("failed_criteria", []), "sha256": (fmus.get(name) or {}).get("sha256")}
            for name, m in d.get("models", {}).items()}


def candidates() -> list[dict]:
    """FMUs on disk with what SPADE can do with each. Never raises."""
    from driveflow.integrations.fmu.adapter import FmuError, inspect_fmu

    eng = engine_report()
    verdicts = b3_verdicts()
    out = []
    d = fmu_dir()
    for path in sorted(d.glob("*.fmu")) if d.is_dir() else []:
        row = {"name": path.stem, "path": str(path), "sha256": None, "platforms": [], "b3": verdicts.get(path.stem), "status": None, "reason": "",
               "publishable": False, "note": REFERENCE_ONLY.get(path.stem)}
        if not eng["fmpy"]:
            row.update(status="Not supported", reason=eng["message"])
            out.append(row)
            continue
        try:
            info = inspect_fmu(path)
        except FmuError as exc:
            row.update(status="Failed", reason=str(exc))
            out.append(row)
            continue
        row.update(sha256=info.sha256, platforms=list(info.platforms), model_name=info.model_name, inputs=[v.name for v in info.inputs],
                   outputs=[{"name": v.name, "unit": v.unit} for v in info.outputs],
                   parameters=[{"name": v.name, "unit": v.unit, "start": v.start} for v in info.parameters])
        b3 = row["b3"]
        if eng["platform"] not in info.platforms:
            row.update(status="Not supported", reason=f"binaries for {info.platforms or 'no platform'}, not {eng['platform']}")
        elif info.fmi_version != "2.0" or info.fmu_type != "CoSimulation":
            row.update(status="Not supported", reason=f"FMI {info.fmi_version} {info.fmu_type}; an FMI 2.0 Co-Simulation FMU is required")
        elif b3 is None:
            row.update(status="Needs attention", reason="no B3 GO/NO-GO verdict for this model: not publishable")
        elif b3["verdict"] != "GO":
            row.update(status="Not publishable", reason=f"B3 verdict {b3['verdict']} (failed criteria {b3['failed_criteria']}); see {B3_LINK}")
        else:
            row.update(status="Ready to map", publishable=True,
                       reason="B3 verdict GO" + ("" if b3["sha256"] == info.sha256 else " for this model; B3 evaluated another build (SHA-256 differs)"))
        out.append(row)
    return out


# --------------------------------------------------------------------------------- profiles
def _store(root=None):
    from driveflow.runs.store import RunStore

    return RunStore(root)


def check_profile_id(profile_id: str) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,39}", profile_id or ""):
        raise ValueError("Profile ID: 1-40 characters, a-z, 0-9, '_' or '-', starting with a letter")
    return profile_id


def new_profile(profile_id: str, display_name: str, fmu_name: str, root=None):
    """IMPORTED profile for one listed FMU. A NO-GO or unsupported FMU can be imported as a draft
    (to inspect it) but stays unpublishable."""
    from driveflow.runs.schemas import EnvironmentCapability, PlantProfile, SpadeError

    check_profile_id(profile_id)
    cand = next((c for c in candidates() if c["name"] == fmu_name), None)
    if cand is None:
        raise ValueError(f"No FMU named {fmu_name!r} in {fmu_dir()}")
    if cand["sha256"] is None:
        raise ValueError(f"{fmu_name}: {cand['reason']}")
    store = _store(root)
    try:
        version = store.get_profile(profile_id).version + 1
    except SpadeError:
        version = 1
    status = "imported" if cand["status"] in ("Ready to map", "Needs attention", "Not publishable") else "unsupported"
    p = PlantProfile(profile_id=profile_id, kind=KIND, engine=ENGINE, display_name=display_name or fmu_name, origin=ORIGIN, file_hash=cand["sha256"],
                     signals=tuple(o["name"] for o in cand.get("outputs", [])), status=status, version=version,
                     capabilities=EnvironmentCapability(diagnostic_messages=("Not verified yet.",)),
                     parameters={"model": fmu_name, "fmu_path": cand["path"], "platforms": cand["platforms"], "b3": cand["b3"], "candidate_status": cand["status"],
                                 "candidate_reason": cand["reason"], "fmu_inputs": cand.get("inputs", []), "fmu_outputs": cand.get("outputs", []),
                                 "fmu_parameters": cand.get("parameters", []), "mapping": None, "gym": None, "verification": None, "published_env_id": None})
    return store.save_profile(p)


def _replace(p, root=None, **params):
    import dataclasses

    status = params.pop("_status", p.status)
    caps = params.pop("_capabilities", p.capabilities)
    q = dataclasses.replace(p, parameters={**p.parameters, **params}, status=status, capabilities=caps)
    return _store(root).save_profile(q)


def suggest_mapping(p) -> dict:
    """Auto-suggested mapping for the user to confirm (never applied silently)."""
    ins = p.parameters["fmu_inputs"]
    outs = p.parameters["fmu_outputs"]
    action = ins[0] if ins else None
    lowhigh = (0.0, 1.0) if action == "duty" else (-1.0, 1.0)
    obs = [{"output": o["name"], "unit": o["unit"] or "1", "scale": 1.0} for o in outs]
    cur = next((o["name"] for o in outs if o["name"].lower().startswith("i")), None)
    volt = next((o["name"] for o in outs if o["name"].lower().startswith("v")), None)
    return {"action": {"input": action, "name": "duty_cycle" if action == "duty" else action, "unit": "1", "low": lowhigh[0], "high": lowhigh[1]},
            "observations": obs, "reference": {"output": volt, "value": 0.0} if volt else None, "safety": {"output": cur, "rated": 1.0} if cur else None}


def validate_mapping(p, mapping: dict) -> list[str]:
    """Row-level problems; [] when the mapping is usable."""
    errs = []
    ins = set(p.parameters["fmu_inputs"])
    outs = {o["name"] for o in p.parameters["fmu_outputs"]}
    a = mapping.get("action") or {}
    if a.get("input") not in ins:
        errs.append(f"Action: {a.get('input')!r} is not an input of the FMU (inputs: {sorted(ins)})")
    try:
        lo, hi = float(a.get("low")), float(a.get("high"))
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            errs.append("Action: bounds must be finite with low < high")
    except (TypeError, ValueError):
        errs.append("Action: bounds must be numbers")
    obs = mapping.get("observations") or []
    if not obs:
        errs.append("Observations: map at least one FMU output")
    names = [o.get("output") for o in obs]
    if len(names) != len(set(names)):
        errs.append("Observations: each output may appear once (order is explicit)")
    for o in obs:
        if o.get("output") not in outs:
            errs.append(f"Observation {o.get('output')!r} is not an output of the FMU (outputs: {sorted(outs)})")
        if not (isinstance(o.get("scale"), (int, float)) and math.isfinite(o["scale"]) and o["scale"] > 0):
            errs.append(f"Observation {o.get('output')!r}: scale must be > 0")
    ref = mapping.get("reference")
    if ref and ref.get("output") not in names:
        errs.append("Reference: the tracked output must be one of the observations")
    if ref and not (isinstance(ref.get("value"), (int, float)) and math.isfinite(ref["value"])):
        errs.append("Reference: value must be a finite number")
    s = mapping.get("safety")
    if s and (s.get("output") not in outs or not (isinstance(s.get("rated"), (int, float)) and s["rated"] > 0)):
        errs.append("Safety: choose an FMU output and a rated value > 0 (episodes end above 2 × rated)")
    return errs


def set_mapping(p, mapping: dict, root=None):
    errs = validate_mapping(p, mapping)
    if errs:
        raise ValueError("; ".join(errs))
    return _replace(p, root, mapping=mapping, gym=None, verification=None, _status="mapped" if p.status != "unsupported" else "unsupported")


def validate_gym(p, gym_cfg: dict) -> list[str]:
    from driveflow.envs import reward as rw

    errs = []
    tau, dur = gym_cfg.get("tau_s"), gym_cfg.get("duration_s")
    if not (isinstance(tau, (int, float)) and 0 < tau <= 1.0):
        errs.append("Control timing: Ts must be in (0, 1] s")
    if not (isinstance(dur, (int, float)) and isinstance(tau, (int, float)) and tau > 0 and dur >= tau and dur / tau <= 1_000_000):
        errs.append("Control timing: the episode must last at least one Ts and at most 1e6 steps")
    known = {q["name"] for q in p.parameters["fmu_parameters"]}
    for k, v in (gym_cfg.get("parameters") or {}).items():
        if k not in known:
            errs.append(f"Episode reset: {k!r} is not a parameter of the FMU")
        elif not (isinstance(v, (int, float)) and math.isfinite(v)):
            errs.append(f"Episode reset: {k} must be a finite number")
    try:
        rw.parse_terms(gym_cfg.get("reward_terms"))
    except (ValueError, TypeError) as exc:
        errs.append(f"Reward: {exc}")
    if gym_cfg.get("action_out_of_range", "clip") not in ("clip", "reject"):
        errs.append("Safety: out-of-range actions must be 'clip' or 'reject'")
    return errs


def set_gym(p, gym_cfg: dict, root=None):
    if not p.parameters.get("mapping"):
        raise ValueError("Map the signals first")
    errs = validate_gym(p, gym_cfg)
    if errs:
        raise ValueError("; ".join(errs))
    return _replace(p, root, gym=gym_cfg, verification=None, _status="specified" if p.status != "unsupported" else "unsupported")


def config_hash(p) -> str:
    return hashlib.sha256(json.dumps({"mapping": p.parameters.get("mapping"), "gym": p.parameters.get("gym")}, sort_keys=True).encode()).hexdigest()


def env_spec(p) -> dict:
    return {"fmu_path": p.parameters["fmu_path"], "model": p.parameters["model"], "plant_profile_id": f"{KIND}:{p.profile_id}:v{p.version}",
            "mapping": p.parameters["mapping"], "gym": p.parameters["gym"]}


def current_file_hash(p) -> str | None:
    path = Path(p.parameters["fmu_path"])
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def is_stale(p) -> bool:
    """The FMU file changed (or is gone) since import, or the verification was run on another
    FMU or another configuration."""
    if current_file_hash(p) != p.file_hash:
        return True
    v = p.parameters.get("verification")
    return bool(v) and (v.get("fmu_sha256") != p.file_hash or v.get("config_hash") != config_hash(p))


def verification_current(p) -> bool:
    v = p.parameters.get("verification")
    return bool(v) and v.get("status") == "passed" and not is_stale(p)


def publish_blockers(p) -> list[str]:
    out = []
    b3 = p.parameters.get("b3") or {}
    if p.status == "unsupported":
        out.append(f"Not supported: {p.parameters.get('candidate_reason')}")
    if b3.get("verdict") != "GO":
        out.append(f"B3 verdict {b3.get('verdict') or 'missing'} for {p.parameters['model']}: not publishable (see {B3_LINK})")
    if p.parameters["model"] in REFERENCE_ONLY:
        out.append(REFERENCE_ONLY[p.parameters["model"]])
    if is_stale(p):
        out.append("Revalidation required: the FMU or the configuration changed since the last verification")
    elif not verification_current(p):
        out.append("Run verification: no passing verification for this FMU and configuration")
    return out


def readiness(p) -> dict:
    """The seven 'Environment readiness' rows: (state, text). States: Not configured, Needs
    attention, Validated, Failed, Not supported."""
    eng = engine_report()
    r = {}
    r["Engine"] = ("Validated", f"fmpy {eng['fmpy_version']} on {eng['platform']}") if eng["fmpy"] else ("Not supported", eng["message"])
    if p is None:
        return {**r, **{k: ("Not configured", "") for k in READINESS_ITEMS[1:]}}
    if p.status == "unsupported":
        r["Circuit"] = ("Not supported", p.parameters.get("candidate_reason", ""))
    elif current_file_hash(p) != p.file_hash:
        r["Circuit"] = ("Needs attention", "Stale — the FMU file changed since import. Revalidate.")
    elif (p.parameters.get("b3") or {}).get("verdict") != "GO":
        r["Circuit"] = ("Needs attention", f"B3 verdict {(p.parameters.get('b3') or {}).get('verdict') or 'missing'}: not publishable")
    else:
        r["Circuit"] = ("Validated", f"{p.parameters['model']} · SHA-256 {p.file_hash[:12]}…")
    m = p.parameters.get("mapping")
    r["Action mapping"] = ("Validated", f"{m['action']['input']} ∈ [{m['action']['low']:g}, {m['action']['high']:g}]") if m else ("Not configured", "")
    r["Observation mapping"] = ("Validated", ", ".join(o["output"] for o in m["observations"])) if m else ("Not configured", "")
    g = p.parameters.get("gym")
    r["Gym dynamics"] = ("Validated", f"Ts {g['tau_s'] * 1e6:g} µs × {round(g['duration_s'] / g['tau_s'])} steps") if g else ("Not configured", "")
    v = p.parameters.get("verification")
    if not v:
        r["Validation"] = ("Not configured", "Not run")
    elif is_stale(p):
        r["Validation"] = ("Needs attention", "Stale — Revalidate")
    elif v["status"] == "passed":
        r["Validation"] = ("Validated", f"{v['n_passed']}/{v['n_checks']} checks passed · run {v['run_id']}")
    else:
        r["Validation"] = ("Failed", f"{v['n_passed']}/{v['n_checks']} checks passed · run {v['run_id']}")
    blockers = publish_blockers(p)
    if p.status == "published" and not blockers:
        r["Publishability"] = ("Validated", f"Published as {p.parameters.get('published_env_id')}")
    elif p.status == "unsupported" or any(b.startswith("B3 verdict") for b in blockers):
        r["Publishability"] = ("Not supported", blockers[0])
    elif blockers:
        r["Publishability"] = ("Needs attention", blockers[0])
    else:
        r["Publishability"] = ("Validated", "Ready to publish")
    return r


def display_state(p) -> str:
    """The profile state shown to the user (spec Sec. 3.8 state machine)."""
    if p is None:
        return "DRAFT"
    if p.status not in ("unsupported",) and p.parameters.get("mapping") and is_stale(p):
        return "STALE"
    return p.status.upper()


# --------------------------------------------------------------------------------- verification
def _episode(env, actions, seed=0):
    obs, _ = env.reset(seed=seed)
    trace = [np.asarray(obs, dtype=float)]
    rewards = []
    for a in actions:
        obs, r, term, trunc, info = env.step(a)
        trace.append(np.append(obs, r))
        rewards.append(r)
        if term or trunc:
            break
    return trace, rewards, info


def _native_counterpart(p):
    """B3 criterion 4 on the profile's own parameters, when a native plant of the same equations
    exists (BuckAveraged ↔ sim.buck_system). None otherwise."""
    if p.parameters["model"] != "BuckAveraged":
        return None
    from driveflow.sim.buck_system import BuckParams, BuckSystem

    g = p.parameters["gym"]
    par = {q["name"]: q["start"] for q in p.parameters["fmu_parameters"]}
    par.update(g.get("parameters") or {})
    bp = BuckParams(v_in_v=par["Vin"], l_h=par["L"], c_f=par["C"], r_l_ohm=par["rL"], load_resistance_ohm=par["R"])
    return BuckSystem(bp, tau=g["tau_s"])


def verify(p, log=print) -> dict:
    """Runs the stage-6 checks on a fresh MappedFmuEnv. Each check: {id, name, status
    (pass/fail/not_applicable), detail}. Status "passed" only if no check fails."""
    from gymnasium.utils.env_checker import check_env

    from driveflow.envs.fmu_env import MappedFmuEnv
    from driveflow.integrations.fmu.adapter import FmuError

    checks = []

    def add(i, name, ok, detail):
        checks.append({"id": i, "name": name, "status": ok if isinstance(ok, str) else ("pass" if ok else "fail"), "detail": detail})
        log(f"[{checks[-1]['status']}] {i}. {name}: {detail}")

    t_start = time.time()
    sha = current_file_hash(p)
    add(1, "Circuit file and hash match", sha == p.file_hash, f"SHA-256 now {sha and sha[:12]}…, profile {p.file_hash[:12]}…")
    eng = engine_report()
    plat_ok = eng["fmpy"] and eng["platform"] in p.parameters["platforms"]
    add(2, "Backend capability supports interactive stepping", plat_ok, eng["message"] if not eng["fmpy"] else f"{eng['platform']} in {p.parameters['platforms']}")
    if not (sha == p.file_hash and plat_ok) or not p.parameters.get("mapping") or not p.parameters.get("gym"):
        for i, n in ((3, "FMU instance initializes and closes correctly"), (4, "Reset can be repeated and state is well defined"), (5, "Observation and action spaces validate"),
                     (6, "Action changes affect the designated FMU input"), (7, "A multi-step episode advances simulated time monotonically"),
                     (8, "Sampling and state synchronization meet configured assumptions"), (9, "Reward terms use available signals and finite values"),
                     (10, "Termination, truncation, invalid action and solver-error handling are tested"), (11, "Seed and scenario reproduction check"), (12, "Performance benchmark")):
            add(i, n, "fail", "not run: the checks above failed or the profile is not specified")
        return _summary(p, checks, t_start)
    spec = env_spec(p)
    g = spec["gym"]
    a = spec["mapping"]["action"]
    lo, hi = float(a["low"]), float(a["high"])
    try:
        env = MappedFmuEnv(spec)
        env.reset(seed=0)
        env.close()
        add(3, "FMU instance initializes and closes correctly", True, f"{env.fmu_info.model_name}, FMI {env.fmu_info.fmi_version} {env.fmu_info.fmu_type}")
    except Exception as exc:
        add(3, "FMU instance initializes and closes correctly", False, f"{type(exc).__name__}: {exc}")
        return _summary(p, checks, t_start)
    n = min(200, max(2, round(g["duration_s"] / g["tau_s"])))
    rng = np.random.default_rng(7)
    acts = [np.array([x]) for x in rng.uniform(lo, hi, n)]
    env = MappedFmuEnv(spec)
    try:
        t1, r1, _ = _episode(env, acts, seed=3)
        t2, r2, _ = _episode(env, acts, seed=3)
        same = len(t1) == len(t2) and all(np.array_equal(x, y) for x, y in zip(t1, t2))
        rel = max((float(np.max(np.abs(x - y) / np.maximum(np.abs(y), 1e-300))) for x, y in zip(t1, t2)), default=0.0) if len(t1) == len(t2) else math.inf
        add(4, "Reset can be repeated and state is well defined", same or rel <= 1e-9,
            "two episodes, same seed and actions: " + ("bit-identical" if same else f"max relative difference {rel:.2e} (≤ 1e-9 declared)") + " (B3 criterion 3)")
        add(11, "Seed and scenario reproduction check", same or rel <= 1e-9, f"seed 3, {len(acts)} random actions, reproduced")
        mid = (lo + hi) / 2
        ta, _, _ = _episode(env, [np.array([lo + 0.25 * (hi - lo)])] * n, seed=0)
        tb, _, _ = _episode(env, [np.array([lo + 0.75 * (hi - lo)])] * n, seed=0)
        diff = float(np.max(np.abs(ta[-1][:-1] - tb[-1][:-1])))
        add(6, "Action changes affect the designated FMU input", diff > 1e-9, f"constant actions at 25 % and 75 % of the range differ by {diff:.3g} (scaled) after {n} steps (B3 criterion 1)")
        env.reset(seed=0)
        terr = 0.0
        for k in range(n):
            env.step(np.array([mid]))
            terr = max(terr, abs(env.adapter.time - (k + 1) * env.tau) / ((k + 1) * env.tau))
        add(7, "A multi-step episode advances simulated time monotonically", terr <= 1e-12, f"t after k steps = k·Ts within {terr:.1e} relative (B3 criterion 2)")
        add(8, "Sampling and state synchronization meet configured assumptions", terr <= 1e-12,
            f"one doStep of Ts = {env.tau:g} s per action, outputs read at t + Ts; the solver's internal step is CVODE's own")
        with __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore")
            check_env(MappedFmuEnv(spec).unwrapped, skip_render_check=True)
        add(5, "Observation and action spaces validate", True, f"gymnasium check_env passed; obs {env.observation_space.shape}, action Box[{lo:g}, {hi:g}]")
        full, rew, info_end = _episode(env, [np.array([mid])] * env.n_steps, seed=0)
        finite = all(math.isfinite(r) for r in rew)
        add(9, "Reward terms use available signals and finite values", finite and info_end.get("termination") != "solver_failure",
            f"{len(rew)} steps at a constant mid-range action: rewards finite={finite}; episode end: {info_end.get('termination') or info_end.get('truncation')}")
        env.reset(seed=0)
        env.step(np.array([mid]))
        bad = []
        try:
            env.step(np.array([np.nan]))
            bad.append("non-finite action accepted")
        except ValueError:
            pass
        *_, info = env.step(np.array([hi + (hi - lo)]))
        if g.get("action_out_of_range", "clip") == "clip" and not info.get("action_clipped"):
            bad.append("out-of-range action not clipped")
        orig = env.adapter._slave.doStep

        def _fail(**_kw):
            raise RuntimeError("injected solver failure (verification)")

        env.adapter._slave.doStep = _fail
        _, r, term, trunc, info = env.step(np.array([mid]))
        env.adapter._slave.doStep = orig
        if not (term and not trunc and info.get("termination") == "solver_failure" and r < 0):
            bad.append("solver failure not reported as termination with a penalty")
        add(10, "Termination, truncation, invalid action and solver-error handling are tested", not bad,
            "; ".join(bad) or f"non-finite action rejected; out-of-range action {g.get('action_out_of_range', 'clip')}ped; injected solver failure → terminated, reward −{env.solver_failure_penalty:g}; horizon → truncated")
        env.reset(seed=0)
        wall = []
        for _ in range(min(env.n_steps, 1000)):
            t0 = time.perf_counter()
            env.step(np.array([mid]))
            wall.append(time.perf_counter() - t0)
        p50, p99 = float(np.percentile(wall, 50) * 1e3), float(np.percentile(wall, 99) * 1e3)
        add(12, "Performance benchmark", p50 <= LATENCY_P50_LIMIT_MS, f"wall time per step p50 {p50:.3f} ms, p99 {p99:.3f} ms over {len(wall)} steps (B3 criterion 6: p50 ≤ {LATENCY_P50_LIMIT_MS} ms)")
        native = _native_counterpart(p)
        if native is not None:
            native.reset()
            d = np.repeat([0.0, 0.3, 0.6, 0.45], 250)
            env.reset(seed=0)
            nat, fm = [], []
            out_names = [o["output"] for o in spec["mapping"]["observations"]]
            for x in d:
                s = native.simulate(float(x))
                nat.append((s.v_out, s.i_l))
                env.step(np.array([x]))
                fm.append((env._outputs["v_out"], env._outputs["i_L"]))
            nat, fm = np.array(nat), np.array(fm)
            nrms = [float(np.sqrt(np.mean((fm[:, j] - nat[:, j]) ** 2)) / np.ptp(nat[:, j])) for j in (0, 1)]
            add("4b", "Agreement with the native plant of the same equations", max(nrms) <= NATIVE_NRMS_LIMIT,
                f"duty levels 0/0.3/0.6/0.45 (B3 sequence): NRMS v_out {nrms[0]:.1e}, i_L {nrms[1]:.1e} (B3 criterion 4: ≤ 1 %); observations {out_names}")
    except FmuError as exc:
        add("x", "FMU error during verification", False, str(exc))
    finally:
        env.close()
    return _summary(p, checks, t_start)


def _summary(p, checks, t_start) -> dict:
    n_fail = sum(c["status"] == "fail" for c in checks)
    return {"status": "passed" if n_fail == 0 else "failed", "n_checks": len(checks), "n_passed": sum(c["status"] == "pass" for c in checks), "checks": checks,
            "fmu_sha256": p.file_hash, "config_hash": config_hash(p), "profile_id": p.profile_id, "profile_version": p.version, "wall_s": round(time.time() - t_start, 2)}


def record_verification(profile_id: str, version: int, report: dict, run_id: str, root=None):
    """Stores the result on that profile version (VALIDATED or FAILED) with the hashes it covers."""
    from driveflow.runs.schemas import EnvironmentCapability, utc_now

    p = _store(root).get_profile(profile_id, version)
    rec = {**{k: report[k] for k in ("status", "n_checks", "n_passed", "fmu_sha256", "config_hash")}, "run_id": run_id, "at": utc_now()}
    passed = report["status"] == "passed"
    caps = EnvironmentCapability(can_simulate_batch=passed, can_step_interactively=passed, can_reset_episode=passed, validated_at=rec["at"],
                                 diagnostic_messages=tuple(f"[{c['status']}] {c['name']}: {c['detail']}" for c in report["checks"]))
    status = p.status if p.status == "unsupported" else ("validated" if passed else "failed")
    return _replace(p, root, verification=rec, _status=status, _capabilities=caps)


def mark_verifying(p, root=None):
    return _replace(p, root, _status="verifying" if p.status != "unsupported" else "unsupported")


# --------------------------------------------------------------------------------- publication
def env_id_for(profile_id: str) -> str:
    return "SPADE/Fmu" + "".join(w.capitalize() for w in re.split(r"[_-]+", profile_id)) + "-v0"


def _index_path() -> Path:
    from driveflow.runs.store import workspace_root

    return workspace_root() / "fmu_envs" / "published.json"


def published() -> dict:
    idx = _index_path()
    try:
        return json.loads(idx.read_text()) if idx.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def publish(p, root=None) -> str:
    """Registers SPADE/Fmu<Profile>-v0. Refused (ValueError) unless publish_blockers(p) is empty."""
    blockers = publish_blockers(p)
    if blockers:
        raise ValueError("Publish to Environments is disabled: " + "; ".join(blockers))
    from driveflow.envs import registry

    env_id = env_id_for(p.profile_id)
    idx = _index_path()
    data = published()
    data[env_id] = {"profile_id": p.profile_id, "version": p.version, "spec": env_spec(p), "fmu_sha256": p.file_hash, "config_hash": config_hash(p)}
    idx.parent.mkdir(parents=True, exist_ok=True)
    idx.write_text(json.dumps(data, indent=2))
    registry.register_fmu_profile(env_id, data[env_id])
    _replace(p, root, published_env_id=env_id, _status="published")
    return env_id
