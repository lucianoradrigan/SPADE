"""FmuAdapter: one FMU driven step by step through fmpy (task B2, D6; FMI 3.0 and Model Exchange: P1).

Supported interfaces (task P1, round 2026-10-26: the interfaces a PLECS 5.0 export can produce):

| FMI | Co-Simulation | Model Exchange |
|---|---|---|
| 2.0 | the FMU's own solver steps (B2) | SPADE integrates (fixed-step RK4) |
| 3.0 | the FMU's own solver steps | SPADE integrates (fixed-step RK4) |

- ``load(path)``: reads modelDescription.xml -- FMI version, the interfaces offered, inputs,
  outputs, parameters, causality, units, declared ranges, GUID, generation tool and the binary
  platforms in the archive. Unlike a .plecs file this introspection is real, so the GUI mapping can
  propose roles from it.
- ``reset(params, initial_state, seed)``: frees and re-instantiates the FMU from the extracted
  archive (more reproducible across FMUs than fmi2Reset), sets parameters and start values, runs
  initialisation (and, for Model Exchange, the initial event iteration), t = 0. ``seed`` is
  recorded; the FMUs here are deterministic.
- ``set_inputs(dict)``, ``do_step(t, dt)``, ``get_outputs()``, ``close()``.
- Model Exchange: ``do_step`` integrates the FMU's continuous states from t to t + dt with
  classical fixed-step RK4 in ``me_substeps`` substeps, stops exactly at the FMU's time events,
  handles step events (completedIntegratorStep) and state events: an event-indicator sign change
  inside a substep is located by bisection of the RK4 step (to ``me_event_tol`` s) before the event
  iteration runs, so a switching instant is not rounded to the substep. A change of a discrete-variability input enters
  Event Mode, as FMI requires for e.g. gate signals.
- Typed errors (FmuError.code): load, platform, solver, nan, out_of_range.
- Instrumentation: wall time of every step (p50, p99) and the count of solver failures.

fmpy is the optional extra ``[fmu]`` (D3); importing this module without it raises FmuError("load").
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

CODES = ("load", "platform", "solver", "nan", "out_of_range")
INTERFACES = {"cs": "CoSimulation", "me": "ModelExchange"}
_FLOAT_TYPES = ("Real", "Float64", "Float32")
_INT_TYPES = ("Integer", "Int8", "UInt8", "Int16", "UInt16", "Int32", "UInt32", "Int64", "UInt64", "Enumeration")
_BOOL_TYPES = ("Boolean",)


class FmuError(RuntimeError):
    def __init__(self, code: str, message: str):
        assert code in CODES
        super().__init__(f"[{code}] {message}")
        self.code = code


@dataclasses.dataclass(frozen=True)
class Variable:
    name: str
    value_reference: int
    causality: str
    variability: str
    unit: str | None
    minimum: float | None
    maximum: float | None
    start: float | None
    type: str = "Real"


@dataclasses.dataclass(frozen=True)
class FmuInfo:
    path: str
    sha256: str
    model_name: str
    guid: str
    fmi_version: str
    #: the interface the adapter uses by default: CoSimulation when offered, else ModelExchange
    fmu_type: str
    platforms: tuple
    variables: tuple
    #: every interface the FMU offers ("CoSimulation", "ModelExchange")
    fmu_types: tuple = ()
    generation_tool: str | None = None
    n_states: int = 0
    n_event_indicators: int = 0
    needs_completed_integrator_step: bool = True
    #: Co-Simulation only: fixedInternalStepSize declared by the FMU (s); the communication step must be a multiple
    fixed_internal_step: float | None = None

    def by_causality(self, causality: str) -> list[Variable]:
        return [v for v in self.variables if v.causality == causality]

    @property
    def inputs(self):
        return self.by_causality("input")

    @property
    def outputs(self):
        return self.by_causality("output")

    @property
    def parameters(self):
        return self.by_causality("parameter")

    @property
    def is_plecs(self) -> bool:
        return bool(self.generation_tool) and "plecs" in self.generation_tool.lower()


def _fmpy():
    try:
        import fmpy  # noqa: F401
        from fmpy import read_model_description
        from fmpy.fmi2 import FMU2Slave

        return fmpy, read_model_description, FMU2Slave
    except ImportError:
        raise FmuError("load", "fmpy is not installed: pip install \"driveflow[fmu]\"") from None


def _num(x):
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def inspect_fmu(path) -> FmuInfo:
    """Model description and archive platforms, without loading the binary."""
    fmpy, read_md, _ = _fmpy()
    p = Path(path)
    if not p.is_file():
        raise FmuError("load", f"no such FMU: {p}")
    try:
        md = read_md(str(p))
        with zipfile.ZipFile(p) as z:
            platforms = tuple(sorted({n.split("/")[1] for n in z.namelist() if n.startswith("binaries/") and n.count("/") >= 2 and n.split("/")[1]}))
    except Exception as exc:
        raise FmuError("load", f"cannot read {p.name}: {exc}") from None
    types = tuple(t for t, attr in (("CoSimulation", md.coSimulation), ("ModelExchange", md.modelExchange)) if attr is not None)
    numeric = _FLOAT_TYPES + _INT_TYPES + _BOOL_TYPES
    variables = tuple(Variable(v.name, int(v.valueReference), v.causality, v.variability, getattr(v, "unit", None), _num(getattr(v, "min", None)),
                               _num(getattr(v, "max", None)), _num(v.start), v.type)
                      for v in md.modelVariables if v.type in numeric)
    me = md.modelExchange
    return FmuInfo(str(p), hashlib.sha256(p.read_bytes()).hexdigest(), md.modelName, md.guid, md.fmiVersion, types[0] if types else "unknown", platforms, variables,
                   fmu_types=types, generation_tool=getattr(md, "generationTool", None), n_states=int(md.numberOfContinuousStates or 0),
                   n_event_indicators=int(md.numberOfEventIndicators or 0),
                   needs_completed_integrator_step=bool(getattr(me, "needsCompletedIntegratorStep", True)) if me is not None else True,
                   fixed_internal_step=_num(getattr(md.coSimulation, "fixedInternalStepSize", None)) if md.coSimulation is not None else None)


def current_platform() -> str:
    """The FMI 2.0 platform folder of this machine (linux64, win64, darwin64)."""
    fmpy, _, _ = _fmpy()
    return fmpy.platform


def current_platform_fmi3() -> str:
    """The FMI 3.0 platform tuple of this machine (x86_64-linux, x86_64-windows, aarch64-darwin, …)."""
    fmpy, _, _ = _fmpy()
    return fmpy.platform_tuple


def platform_for(fmi_version: str) -> str:
    return current_platform_fmi3() if str(fmi_version).startswith("3") else current_platform()


def supported(info: FmuInfo) -> tuple[bool, str]:
    """(True, "") when this adapter can drive the FMU here; else (False, the reason)."""
    if info.fmi_version not in ("2.0",) and not str(info.fmi_version).startswith("3"):
        return False, f"FMI {info.fmi_version}: only FMI 2.0 and 3.0 are supported"
    if not info.fmu_types:
        return False, "the FMU offers neither Co-Simulation nor Model Exchange"
    try:
        plat = platform_for(info.fmi_version)
    except FmuError as exc:
        return False, str(exc)
    if plat not in info.platforms:
        return False, f"binaries for {list(info.platforms) or 'no platform'}, not {plat}: Not supported on this machine"
    return True, ""


class FmuAdapter:
    def __init__(self, slave_factory=None, interface: str = "auto", me_substeps: int = 10, me_event_tol: float = 1e-9):
        """slave_factory(info, unzip_dir) -> an object with fmpy's FMU2Slave methods (tests inject a fake).
        interface: "auto" (Co-Simulation when offered, else Model Exchange), "cs" or "me".
        me_substeps: RK4 substeps per do_step for Model Exchange."""
        if interface not in ("auto", "cs", "me"):
            raise ValueError("interface must be 'auto', 'cs' or 'me'")
        if not (isinstance(me_substeps, int) and 1 <= me_substeps <= 100_000):
            raise ValueError("me_substeps must be an int in [1, 100000]")
        self._factory = slave_factory
        self.interface_request = interface
        self.me_substeps = me_substeps
        self.me_event_tol = float(me_event_tol)
        self.info: FmuInfo | None = None
        self.interface: str | None = None
        self._dir: str | None = None
        self._slave = None
        self._t = 0.0
        self._inputs: dict = {}
        self._me = None
        self.step_wall_s: list[float] = []
        self.solver_failures = 0
        self.events = 0
        self.seed = None

    # ------------------------------------------------------------------ life cycle
    @property
    def fmi3(self) -> bool:
        return self.info is not None and str(self.info.fmi_version).startswith("3")

    def _choose_interface(self, info: FmuInfo) -> str:
        want = self.interface_request
        if want == "auto":
            return "CoSimulation" if "CoSimulation" in info.fmu_types else "ModelExchange"
        if INTERFACES[want] not in info.fmu_types:
            raise FmuError("load", f"{Path(info.path).name} does not offer {INTERFACES[want]} (it offers {list(info.fmu_types)})")
        return INTERFACES[want]

    def load(self, path) -> FmuInfo:
        info = inspect_fmu(path)
        if self._factory is None:
            if info.fmi_version != "2.0" and not str(info.fmi_version).startswith("3"):
                raise FmuError("load", f"{Path(path).name} is FMI {info.fmi_version}; FMI 2.0 or 3.0 is required")
            plat = platform_for(info.fmi_version)
            if plat not in info.platforms:
                raise FmuError("platform", f"{Path(path).name} has binaries for {list(info.platforms) or 'no platform'}, not {plat}: Not supported on this machine")
            self.interface = self._choose_interface(info)
        else:
            if info.fmi_version != "2.0" or info.fmu_type != "CoSimulation":
                raise FmuError("load", f"{Path(path).name} is FMI {info.fmi_version} {info.fmu_type}; an FMI 2.0 Co-Simulation FMU is required")
            self.interface = "CoSimulation"
        self.info = info
        self._dir = tempfile.mkdtemp(prefix="spade_fmu_")
        try:
            with zipfile.ZipFile(path) as z:
                z.extractall(self._dir)
        except Exception as exc:
            raise FmuError("load", f"cannot extract {Path(path).name}: {exc}") from None
        return info

    def _make_slave(self):
        if self._factory is not None:
            return self._factory(self.info, self._dir)
        _, read_md, FMU2Slave = _fmpy()
        md = read_md(self.info.path)
        cs = self.interface == "CoSimulation"
        ident = (md.coSimulation if cs else md.modelExchange).modelIdentifier
        kw = dict(guid=md.guid, unzipDirectory=self._dir, modelIdentifier=ident, instanceName="spade")
        if self.fmi3:
            from fmpy.fmi3 import FMU3Model, FMU3Slave

            return (FMU3Slave if cs else FMU3Model)(**kw)
        from fmpy.fmi2 import FMU2Model

        return (FMU2Slave if cs else FMU2Model)(**kw)

    def reset(self, params: dict | None = None, initial_state: dict | None = None, seed: int | None = None, start_time: float = 0.0) -> None:
        if self.info is None:
            raise FmuError("load", "load() an FMU first")
        self._free()
        self.seed = seed
        try:
            s = self._make_slave()
            values = {**(initial_state or {}), **(params or {})}
            if self.fmi3:
                s.instantiate()
                if values:
                    self._set(s, values)
                s.enterInitializationMode(startTime=start_time)
                s.exitInitializationMode()
            else:
                s.instantiate()
                s.setupExperiment(startTime=start_time)
                if values:
                    self._set(s, values)
                s.enterInitializationMode()
                s.exitInitializationMode()
            self._slave, self._t, self._inputs = s, float(start_time), {}
            self._me = None
            if self.interface == "ModelExchange":
                self._me_init(start_time)
        except FmuError:
            raise
        except Exception as exc:
            raise FmuError("solver", f"initialisation failed: {exc}") from None

    def _vrs(self, values: dict) -> list[int]:
        idx = {v.name: v.value_reference for v in self.info.variables}
        unknown = [n for n in values if n not in idx]
        if unknown:
            raise FmuError("load", f"unknown variable(s) {unknown} in {self.info.model_name}")
        return [idx[n] for n in values]

    def _set(self, s, values: dict) -> None:
        """Sets values grouped by FMI type (floats, integers, booleans)."""
        by = {v.name: v for v in self.info.variables}
        self._vrs(values)  # raises on unknown names
        groups = {"float": {}, "int": {}, "bool": {}}
        for n, x in values.items():
            t = by[n].type
            groups["float" if t in _FLOAT_TYPES else "int" if t in _INT_TYPES else "bool"][n] = x
        if groups["float"]:
            vr, val = [by[n].value_reference for n in groups["float"]], [float(x) for x in groups["float"].values()]
            (s.setFloat64 if self.fmi3 else s.setReal)(vr, val)
        if groups["int"]:
            vr, val = [by[n].value_reference for n in groups["int"]], [int(round(float(x))) for x in groups["int"].values()]
            (s.setInt32 if self.fmi3 else s.setInteger)(vr, val)
        if groups["bool"]:
            vr, val = [by[n].value_reference for n in groups["bool"]], [bool(x) for x in groups["bool"].values()]
            s.setBoolean(vr, val)

    def _get(self, variables: list[Variable]) -> list[float]:
        s, out = self._slave, {}
        floats = [v for v in variables if v.type in _FLOAT_TYPES]
        ints = [v for v in variables if v.type in _INT_TYPES]
        bools = [v for v in variables if v.type in _BOOL_TYPES]
        if floats:
            vals = (s.getFloat64 if self.fmi3 else s.getReal)([v.value_reference for v in floats])
            out.update({v.name: float(x) for v, x in zip(floats, vals)})
        if ints:
            vals = (s.getInt32 if self.fmi3 else s.getInteger)([v.value_reference for v in ints])
            out.update({v.name: float(x) for v, x in zip(ints, vals)})
        if bools:
            vals = s.getBoolean([v.value_reference for v in bools])
            out.update({v.name: float(bool(x)) for v, x in zip(bools, vals)})
        return [out[v.name] for v in variables]

    def set_inputs(self, values: dict) -> None:
        names = {v.name: v for v in self.info.inputs}
        bad = [n for n in values if n not in names]
        if bad:
            raise FmuError("load", f"{bad} are not inputs of {self.info.model_name} (inputs: {sorted(names)})")
        for n, x in values.items():
            if not math.isfinite(float(x)):
                raise FmuError("nan", f"input {n} = {x!r}")
        if self._factory is not None:  # fake slaves implement the FMI 2.0 float API only
            self._slave.setReal(self._vrs(values), [float(x) for x in values.values()])
            self._inputs.update(values)
            return
        discrete_change = self.interface == "ModelExchange" and any(names[n].variability == "discrete" and self._inputs.get(n) != values[n] for n in values)
        self._set(self._slave, values)
        self._inputs.update(values)
        if discrete_change:
            try:
                self._me_event(input_event=True)
            except FmuError:
                raise
            except Exception as exc:
                self.solver_failures += 1
                raise FmuError("solver", f"event iteration after a discrete input at t = {self._t:g} s failed: {exc}") from None

    def do_step(self, t: float, dt: float) -> None:
        if self._slave is None:
            raise FmuError("load", "reset() before do_step()")
        if abs(t - self._t) > 1e-12 * max(1.0, abs(t)):
            raise FmuError("solver", f"non-monotone step: FMU at t = {self._t!r}, asked to step from {t!r}")
        t0 = time.perf_counter()
        try:
            if self.interface == "ModelExchange":
                self._me_step(t, dt)
                status = 0
            elif self.fmi3:
                _event, terminate, _early, last = self._slave.doStep(currentCommunicationPoint=t, communicationStepSize=dt)
                if terminate:
                    raise FmuError("solver", f"the FMU requested termination at t = {last:g} s")
                if abs(last - (t + dt)) > 1e-12 * max(1.0, abs(t + dt)):
                    fixed = self.info.fixed_internal_step
                    raise FmuError("solver", f"doStep from t = {t:g} s stopped at {last:g} s instead of {t + dt:g} s"
                                   + (f"; the FMU declares fixedInternalStepSize = {fixed:g} s: use a communication step that is a multiple of it" if fixed else ""))
                status = 0
            else:
                status = self._slave.doStep(currentCommunicationPoint=t, communicationStepSize=dt)
        except FmuError:
            self.solver_failures += 1
            raise
        except Exception as exc:
            self.solver_failures += 1
            raise FmuError("solver", f"doStep at t = {t:g} s failed: {exc}") from None
        finally:
            self.step_wall_s.append(time.perf_counter() - t0)
        if status not in (None, 0):
            self.solver_failures += 1
            raise FmuError("solver", f"doStep at t = {t:g} s returned status {status}")
        self._t = t + dt

    # ------------------------------------------------------------------ Model Exchange
    def _me_init(self, start_time: float) -> None:
        nx, nz = self.info.n_states, self.info.n_event_indicators
        self._me = {"nx": nx, "nz": nz, "x": np.zeros(nx), "dx": np.zeros(nx), "z": np.zeros(nz), "next_event": None}
        if self.fmi3:
            self._me_iterate(self._slave.updateDiscreteStates)
        else:
            self._me_iterate(self._slave.newDiscreteStates)
        self._slave.enterContinuousTimeMode()
        if nz:
            self._me_z()

    def _ptr(self, a: np.ndarray):
        from ctypes import POINTER, c_double

        return a.ctypes.data_as(POINTER(c_double))

    def _me_z(self) -> np.ndarray:
        m = self._me
        self._slave.getEventIndicators(self._ptr(m["z"]), m["nz"])
        return m["z"].copy()

    def _me_iterate(self, update) -> None:
        needed, terminate = True, False
        while needed and not terminate:
            needed, terminate, _nom, _vals, defined, nxt = update()
        if terminate:
            raise FmuError("solver", f"the FMU requested termination at t = {self._t:g} s")
        self._me["next_event"] = nxt if defined else None

    def _me_event(self, input_event: bool = False) -> None:
        s = self._slave
        s.enterEventMode()
        self._me_iterate(s.updateDiscreteStates if self.fmi3 else s.newDiscreteStates)
        s.enterContinuousTimeMode()
        self.events += 1
        if self._me["nz"]:
            self._me_z()

    def _derivatives(self, t: float, x: np.ndarray) -> np.ndarray:
        s, m = self._slave, self._me
        s.setTime(t)
        if m["nx"]:
            xx = np.ascontiguousarray(x, dtype=float)
            s.setContinuousStates(self._ptr(xx), m["nx"])
            dx = np.zeros(m["nx"])
            (s.getContinuousStateDerivatives if self.fmi3 else s.getDerivatives)(self._ptr(dx), m["nx"])
            return dx
        return np.zeros(0)

    def _rk4(self, t: float, x: np.ndarray, h: float) -> np.ndarray:
        k1 = self._derivatives(t, x)
        k2 = self._derivatives(t + h / 2, x + h / 2 * k1)
        k3 = self._derivatives(t + h / 2, x + h / 2 * k2)
        k4 = self._derivatives(t + h, x + h * k3)
        return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def _z_at(self, t: float, x: np.ndarray) -> np.ndarray:
        s, m = self._slave, self._me
        s.setTime(t)
        if m["nx"]:
            s.setContinuousStates(self._ptr(np.ascontiguousarray(x)), m["nx"])
        z = np.zeros(m["nz"])
        s.getEventIndicators(self._ptr(z), m["nz"])
        return z

    @staticmethod
    def _crossed(z0: np.ndarray, z1: np.ndarray) -> bool:
        return bool(np.any(np.sign(z0) * np.sign(z1) < 0) or np.any((z0 != 0) & (z1 == 0)))

    def _me_step(self, t: float, dt: float) -> None:
        s, m = self._slave, self._me
        t_end, h_max, tt = t + dt, dt / self.me_substeps, t
        x = np.zeros(m["nx"])
        if m["nx"]:
            s.getContinuousStates(self._ptr(x), m["nx"])
        tol = 1e-12 * max(1.0, abs(t_end))
        while tt < t_end - tol:
            h = min(h_max, t_end - tt)
            nxt = m["next_event"]
            time_event = nxt is not None and nxt < tt + h - tol
            if time_event:
                h = max(nxt - tt, 0.0)
            elif nxt is not None and abs(nxt - (tt + h)) <= tol:
                time_event = True
            z0 = m["z"].copy() if m["nz"] else None
            x_new = self._rk4(tt, x, h) if (m["nx"] and h > 0) else x
            state_event = False
            if m["nz"]:
                z1 = self._z_at(tt + h, x_new)
                if self._crossed(z0, z1):
                    # locate the first crossing by bisection of the RK4 step
                    lo, hi = 0.0, h
                    while hi - lo > self.me_event_tol:
                        mid = (lo + hi) / 2
                        if self._crossed(z0, self._z_at(tt + mid, self._rk4(tt, x, mid) if m["nx"] else x)):
                            hi = mid
                        else:
                            lo = mid
                    h, x_new, state_event, time_event = hi, (self._rk4(tt, x, hi) if m["nx"] else x), True, False
            if m["nx"] and h > 0:
                if not np.all(np.isfinite(x_new)):
                    raise FmuError("nan", f"continuous states became non-finite at t = {tt + h:g} s")
                s.setContinuousStates(self._ptr(np.ascontiguousarray(x_new)), m["nx"])
            x = x_new
            tt = tt + h
            s.setTime(tt)
            if m["nz"]:
                self._me_z()
            step_event = False
            if self.info.needs_completed_integrator_step:
                r = s.completedIntegratorStep()
                step_event, terminate = (r[0], r[1]) if isinstance(r, tuple) else (bool(r), False)
                if terminate:
                    raise FmuError("solver", f"the FMU requested termination at t = {tt:g} s")
            if time_event or state_event or step_event:
                self._me_event()
                if m["nx"]:
                    s.getContinuousStates(self._ptr(x), m["nx"])
        s.setTime(t_end)

    # ------------------------------------------------------------------ outputs, stats, close
    def get_outputs(self, check_range: bool = True) -> dict:
        outs = self.info.outputs
        if self._factory is not None:
            vals = self._slave.getReal([v.value_reference for v in outs])
        else:
            vals = self._get(outs)
        res = dict(zip([v.name for v in outs], (float(x) for x in vals)))
        for v in outs:
            x = res[v.name]
            if not math.isfinite(x):
                raise FmuError("nan", f"output {v.name} is {x!r} at t = {self._t:g} s")
            if check_range and ((v.minimum is not None and x < v.minimum) or (v.maximum is not None and x > v.maximum)):
                raise FmuError("out_of_range", f"output {v.name} = {x:g} outside its declared [{v.minimum}, {v.maximum}]")
        return res

    @property
    def time(self) -> float:
        return self._t

    def stats(self) -> dict:
        w = np.asarray(self.step_wall_s)
        return {"n_steps": int(len(w)), "p50_ms": float(np.percentile(w, 50) * 1e3) if len(w) else None,
                "p99_ms": float(np.percentile(w, 99) * 1e3) if len(w) else None, "solver_failures": self.solver_failures,
                "interface": self.interface, "fmi_version": self.info.fmi_version if self.info else None, "events": self.events}

    def _free(self) -> None:
        if self._slave is not None:
            try:
                self._slave.terminate()
            except Exception:
                pass
            try:
                self._slave.freeInstance()
            except Exception:
                pass
            self._slave = None

    def close(self) -> None:
        self._free()
        if self._dir:
            shutil.rmtree(self._dir, ignore_errors=True)
            self._dir = None
