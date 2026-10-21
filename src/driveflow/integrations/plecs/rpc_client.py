"""Client for the PLECS Standalone RPC interface -- expansion task E2 (spec Sec. 3.4 stage 2, 3.14.5).

Pattern from Fund. ``8_PE_Simulation_Automation/PlecsAutomation/Data acquisition.ipynb``
(``plecs.load`` / ``plecs.simulate(model, {"ModelVars": ...})`` / ``plecs.close``), rewritten on the
standard library's ``xmlrpc.client`` without that notebook's defects: each result is returned with
the ModelVars that produced it (the notebook wrote every row with combination 0's parameters), the
URL is built and checked instead of hand-typed (it had a stray quote), paths are pathlib objects, and
there are no threads -- PLECS executes one blocking command at a time.

What the PLECS documentation states (https://docs.plexim.com/plecs/latest/scripting/, read
2026-10-21), and this client relies on:
- RPC is DISABLED by default (PLECS preferences) and accepts clients on the SAME computer only;
  the examples use http://localhost:1080.
- ``plecs.load(absolute_path)``, ``plecs.close(model_name)``, ``plecs.simulate(model_name, opts)``
  where opts may hold ``ModelVars`` and ``SolverOpts`` (``TimeSpan``, ``Timeout``, ``OutputTimes``,
  ...; ``StopTime`` is obsolete). ``simulate`` blocks; a second blocking call while one runs is an
  error.
- With outports on the top level, ``simulate`` returns ``{"Time": [...], "Values": [...]}``, one
  signal per outport, in port order.
- There is NO documented RPC command for the PLECS version, so this client never reports one:
  ``EngineCapabilityReport.plecs_version`` stays "not reported by the RPC interface".

Every failure is a ``PlecsRpcError`` with a ``RpcErrorCode`` whose message is the spec's diagnostic
text (e.g. "Cannot reach RPC endpoint"). Results never pass silently: non-finite values, a wrong
shape or a missing output are errors, and a simulation that does not finish within the per-run
time limit is ``TIMEOUT``.
"""

from __future__ import annotations

import dataclasses
import http.client
import ipaddress
import math
import socket
import threading
import time
import xmlrpc.client
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_URL = "http://localhost:1080"
#: Client-side socket margin over the per-run limit PLECS enforces itself (SolverOpts.Timeout).
SOCKET_MARGIN_S = 5.0
VERSION_NOT_REPORTED = "not reported by the RPC interface"


class RpcErrorCode(str, Enum):
    UNREACHABLE = "unreachable"
    REMOTE_HOST = "remote_host"
    BAD_URL = "bad_url"
    TIMEOUT = "timeout"
    SIMULATION_FAILED = "simulation_failed"
    BUSY = "busy"
    CONNECTION_LOST = "connection_lost"
    MODEL_NOT_FOUND = "model_not_found"
    BAD_RESULT = "bad_result"


MESSAGES = {
    RpcErrorCode.UNREACHABLE: "Cannot reach RPC endpoint",
    RpcErrorCode.REMOTE_HOST: "Backend running on a different host",
    RpcErrorCode.BAD_URL: "Invalid RPC endpoint address",
    RpcErrorCode.TIMEOUT: "Simulation exceeded its time limit",
    RpcErrorCode.SIMULATION_FAILED: "Simulation failed in PLECS",
    RpcErrorCode.BUSY: "PLECS is busy with another blocking command",
    RpcErrorCode.CONNECTION_LOST: "Connection to PLECS lost",
    RpcErrorCode.MODEL_NOT_FOUND: "Circuit file not found",
    RpcErrorCode.BAD_RESULT: "PLECS returned an unusable result",
}
HINTS = {
    RpcErrorCode.UNREACHABLE: "Start PLECS on the computer that runs SPADE's backend and enable RPC in its preferences (it is disabled by default); check the port.",
    RpcErrorCode.REMOTE_HOST: "PLECS RPC is available only when SPADE's backend runs on the same computer. Use localhost.",
    RpcErrorCode.BAD_URL: "Use http://localhost:<port>, e.g. http://localhost:1080.",
    RpcErrorCode.TIMEOUT: "Increase the per-run time limit or shorten the simulated time; the run is recorded as failed, not interpolated.",
    RpcErrorCode.SIMULATION_FAILED: "Open the circuit in PLECS with these parameters to see the solver message.",
    RpcErrorCode.BUSY: "Another simulation is running on this PLECS instance; SPADE serialises its own jobs per engine.",
    RpcErrorCode.CONNECTION_LOST: "PLECS stopped answering during the call; restart it and re-run the failed conditions.",
    RpcErrorCode.MODEL_NOT_FOUND: "The .plecs file must exist on the computer where PLECS runs.",
    RpcErrorCode.BAD_RESULT: "The circuit's top-level outports must match the interface manifest.",
}


class PlecsRpcError(Exception):
    """``answered``: the server replied (an XML-RPC fault), so it is reachable even though the
    call failed."""

    def __init__(self, code: RpcErrorCode, detail: str = "", answered: bool = False):
        self.code = RpcErrorCode(code)
        self.detail = detail
        self.answered = answered
        super().__init__(f"{MESSAGES[self.code]}{': ' + detail if detail else ''}")

    @property
    def hint(self) -> str:
        return HINTS[self.code]


def check_local_url(url: str) -> str:
    """The normalised URL, or PlecsRpcError: only http://localhost / loopback addresses are
    accepted, because PLECS refuses RPC from other computers."""
    try:
        parsed = urlparse(str(url).strip())
        parsed.port  # noqa: B018 -- raises ValueError for a malformed port
    except ValueError:
        raise PlecsRpcError(RpcErrorCode.BAD_URL, f"{url!r}") from None
    if parsed.scheme != "http" or not parsed.hostname:
        raise PlecsRpcError(RpcErrorCode.BAD_URL, f"{url!r}")
    host = parsed.hostname
    if host != "localhost":
        try:
            if not ipaddress.ip_address(host).is_loopback:
                raise PlecsRpcError(RpcErrorCode.REMOTE_HOST, f"{host} is not this computer")
        except ValueError:
            raise PlecsRpcError(RpcErrorCode.REMOTE_HOST, f"{host} is not localhost") from None
    return f"http://{parsed.netloc}{parsed.path or ''}"


class _TimeoutTransport(xmlrpc.client.Transport):
    def __init__(self, timeout_s: float):
        super().__init__()
        self.timeout_s = timeout_s

    def make_connection(self, host):
        conn = super().make_connection(host)
        conn.timeout = self.timeout_s
        return conn


#: One lock per endpoint: PLECS executes a single blocking command at a time, so calls from this
#: process are serialised; jobs from other processes are serialised by the job engine lock.
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(url: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(url, threading.Lock())


@dataclasses.dataclass(frozen=True)
class EngineCapabilityReport:
    """What a connection test actually showed (spec Sec. 3.4 stage 2). Nothing is inferred."""

    url: str
    reachable: bool
    plecs_version: str = VERSION_NOT_REPORTED
    methods: tuple | None = None
    can_simulate_batch: bool = False
    can_step_interactively: bool = False
    checked: tuple = ()
    not_checked: tuple = ()
    error_code: str | None = None
    message: str = ""
    hint: str = ""
    latency_ms: float | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class SimulationResult:
    time: list
    outputs: dict  # name -> list, aligned with time
    model_vars: dict
    wall_s: float


class PlecsRpcClient:
    def __init__(self, url: str = DEFAULT_URL, call_timeout_s: float = 10.0):
        self.url = check_local_url(url)
        self.call_timeout_s = float(call_timeout_s)

    def _proxy(self, timeout_s: float) -> xmlrpc.client.ServerProxy:
        return xmlrpc.client.ServerProxy(self.url, transport=_TimeoutTransport(timeout_s), allow_none=True)

    def _call(self, method: str, *args, timeout_s: float | None = None):
        proxy = self._proxy(self.call_timeout_s if timeout_s is None else timeout_s)
        target = proxy
        for part in method.split("."):
            target = getattr(target, part)
        with _lock_for(self.url):
            try:
                return target(*args)
            except ConnectionRefusedError as exc:
                raise PlecsRpcError(RpcErrorCode.UNREACHABLE, f"{self.url} ({exc.strerror or exc})") from None
            except socket.timeout:
                raise PlecsRpcError(RpcErrorCode.TIMEOUT, f"no answer from {self.url} within {timeout_s or self.call_timeout_s:g} s") from None
            except (ConnectionResetError, BrokenPipeError, http.client.RemoteDisconnected, http.client.IncompleteRead) as exc:
                raise PlecsRpcError(RpcErrorCode.CONNECTION_LOST, f"{type(exc).__name__} during {method}") from None
            except OSError as exc:
                raise PlecsRpcError(RpcErrorCode.UNREACHABLE, f"{self.url} ({exc})") from None
            except xmlrpc.client.Fault as fault:
                raise _classify_fault(fault, method) from None
            except xmlrpc.client.ProtocolError as exc:
                raise PlecsRpcError(RpcErrorCode.UNREACHABLE, f"HTTP {exc.errcode} {exc.errmsg} from {self.url}") from None

    def test_connection(self) -> EngineCapabilityReport:
        """Reachability and what the server answers to. ``system.listMethods`` is tried; a server
        that answers with a fault is still reachable (PLECS does not document introspection)."""
        not_checked = ("PLECS version (no documented RPC command)", "license (not exposed over RPC)",
                       "interactive stepping (needs a Co-Simulation FMU, task E3)", "circuit loading (needs a circuit file)")
        t0 = time.perf_counter()
        try:
            methods = tuple(self._call("system.listMethods"))
        except PlecsRpcError as exc:
            if exc.answered:  # a fault: the server answered, it just does not list its methods
                methods = None
            else:
                return EngineCapabilityReport(url=self.url, reachable=False, error_code=exc.code.value, message=str(exc), hint=exc.hint,
                                              checked=("RPC connectivity",), not_checked=not_checked)
        latency = (time.perf_counter() - t0) * 1e3
        has_plecs = methods is None or any(m.startswith("plecs.") for m in methods)
        msg = "RPC endpoint answered." + ("" if methods is None else f" {len(methods)} method(s) listed.")
        if methods is not None and not has_plecs:
            msg += " None of them is a plecs.* command: this may not be PLECS."
        return EngineCapabilityReport(url=self.url, reachable=True, methods=methods, can_simulate_batch=False, checked=("RPC connectivity",),
                                      not_checked=not_checked + ("batch simulation (shown only by a completed sweep)",), message=msg, latency_ms=round(latency, 2))

    def load(self, circuit_path: str | Path) -> str:
        """Loads the circuit; returns the model name PLECS uses (the file stem)."""
        path = Path(circuit_path).expanduser().resolve()
        if not path.is_file():
            raise PlecsRpcError(RpcErrorCode.MODEL_NOT_FOUND, str(path.name))
        self._call("plecs.load", str(path))
        return path.stem

    def close(self, model_name: str) -> None:
        try:
            self._call("plecs.close", model_name)
        except PlecsRpcError as exc:
            if exc.code not in (RpcErrorCode.UNREACHABLE, RpcErrorCode.CONNECTION_LOST):
                raise

    def simulate(self, model_name: str, output_names: list[str], model_vars: dict | None = None, solver_opts: dict | None = None,
                 time_limit_s: float = 60.0) -> SimulationResult:
        """One blocking run. ``time_limit_s`` is passed to PLECS as SolverOpts.Timeout and bounds
        the socket wait (plus a margin). The result is checked against ``output_names`` (the
        manifest's outports, in port order); any mismatch or non-finite sample is an error."""
        if not (math.isfinite(time_limit_s) and time_limit_s > 0):
            raise ValueError(f"time_limit_s must be > 0, got {time_limit_s!r}")
        opts = {"SolverOpts": {**(solver_opts or {}), "Timeout": float(time_limit_s)}}
        if model_vars:
            opts["ModelVars"] = {k: float(v) for k, v in model_vars.items()}
        t0 = time.perf_counter()
        raw = self._call("plecs.simulate", model_name, opts, timeout_s=time_limit_s + SOCKET_MARGIN_S)
        wall = time.perf_counter() - t0
        return SimulationResult(*_parse_result(raw, output_names), model_vars=dict(opts.get("ModelVars", {})), wall_s=wall)


def _classify_fault(fault: xmlrpc.client.Fault, method: str) -> PlecsRpcError:
    text = str(fault.faultString)
    low = text.lower()
    if "timeout" in low or "time limit" in low:
        return PlecsRpcError(RpcErrorCode.TIMEOUT, text, answered=True)
    if "busy" in low or "already running" in low or "blocking command" in low:
        return PlecsRpcError(RpcErrorCode.BUSY, text, answered=True)
    if method == "plecs.load" and ("not found" in low or "no such file" in low or "cannot open" in low):
        return PlecsRpcError(RpcErrorCode.MODEL_NOT_FOUND, text, answered=True)
    return PlecsRpcError(RpcErrorCode.SIMULATION_FAILED, f"{method}: {text}", answered=True)


def _parse_result(raw, output_names: list[str]) -> tuple[list, dict]:
    """(time, {name: values}). PLECS returns one row per output signal; a time-major matrix
    (one row per time stamp) is also accepted when unambiguous."""
    if isinstance(raw, str):  # PLECS returns runtime errors of parallel runs as a string
        raise PlecsRpcError(RpcErrorCode.SIMULATION_FAILED, raw, answered=True)
    if not isinstance(raw, dict) or "Time" not in raw or "Values" not in raw:
        raise PlecsRpcError(RpcErrorCode.BAD_RESULT, "expected a struct with Time and Values (does the circuit have top-level outports?)")
    t = [float(x) for x in raw["Time"]]
    values = raw["Values"]
    n_out, n_t = len(output_names), len(t)
    if n_t == 0:
        raise PlecsRpcError(RpcErrorCode.BAD_RESULT, "empty Time vector")
    rows = [list(r) if isinstance(r, (list, tuple)) else [r] for r in values] if isinstance(values, (list, tuple)) else None
    if rows is None:
        raise PlecsRpcError(RpcErrorCode.BAD_RESULT, "Values is not a matrix")
    if n_out == 1 and len(rows) == n_t and all(len(r) == 1 for r in rows):
        rows = [[r[0] for r in rows]]
    if len(rows) == n_out and all(len(r) == n_t for r in rows):
        signals = rows
    elif len(rows) == n_t and all(len(r) == n_out for r in rows) and n_out != n_t:
        signals = [list(col) for col in zip(*rows)]
    else:
        shape = f"{len(rows)}x{len(rows[0]) if rows else 0}"
        raise PlecsRpcError(RpcErrorCode.BAD_RESULT, f"Values is {shape}, expected {n_out} output(s) x {n_t} time stamps (manifest: {', '.join(output_names)})")
    out = {}
    for name, sig in zip(output_names, signals):
        vals = [float(v) for v in sig]
        if not all(math.isfinite(v) for v in vals):
            raise PlecsRpcError(RpcErrorCode.BAD_RESULT, f"output {name!r} has non-finite values")
        out[name] = vals
    # equal consecutive stamps are legal (PLECS outputs both sides of a switching event)
    if any(b < a for a, b in zip(t, t[1:])):
        raise PlecsRpcError(RpcErrorCode.BAD_RESULT, "Time decreases")
    return t, out
