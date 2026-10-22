"""A fake PLECS Standalone RPC server for tests (expansion task E2): xmlrpc.server in a thread that
imitates plecs.load / plecs.simulate / plecs.close as the PLECS documentation describes them.

The "circuit" is an averaged buck regulated by its own (first-order) controller: with ModelVars
Vin, Rload and Vref, v_out(t) = Vref (1 - exp(-t/tau)) and i_L = v_out / Rload, sampled at
OutputTimes (or every 1e-4 s). Behaviours a test can trigger:
- ``fail_if(model_vars) -> str | None``: a non-convergence fault with that message;
- ``delay_s(model_vars) -> float``: wall time the run takes; above SolverOpts.Timeout the server
  returns PLECS's timeout fault instead of a result (PLECS enforces its own Timeout);
- ``ignore_timeout``: sleep the full delay even past Timeout (a hung server);
- ``drop_after``: shut down after that many simulate calls (connection lost mid-sweep);
- ``time_major``: return Values as rows per time stamp instead of rows per signal.

Excited mode (task A1, TEST DOUBLE of a PLECS buck): when the ModelVars carry an action-excitation
profile (``prbs_seed``/``D_min``/``D_max``/``T_hold`` or ``D0``/``D1``/``t_step``, see
integrations/plecs/excitation.py), the "circuit" is SPADE's own averaged buck (sim/buck_system.py)
in open loop with Vin and Rload, its duty following the profile, and it returns three signals:
v_out, i_L and the duty applied. This checks the surrogate MECHANICS end to end; it is not PLECS.
Busy handling is real: a second blocking call while one runs gets the PLECS busy fault.
"""

from __future__ import annotations

import math
import socket
import socketserver
import threading
import time
from pathlib import Path
from xmlrpc.server import SimpleXMLRPCRequestHandler, SimpleXMLRPCServer

TAU_S = 2e-3


class _ThreadedServer(socketserver.ThreadingMixIn, SimpleXMLRPCServer):
    """Answers concurrent requests, so a second blocking call can be refused as busy, like PLECS."""

    daemon_threads = True


class _Quiet(SimpleXMLRPCRequestHandler):
    def log_message(self, *args):  # keep test output clean
        pass


class FakePlecs:
    def __init__(self, fail_if=None, delay_s=None, ignore_timeout=False, drop_after=None, time_major=False, introspection=True):
        self.fail_if = fail_if or (lambda mv: None)
        self.delay_s = delay_s or (lambda mv: 0.0)
        self.ignore_timeout = ignore_timeout
        self.drop_after = drop_after
        self.time_major = time_major
        self.loaded: dict[str, str] = {}
        self.calls: list = []
        self._busy = threading.Lock()
        self.server = _ThreadedServer(("127.0.0.1", 0), requestHandler=_Quiet, allow_none=True, logRequests=False)
        if introspection:
            self.server.register_introspection_functions()
        self.server.register_function(self.load, "plecs.load")
        self.server.register_function(self.simulate, "plecs.simulate")
        self.server.register_function(self.close, "plecs.close")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.n_sim = 0

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def stop(self):
        try:
            self.server.shutdown()
        except Exception:
            pass
        self.server.server_close()

    # ------------------------------------------------------------------ PLECS commands
    def load(self, path):
        self.calls.append(("load", path))
        p = Path(path)
        if not p.is_file():
            raise RuntimeError(f"File not found: {path}")
        self.loaded[p.stem] = path
        return True

    def close(self, name):
        self.calls.append(("close", name))
        self.loaded.pop(name, None)
        return True

    def simulate(self, name, opts=None):
        if not self._busy.acquire(blocking=False):
            raise RuntimeError("Cannot execute a blocking command while another blocking command is running")
        try:
            return self._simulate(name, opts or {})
        finally:
            self._busy.release()

    def _simulate(self, name, opts):
        self.calls.append(("simulate", name, opts))
        self.n_sim += 1
        if name not in self.loaded:
            raise RuntimeError(f"Model '{name}' is not loaded")
        mv = dict(opts.get("ModelVars") or {})
        solver = dict(opts.get("SolverOpts") or {})
        delay, timeout = float(self.delay_s(mv)), float(solver.get("Timeout", math.inf))
        if delay > timeout and not self.ignore_timeout:
            time.sleep(timeout)
            raise RuntimeError("Simulation timeout: the simulation exceeded the specified timeout")
        time.sleep(delay)
        msg = self.fail_if(mv)
        if msg:
            raise RuntimeError(msg)
        if self.drop_after is not None and self.n_sim > self.drop_after:
            threading.Thread(target=self.stop, daemon=True).start()
            raise ConnectionAbortedError("server going away")
        span = float(solver.get("TimeSpan", 0.01))
        times = solver.get("OutputTimes") or [k * 1e-4 for k in range(int(round(span / 1e-4)) + 1)]
        if "prbs_seed" in mv or "D0" in mv:
            return self._excited_buck(mv, span, times)
        vref, rload = float(mv.get("Vref", 24.0)), float(mv.get("Rload", 6.0))
        v = [vref * (1 - math.exp(-t / TAU_S)) for t in times]
        i = [x / rload for x in v]
        values = [list(r) for r in zip(v, i)] if self.time_major else [v, i]
        return {"Time": list(times), "Values": values}

    def _excited_buck(self, mv, span, times):
        sig = _excited_buck_signals(mv, span, times)
        rows = [sig["v"], sig["i"], sig["d"]]
        values = [list(r) for r in zip(*rows)] if self.time_major else rows
        return {"Time": list(times), "Values": values}


def _excited_buck_signals(mv: dict, span: float, times) -> dict:
    import numpy as np

    from driveflow.integrations.plecs import excitation as ex
    from driveflow.sim.buck_system import TAU, BuckParams, BuckSystem

    kind, roles = ("prbs", ex.KINDS["prbs"]) if "prbs_seed" in mv else ("step", ex.KINDS["step"])
    n = int(round(span / TAU))
    duty = ex.sequence(kind, {r: float(mv[r]) for r in roles}, np.arange(n + 1) * TAU)
    plant = BuckSystem(BuckParams(v_in_v=float(mv.get("Vin", 48.0)), load_resistance_ohm=float(mv.get("Rload", 6.0))))
    st = plant.reset()
    x = np.empty((n + 1, 2))
    x[0] = (st.v_out, st.i_l)
    for k in range(n):
        st = plant.simulate(float(duty[k]))
        x[k + 1] = (st.v_out, st.i_l)
    idx = np.clip(np.rint(np.asarray(times, dtype=float) / TAU).astype(int), 0, n)
    return {"v": x[idx, 0].tolist(), "i": x[idx, 1].tolist(), "d": duty[idx].tolist()}


def closed_port_url() -> str:
    """A localhost URL on which nothing listens."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}"
