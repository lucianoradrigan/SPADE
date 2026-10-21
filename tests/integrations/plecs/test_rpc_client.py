"""integrations/plecs/rpc_client.py against a fake PLECS RPC server (expansion task E2)."""

import threading

import pytest

from driveflow.integrations.plecs import rpc_client as rc
from fake_plecs import FakePlecs, closed_port_url

OUTS = ["v_out", "i_L"]


@pytest.fixture
def circuit(tmp_path):
    p = tmp_path / "buck_demo.plecs"
    p.write_text("Plecs { Name \"buck_demo\" }\n")
    return p


def test_connection_report_reachable_and_honest_about_what_was_not_checked():
    with FakePlecs() as fake:
        rep = rc.PlecsRpcClient(fake.url).test_connection()
    assert rep.reachable and "plecs.simulate" in rep.methods and rep.latency_ms is not None
    assert rep.plecs_version == rc.VERSION_NOT_REPORTED
    assert not rep.can_simulate_batch and not rep.can_step_interactively
    assert any("interactive stepping" in x for x in rep.not_checked) and any("license" in x for x in rep.not_checked)


def test_connection_without_introspection_is_still_reachable():
    with FakePlecs(introspection=False) as fake:
        rep = rc.PlecsRpcClient(fake.url).test_connection()
    assert rep.reachable and rep.methods is None


def test_unreachable_endpoint():
    rep = rc.PlecsRpcClient(closed_port_url()).test_connection()
    assert not rep.reachable and rep.error_code == "unreachable" and rep.message.startswith("Cannot reach RPC endpoint")
    assert "disabled by default" in rep.hint


@pytest.mark.parametrize("url, code", [("http://192.168.1.20:1080", "remote_host"), ("http://plecs-box:1080", "remote_host"),
                                       ("https://localhost:1080", "bad_url"), ("localhost:1080", "bad_url")])
def test_only_local_http_endpoints(url, code):
    with pytest.raises(rc.PlecsRpcError) as exc:
        rc.PlecsRpcClient(url)
    assert exc.value.code.value == code
    if code == "remote_host":
        assert str(exc.value).startswith("Backend running on a different host")


def test_load_simulate_close(circuit):
    with FakePlecs() as fake:
        c = rc.PlecsRpcClient(fake.url)
        name = c.load(circuit)
        res = c.simulate(name, OUTS, {"Vref": 12.0, "Rload": 4.0}, {"TimeSpan": 0.01}, time_limit_s=5)
        c.close(name)
        sim_call = next(x for x in fake.calls if x[0] == "simulate")
    assert name == "buck_demo" and len(res.time) == 101
    assert res.outputs["v_out"][-1] == pytest.approx(12 * (1 - 2.718281828459045 ** -5), rel=1e-9)
    assert res.outputs["i_L"][-1] == pytest.approx(res.outputs["v_out"][-1] / 4.0)
    assert res.model_vars == {"Vref": 12.0, "Rload": 4.0}
    assert sim_call[2]["SolverOpts"]["Timeout"] == 5.0 and "StopTime" not in sim_call[2]["SolverOpts"]
    assert ("close", "buck_demo") in fake.calls


def test_time_major_values_are_accepted(circuit):
    with FakePlecs(time_major=True) as fake:
        c = rc.PlecsRpcClient(fake.url)
        res = c.simulate(c.load(circuit), OUTS, {"Vref": 10.0}, {"TimeSpan": 0.005}, 5)
    assert len(res.outputs["v_out"]) == len(res.time) == 51


def test_missing_circuit_file(tmp_path):
    with FakePlecs() as fake, pytest.raises(rc.PlecsRpcError) as exc:
        rc.PlecsRpcClient(fake.url).load(tmp_path / "nope.plecs")
    assert exc.value.code is rc.RpcErrorCode.MODEL_NOT_FOUND


def test_non_convergence_is_a_typed_failure(circuit):
    with FakePlecs(fail_if=lambda mv: "Solver did not converge at t=0.004" if mv.get("Rload", 6) < 1.5 else None) as fake:
        c = rc.PlecsRpcClient(fake.url)
        name = c.load(circuit)
        with pytest.raises(rc.PlecsRpcError) as exc:
            c.simulate(name, OUTS, {"Rload": 1.0}, {"TimeSpan": 0.01}, 5)
    assert exc.value.code is rc.RpcErrorCode.SIMULATION_FAILED and "did not converge" in str(exc.value) and exc.value.answered


def test_plecs_side_timeout(circuit):
    with FakePlecs(delay_s=lambda mv: 2.0) as fake:
        c = rc.PlecsRpcClient(fake.url)
        name = c.load(circuit)
        with pytest.raises(rc.PlecsRpcError) as exc:
            c.simulate(name, OUTS, {}, {"TimeSpan": 0.01}, time_limit_s=0.3)
    assert exc.value.code is rc.RpcErrorCode.TIMEOUT


def test_client_side_timeout_when_the_server_hangs(circuit, monkeypatch):
    monkeypatch.setattr(rc, "SOCKET_MARGIN_S", 0.2)
    with FakePlecs(delay_s=lambda mv: 2.0, ignore_timeout=True) as fake:
        c = rc.PlecsRpcClient(fake.url)
        name = c.load(circuit)
        with pytest.raises(rc.PlecsRpcError) as exc:
            c.simulate(name, OUTS, {}, {"TimeSpan": 0.01}, time_limit_s=0.3)
    assert exc.value.code is rc.RpcErrorCode.TIMEOUT and "no answer" in str(exc.value)


def test_connection_dropped_mid_sweep(circuit):
    with FakePlecs(drop_after=1) as fake:
        c = rc.PlecsRpcClient(fake.url)
        name = c.load(circuit)
        c.simulate(name, OUTS, {}, {"TimeSpan": 0.002}, 5)
        codes = []
        for _ in range(2):
            with pytest.raises(rc.PlecsRpcError) as exc:
                c.simulate(name, OUTS, {}, {"TimeSpan": 0.002}, 5)
            codes.append(exc.value.code)
    assert codes[-1] in (rc.RpcErrorCode.UNREACHABLE, rc.RpcErrorCode.CONNECTION_LOST)


def test_busy_server_gets_the_busy_error(circuit):
    """Two clients (e.g. two processes) on one PLECS: the second blocking call is refused."""
    with FakePlecs(delay_s=lambda mv: 0.8) as fake:
        a, b = rc.PlecsRpcClient(fake.url), rc.PlecsRpcClient(fake.url)
        name = a.load(circuit)
        rc._LOCKS.clear()  # bypass this process's own serialisation to emulate another process
        t = threading.Thread(target=lambda: a.simulate(name, OUTS, {}, {"TimeSpan": 0.001}, 5))
        t.start()
        import time

        time.sleep(0.2)
        rc._LOCKS.clear()
        with pytest.raises(rc.PlecsRpcError) as exc:
            b.simulate(name, OUTS, {}, {"TimeSpan": 0.001}, 5)
        t.join()
    assert exc.value.code is rc.RpcErrorCode.BUSY


@pytest.mark.parametrize("raw, msg", [
    ({"Time": [0, 1e-4], "Values": [[1, 2]]}, "expected 2 output"),
    ({"Time": [0, 1e-4], "Values": [[1, float("nan")], [0, 0]]}, "non-finite"),
    ({"Time": [0, 2e-4, 1e-4], "Values": [[1, 2, 3], [0, 0, 0]]}, "Time decreases"),
    ({"Values": [[1]]}, "Time and Values"),
    ("Error in parallel run 2: singular matrix", "singular matrix"),
])
def test_bad_results_are_rejected(raw, msg):
    with pytest.raises(rc.PlecsRpcError) as exc:
        rc._parse_result(raw, OUTS)
    assert msg in str(exc.value)


def test_equal_time_stamps_at_switching_events_are_legal():
    t, out = rc._parse_result({"Time": [0, 1e-4, 1e-4, 2e-4], "Values": [[0, 1, 2, 3], [0, 0, 0, 0]]}, OUTS)
    assert t == [0, 1e-4, 1e-4, 2e-4] and out["v_out"] == [0, 1, 2, 3]
