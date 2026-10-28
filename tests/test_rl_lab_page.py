"""Design & Control / RL Lab (expansion task E8): reward preview only after a real episode, a real
training job through the worker, playback of recorded episodes, the benchmark and the blocked default."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from driveflow.jobs.worker import Worker
from driveflow.runs.store import workspace_root
from driveflow.viz.adapters import rl_adapter as ra
from tests.streamlit_compat import chart_keys

APP_PATH = str(Path(__file__).resolve().parents[1] / "src" / "driveflow" / "viz" / "app.py")


def _open():
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.run()
    at.switch_page("pages/rl_lab.py").run()
    assert not at.exception
    return at


def _text(at):
    return " ".join(e.value for kind in ("markdown", "caption", "info", "warning", "error", "success") for e in at.get(kind))


def test_empty_states_and_reward_preview():
    at = _open()
    text = _text(at)
    assert "No evaluated episode yet." in text and "No simulation has been executed for this configuration." in text
    at.button(key="rl_preview").click().run()
    assert not at.exception
    table = next(d.value for d in at.dataframe if "Sum of weighted penalties" in getattr(d.value, "columns", []))
    assert set(table["Term"]) == {"tracking", "overcurrent", "action_rate"}
    assert "Baseline: PI (BuckVoltageController)" in _text(at)


def test_dqn_asks_for_a_discretization():
    at = _open()
    at.radio(key="rl_algo").set_value("dqn").run()
    assert at.slider(key="rl_levels").value == 11 and "11 discrete actions" in _text(at)


def test_training_job_end_to_end_and_blocked_default():
    at = _open()
    at.number_input(key="rl_episodes").set_value(2).run()
    at.number_input(key="rl_eval_every").set_value(1).run()
    at.button(key="rl_start").click().run()
    assert any("Queued as job" in s.value for s in at.success)
    Worker(workspace_root()).serve(once=True, poll_s=0.05)
    at.button(key="rl_refresh").click().run()
    assert not at.exception
    keys = chart_keys(at)
    run_id = ra.runs()[0].run_id
    assert {f"rl_reward_{run_id}", f"rl_loss_{run_id}", f"rl_viol_{run_id}", f"rl_pb_fig_{run_id}"} <= keys
    bench = next(d.value for d in at.dataframe if "Median tracking RMSE (V)" in getattr(d.value, "columns", []))
    assert set(bench["Controller"]) == {"RL (DDPG)", "PI (BuckVoltageController)"}
    assert at.button(key=f"rl_gate_{run_id}_approve").disabled
    at.button(key=f"rl_save_{run_id}").click().run()
    assert any("Nothing was deployed" in s.value for s in at.success)
