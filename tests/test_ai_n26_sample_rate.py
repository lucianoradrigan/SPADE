"""AI-N26 (software part): the edge examples wait one sample period of the run's own data, every run
records the rate it was trained at, and no GUI or example text claims the data are sampled at 5 kHz."""

import inspect
import json
import re
from pathlib import Path

import pytest

from driveflow.ai.edge_examples import ModelFacts, c2000_example, esp32_example, facts_from_run, rpi_cpp_example, rpi_python_example
from driveflow.models.common.windowing import SIMULATION_FS_HZ

ROOT = Path(__file__).resolve().parents[1]
RUNS = sorted((ROOT / "configs").glob("*/*/*/run_metadata.json"))
#: the only texts allowed to name 5 kHz: the post-hoc correction of the pre-registered C2000 window.
CORRECTION = re.compile(r"AI-N26|assumed 5 kHz|5 kHz budget")


@pytest.mark.parametrize("fs_hz", [10_000.0, 5_000.0, 20_000.0, 12_800.0])
@pytest.mark.parametrize("generator", [esp32_example, c2000_example])
def test_the_delay_is_one_sample_period_of_the_run(generator, fs_hz):
    facts = ModelFacts(channels=["a", "b"], window=32, classes=["x", "y"], fs_hz=fs_hz)
    delays = re.findall(r"(?:esp_rom_delay_us|DEVICE_DELAY_US)\((\d+)\)", generator(facts))
    assert delays == [str(round(1e6 / fs_hz))]


def test_facts_from_a_run_take_its_rate_or_the_simulator_rate():
    config = {"input_window": 64}
    assert facts_from_run({"channels": ["a"], "fs_hz": 5000.0}, config).sample_period_us == 200
    assert facts_from_run({"channels": ["a"]}, config).fs_hz == SIMULATION_FS_HZ == 1 / 1e-4


@pytest.mark.parametrize("path", RUNS, ids=lambda p: str(p.parent.relative_to(ROOT / "configs")))
def test_every_run_records_its_sample_rate(path):
    meta = json.loads(path.read_text())
    metrics_path = path.parent / "metrics.json"
    recorded = json.loads(metrics_path.read_text()).get("fs_hz") if metrics_path.exists() else None
    assert meta["fs_hz"] == (recorded or SIMULATION_FS_HZ)
    assert meta["fs_hz_source"]


def test_runs_exist():
    assert len(RUNS) >= 30


def test_the_run_writers_record_fs_hz():
    from driveflow.ai import custom_training
    from driveflow.ai.transfer import workflow
    from driveflow.models import training

    for module in (training, custom_training, workflow):
        assert '"fs_hz"' in inspect.getsource(module), module.__name__


def test_no_gui_or_example_text_claims_5_khz():
    facts = ModelFacts(channels=["a"], window=32, classes=["x", "y"])
    texts = {g.__name__: g(facts) for g in (esp32_example, c2000_example, rpi_cpp_example, rpi_python_example)}
    texts |= {str(p.relative_to(ROOT)): p.read_text() for p in (ROOT / "src/driveflow/viz").rglob("*.py")}
    texts[str(Path("src/driveflow/ai/edge_examples.py"))] = (ROOT / "src/driveflow/ai/edge_examples.py").read_text()
    offending = [f"{name}: {line.strip()}" for name, text in texts.items() for line in text.splitlines()
                 if re.search(r"\b5\s?kHz|5000\s?Hz|5_000", line) and not CORRECTION.search(line)]
    assert not offending, offending
