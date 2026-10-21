"""Help / Algorithm Guide (expansion task E12, spec Sec. 8.6): every combination of the three
questions renders a recommendation whose link is a registered, existing page, or says "Not available
in this version" -- never a broken link."""

from pathlib import Path

import pytest
import yaml
from streamlit.testing.v1 import AppTest

from driveflow.viz.adapters import guide_adapter as ga

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = str(ROOT / "src" / "driveflow" / "viz" / "app.py")
PAGES_DIR = ROOT / "src" / "driveflow" / "viz"


def test_table_is_valid_and_cites_the_paper():
    guide = ga.load_guide()
    assert ga.validate(guide, ga.registered_pages()) == []
    assert guide["citation"]["venue"] == "IEEE Transactions on Industrial Electronics" and guide["citation"]["year"] == 2026
    assert "Li" in ga.citation_text(guide)


def test_validation_catches_broken_entries():
    guide = yaml.safe_load(ga.GUIDE_PATH.read_text())
    guide["tasks"][0] = {**guide["tasks"][0], "page": "pages/does_not_exist.py", "accepts": ["simulated"]}
    guide["tasks"].append({**guide["tasks"][1], "phase": "deployment"})
    problems = " ".join(ga.validate(guide, ga.registered_pages()))
    assert "not registered" in problems and "duplicate id" in problems and "unknown phase" in problems


def _combinations():
    guide = ga.load_guide()
    return [(p, t, s) for p in guide["phases"] for t, _ in ga.tasks_for(p, guide) for s in guide["data_sources"]]


@pytest.mark.parametrize("phase, task, source", _combinations())
def test_every_recommendation_resolves(phase, task, source):
    rec = ga.recommend(phase, task, source)
    assert rec.family and rec.why and len(rec.assumptions) == 2 and len(rec.limitations) == 2
    if rec.page is None:
        assert rec.unavailable_reason.startswith(ga.NOT_AVAILABLE)
    else:
        assert rec.unavailable_reason is None and rec.page in ga.registered_pages() and (PAGES_DIR / rec.page).is_file()


def test_no_page_reads_plecs_data_yet():
    assert all(ga.recommend(p, t, "plecs").page is None for p, t, s in _combinations() if s == "plecs")


def test_apptest_walks_every_combination():
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.switch_page("pages/algorithm_guide.py").run()
    assert not at.exception
    guide = ga.load_guide()
    registered = ga.registered_pages()
    n = 0
    for phase in guide["phases"]:
        at.radio(key="guide_phase").set_value(phase).run()
        for task, _ in ga.tasks_for(phase, guide):
            at.radio(key=f"guide_task_{phase}").set_value(task).run()
            for source in guide["data_sources"]:
                at.radio(key="guide_source").set_value(source).run()
                assert not at.exception, (phase, task, source)
                rec = ga.recommend(phase, task, source, guide)
                assert any(m.value == f"#### {rec.family}" for m in at.markdown), (phase, task, source)
                links = [el for el in at.get("page_link")]
                infos = " ".join(i.value for i in at.info)
                if rec.page:
                    targets = [el.proto.page for el in links]
                    assert any(t.endswith(Path(rec.page).stem) or t == rec.page for t in targets), (phase, task, source, targets)
                    assert all(Path(t).stem in {Path(p).stem for p in registered} or t == "" for t in targets)
                else:
                    assert ga.NOT_AVAILABLE in infos, (phase, task, source)
                n += 1
    assert n == len(_combinations())
