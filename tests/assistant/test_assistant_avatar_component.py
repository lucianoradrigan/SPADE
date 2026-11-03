"""FE-3: the radrigan_avatar component renders every state, rejects an unknown one clearly, and
falls back to a silhouette without the layers."""

from pathlib import Path

import pytest

from driveflow.viz.adapters.assistant_adapter import STATES
from driveflow.viz.components.radrigan_avatar import UnknownAvatarState, avatar_props

DEMO = str(Path(__file__).resolve().parents[2] / "scripts" / "assistant_avatar_demo.py")


def test_props_for_every_state_and_assets():
    for s in STATES:
        p = avatar_props(s)
        assert p["state"] == s and p["name"] == "PhD. Radrigan" and p["tagline"] == "AI assistant · local model"
    p = avatar_props("idle")
    assert p["images"]["base"].startswith("data:image/webp;base64,") and len(p["images"]["digest"]) == 12


def test_unknown_state_and_bad_progress():
    with pytest.raises(UnknownAvatarState, match="unknown avatar state 'celebrating'"):
        avatar_props("celebrating")
    with pytest.raises(ValueError, match="progress"):
        avatar_props("running", progress=1.5)


def test_missing_assets_give_a_silhouette(tmp_path):
    assert avatar_props("idle", avatar_dir=str(tmp_path))["images"] is None


@pytest.mark.parametrize("state", STATES)
def test_demo_page_renders_each_state(state):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(DEMO, default_timeout=60)
    at.query_params["state"] = state
    at.run()
    assert not at.exception, at.exception


def test_demo_page_without_assets_and_compact():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(DEMO, default_timeout=60)
    at.query_params["no_assets"] = "1"
    at.query_params["compact"] = "1"
    at.run()
    assert not at.exception, at.exception
