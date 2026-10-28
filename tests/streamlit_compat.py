"""AppTest helpers that behave the same across the Streamlit versions SPADE installs.

Streamlit <= 1.62 (the version pinned in requirements-lock.txt, and the newest one the macOS
Intel constraints allow) returns st.plotly_chart as an UnknownElement whose .key is always None;
newer releases fill it in. The user key is still carried in the element id
("$$ID-<hash>-<key>"), so read it from there when .key is missing."""

from streamlit.runtime.state.common import user_key_from_element_id


def chart_key(element) -> str | None:
    """The user key of a chart element returned by AppTest.get("plotly_chart")."""
    if element.key is not None:
        return element.key
    element_id = getattr(element.proto, "id", "")
    return user_key_from_element_id(element_id) if element_id else None


def chart_keys(at) -> set:
    """User keys of every Plotly chart on the page."""
    return {chart_key(c) for c in at.get("plotly_chart")}
