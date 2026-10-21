"""SPADE -- Engineering Workbench: Streamlit multipage entry point (guide Sec. 3.2 / 5.1).

Run with either of:
    streamlit run src/driveflow/viz/app.py
    streamlit run src/driveflow/viz/dashboard.py   # legacy command, same app

Native st.navigation (Streamlit >= 1.36). Each page is a thin script in pages/ that calls one
render() function in views/ -- the views hold the UI logic and are importable/testable; numerics
stay in the library packages (sim/, control/, models/, ai/, monitoring/, analysis/), reached via
adapters/. The five original macro-phases map onto sections: A -> Simulation (DC Motor, PMSM),
B -> Simulation (VSC / DPC), IA -> AI Diagnostics + Model Registry, TL -> Transfer Learning,
LM -> Monitoring + Alert History.
"""

import streamlit as st

from driveflow.viz.adapters.jobs_adapter import queue_summary
from driveflow.viz.adapters.workspace import workspace_status
from driveflow.viz.theme.css import inject_base_css

PAGES = {
    "Workspace": [
        ("pages/overview.py", "Overview", ":material/home:"),
        ("pages/experiment_runs.py", "Experiment Runs", ":material/work_history:"),
        ("pages/environments.py", "Environments", ":material/sports_esports:"),
    ],
    "Simulation": [
        ("pages/dc_motor.py", "DC Motor", ":material/electric_bolt:"),
        ("pages/pmsm.py", "PMSM / FOC", ":material/radio_button_checked:"),
        ("pages/vsc_dpc.py", "VSC / DPC", ":material/power:"),
        ("pages/dcdc_converters.py", "DC-DC Converters", ":material/bolt:"),
        ("pages/plecs_plant_studio.py", "PLECS Plant Studio", ":material/schema:"),
    ],
    "Intelligence": [
        ("pages/ai_diagnostics.py", "AI Diagnostics", ":material/psychology:"),
        ("pages/transfer_learning.py", "Transfer Learning", ":material/sync_alt:"),
        ("pages/model_registry.py", "Model Registry", ":material/inventory_2:"),
    ],
    "Operations": [
        ("pages/monitoring.py", "Monitoring", ":material/monitor_heart:"),
        ("pages/alert_history.py", "Alert History", ":material/list_alt:"),
    ],
    "Help": [
        ("pages/documentation.py", "Documentation", ":material/menu_book:"),
        ("pages/algorithm_guide.py", "Algorithm Guide", ":material/alt_route:"),
    ],
}


def _sidebar_status() -> None:
    status = workspace_status()
    with st.sidebar:
        st.caption("SPADE · Engineering Workbench")
        n_ok = sum(1 for c in status["checks"] if c["ok"])
        icon = ":material/check_circle:" if n_ok == len(status["checks"]) else ":material/warning:"
        st.caption(f"{icon} {n_ok}/{len(status['checks'])} workspace checks pass · data source: simulation only · v{status['version']}")
        jobs = queue_summary()
        if jobs.get("error"):
            st.caption(":material/warning: Jobs: store unavailable")
        else:
            st.caption(f":material/work_history: Jobs: {jobs['running']} running · {jobs['queued']} queued · {jobs['workers']} worker(s)")
        st.page_link("pages/experiment_runs.py", label="Open queue", icon=":material/list:")


def main() -> None:
    st.set_page_config(page_title="SPADE · Engineering Workbench", layout="wide", page_icon=":material/settings_input_component:")
    inject_base_css()
    nav = {section: [st.Page(path, title=title, icon=icon, default=(path == "pages/overview.py")) for path, title, icon in items] for section, items in PAGES.items()}
    page = st.navigation(nav, position="sidebar", expanded=True)
    _sidebar_status()
    page.run()


if __name__ == "__main__":
    main()
