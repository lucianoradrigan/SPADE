"""Legacy entry point, kept so the documented command keeps working:

    streamlit run src/driveflow/viz/dashboard.py

The single-file dashboard that used to live here (landing page + five macro-phase dispatch,
~2100 lines) was split into the multipage app in driveflow.viz.app (pages/ + views/ +
components/ + theme/ + adapters/). Every former macro-phase is still reachable -- see app.py's
module docstring for the mapping. Page paths in app.PAGES are relative to this directory, which
is the same directory as app.py, so both entry points resolve them identically.
"""

from driveflow.viz.app import main

main()
