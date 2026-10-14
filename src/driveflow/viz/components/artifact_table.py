"""Downloadable run artifacts with size and SHA-256, read from disk (registry_adapter.artifacts) --
only files that exist are listed, so no export is ever shown that was not produced."""

import streamlit as st

from driveflow.viz.adapters.registry_adapter import artifacts


def artifact_table(run_dir, run_name: str) -> None:
    for a in artifacts(run_dir):
        c1, c2 = st.columns([3, 2])
        c1.download_button(a["file"], a["path"].read_bytes(), file_name=f"{run_name.replace('/', '_')}_{a['file']}", key=f"reg_dl_{run_name}_{a['file']}")
        c2.caption(f"{a['size_bytes'] / 1024:.1f} KB · sha256 {a['sha256'][:16]}…")
