"""User presets: saved motor-characteristic profiles and user-defined fault types (ported from
the legacy dashboard). Both are a user's local experimentation state persisted to gitignored JSON
files under configs/, not source-controlled calibration data."""

import json
from pathlib import Path

import streamlit as st

from driveflow.sim.vibration import bearing_frequencies as bf

#: Saved motor-characteristic presets (both DC and PMSM), so repeated iteration doesn't mean
#: retyping r_a/l_a/psi_e (or p/l_d/l_q/r_s/psi_p) by hand every time -- a user's own local
#: experimentation state, not source-controlled calibration, so it's gitignored (see .gitignore).
_PROFILES_PATH = Path(__file__).resolve().parents[4] / "configs" / "motor_profiles.json"


def _load_profiles() -> dict:
    if _PROFILES_PATH.exists():
        try:
            return json.loads(_PROFILES_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_profiles(profiles: dict):
    _PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PROFILES_PATH.write_text(json.dumps(profiles, indent=2, sort_keys=True))


#: Built-in fault types (Fault type selectbox always offers these -- "Custom fault types" below
#: only ever APPENDS to this list, never hides/replaces it, so it's always clear what's built-in
#: vs. user-added).
BUILTIN_FAULT_TYPES = ["healthy", "outer_race", "inner_race", "ball", "cage"]

#: User-defined fault types (name -> {"order": float}), gitignored for the same reason as
#: motor_profiles.json above -- personal experimentation state, not source-controlled data.
_CUSTOM_FAULTS_PATH = Path(__file__).resolve().parents[4] / "configs" / "custom_fault_types.json"


def load_custom_faults() -> dict:
    if _CUSTOM_FAULTS_PATH.exists():
        try:
            return json.loads(_CUSTOM_FAULTS_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_custom_faults(custom_faults: dict):
    _CUSTOM_FAULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CUSTOM_FAULTS_PATH.write_text(json.dumps(custom_faults, indent=2, sort_keys=True))


def _resolve_builtin_order(fault_type: str) -> float:
    return bf.fault_order(fault_type, bf.KAT_DATACENTER_6203_GEOMETRY)


def custom_fault_manager(container):
    """Renders the save/delete UI for user-defined fault types, right below the built-in Fault
    type selectbox. A saved custom fault type is a NAMED list of one or more characteristic
    orders (see Scenario.fault_order_override's docstring for what "order" means -- a multiple of
    shaft speed, not raw data or a trained model) -- pick one or more EXISTING fault types (built-
    in or already-saved custom) to merge, and/or type a manual order, then save under one new
    name. That name then behaves like any other single Fault type entry above: selecting it once
    injects every one of its component orders together, with no separate combining step at
    generate time (Scenario/runner.py already support N simultaneous orders per run, see
    BearingFaultLoad.extra_faults -- this just resolves a saved name into that list up front).
    """
    custom_faults = load_custom_faults()

    header_col, help_col = container.columns([4, 1])
    header_col.markdown('<div class="df-card-header">Custom fault types</div>', unsafe_allow_html=True)
    with help_col.popover("❓", help="How do I pick an order?"):
        st.markdown(
            """
**Order** = how many times the fault event repeats per shaft revolution
(`fault_frequency_hz / shaft_frequency_hz`). That single number is all this needs -- no
geometry, dataset, or trained model.

| Fault | Order to use |
|---|---|
| Mass unbalance | `1.0` |
| Shaft misalignment | `2.0` (sometimes a 1.0 component too) |
| Gear-mesh fault | the gear's tooth count on that shaft (e.g. `24.0`) |
| A different bearing's known BPFO/BPFI/BSF | `known_fault_freq_hz / shaft_freq_hz` -- or give the bearing's geometry (ball count, diameters, contact angle) and it can be computed the same way the built-in types are |
| A peak seen in a real FFT spectrum | `peak_freq_hz / shaft_freq_hz`, both read at the same run's speed |

**Combining**: pick existing fault types below to merge their orders into one new named fault
type -- e.g. outer_race + a custom gear_mesh order together, saved as "outer_gear_combo". Select
it later like any other single fault type; every component order injects simultaneously.

Severity (amplitude) isn't set here -- use the **Fault severity** sliders above, same as the built-in types. Amplitude is always constant here (no load-zone modulation, unlike inner_race/ball).
            """
        )
    container.caption(
        "Save a new fault type: pick existing ones to combine and/or type a manual order (a multiple of shaft "
        "speed). Tap ❓ above for examples."
    )

    new_fault_name = container.text_input("Name", key="custom_fault_name", placeholder="e.g. outer_gear_combo")
    component_options = BUILTIN_FAULT_TYPES[1:] + sorted(custom_faults)  # exclude "healthy"
    selected_components = container.multiselect(
        "Combine these (optional)",
        component_options,
        key="custom_fault_components",
        help="Existing fault types (built-in or already-saved custom) to merge into the new one -- each contributes its own order(s).",
    )
    col_toggle, col_order = container.columns([1, 2])
    add_manual = col_toggle.checkbox("+ manual order", key="custom_fault_add_manual", value=True, help="Also add a literal order you type in, on top of anything selected above.")
    new_fault_order = col_order.number_input("Manual order", key="custom_fault_order_input", min_value=0.01, value=3.5, step=0.1, format="%.2f", label_visibility="collapsed", disabled=not add_manual)

    name_conflicts_builtin = new_fault_name.strip() in BUILTIN_FAULT_TYPES
    has_content = bool(selected_components) or add_manual
    if container.button(
        "💾 Save fault type",
        key="custom_fault_save",
        disabled=not new_fault_name.strip() or name_conflicts_builtin or not has_content,
        help=("That name is already a built-in fault type -- pick another." if name_conflicts_builtin else "Pick at least one component or add a manual order." if not has_content else "Save this fault type."),
    ):
        orders = []
        for comp in selected_components:
            orders.extend(custom_faults[comp]["orders"] if comp in custom_faults else [_resolve_builtin_order(comp)])
        if add_manual:
            orders.append(new_fault_order)
        custom_faults[new_fault_name.strip()] = dict(orders=orders)
        _save_custom_faults(custom_faults)
        st.toast(f"Saved '{new_fault_name.strip()}' -- {len(orders)} order(s): {', '.join(f'{o:.2f}' for o in orders)}.")
        st.rerun()

    if not custom_faults:
        container.caption("No custom fault types yet -- the built-ins above (healthy/outer_race/inner_race/ball/cage) are always available regardless.")
        return

    summary = ", ".join(f"{n} ({len(v['orders'])} order{'s' if len(v['orders']) != 1 else ''})" for n, v in sorted(custom_faults.items()))
    container.caption(f"Saved: {summary}")
    col_select, col_delete = container.columns([3, 1])
    selected = col_select.selectbox("Delete a custom fault type", sorted(custom_faults), key="custom_fault_delete_select", label_visibility="collapsed")
    if col_delete.button(
        "🗑️ Delete",
        key="custom_fault_delete",
        help=f"Delete '{selected}'. Any saved scenario history referencing it keeps working (its orders are baked into that history entry already) -- only future runs lose access to it, including as a component of other combos.",
    ):
        del custom_faults[selected]
        _save_custom_faults(custom_faults)
        st.rerun()


#: {group: {characteristic_name: number_input's key=}} -- the single source of truth for which
#: widget each saved field maps to, used both by _profile_manager (to know what "Save" reads) and
#: by _apply_pending_profile_load (to know what "Load" must write into session_state).
_PROFILE_FIELD_KEYS = {
    "dc": dict(r_a="dc_r_a", l_a="dc_l_a", psi_e="dc_psi_e"),
    "pmsm": dict(p="pmsm_p", l_d="pmsm_l_d", l_q="pmsm_l_q", r_s="pmsm_r_s", psi_p="pmsm_psi_p"),
}


def apply_pending_profile_load():
    """Must be called BEFORE any motor-characteristic number_input is instantiated this script
    pass. Streamlit forbids writing session_state[key] for a widget that has already rendered in
    the same run ("cannot be modified after the widget with key ... is instantiated") -- but the
    "Load" button lives inside the expander, AFTER those widgets. So Load doesn't write
    session_state directly; it stashes (group, name) here and reruns, and this function (called
    near the top of the script, before tab1/tab3's widgets exist for this pass) applies it."""
    pending = st.session_state.pop("_pending_profile_load", None)
    if pending is None:
        return
    group, name = pending
    values = _load_profiles().get(group, {}).get(name)
    if values is None:
        return
    for field, value in values.items():
        key = _PROFILE_FIELD_KEYS.get(group, {}).get(field)
        if key is not None:
            st.session_state[key] = value


def profile_manager(container, group: str, current_values: dict):
    """Renders a compact save/load/delete UI for named motor-characteristic presets, persisted to
    configs/motor_profiles.json (survives server restarts, unlike the in-memory undo history).

    Args:
        container: Where to render (an st.expander or similar) -- placed right below the
            characteristic number_inputs it saves/overwrites.
        group: "dc" or "pmsm" -- a separate namespace within the shared JSON file, and a key into
            _PROFILE_FIELD_KEYS.
        current_values: {characteristic_name: current_value}, e.g. {"r_a": r_a, "l_a": l_a, ...}
            -- what "Save" writes under the chosen name.
    """
    all_profiles = _load_profiles()
    group_profiles = all_profiles.setdefault(group, {})

    container.markdown('<div class="df-card-header">Motor profiles</div>', unsafe_allow_html=True)
    col_name, col_save = container.columns([3, 1])
    new_name = col_name.text_input("Profile name", key=f"{group}_profile_name", placeholder="e.g. worn-bearing spare motor", label_visibility="collapsed")
    if col_save.button("💾 Save", key=f"{group}_profile_save", disabled=not new_name.strip(), help="Save the current values above under this name."):
        group_profiles[new_name.strip()] = dict(current_values)
        _save_profiles(all_profiles)
        st.toast(f"Saved profile '{new_name.strip()}'.")
        st.rerun()

    if not group_profiles:
        container.caption("No saved profiles yet -- name one above and click Save.")
        return

    col_select, col_load, col_delete = container.columns([3, 1, 1])
    selected = col_select.selectbox("Load profile", sorted(group_profiles), key=f"{group}_profile_select", label_visibility="collapsed")
    if col_load.button("Load", key=f"{group}_profile_load", help=f"Overwrite the fields above with '{selected}'."):
        st.session_state["_pending_profile_load"] = (group, selected)
        st.rerun()
    if col_delete.button("🗑️", key=f"{group}_profile_delete", help=f"Delete '{selected}'."):
        del group_profiles[selected]
        _save_profiles(all_profiles)
        st.rerun()
