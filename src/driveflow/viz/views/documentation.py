"""Documentation page: data formats, status semantics, design decisions, known limitations."""

import streamlit as st

from driveflow.viz.components.page_header import page_header

_STATUS = """
| Status | Meaning in SPADE | Evidence required |
|---|---|---|
| **HEALTHY** | The domain was evaluated (a monitoring pass ran / a connected source reported) and **no** alert is active | evaluation timestamp + empty active-alert set |
| **WARNING** | At least one active alert of severity low/medium, none critical | an unresolved alert |
| **CRITICAL** | At least one active alert of severity high/critical | an unresolved alert |
| **OFFLINE** | A *connected* source missed its heartbeat timeout | heartbeat older than its timeout (no connected source exists today) |
| **UNKNOWN** | Nothing evaluated yet, demo data, or no source | default — never HEALTHY by default |

Severity mapping (configurable in `ServerAgent`): critical/high → CRITICAL; medium/warning/low/info → WARNING.
A classifier prediction is an **AI suggestion**: it is shown with model version and confidence and never raises an alert by itself.
"""

_SOURCES = """
| Label | Meaning |
|---|---|
| **SIMULATED** | Produced by SPADE's physics simulation (`run_scenario`) in this session |
| **EXTERNAL FILE** | Uploaded by the user; origin not verified by SPADE (content hash shown) |
| **DEMO** | Illustrative content (none is used in the app) |
| **CONNECTED / LIVE** | Reserved for real telemetry with heartbeat — **not implemented** in this repository |
"""

_FORMATS = """
**Inference upload (AI Diagnostics)** — CSV/JSON with at least one known channel (`acc_x, acc_y, acc_z, current_r, current_s, current_t, rpm, torque_nm` for dc_motor; `vc_real, vc_imag, i_f_real, i_f_imag` for vsc_dpc). Only channels are kept.

**Training upload (Transfer Learning)** — must also contain:
- `source_file` — one ID per contiguous recording (acquisition); windows never cross recordings,
- `timestamp_s` — strictly increasing inside each recording, sampled at **10 000 Hz** (no resampling is implemented; other rates are rejected),
- `label` (classification only) — one of the base model's classes, one label per recording,
- every channel the base model was trained on.
Rows with non-finite or implausible values are dropped and the gap is treated as a discontinuity. Size limit 50 MB.

**DPC upload (VSC / DPC page)** — the trained network's 15-column input row (7 required, 8 auto-fillable).

**Hardware evidence** — `deployment_evidence.json` in a run directory, e.g.
`{"static_compatibility_passed": {...}, "bench_verified": {"board": "...", "runtime": "...", "latency_ms": ..., "ram_kb": ..., "flash_kb": ..., "accuracy_after_quantization": ..., "by": "...", "date": "..."}}`.
Without it, no run is shown as hardware-verified.
"""

_LIMITS = """
- **All data are simulated.** No hardware ingestion, streaming or heartbeat exists; Monitoring is SIMULATED MONITORING.
- **Vibration is synthetic_approximate.** Module B was calibrated against one measured accelerometer axis (KAt-DataCenter); y/z use assumed scale factors. Leave-bearing-out CV shows the calibration does **not** generalise to unseen bearings (model RMSE > flat baseline in 6/6 folds, `configs/vibration_module_b.yaml`).
- **PMSM** is a stand-alone FOC/MTPA control demo: no fault model, dataset, classifier or registry entry.
- **Edge (.tflite)** artifacts are exported (quantization read from the files) but never executed on a Raspberry Pi 5 / ESP32 by this project.
- **Classifier confidence** of the promoted dc_motor classifier is temperature-calibrated on its validation split, and windows below its abstention threshold are reported as *uncertain*. This does not cover unseen fault types: ball faults are still called outer_race with high confidence (docs/AI_TECHNICAL_AUDIT.md §7.2).
- **Thresholds** (TL acceptance, plausibility bounds) are proposed defaults that require researcher confirmation. The anomaly-score thresholds 0.70/0.85 have a measured operating point on simulated runs only (AUC 1.0, no false alarm or missed detection at 0.70; docs/audit/ai/anomaly_agent_roc.json).
- **Theme:** dark by default; switch to light in the app menu (⋮ → Settings → Theme).
"""


def render() -> None:
    page_header("Help / Documentation", "How to read SPADE", "Status semantics, data formats, design decisions and the limits of what this workbench demonstrates.", [])
    t1, t2, t3, t4 = st.tabs(["Status & sources", "Data formats", "Known limitations", "Platform"])
    with t1:
        st.markdown(_STATUS)
        st.markdown(_SOURCES)
    with t2:
        st.markdown(_FORMATS)
    with t3:
        st.markdown(_LIMITS)
    with t4:
        st.markdown(
            "**Systems.** (1) DC motor — permanently-excited DC motor (GEM parameters) under PI or MPC, bearing-fault injection on an electrical (MCSA torque-ripple) path and a synthetic vibration path. "
            "(2) PMSM — FOC with MTPA vs. i_d = 0. (3) VSC — trained Direct Power Control network on an identified converter plant (ported from DPC4PowerElectronics).\n\n"
            "**AI layer.** Config-driven classifiers (dc_motor) and forecasters (vsc_dpc) per domain and tier (PC / Raspberry Pi 5 / ESP32), a registry with audited promotion and rollback, "
            "fail-closed transfer learning, and monitoring agents (rule-based gateway, simulation-based detector, confidence-drift server).\n\n"
            "**Macro-phase map.** A → DC Motor + PMSM · B → VSC / DPC · IA → AI Diagnostics + Model Registry · TL → Transfer Learning · LM → Monitoring + Alert History."
        )
