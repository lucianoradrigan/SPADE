"""Edge Deployment page: one tab per board with what it is, what it can and cannot do, the model
files (and, for the C2000, the export), copy-paste inference code generated from the run itself,
and a short tutorial."""

import streamlit as st

from driveflow.ai.c2000_export import C2000ExportError, export_run, kit_zip
from driveflow.ai.deployment_status import tflite_op_names
from driveflow.ai.edge_examples import VERIFICATION, c2000_example, esp32_example, facts_from_run, rpi_cpp_example, rpi_python_example
from driveflow.ai.registry import RegistryError
from driveflow.viz.adapters.registry_adapter import c2000_summary, list_runs
from driveflow.viz.components.board_card import board_banner, board_cards, panel, tutorial_step
from driveflow.viz.components.metric_cards import kpi_row
from driveflow.viz.components.page_header import badge_html, page_header

DOMAIN_LABELS = {"dc_motor": "DC motor (dc_motor)", "vsc_dpc": "VSC / DPC (vsc_dpc)"}
BLOCK_LABELS = {"classifier": "Fault classifier", "regressor": "Forecaster"}

_ABOUT = {
    "esp32": ("A low-cost Wi-Fi microcontroller, suited to a sensor node next to the machine. The model is quantized to 8-bit integers so it fits in a few hundred KB of RAM.",
              ["Compact models, with no recurrent layers.", "Runs on the node itself, with no connection.", "Integer inference, with no need for an FPU."],
              ["Not run on a board: latency, RAM and accuracy on the device are unknown.", "Quantization can cost accuracy: the 5-class int8 export is not usable, so this tier keeps 3 classes.", "A ball or cage fault is answered as one of those 3 classes."]),
    "rpi5": ("A small Linux computer, suited to a gateway that gathers several machines. It has the most headroom of the three: the models keep float weights and can be larger.",
             ["The largest edge models, including the 5-class classifier.", "Python for prototyping, C++ for a deployed service.", "Works offline, and can talk to the PC tier."],
             ["Not run on a board: latency and memory on the device are unknown.", "A model with Flex ops needs the Flex delegate, and does not load on plain LiteRT.", "Linux is not real-time: do not put it inside a control loop."]),
    "c2000": ("Texas Instruments' real-time microcontroller for motor drives and power converters. The model runs next to the control code. The C28x has no 8-bit type, so TFLite Micro cannot run here: SPADE generates plain float32 C instead.",
              ["Same arithmetic as the Keras model, with no quantization loss.", "No runtime or library to install, and fixed memory use.", "Window normalization and calibration included in the C code."],
              ["Measured with simulated windows replayed from flash: no ADC or real signal yet.", "Only small models: 128 KB for tensors. The 5-class classifier does not fit.", "Inference takes tens of ms: run it apart from the control ISR, never inside it.", "No out-of-distribution check: the device never answers \"unknown\"."]),
}


def _steps(items: list, target: str) -> None:
    for i, (text, code, language) in enumerate(items, 1):
        tutorial_step(i, text, target)
        if code:
            st.code(code, language=language)


def _tutorial(target: str, facts) -> None:
    order = ", ".join(f"`{c}`" for c in facts.channels)
    if target == "esp32":
        _steps([
            ("Install ESP-IDF, then create the project and add TensorFlow Lite Micro.", 'idf.py create-project spade_infer\ncd spade_infer\nidf.py add-dependency "espressif/esp-tflite-micro"', "bash"),
            ("Download the `.tflite` above, rename it `model.tflite`, and turn it into a C array.", "xxd -i model.tflite | sed 's/^unsigned char model_tflite\\[\\]/const unsigned char model_tflite[] __attribute__((aligned(16)))/' > main/model_data.c", "bash"),
            ("Paste the example into `main/main.cc`, delete the generated `main/spade_infer.c`, and list both sources.", 'idf_component_register(SRCS "main.cc" "model_data.c" INCLUDE_DIRS ".")', "cmake"),
            (f"Replace `read_sample()` with your driver. Channel order: {order}. Sample at 5 kHz, the rate of the training data.", None, None),
            ("Build, flash and watch the serial output. Then set `kArenaSize` from the `arena used` line.", "idf.py build flash monitor", "bash"),
        ], target)
    elif target == "rpi5":
        _steps([
            ("**Python.** Create an environment on the Pi and install the runtime.", "python3 -m venv spade && source spade/bin/activate\npip install ai-edge-litert numpy", "bash"),
            ("Download the `.tflite` above, rename it `model.tflite`, and save the example next to it as `spade_infer.py`.", "python3 spade_infer.py", "bash"),
            (f"Replace the random window with real data. Shape: ({facts.window}, {facts.n_channels}). Channel order: {order}. Sample at 5 kHz.", None, None),
            ("**C++.** Get the TensorFlow sources, and save the example as `main.cpp` with this `CMakeLists.txt`.", "git clone --depth 1 --branch v2.17.0 https://github.com/tensorflow/tensorflow.git tensorflow_src", "bash"),
            ("Create `CMakeLists.txt`.", "cmake_minimum_required(VERSION 3.16)\nproject(spade_infer CXX)\nset(CMAKE_CXX_STANDARD 17)\nadd_subdirectory(tensorflow_src/tensorflow/lite tflite_build EXCLUDE_FROM_ALL)\nadd_executable(spade_infer main.cpp)\ntarget_link_libraries(spade_infer tensorflow-lite)", "cmake"),
            ("Build and run. The first build takes a while on the Pi.", "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release\ncmake --build build -j4\ncp model.tflite build/ && cd build && ./spade_infer", "bash"),
        ], target)
    else:
        _steps([
            ("Install TI's C2000 compiler (CGT), UniFlash and C2000Ware. The kit's Makefile looks for them in `~/ti`.", None, None),
            ("Export the model above, download the kit, and check it builds. It first runs a benchmark with reference windows.", "unzip <downloaded_kit>.zip && cd spade_c2000\nmake CCXML=/path/to/TMS320F28379D.ccxml flash\npython3 -m serial.tools.miniterm /dev/ttyUSB0 115200", "bash"),
            ("Replace the kit's `main.c` with the example. It uses the kit's `spade_model_run()`, which normalizes the window and runs the model.", None, None),
            (f"Replace `read_sample()` with your ADC reads. Channel order: {order}. Sample at 5 kHz.", None, None),
            ("Keep inference out of the control ISR: fill the window from an ADC interrupt and call `spade_model_run()` from the main loop.", None, None),
            ("Build and flash again.", "make clean\nmake CCXML=/path/to/TMS320F28379D.ccxml flash", "bash"),
        ], target)


def _about(target: str) -> None:
    text, scope, limits = _ABOUT[target]
    board_banner(target, text)
    left, right = st.columns(2, gap="medium")
    with left:
        panel("Scope", scope, "healthy", "✓")
    with right:
        panel("Limitations", limits, "warning", "▲")


def _model_facts(facts, memory: tuple | None) -> None:
    """memory: (label, value, unit) of what this board's artifact weighs, or None if not exported."""
    outputs = {"label": "Output", "value": str(len(facts.classes)), "unit": "classes"} if facts.is_classifier else {"label": "Output", "value": "—" if facts.horizon is None else str(facts.horizon), "unit": "steps ahead"}
    items = [{"label": "Input window", "value": str(facts.window), "unit": "samples"}, {"label": "Channels", "value": str(facts.n_channels), "tone": "training"}, {**outputs, "tone": "primary"}]
    if memory:
        items.append({"label": memory[0], "value": memory[1], "unit": memory[2], "tone": "training"})
    kpi_row(items)


def _promoted(rows: list, tier: str):
    return next((r for r in rows if r["tier"] == tier and r["status"].startswith("Promoted")), None)


def _facts(row):
    tflite = row["run_dir"] / "model.tflite"
    ops = tflite_op_names(tflite) if tflite.exists() else []
    return facts_from_run(row["metrics"], row["config"], ops)


def _tflite_tab(rows: list, target: str, domain: str, block: str, label: str) -> None:
    _about(target)
    st.markdown("##### :material/deployed_code: Model")
    row = _promoted(rows, target)
    if row is None:
        st.info(f"No {BLOCK_LABELS[block].lower()} is promoted for this board and domain.", icon=":material/info:")
        return
    st.caption(f"Promoted run {row['run']} · {row['headline']}. Fixed, already-trained artifact: not re-generated from the sample-run controls of AI Diagnostics.")
    tflite = row["run_dir"] / "model.tflite"
    if tflite.exists():
        st.download_button(f"Download {label} model (.tflite)", tflite.read_bytes(), file_name=f"{domain}_{target}_{block}.tflite", key=f"ia_edge_dl_{domain}_{target}_{block}", icon=":material/download:", type="primary")
    else:
        st.caption("Promoted, not exported.")
    facts = _facts(row)
    _model_facts(facts, ("Model file", f"{tflite.stat().st_size / 1024:.1f}", "KB") if tflite.exists() else None)
    flex = [o for o in facts.tflite_ops if o.startswith("Flex")]
    if flex:
        st.warning(f"This model has Flex ops ({', '.join(flex)}): the examples need a runtime with the Flex delegate.", icon=":material/warning:")
    st.markdown("##### :material/code: Inference code")
    if target == "esp32":
        st.code(esp32_example(facts), language="cpp")
        st.caption(VERIFICATION["esp32"])
    else:
        python_tab, cpp_tab = st.tabs([":material/terminal: Python", ":material/data_object: C++"])
        with python_tab:
            st.code(rpi_python_example(facts), language="python")
            st.caption(VERIFICATION["rpi5_python"])
        with cpp_tab:
            st.code(rpi_cpp_example(facts), language="cpp")
            st.caption(VERIFICATION["rpi5_cpp"])
    st.markdown("##### :material/school: Tutorial")
    _tutorial(target, facts)


def _c2000_export(row: dict, domain: str, block: str) -> None:
    kit = row["deployment"].get("c2000")
    measured = bool(kit and kit["run"])
    confirmed = True
    if measured:
        confirmed = st.checkbox("Replace this kit and discard its board measurements", key=f"edge_c2000_replace_{row['run']}")
    if st.button("Export to C2000", key=f"edge_c2000_export_{row['run']}", icon=":material/memory:", type="primary", disabled=not confirmed):
        with st.spinner("Generating C code and checking it against Keras…"):
            try:
                result = export_run(row["run_dir"], domain, block)
            except (C2000ExportError, RegistryError, OSError) as exc:
                st.session_state["edge_c2000_result"] = ("error", str(exc))
            else:
                parity = result["host_parity"]
                detail = f"max error vs. Keras on this PC {parity['max_abs_err']:.1e}" if parity else "gcc not found, PC check skipped"
                st.session_state["edge_c2000_result"] = ("success", f"Kit ready: {result['memory']['arena_bytes'] / 1024:.1f} KB of RAM for tensors, {detail}. Not measured on the board yet.")
        st.rerun()


def _c2000_tab(rows: list, domain: str, block: str) -> None:
    _about("c2000")
    st.markdown("##### :material/deployed_code: Model")
    candidates = sorted((r for r in rows if (r["run_dir"] / "model.weights.h5").exists()), key=lambda r: (not r["status"].startswith("Promoted"), r["tier"] != "esp32", r["run"]))
    if not candidates:
        st.info("No trained run with weights for this domain.", icon=":material/info:")
        return
    by_run = {r["run"]: r for r in candidates}
    run = st.selectbox("Run to export", list(by_run), format_func=lambda k: f"{k} · {by_run[k]['tier']} · {by_run[k]['status']}", key=f"edge_c2000_run_{domain}_{block}",
                       help="Any trained run can be tried. Models with unsupported layers, or too large for the board, are refused with the reason.")
    row = by_run[run]
    outcome = st.session_state.pop("edge_c2000_result", None)
    if outcome:
        getattr(st, outcome[0])(outcome[1])
    _c2000_export(row, domain, block)
    kit = row["deployment"].get("c2000")
    if kit:
        st.caption(c2000_summary(kit))
        st.download_button("Download C2000 kit (.zip)", kit_zip(kit["kit_dir"]), file_name=f"{domain}_c2000_{block}.zip", key=f"ia_edge_dl_{domain}_c2000_{block}", icon=":material/download:")
    facts = _facts(row)
    _model_facts(facts, ("Tensor RAM", f"{kit['manifest']['memory']['arena_bytes'] / 1024:.1f}", "KB") if kit else None)
    st.markdown("##### :material/code: Inference code")
    st.code(c2000_example(facts), language="c")
    st.caption(VERIFICATION["c2000"])
    st.markdown("##### :material/school: Tutorial")
    _tutorial("c2000", facts)


def render() -> None:
    page_header(
        "Intelligence / Edge deployment",
        "Run your models on edge hardware",
        "Take a trained SPADE model to an ESP32, a Raspberry Pi 5 or a TI C2000, with ready-to-copy inference code for each one.",
        [badge_html("ALL SIGNALS SIMULATED", "warning", "▲", "No board has processed real sensor data yet.")],
    )
    board_cards()
    c1, c2 = st.columns(2)
    domain = c1.selectbox("Domain", list(DOMAIN_LABELS), format_func=DOMAIN_LABELS.get, key="edge_domain")
    all_rows = [r for r in list_runs() if "error" not in r and r["domain"] == domain]
    blocks = [b for b in BLOCK_LABELS if any(r["block"] == b for r in all_rows)]
    if not blocks:
        st.info("No trained model for this domain.", icon=":material/info:")
        return
    block = c2.selectbox("Model", blocks, format_func=BLOCK_LABELS.get, key="edge_block")
    rows = [r for r in all_rows if r["block"] == block]
    esp32, rpi, c2000 = st.tabs([":material/sensors: ESP32", ":material/developer_board: Raspberry Pi 5", ":material/memory: TI C2000"])
    with esp32:
        _tflite_tab(rows, "esp32", domain, block, "ESP32")
    with rpi:
        _tflite_tab(rows, "rpi5", domain, block, "Raspberry Pi 5")
    with c2000:
        _c2000_tab(rows, domain, block)
