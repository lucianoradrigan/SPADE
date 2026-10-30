"""C export of a trained keras.Model for the TI C2000 (LAUNCHXL-F28379D, TMS320F28379D CPU1).

Not a .tflite: the C28x has no 8-bit type (CHAR_BIT = 16), so TFLite Micro and its byte-addressed
flatbuffer do not apply. The FPU is 32-bit, so the model is emitted as plain float32 C: the weights
as const arrays, the graph as one kernel call per layer (c2000_kit/spade_nn.c), the tensors in one
static arena with a liveness-based layout. The arithmetic is the Keras model's, with no quantization.

The window preprocessing is emitted too (spade_normalize), with the run's own level_scaling and
calibration, because the model is meaningless on raw signals. Out-of-distribution scoring is not
ported: a run's metrics.json "ood" has no C counterpart, so the device never answers UNKNOWN.

Memory is checked against the kit's linker regions; a model that does not fit raises
C2000MemoryError instead of producing a kit that cannot link.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import shutil
import subprocess
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from driveflow.models.common.windowing import normalize_window_like_training

KIT_DIR = Path(__file__).parent / "c2000_kit"
KIT_FILES = ("spade_nn.h", "spade_nn.c", "selftest.h", "selftest.c", "host_main.c", "main.c", "board.h", "board.c", "lnk_spade_f28379d.cmd", "Makefile")
EXPORT_DIRNAME = "c2000"
TARGET = "LAUNCHXL-F28379D"

#: lnk_spade_f28379d.cmd: ARENA is 0xFFF8 16-bit words, FLASHDATA 0x28000.
ARENA_LIMIT_BYTES = 0xFFF8 * 2
FLASHDATA_LIMIT_BYTES = 0x28000 * 2

ALIAS_LAYERS = ("Dropout", "Reshape")
ACTIVATIONS = ("linear", "relu", "sigmoid", "softmax")


class C2000ExportError(ValueError):
    pass


class UnsupportedLayerError(C2000ExportError):
    pass


class C2000MemoryError(C2000ExportError):
    pass


@dataclass
class _Root:
    """One arena buffer; several tensors may view it (reshape, last time step, in-place ops)."""

    floats: int
    first: int
    last: int
    offset: int = 0


@dataclass
class _View:
    root: _Root
    shape: tuple  # without the batch axis
    start: int = 0  # float offset inside the root

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))


@dataclass
class ModelPlan:
    calls: list = field(default_factory=list)  # C statements, with ${tensor} placeholders resolved at emit
    weights: dict = field(default_factory=dict)  # C name -> float32 array
    views: dict = field(default_factory=dict)  # Keras tensor id -> _View
    input_shape: tuple = ()
    output: _View | None = None
    arena_floats: int = 0

    @property
    def arena_bytes(self) -> int:
        return self.arena_floats * 4

    @property
    def weight_bytes(self) -> int:
        return sum(w.size for w in self.weights.values()) * 4


def _history(tensor):
    return id(tensor._keras_history[0]), tensor._keras_history[1], tensor._keras_history[2]


def _as_list(x) -> list:
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _single(cfg_value) -> int:
    return int(cfg_value[0]) if isinstance(cfg_value, (list, tuple)) else int(cfg_value)


def _is_last_step(layer) -> bool:
    """A Lambda is opaque code, so what it does is measured: only the last-time-step slice passes."""
    probe = np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3)
    try:
        return bool(np.array_equal(np.asarray(layer(probe)), probe[:, -1, :]))
    except Exception:  # noqa: BLE001 -- anything else it does is unsupported, whatever it raises
        return False


def _pad_left(padding: str, k: int, dilation: int) -> int:
    span = dilation * (k - 1)
    if padding == "causal":
        return span
    if padding == "same":
        return span // 2
    if padding == "valid":
        raise UnsupportedLayerError("padding='valid' on a convolution is not supported by the C2000 kernels")
    raise UnsupportedLayerError(f"padding {padding!r} is not supported by the C2000 kernels")


def plan_model(model) -> ModelPlan:
    """Walks the functional graph into kernel calls. Raises UnsupportedLayerError on anything the
    kernels do not reproduce exactly (recurrent, attention, strided convolution, ...)."""
    plan = ModelPlan()
    consumers: dict = {}
    order = [layer for layer in model.layers if type(layer).__name__ != "InputLayer"]
    for step, layer in enumerate(order):
        for tensor in _as_list(layer.input):
            consumers[_history(tensor)] = step
    final = len(order)
    outputs = _as_list(model.output)
    if len(outputs) != 1 or len(_as_list(model.input)) != 1:
        raise UnsupportedLayerError("only single-input, single-output models are supported")

    roots: list = []

    def new_view(tensor, step) -> _View:
        shape = tuple(int(d) for d in tensor.shape[1:])
        last = final if _history(tensor) == _history(outputs[0]) else consumers.get(_history(tensor), step)
        root = _Root(int(np.prod(shape)), step, last)
        roots.append(root)
        return _View(root, shape)

    def bind(tensor, view) -> None:
        plan.views[_history(tensor)] = view

    def extend(view, tensor) -> None:
        last = final if _history(tensor) == _history(outputs[0]) else consumers.get(_history(tensor), 0)
        view.root.last = max(view.root.last, last)

    def inplace_or_new(src: _View, tensor, step) -> _View:
        """Element-wise kernels run forward, so they may overwrite an input nobody reads later."""
        if src.root.last == step and src.start == 0 and src.size == src.root.floats:
            view = _View(src.root, tuple(int(d) for d in tensor.shape[1:]))
            extend(view, tensor)
            return view
        return new_view(tensor, step)

    def weight(name: str, values) -> str:
        plan.weights[name] = np.asarray(values, dtype=np.float32)
        return name

    def activation(act: str, view: _View, step: int) -> None:
        if act == "relu":
            plan.calls.append(("spade_relu", [view, view.size, view]))
        elif act == "sigmoid":
            plan.calls.append(("spade_sigmoid", [view, view.size, view]))
        elif act == "softmax":
            if len(view.shape) != 1:
                raise UnsupportedLayerError("softmax is only supported on a rank-1 output")
            plan.calls.append(("spade_softmax", [view, view.size, view]))

    input_tensor = _as_list(model.input)[0]
    plan.input_shape = tuple(int(d) for d in input_tensor.shape[1:])
    if len(plan.input_shape) != 2:
        raise UnsupportedLayerError(f"expected (steps, channels) input, got {plan.input_shape}")
    in_view = new_view(input_tensor, -1)
    bind(input_tensor, in_view)

    for step, layer in enumerate(order):
        kind = type(layer).__name__
        cfg = layer.get_config()
        inputs = [plan.views[_history(t)] for t in _as_list(layer.input)]
        tensor = _as_list(layer.output)[0]
        src = inputs[0]
        act = cfg.get("activation", "linear")
        if not isinstance(act, str) or act not in ACTIVATIONS:
            raise UnsupportedLayerError(f"layer {layer.name}: activation {act!r} is not supported")

        if kind in ("Conv1D", "DepthwiseConv1D"):
            if _single(cfg["strides"]) != 1:
                raise UnsupportedLayerError(f"layer {layer.name}: strided convolutions are not supported")
            if cfg.get("data_format", "channels_last") != "channels_last":
                raise UnsupportedLayerError(f"layer {layer.name}: only channels_last is supported")
            if not cfg.get("use_bias", True):
                raise UnsupportedLayerError(f"layer {layer.name}: use_bias=False is not supported")
            k, dilation = _single(cfg["kernel_size"]), _single(cfg["dilation_rate"])
            pad = _pad_left(cfg["padding"], k, dilation)
            kernel, bias = layer.get_weights()
            steps, in_ch = src.shape
            view = new_view(tensor, step)
            width = int(tensor.shape[-1]) if kind == "Conv1D" else int(cfg["depth_multiplier"])
            fn = "spade_conv1d" if kind == "Conv1D" else "spade_depthwise1d"
            plan.calls.append((fn, [src, steps, in_ch, view, width, weight(f"w{step}_kernel", kernel), weight(f"w{step}_bias", bias), k, dilation, pad]))
            activation(act, view, step)
        elif kind == "Dense":
            if len(src.shape) != 1:
                raise UnsupportedLayerError(f"layer {layer.name}: Dense on a rank-{len(src.shape)} tensor is not supported")
            if not cfg.get("use_bias", True):
                raise UnsupportedLayerError(f"layer {layer.name}: use_bias=False is not supported")
            kernel, bias = layer.get_weights()
            view = new_view(tensor, step)
            plan.calls.append(("spade_dense", [src, src.size, view, int(cfg["units"]), weight(f"w{step}_kernel", kernel), weight(f"w{step}_bias", bias)]))
            activation(act, view, step)
        elif kind == "BatchNormalization":
            if not (cfg.get("center", True) and cfg.get("scale", True)) or len(src.shape) != 2:
                raise UnsupportedLayerError(f"layer {layer.name}: only a full center+scale BatchNormalization on (T, C) is supported")
            gamma, beta, mean, var = layer.get_weights()
            scale = gamma / np.sqrt(var + float(cfg["epsilon"]))
            view = inplace_or_new(src, tensor, step)
            plan.calls.append(("spade_batchnorm", [src, src.shape[0], src.shape[1], view, weight(f"w{step}_scale", scale), weight(f"w{step}_offset", beta - mean * scale)]))
        elif kind == "ReLU":
            if cfg.get("negative_slope") or cfg.get("threshold") or cfg.get("max_value") is not None:
                raise UnsupportedLayerError(f"layer {layer.name}: only a plain ReLU is supported")
            view = inplace_or_new(src, tensor, step)
            plan.calls.append(("spade_relu", [src, src.size, view]))
        elif kind == "GlobalAveragePooling1D":
            view = new_view(tensor, step)
            plan.calls.append(("spade_gap1d", [src, src.shape[0], src.shape[1], view]))
        elif kind == "MaxPooling1D":
            pool, stride = _single(cfg["pool_size"]), _single(cfg["strides"])
            out_steps = int(tensor.shape[1])
            pad = max((out_steps - 1) * stride + pool - src.shape[0], 0) // 2 if cfg["padding"] == "same" else 0
            view = new_view(tensor, step)
            plan.calls.append(("spade_maxpool1d", [src, src.shape[0], src.shape[1], view, out_steps, pool, stride, pad]))
        elif kind == "Multiply":
            a, b = inputs
            if b.size > a.size:
                a, b = b, a
            if a.shape[-1] != b.shape[-1] or b.shape[0] not in (1, a.shape[0]):
                raise UnsupportedLayerError(f"layer {layer.name}: unsupported broadcast {a.shape} * {b.shape}")
            view = inplace_or_new(a, tensor, step)
            plan.calls.append(("spade_mul", [a, b, a.shape[0], a.shape[1], b.shape[0], view]))
        elif kind == "Add":
            a, b = inputs
            if a.shape != b.shape:
                raise UnsupportedLayerError(f"layer {layer.name}: Add needs equal shapes, got {a.shape} and {b.shape}")
            view = inplace_or_new(a, tensor, step)
            plan.calls.append(("spade_add", [a, b, a.size, view]))
        elif kind in ALIAS_LAYERS:
            view = _View(src.root, tuple(int(d) for d in tensor.shape[1:]), src.start)
            extend(view, tensor)
        elif kind == "Lambda":
            if not _is_last_step(layer):
                raise UnsupportedLayerError(f"layer {layer.name}: a Lambda other than x[:, -1, :] is not supported")
            view = _View(src.root, tuple(int(d) for d in tensor.shape[1:]), src.start + (src.shape[0] - 1) * src.shape[1])
            extend(view, tensor)
        else:
            raise UnsupportedLayerError(f"layer {layer.name}: {kind} has no C2000 kernel")
        bind(tensor, view)

    plan.output = plan.views[_history(outputs[0])]
    _layout(roots, plan)
    return plan


def _layout(roots: list, plan: ModelPlan) -> None:
    """First-fit, largest first: two buffers may share space only if their lifetimes do not overlap."""
    placed: list = []
    for root in sorted(roots, key=lambda r: -r.floats):
        offset = 0
        for other in sorted(placed, key=lambda r: r.offset):
            overlap = root.first <= other.last and other.first <= root.last
            if overlap and offset < other.offset + other.floats and other.offset < offset + root.floats:
                offset = other.offset + other.floats
        root.offset = offset
        placed.append(root)
    plan.arena_floats = max((r.offset + r.floats for r in placed), default=0)


def check_fits(plan: ModelPlan, reference_bytes: int = 0) -> dict:
    """The memory verdict. reference_bytes: the reference windows stored next to the weights."""
    return {
        "arena_bytes": plan.arena_bytes, "arena_limit_bytes": ARENA_LIMIT_BYTES,
        "weights_bytes": plan.weight_bytes, "reference_bytes": reference_bytes, "flash_data_limit_bytes": FLASHDATA_LIMIT_BYTES,
        "fits": plan.arena_bytes <= ARENA_LIMIT_BYTES and plan.weight_bytes + reference_bytes <= FLASHDATA_LIMIT_BYTES,
    }


def _ref(arg) -> str:
    if isinstance(arg, _View):
        return f"(spade_arena + {arg.root.offset + arg.start}L)"
    return f"{arg}L" if isinstance(arg, int) else str(arg)


def _c_floats(values) -> str:
    flat = np.asarray(values, dtype=np.float32).ravel()
    if not np.isfinite(flat).all():
        raise C2000ExportError("non-finite value in the exported data")
    cells = [f"{float(v):.9e}f" for v in flat]
    return ",\n".join("    " + ", ".join(cells[i : i + 6]) for i in range(0, len(cells), 6))


def _model_h(plan: ModelPlan, model_id: str, num_classes: int) -> str:
    steps, channels = plan.input_shape
    return f"""/* Generated by driveflow.ai.c2000_export. Do not edit. */
#ifndef SPADE_MODEL_H
#define SPADE_MODEL_H

#include <stdint.h>

#define SPADE_MODEL_ID "{model_id}"
#define SPADE_INPUT_STEPS {steps}L
#define SPADE_INPUT_CHANNELS {channels}L
#define SPADE_INPUT_LEN {steps * channels}L
#define SPADE_OUTPUT_LEN {plan.output.size}L
#define SPADE_NUM_CLASSES {num_classes}L
#define SPADE_ARENA_FLOATS {plan.arena_floats}L

/* raw: (SPADE_INPUT_STEPS, SPADE_INPUT_CHANNELS) physical samples, row by row. Normalizes the
 * window exactly as training did and runs the model. */
void spade_model_run(const float *raw);
const float *spade_model_output(void);

/* Per-channel statistics of the last window: a forecast y returns to physical units as
 * y * sigma + mu. */
const float *spade_model_mu(void);
const float *spade_model_sigma(void);

/* Classifier only: the run's temperature calibration and abstention, applied to the last output
 * in place. The class index, or -1 for "uncertain". */
int32_t spade_model_decide(void);

#endif
"""


def _model_c(plan: ModelPlan, levels: list, calibration: dict | None, channels: list) -> str:
    centers = _c_floats([c for c, _ in levels])
    scales = _c_floats([s for _, s in levels])
    temperature, abstain = (float(calibration["temperature"]), float(calibration["abstain_below"])) if calibration else (0.0, 0.0)
    parts = ["/* Generated by driveflow.ai.c2000_export. Do not edit. */", '#include "spade_model.h"', '#include "spade_nn.h"', "",
             f"/* Input channels, in order: {', '.join(channels)} */",
             "#ifdef __TI_COMPILER_VERSION__", '#pragma DATA_SECTION(spade_arena, ".spade_arena")', "#endif",
             "static float spade_arena[SPADE_ARENA_FLOATS];", "static float spade_mu[SPADE_INPUT_CHANNELS], spade_sigma[SPADE_INPUT_CHANNELS];", ""]
    parts.append(f"static const float level_center[SPADE_INPUT_CHANNELS] = {{\n{centers}\n}};")
    parts.append(f"static const float level_scale[SPADE_INPUT_CHANNELS] = {{\n{scales}\n}};\n")
    for name, values in plan.weights.items():
        parts.append(f"static const float {name}[{values.size}] = {{\n{_c_floats(values)}\n}};\n")
    input_view = plan.views[next(iter(plan.views))]
    parts.append("void spade_model_run(const float *raw)\n{")
    parts.append(f"    spade_normalize(raw, SPADE_INPUT_STEPS, SPADE_INPUT_CHANNELS, level_center, level_scale, {_ref(input_view)}, spade_mu, spade_sigma);")
    parts.extend(f"    {fn}({', '.join(_ref(a) for a in args)});" for fn, args in plan.calls)
    parts.append("}\n")
    parts.append(f"const float *spade_model_output(void)\n{{\n    return {_ref(plan.output)};\n}}\n")
    parts.append("const float *spade_model_mu(void)\n{\n    return spade_mu;\n}\n")
    parts.append("const float *spade_model_sigma(void)\n{\n    return spade_sigma;\n}\n")
    parts.append(f"int32_t spade_model_decide(void)\n{{\n    return spade_decide({_ref(plan.output)}, SPADE_OUTPUT_LEN, {temperature:.9e}f, {abstain:.9e}f);\n}}")
    return "\n".join(parts) + "\n"


def _vectors_h(raw: np.ndarray, expected: np.ndarray) -> str:
    return f"""/* Generated by driveflow.ai.c2000_export: simulated raw windows and the Keras outputs for them. */
#ifndef SPADE_TEST_VECTORS_H
#define SPADE_TEST_VECTORS_H

#define SPADE_TEST_VECTORS {len(raw)}L

static const float spade_test_raw[{raw.size}] = {{
{_c_floats(raw)}
}};

static const float spade_test_expected[{expected.size}] = {{
{_c_floats(expected)}
}};

#endif
"""


def _levels(channels: list, level_scaling: dict | None) -> list:
    """(center, scale) per channel; scale 0 means the window's own z-score."""
    return [tuple(float(v) for v in (level_scaling or {}).get(c, (0.0, 0.0))) for c in channels]


def export_c2000(model, metrics: dict, raw_windows: np.ndarray, out_dir, source_run: str = "") -> dict:
    """Writes the kit to out_dir and returns its manifest. raw_windows: (n, steps, channels) in
    physical units, float32; it becomes the reference set the firmware replays. As many windows as
    the flash region holds are kept (at least one)."""
    plan = plan_model(model)
    channels = list(metrics["channels"])
    if raw_windows.shape[1:] != plan.input_shape or len(channels) != plan.input_shape[1]:
        raise C2000ExportError(f"raw windows {raw_windows.shape[1:]} / channels {len(channels)} do not match the model input {plan.input_shape}")
    per_window = (raw_windows[0].size + plan.output.size) * 4
    room = (FLASHDATA_LIMIT_BYTES - plan.weight_bytes) // per_window
    verdict = check_fits(plan, per_window)
    if not verdict["fits"]:
        raise C2000MemoryError(f"does not fit {TARGET}: arena {plan.arena_bytes} B of {ARENA_LIMIT_BYTES} B, weights + one reference window {plan.weight_bytes + per_window} B of {FLASHDATA_LIMIT_BYTES} B")
    raw = raw_windows[: min(room, len(raw_windows))].astype(np.float32)

    normalized = np.stack([normalize_window_like_training(w, channels, metrics.get("level_scaling"))[0] for w in raw])
    expected = np.asarray(model.predict(normalized, verbose=0), dtype=np.float32).reshape(len(raw), -1)

    weights_digest = hashlib.sha256(b"".join(w.tobytes() for w in plan.weights.values())).hexdigest()
    classes = metrics.get("classes") or []
    calibration = metrics.get("calibration") if classes else None
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for name in KIT_FILES:
        shutil.copy2(KIT_DIR / name, out_dir / name)
    (out_dir / "spade_model.h").write_text(_model_h(plan, weights_digest[:12], len(classes)))
    (out_dir / "spade_model.c").write_text(_model_c(plan, _levels(channels, metrics.get("level_scaling")), calibration, channels))
    (out_dir / "test_vectors.h").write_text(_vectors_h(raw, expected))

    manifest = {
        "target": TARGET, "format": "float32 C (no quantization)", "source_run": source_run, "model_id": weights_digest[:12], "weights_sha256": weights_digest,
        "channels": channels, "classes": classes, "input_shape": list(plan.input_shape), "output_shape": list(plan.output.shape),
        "level_scaling": metrics.get("level_scaling"), "calibration": calibration, "ood_ported": False,
        "reference_windows": len(raw), "memory": check_fits(plan, per_window * len(raw)),
        "macs_per_inference": _macs(plan), "kernels": sorted({fn for fn, _ in plan.calls}),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def simulated_raw_windows(domain: str, channels: list, steps: int, n: int) -> np.ndarray:
    """Physical-unit windows from fresh simulations, one scenario after another so that every
    class is represented: the same scenarios as scripts/edge_benchmark/make_reference_vectors.py."""
    import pandas as pd

    from driveflow.datagen import Scenario, run_scenario
    from driveflow.datagen.export_parquet import records_to_dataframe

    if domain == "dc_motor":
        frames = [records_to_dataframe(run_scenario(Scenario(scenario_id=f"edge_{f or 'normal'}", fault_type=f, seed=7, duration_s=2.0))) for f in (None, "outer_race", "inner_race")]
        frames = [d[d["timestamp_s"] >= 0.5] for d in frames]
    else:
        frames = [records_to_dataframe(run_scenario(Scenario(scenario_id=f"edge_vsc_{m:g}", controller_type="DPC", plant_config_id="vsc_dpc_v1", reference_magnitude_v=m, duration_s=0.3, seed=7))) for m in (35.0, 50.0, 75.0)]
    per_frame = []
    for frame in frames:
        cols = pd.DataFrame({c: frame[c] if c in frame.columns else 0.0 for c in channels}).fillna(0.0).to_numpy(dtype=np.float32)
        per_frame.append([cols[i : i + steps] for i in range(0, len(cols) - steps + 1, steps)])
    picked = [w for group in zip(*per_frame) for w in group][:n]
    if not picked:
        raise C2000ExportError(f"the simulated runs are shorter than the model window ({steps} samples)")
    return np.stack(picked)


def export_run(run_dir, domain: str, block: str, reference_windows: int = 6) -> dict:
    """Exports one run directory to <run_dir>/c2000 and returns the manifest + the gcc parity."""
    from driveflow.ai.registry import load_run_model

    run_dir = Path(run_dir)
    model, _, metrics = load_run_model(run_dir, block)
    plan = plan_model(model)
    raw = simulated_raw_windows(domain, metrics["channels"], plan.input_shape[0], reference_windows)
    manifest = export_c2000(model, metrics, raw, run_dir / EXPORT_DIRNAME, source_run=f"{run_dir.parent.name}/{run_dir.name}")
    return {**manifest, "host_parity": host_parity(run_dir / EXPORT_DIRNAME) if shutil.which("gcc") else None}


def kit_zip(kit_dir) -> bytes:
    """The kit's sources and evidence as one download, without the local build output."""
    kit_dir = Path(kit_dir)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(kit_dir.iterdir()):
            if path.is_file():
                archive.write(path, f"spade_c2000/{path.name}")
    return buffer.getvalue()


def _macs(plan: ModelPlan) -> int:
    total = 0
    for fn, args in plan.calls:
        if fn == "spade_conv1d":
            total += args[1] * args[2] * args[4] * args[7]
        elif fn == "spade_depthwise1d":
            total += args[1] * args[2] * args[4] * args[7]
        elif fn == "spade_dense":
            total += args[1] * args[3]
    return int(total)


def host_parity(kit_dir, tolerance: float = 1e-3) -> dict:
    """Compiles the kit with gcc and replays its reference windows on this PC. A host measurement,
    never evidence about the board."""
    kit_dir = Path(kit_dir)
    build = kit_dir / "build"
    build.mkdir(exist_ok=True)
    exe = build / "host_selftest"
    sources = ["host_main.c", "selftest.c", "spade_nn.c", "spade_model.c"]
    subprocess.run(["gcc", "-O2", "-ffp-contract=off", "-I.", *sources, "-lm", "-o", str(exe.relative_to(kit_dir))], cwd=kit_dir, check=True, capture_output=True, text=True)
    lines = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout.splitlines()
    fields = [dict(item.split("=") for item in line.split()[2:]) for line in lines]
    errors = [float(f["err"]) for f in fields]
    if not errors or any(math.isnan(e) for e in errors):
        raise C2000ExportError(f"host self-test produced no usable output: {lines!r}")
    worst = max(errors)
    top1 = sum(f["top"] == f["exp"] for f in fields) / len(fields)
    return {"windows": len(fields), "max_abs_err": worst, "top1_agreement": top1, "passed": worst <= tolerance and top1 == 1.0, "tolerance": tolerance}
