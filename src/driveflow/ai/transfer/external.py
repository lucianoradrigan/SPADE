"""Importing a third-party model as a transfer-learning starting point.

An uploaded model file is checked, loaded in Keras safe mode and saved as an ordinary run
directory (configs/<block>s/external/<name>_<sha12>/) with:

    config.yaml    model_format: keras, block, domain, tier, input_window, num_classes, source
    model.keras    the full model, re-saved by Keras after the safe load
    metrics.json   kind, domain, channels and classes as DECLARED by the user, origin, sha256

From then on it is one more run on this machine: it can be evaluated in AI Diagnostics, used as
the base on the Transfer Learning page, and (after fine-tuning and the promotion gate) promoted.

Supported formats (SUPPORTED_FORMATS is what the GUI shows):
- Keras v3 ``.keras`` and legacy HDF5 ``.h5`` FULL models (architecture + weights), classifiers
  with one input of shape (window, channels) and one Dense softmax output of shape (classes,).
- Refused: Lambda / TFOpLambda layers and any layer outside the keras namespace (they would run
  Python from the file, or need code SPADE does not have), weights-only files, SavedModel, TFLite,
  ONNX and PyTorch (see SUPPORTED_FORMATS for why).

The model's own preprocessing is unknown to SPADE: windows reach it with SPADE's per-window
z-score per channel (models.common.windowing). A model trained with another normalisation will
start from a poor accuracy; fine-tuning on SPADE windows is what adapts it.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import yaml

from driveflow.ai.audit_findings import ensure_synthetic_class_finding
from driveflow.ai.registry import FULL_MODEL_FILE, KERAS_FORMAT, REPO_ROOT

MAX_MODEL_MB = 200
EXTERNAL_ROOT = REPO_ROOT / "configs" / "classifiers" / "external"
_FORBIDDEN_LAYERS = {"Lambda", "TFOpLambda", "TFSMLayer"}
_NAME_RE = re.compile(r"[^A-Za-z0-9_-]+")

SUPPORTED_FORMATS = [
    {"Format": "SPADE run directory", "Files": "config.yaml + model.weights.h5 + metrics.json", "Status": "Supported",
     "Notes": "Any run on this machine: promoted, custom-trained (AI Diagnostics → Train a model), transfer-learning candidates, imported models."},
    {"Format": "Keras v3 model", "Files": ".keras", "Status": "Supported",
     "Notes": "Full model (architecture + weights). Loaded in safe mode; Lambda layers and custom (non-Keras) layers are refused."},
    {"Format": "Keras HDF5 full model", "Files": ".h5 / .hdf5 saved with model.save()", "Status": "Supported",
     "Notes": "Same checks as .keras. A weights-only .h5 has no architecture and is refused."},
    {"Format": "Weights only", "Files": ".weights.h5", "Status": "Not supported",
     "Notes": "No architecture to rebuild. Import the full model, or train it in SPADE."},
    {"Format": "TensorFlow SavedModel", "Files": "directory / .pb", "Status": "Not supported",
     "Notes": "Keras 3 only loads it as an inference-only layer; it cannot be fine-tuned. Re-save it as .keras."},
    {"Format": "TensorFlow Lite", "Files": ".tflite", "Status": "Not supported", "Notes": "Inference-only format; it has no trainable graph."},
    {"Format": "ONNX / PyTorch", "Files": ".onnx / .pt / .pth", "Status": "Not supported",
     "Notes": "Would need converter dependencies (onnx2tf, torch) that SPADE does not install. Convert to .keras first."},
]


class ExternalModelError(ValueError):
    """The file cannot be imported; the message says why."""


def _model_config(path: Path) -> dict:
    """The serialized architecture, read WITHOUT deserializing any layer."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            if "config.json" not in z.namelist():
                raise ExternalModelError("not a Keras v3 .keras archive (no config.json inside)")
            return json.loads(z.read("config.json"))
    import h5py

    try:
        with h5py.File(path, "r") as f:
            raw = f.attrs.get("model_config")
    except OSError:
        raise ExternalModelError("not a .keras archive nor an HDF5 file") from None
    if raw is None:
        raise ExternalModelError("this HDF5 file has no model architecture (weights only?): save the full model with model.save()")
    return json.loads(raw.decode() if isinstance(raw, bytes) else raw)


def _layers(cfg) -> list:
    """Every (class_name, module) pair in a serialized model config, nested models included."""
    out = []
    if isinstance(cfg, dict):
        if "class_name" in cfg:
            out.append((cfg["class_name"], cfg.get("module")))
        for v in cfg.values():
            out += _layers(v)
    elif isinstance(cfg, list):
        for v in cfg:
            out += _layers(v)
    return out


def inspect_file(path) -> dict:
    """Static checks before anything is deserialized. Raises ExternalModelError."""
    path = Path(path)
    if path.stat().st_size > MAX_MODEL_MB * 1024 * 1024:
        raise ExternalModelError(f"file is larger than {MAX_MODEL_MB} MB")
    cfg = _model_config(path)
    layers = _layers(cfg)
    bad = sorted({c for c, _ in layers if c in _FORBIDDEN_LAYERS})
    if bad:
        raise ExternalModelError(f"layers {bad} execute Python code stored in the file; SPADE refuses them. Rebuild the model with standard Keras layers.")
    custom = sorted({f"{m}.{c}" for c, m in layers if m and not str(m).startswith("keras")})
    if custom:
        raise ExternalModelError(f"custom (non-Keras) objects {custom[:5]} need code SPADE does not have")
    return {"format": "keras_v3" if zipfile.is_zipfile(path) else "hdf5", "n_layers": len(layers)}


def load_checked(path):
    """inspect_file + keras.models.load_model(safe_mode=True) + shape checks for a classifier.
    Returns (model, {"input_window", "n_channels", "n_classes", "format"})."""
    import keras

    info = inspect_file(path)
    try:
        model = keras.models.load_model(path, compile=False, safe_mode=True)
    except Exception as exc:
        raise ExternalModelError(f"Keras could not load the model: {type(exc).__name__}: {exc}") from None
    if len(model.inputs) != 1 or len(model.outputs) != 1:
        raise ExternalModelError(f"expected one input and one output, got {len(model.inputs)} and {len(model.outputs)}")
    shape_in, shape_out = tuple(model.inputs[0].shape), tuple(model.outputs[0].shape)
    if len(shape_in) != 3 or shape_in[1] is None or shape_in[2] is None:
        raise ExternalModelError(f"input must be (batch, window, channels) with a fixed window and channel count, got {shape_in}")
    if len(shape_out) != 2 or shape_out[1] is None or shape_out[1] < 2:
        raise ExternalModelError(f"output must be (batch, classes) with at least 2 classes, got {shape_out}")
    from driveflow.ai.transfer.pipeline import output_layer

    try:
        head = output_layer(model)
    except ValueError as exc:
        raise ExternalModelError(str(exc)) from None
    act = getattr(head.activation, "__name__", str(head.activation))
    if act != "softmax":
        raise ExternalModelError(f"the output layer's activation is {act!r}; a classifier must end in softmax (SPADE reads its output as class probabilities)")
    return model, {**info, "input_window": int(shape_in[1]), "n_channels": int(shape_in[2]), "n_classes": int(shape_out[1])}


def import_model(file_bytes: bytes, filename: str, domain: str, channels: list, classes: list, root: Path | None = None) -> Path:
    """Checks, loads and saves an uploaded classifier as a run directory; returns it. channels and
    classes are the user's declaration of what the model's input channels and output units mean."""
    if len(file_bytes) > MAX_MODEL_MB * 1024 * 1024:
        raise ExternalModelError(f"file is larger than {MAX_MODEL_MB} MB")
    suffix = Path(filename).suffix.lower()
    if suffix not in (".keras", ".h5", ".hdf5"):
        raise ExternalModelError(f"unsupported file type {suffix!r}: upload a .keras or .h5 full model (see the supported-formats table)")
    sha = hashlib.sha256(file_bytes).hexdigest()
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"model{suffix}"
        src.write_bytes(file_bytes)
        model, info = load_checked(src)
        if len(channels) != info["n_channels"] or len(set(channels)) != len(channels):
            raise ExternalModelError(f"the model takes {info['n_channels']} channel(s); declare exactly that many distinct SPADE channels, in input order")
        classes = [str(c).strip() for c in classes]
        if len(classes) != info["n_classes"] or len(set(classes)) != len(classes) or not all(classes):
            raise ExternalModelError(f"the model has {info['n_classes']} output unit(s); declare exactly that many distinct class names, in output order")
        name = _NAME_RE.sub("_", Path(filename).stem)[:40] or "model"
        run_dir = Path(root or EXTERNAL_ROOT) / f"{name}_{sha[:12]}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
        try:
            model.save(run_dir / FULL_MODEL_FILE)
            model.save_weights(run_dir / "model.weights.h5")
            (run_dir / "config.yaml").write_text("# Imported third-party model (driveflow.ai.transfer.external). Architecture: model.keras.\n" + yaml.safe_dump({
                "model_format": KERAS_FORMAT, "block": "classifier", "domain": domain, "tier": "pc", "input_window": info["input_window"],
                "num_classes": info["n_classes"], "source": f"upload:{filename}"}, sort_keys=False))
            (run_dir / "metrics.json").write_text(json.dumps({
                "kind": "classifier", "domain": domain, "tier": "pc", "channels": list(channels), "classes": classes, "origin": "EXTERNAL",
                "status": "imported (not evaluated in SPADE, not promoted)", "source_file": filename, "source_sha256": sha, "source_format": info["format"],
                "preprocessing": "SPADE per-window z-score per channel (the model's original preprocessing is unknown)"}, indent=2))
            ensure_synthetic_class_finding(run_dir, classes)
        except Exception:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise
    return run_dir
