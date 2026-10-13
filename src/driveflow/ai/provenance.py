"""Run provenance and reproducibility helpers (AUDIT AI-11, AUDIT_ANSWERS.md Q20/Q23).

A seed that only feeds the data split does not reproduce a training run: weight initialisation,
dropout and shuffling draw from Python's `random`, NumPy's global RNG and TensorFlow's RNG.
set_global_seed() seeds all three through keras.utils.set_random_seed (which does exactly that).
Bit-exact reproducibility additionally needs deterministic TF kernels; enabling them globally can
slow training and is therefore opt-in (`deterministic_ops=True`), and recorded either way.

run_metadata() captures what is needed to explain a run later -- commit (+ dirty flag), library
versions, interpreter, platform -- without failing when git is unavailable (it records why).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import platform
import subprocess
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
_TRACKED_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "tensorflow", "keras", "streamlit", "plotly")


def set_global_seed(seed: int, deterministic_ops: bool = False) -> dict:
    """Seeds Python, NumPy and TensorFlow/Keras RNGs. Returns what was set, for the run record."""
    import keras

    keras.utils.set_random_seed(int(seed))
    applied = {"seed": int(seed), "python_numpy_tf_seeded": True, "deterministic_ops": False}
    if deterministic_ops:
        import tensorflow as tf

        try:
            tf.config.experimental.enable_op_determinism()
            applied["deterministic_ops"] = True
        except Exception as exc:  # pragma: no cover - depends on TF build
            applied["deterministic_ops_error"] = str(exc)
    return applied


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def run_metadata() -> dict:
    versions = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            versions[pkg] = None
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
        "git_note": None if commit else "git unavailable -- commit not recorded",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
    }


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def weights_fingerprint(model) -> str:
    """SHA-256 over every weight tensor (shape + dtype + bytes), in layer order. Two models with
    the same architecture have the same fingerprint iff their weights are bit-identical -- used to
    prove a base model was NOT modified by fine-tuning (AUDIT AI-04)."""
    h = hashlib.sha256()
    for w in model.get_weights():
        a = np.ascontiguousarray(w)
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()
