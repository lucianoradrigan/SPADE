"""Out-of-distribution (OOD) detection for the fault classifiers -- TODO_CLAUDE_CODE.md A1.

AI-N12 left one gap open: a fault the model has no class for (ball, cage) gets a confident answer.
Temperature calibration and the abstention threshold only look at the softmax, and the softmax of a
ball-fault window is as peaked as an outer-race one. This module scores how far a window's
*features* are from what the classifier was trained on, independently of which class it picks.

Two scores, both "larger = more out-of-distribution":
- ``mahalanobis``: the minimum over classes of the squared Mahalanobis distance of the penultimate
  feature vector (the last hidden Dense layer, ``dense1`` in models/classifiers/builder.py) to the
  class mean, with one shared covariance (Lee et al., 2018). Means and covariance are fitted on the
  training windows.
- ``energy``: minus the log-sum-exp of the logits (Liu et al., 2020). The head is
  ``Dense(softmax)``, so the logits are recomputed from the penultimate features and the head's
  kernel and bias.

Each threshold is the quantile of the score on the *validation* windows that keeps
``target_coverage`` (95 %) of in-distribution windows, the same convention as the abstention
threshold (ai/calibration.py). Nothing from an OOD fault is used to fit anything.

The fitted parameters live in a run's ``metrics.json["ood"]``; runs without it skip the check.
"""

from __future__ import annotations

import numpy as np

UNKNOWN = "unknown"
METHODS = ("mahalanobis", "energy")
#: Ridge added to the shared covariance, relative to its mean diagonal, so near-constant ReLU
#: units do not make it singular.
_RIDGE = 1e-3


def penultimate_layer(model):
    """The last hidden layer feeding the head (the input of ``head``)."""
    head = model.get_layer("head")
    return head.input


def feature_extractor(model, layer: str | None = None):
    """Features for the Mahalanobis score: the head's input (default) or a named layer's output
    (e.g. ``gap``, the pooled convolutional features before the dense layers)."""
    import keras

    return keras.Model(model.inputs, model.get_layer(layer).output if layer else penultimate_layer(model))


def head_logits(model, features: np.ndarray) -> np.ndarray:
    kernel, bias = model.get_layer("head").get_weights()
    return np.asarray(features, dtype=np.float64) @ kernel + bias


def fit_mahalanobis(features: np.ndarray, y: np.ndarray, n_classes: int) -> dict:
    f = np.asarray(features, dtype=np.float64)
    means = np.stack([f[y == k].mean(axis=0) for k in range(n_classes)])
    centered = f - means[y]
    cov = centered.T @ centered / len(f)
    cov += _RIDGE * np.mean(np.diag(cov)) * np.eye(cov.shape[0])
    return {"means": means.tolist(), "precision": np.linalg.inv(cov).tolist()}


def mahalanobis_score(features: np.ndarray, params: dict) -> np.ndarray:
    f = np.asarray(features, dtype=np.float64)
    means, prec = np.asarray(params["means"]), np.asarray(params["precision"])
    d = f[:, None, :] - means[None, :, :]  # (n, classes, dim)
    return np.min(np.einsum("ncd,de,nce->nc", d, prec, d), axis=1)


def energy_score(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    m = z.max(axis=1, keepdims=True)
    return -(m[:, 0] + np.log(np.exp(z - m).sum(axis=1)))


def _features(model, x: np.ndarray, layer: str | None, batch_size: int) -> np.ndarray:
    fx = feature_extractor(model, layer)
    return np.asarray(fx(x, training=False)) if len(x) <= batch_size else fx.predict(x, batch_size=batch_size, verbose=0)


def scores(model, x: np.ndarray, ood: dict, batch_size: int = 256) -> dict:
    """Every fitted method's score for windows ``x`` (shape (n, window, channels))."""
    out = {}
    if "energy" in ood:
        out["energy"] = energy_score(head_logits(model, _features(model, x, None, batch_size)))
    if "mahalanobis" in ood:
        out["mahalanobis"] = mahalanobis_score(_features(model, x, ood["mahalanobis"].get("layer"), batch_size), ood["mahalanobis"])
    return out


def fit_ood(model, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, target_coverage: float = 0.95, method: str = "mahalanobis",
            layer: str | None = None) -> dict:
    """Fit both scores on train, both thresholds on val; ``method`` is the one decide() applies.
    ``layer``: features for Mahalanobis (None = the head's input)."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    n_classes = model.get_layer("head").get_weights()[1].shape[0]
    feats_train = _features(model, x_train, layer, 256)
    maha = fit_mahalanobis(feats_train, np.asarray(y_train), n_classes)
    maha["layer"] = layer
    ood = {"method": method, "target_coverage": target_coverage, "fitted_on": {"statistics": "train", "thresholds": "val"},
           "mahalanobis": maha, "energy": {}}
    val_scores = scores(model, x_val, ood)
    for m in METHODS:
        ood[m]["threshold"] = float(np.quantile(val_scores[m], target_coverage))
    return ood


def window_score(model, window: np.ndarray, ood: dict) -> float:
    """Score of ONE normalized window (shape (window, channels)) under the run's chosen method."""
    return float(scores(model, np.asarray(window)[np.newaxis, ...], {ood["method"]: ood[ood["method"]]})[ood["method"]][0])


def is_ood(score: np.ndarray | float, ood: dict, method: str | None = None) -> np.ndarray:
    m = method or ood["method"]
    return np.asarray(score) > ood[m]["threshold"]
