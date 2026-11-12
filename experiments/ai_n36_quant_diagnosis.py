"""AI-N36 diagnosis (decision M-D3, 2026-11-12): where does the B4 ESP32 int8 lose inner_race, and what recovers it?
VALIDATION split of the B4 3-class training data only; E4 and E4c are not read. A diagnosis, not a candidate:
nothing is trained or promoted.

    python experiments/ai_n36_quant_diagnosis.py [--work /mnt/datos/spade/m4_work]

Model: esp32_dscnn/2026-11-10_run01 (B4 4a, Keras) and its selected int8 variant V1 (per tensor, noise-free stratified
calibration C2, model.tflite in the run). The int8 effects are measured with an exact float emulation of TFLite's
post-training quantization: BatchNorm folded into the pointwise convolutions, weights symmetric per tensor
(round(w / s) * s, s = max|w| / 127), activations asymmetric per tensor with fake_quant over the min/max of the
calibration set, at the tensors TFLite quantizes (input, each conv output after ReLU fusion, GAP, logits).
The emulation is checked against the real V1 file (top-1 agreement). Hypotheses:
  (a) calibration that includes 40 and 60 rad/s windows in the validation set's proportion;
  (b) per-channel scaling of the input before quantizing it, folded exactly into the first (depthwise) layer.
      Exact in float, but it changes the deployed preprocessing (divide each channel by s_c): a proposal.
Both are also exported as real per-tensor int8 TFLite files (work dir, not committed) and scored.
Results by subset of the validation windows: low speed (40, 60 rad/s), grid (80-220 rad/s), off-grid healthy.
Writes docs/audit/b4/ai_n36_quant_diagnosis.{md,json}.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import report_ai_n27_regen as RG  # noqa: E402
import report_esp32_repair as ER  # noqa: E402

ER.GRID = REPO / "data" / "b4_train_3class.parquet"
RUN = "configs/classifiers/esp32_dscnn/2026-11-10_run01"
OUT = REPO / "docs" / "audit" / "b4" / "ai_n36_quant_diagnosis"
LOW = (40.0, 60.0)
POINTS = ("A1_dw1", "A2_pw1_relu", "A3_dw2", "A4_pw2_relu", "A5_gap", "A6_logits")


def _fold(model):
    W = {l.name: l.get_weights() for l in model.layers}
    out = {}
    for b in ("block1", "block2"):
        dk, db = W[f"{b}_dw"]
        pk, pb = W[f"{b}_pw"]
        g, beta, mean, var = W[f"{b}_bn"]
        k = g / np.sqrt(var + 1e-3)
        out[b] = {"dw_k": dk[:, :, 0], "dw_b": db, "pw_k": pk[0] * k, "pw_b": (pb - mean) * k + beta}
    out["head"] = {"k": W["head"][0], "b": W["head"][1]}
    return out


def _qw(w):
    s = np.max(np.abs(w)) / 127.0
    return np.round(w / s) * s if s > 0 else w


BITS = {"n": 8}


def _fq(x, r):
    import tensorflow as tf
    lo, hi = min(r[0], 0.0), max(r[1], 0.0)
    return tf.quantization.fake_quant_with_min_max_args(x, min=lo, max=hi, num_bits=BITS["n"]).numpy()


def _dw(x, k, b):
    import tensorflow as tf
    y = tf.nn.depthwise_conv2d(x[:, None, :, :], k[None, :, :, None], strides=[1, 1, 1, 1], padding="SAME")[:, 0].numpy()
    return y + b


def forward(F, X, qin=None, qw=False, qact=None, in_scale=None):
    """qin: input range (or None); qw: quantize weights; qact: {point: range} activations to quantize; in_scale: per-channel s_c (b)."""
    w = (lambda a: _qw(a)) if qw else (lambda a: a)
    x = (X / in_scale).astype(np.float32) if in_scale is not None else X
    if qin is not None:
        x = _fq(x, qin)
    acts = {}
    k1 = F["block1"]["dw_k"] * (in_scale[None, :] if in_scale is not None else 1.0)
    h = _dw(x, w(k1), F["block1"]["dw_b"]); acts["A1_dw1"] = h
    h = _fq(h, qact["A1_dw1"]) if qact and "A1_dw1" in qact else h
    h = np.maximum(h @ w(F["block1"]["pw_k"]) + F["block1"]["pw_b"], 0); acts["A2_pw1_relu"] = h
    h = _fq(h, qact["A2_pw1_relu"]) if qact and "A2_pw1_relu" in qact else h
    h = _dw(h, w(F["block2"]["dw_k"]), F["block2"]["dw_b"]); acts["A3_dw2"] = h
    h = _fq(h, qact["A3_dw2"]) if qact and "A3_dw2" in qact else h
    h = np.maximum(h @ w(F["block2"]["pw_k"]) + F["block2"]["pw_b"], 0); acts["A4_pw2_relu"] = h
    h = _fq(h, qact["A4_pw2_relu"]) if qact and "A4_pw2_relu" in qact else h
    h = h.mean(axis=1); acts["A5_gap"] = h
    h = _fq(h, qact["A5_gap"]) if qact and "A5_gap" in qact else h
    z = h @ w(F["head"]["k"]) + F["head"]["b"]; acts["A6_logits"] = z
    z = _fq(z, qact["A6_logits"]) if qact and "A6_logits" in qact else z
    return z, acts


def _ranges(F, Xcal, in_scale=None):
    x = Xcal / in_scale if in_scale is not None else Xcal
    _, acts = forward(F, Xcal, in_scale=in_scale)
    r = {p: (float(a.min()), float(a.max())) for p, a in acts.items()}
    return (float(x.min()), float(x.max())), r


def _score(z, y, classes):
    pred = z.argmax(1)
    k = len(classes)
    return {"n": int(len(y)), "macro_f1": RG._macro_f1(y, pred, k), "accuracy": float(np.mean(pred == y)),
            "recall": {c: (float(np.mean(pred[y == i] == i)) if np.any(y == i) else None) for i, c in enumerate(classes)},
            "confusion": [[int(np.sum((y == t) & (pred == q))) for q in range(k)] for t in range(k)]}


def _by_subset(z, y, subset, classes):
    return {s: _score(z[subset == s], y[subset == s], classes) for s in ("low_40_60", "grid_80_220", "offgrid_healthy")} | {"all": _score(z, y, classes)}


def _export(work: Path, name: str, model, Xcal, scale=None) -> Path:
    from driveflow.ai.tflite_export import export_int8
    import keras

    d = work / name
    d.mkdir(parents=True, exist_ok=True)
    for f in ("metrics.json", "config.yaml", "run_metadata.json", "model.weights.h5"):
        shutil.copy2(REPO / RUN / f, d / f)
    m = model
    if scale is not None:
        m = keras.models.clone_model(model)
        m.set_weights(model.get_weights())
        lay = m.get_layer("block1_dw")
        k, b = lay.get_weights()
        lay.set_weights([k * scale[None, :, None], b])
    export_int8(m, (Xcal / scale if scale is not None else Xcal).astype(np.float32), d / "model.tflite", n_calibration_samples=len(Xcal), per_channel=False)
    return d


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, default=REPO / "workspace" / "ai_n36_work")
    args = ap.parse_args(argv)
    t0, c0 = time.time(), time.process_time()
    model, _ = ER._keras(RUN)
    info = RG._model_info(RUN)
    classes = info["metrics"]["classes"]
    grid = ER._grid_windows(info)
    df, Xg, yg, gg = grid
    Xv, yv_s, gv = ER._val_windows(info, grid)
    yv = np.array([classes.index(c) for c in yv_s])
    meta = df.groupby("source_file")[["grid_speed"]].first()
    sp = meta["grid_speed"].reindex(gv).to_numpy()
    off = np.array([str(g).startswith("b4_offgrid") for g in gv])
    subset = np.where(off, "offgrid_healthy", np.where(np.isin(sp, LOW), "low_40_60", "grid_80_220"))
    # V1's own calibration (C2: train split, noise-free, stratified 334/333/333, rng 0)
    Xc2, counts_c2 = ER._calibration("C2", RUN, info, grid)
    F = _fold(model)
    res = {"command": "python experiments/ai_n36_quant_diagnosis.py", "simulated": True, "decision": "M-D3 (validation split only; E4/E4c not read)",
           "run": RUN, "classes": classes, "n_val_windows": int(len(yv)),
           "val_windows_per_subset": {s: int(np.sum(subset == s)) for s in ("low_40_60", "grid_80_220", "offgrid_healthy")}, "configs": {}}
    # emulation check
    zf, _ = forward(F, Xv)
    pk = model.predict(Xv, verbose=0)
    res["emulation_check"] = {"float_folded_vs_keras_top1_agreement": float(np.mean(zf.argmax(1) == pk.argmax(1)))}
    qin, qact = _ranges(F, Xc2)
    P_v1 = RG._predict(RUN, "tflite", Xv)
    zq, _ = forward(F, Xv, qin=qin, qw=True, qact=qact)
    res["emulation_check"]["full_int8_emulated_vs_real_V1_top1_agreement"] = float(np.mean(zq.argmax(1) == P_v1.argmax(1)))
    cfg = {"keras_float": forward(F, Xv)[0], "weights_int8_only": forward(F, Xv, qw=True)[0],
           "activations_int8_only": forward(F, Xv, qact=qact)[0], "input_int8_only": forward(F, Xv, qin=qin)[0],
           "full_int8_emulated": zq, "full_int8_real_V1_tflite": P_v1}
    for p in POINTS:
        cfg[f"single_activation_{p}"] = forward(F, Xv, qact={p: qact[p]})[0]
    for name, z in cfg.items():
        res["configs"][name] = _by_subset(z, yv, subset, classes)
    res["ranges_C2"] = {"input": qin, **qact}
    # calibration composition (share of 40/60 rad/s windows)
    train = set(info["metrics"]["split"]["group_hashes"]["train"])
    keep_tr = np.array([RG._hash(x) in train for x in gg])
    sp_all = meta["grid_speed"].reindex(gg).to_numpy()
    low_val_share = float(np.mean(subset == "low_40_60"))
    # (a) C2 pool (train split, noise-free), stratified by class AND with the validation share of 40/60 rad/s windows
    noise = df.groupby("source_file")[["grid_electrical_noise_pct", "grid_mechanical_noise_pct"]].first()
    free = set(noise.index[(noise["grid_electrical_noise_pct"] == 0) & (noise["grid_mechanical_noise_pct"] == 0)])
    pool = keep_tr & np.isin(gg, list(free))
    rng = np.random.default_rng(0)
    parts, counts_a = [], {}
    for cls, n in zip(classes, ER.CAL_SPLIT):
        val_low_share_cls = float(np.mean(subset[yv == classes.index(cls)] == "low_40_60"))
        n_low = int(round(n * val_low_share_cls))
        lo_idx = np.flatnonzero(pool & (yg == cls) & np.isin(sp_all, LOW))
        hi_idx = np.flatnonzero(pool & (yg == cls) & ~np.isin(sp_all, LOW))
        take = np.concatenate([rng.choice(lo_idx, size=min(n_low, len(lo_idx)), replace=False), rng.choice(hi_idx, size=min(n - n_low, len(hi_idx)), replace=False)])
        counts_a[cls] = {"total": int(len(take)), "low_40_60": int(min(n_low, len(lo_idx))), "validation_low_share": val_low_share_cls}
        parts.append(take)
    Xa = Xg[rng.permutation(np.concatenate(parts))]
    qin_a, qact_a = _ranges(F, Xa)
    res["hypothesis_a"] = {"calibration_counts": counts_a, "validation_low_share_overall": low_val_share, "ranges": {"input": qin_a, **qact_a},
                           "emulated": _by_subset(forward(F, Xv, qin=qin_a, qw=True, qact=qact_a)[0], yv, subset, classes)}
    da = _export(args.work, "hyp_a_calibration_low_speed", model, Xa)
    res["hypothesis_a"]["real_tflite"] = _by_subset(RG._predict(str(da), "tflite", Xv), yv, subset, classes)
    # (b) per-channel input scaling before quantization (s_c = max|x_c| on the C2 calibration set), folded into block1_dw
    s = np.max(np.abs(Xc2), axis=(0, 1)).astype(np.float32)
    s[s == 0] = 1.0
    qin_b, qact_b = _ranges(F, Xc2, in_scale=s)
    res["hypothesis_b"] = {"per_channel_scale": dict(zip(info["metrics"]["channels"], map(float, s))), "ranges": {"input": qin_b, **qact_b},
                           "float_check_top1_agreement": float(np.mean(forward(F, Xv, in_scale=s)[0].argmax(1) == zf.argmax(1))),
                           "emulated": _by_subset(forward(F, Xv, qin=qin_b, qw=True, qact=qact_b, in_scale=s)[0], yv, subset, classes),
                           "input_only_emulated": _by_subset(forward(F, Xv, qin=qin_b, in_scale=s)[0], yv, subset, classes)}
    db = _export(args.work, "hyp_b_per_channel_input_scale", model, Xc2, scale=s.astype(np.float32))
    res["hypothesis_b"]["real_tflite"] = _by_subset(RG._predict(str(db), "tflite", (Xv / s).astype(np.float32)), yv, subset, classes)
    # Exploratory (not one of the two hypotheses; labelled as such in the report): clipping and 16-bit activations.
    _, acts_v = forward(F, Xv)
    res["exploratory"] = {"note": "not requested by M-D3; informs the next pre-registration only",
                          "clip_fraction_low_40_60": {p: float(np.mean((acts_v[p][subset == "low_40_60"] < qact[p][0]) | (acts_v[p][subset == "low_40_60"] > qact[p][1]))) for p in POINTS},
                          "clip_fraction_grid": {p: float(np.mean((acts_v[p][subset == "grid_80_220"] < qact[p][0]) | (acts_v[p][subset == "grid_80_220"] > qact[p][1]))) for p in POINTS}}
    BITS["n"] = 16
    res["exploratory"]["activations_16bit_weights_int8_emulated"] = _by_subset(forward(F, Xv, qin=qin, qw=True, qact=qact)[0], yv, subset, classes)
    res["exploratory"]["A3_only_16bit_rest_int8_emulated"] = None
    BITS["n"] = 8
    z16 = None
    # A3 at 16 bits, everything else at 8: emulate by quantizing A3 separately
    qact_noA3 = {k: v for k, v in qact.items() if k != "A3_dw2"}
    def fwd_mixed(X):
        BITS["n"] = 8
        x = _fq(X, qin)
        h = _fq(_dw(x, _qw(F["block1"]["dw_k"]), F["block1"]["dw_b"]), qact["A1_dw1"])
        h = _fq(np.maximum(h @ _qw(F["block1"]["pw_k"]) + F["block1"]["pw_b"], 0), qact["A2_pw1_relu"])
        h = _dw(h, _qw(F["block2"]["dw_k"]), F["block2"]["dw_b"])
        BITS["n"] = 16; h = _fq(h, qact["A3_dw2"]); BITS["n"] = 8
        h = _fq(np.maximum(h @ _qw(F["block2"]["pw_k"]) + F["block2"]["pw_b"], 0), qact["A4_pw2_relu"])
        h = _fq(h.mean(axis=1), qact["A5_gap"])
        return _fq(h @ _qw(F["head"]["k"]) + F["head"]["b"], qact["A6_logits"])
    res["exploratory"]["A3_only_16bit_rest_int8_emulated"] = _by_subset(fwd_mixed(Xv), yv, subset, classes)
    res["input_channel_ranges_C2"] = {c: [float(Xc2[..., i].min()), float(Xc2[..., i].max())] for i, c in enumerate(info["metrics"]["channels"])}
    res["calibration_C2_counts"] = counts_c2
    res["cpu"] = {"wall_s": round(time.time() - t0, 1), "process_cpu_s": round(time.process_time() - c0, 1)}
    OUT.with_suffix(".json").write_text(json.dumps(res, indent=2, default=float) + "\n")
    print(json.dumps({k: {s: (None if v["recall"]["inner_race"] is None else round(v["recall"]["inner_race"], 3)) for s, v in r.items() if s != "all"} for k, r in res["configs"].items()}, indent=1))
    print("a", {s: (None if v["recall"]["inner_race"] is None else round(v["recall"]["inner_race"], 3)) for s, v in res["hypothesis_a"]["real_tflite"].items()})
    print("b", {s: (None if v["recall"]["inner_race"] is None else round(v["recall"]["inner_race"], 3)) for s, v in res["hypothesis_b"]["real_tflite"].items()})
    print(res["emulation_check"])


if __name__ == "__main__":
    main()
