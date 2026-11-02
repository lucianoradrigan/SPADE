"""AI-N26 (2026-11-02): the C2000 real-time criterion C5 assumed 5 kHz; the training data are 10 kHz.

POST-HOC ANALYSIS, NOT PRE-REGISTERED. docs/audit/c2000/criteria.md defined C5 as "worst latency
below the window duration: 102.4 ms for 512 samples at 5 kHz, 51.2 ms for 256 samples at 5 kHz"
and the board results (2026-10-30) were judged against it. Those verdicts are kept as recorded.
This script only re-reads what exists:

1. the sample rate of each measured run's training dataset (its fs_hz column and the timestamp
   step), found through the run's run_metadata.json and checked against the recorded SHA-256;
2. the latency the board itself reported (the kit's hardware_run.json, unchanged);
3. the window duration at the dataset's real rate, and whether that latency fits it.

Nothing is re-measured on the board.

    python experiments/report_c2000_sample_rate.py
Writes docs/audit/c2000/ai_n26_sample_rate.{md,json}.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "audit" / "c2000" / "ai_n26_sample_rate"
RUNS = (
    "configs/classifiers/esp32_dscnn/2026-10-18_run01",
    "configs/regressors/esp32_tiny/2026-10-19_run01",
    "configs/regressors/rpi5_tcn/2026-10-20_run01",
)
ASSUMED_FS_HZ = 5000.0  # the rate scripts/edge_benchmark/c2000_collect.py used until 2026-11-02


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_rate(path: Path) -> dict:
    table = pq.read_table(path, columns=["fs_hz", "timestamp_s", "source_file"])
    fs = np.unique(table.column("fs_hz").to_numpy(zero_copy_only=False))
    first = table.column("source_file").to_numpy(zero_copy_only=False)
    t = table.column("timestamp_s").to_numpy()
    same = first[1:] == first[:-1]
    dt = np.diff(t)[same]
    return {"fs_hz_values": [float(v) for v in fs], "median_dt_s": float(np.median(dt)), "rate_from_dt_hz": float(1.0 / np.median(dt)), "rows": table.num_rows}


def main() -> None:
    rows = []
    for rel in RUNS:
        run_dir = REPO / rel
        meta = json.loads((run_dir / "run_metadata.json").read_text())
        data_path = REPO / meta["dataset_path"]
        sha_ok = _sha256(data_path) == meta["dataset_sha256"] if data_path.exists() else None
        rate = dataset_rate(data_path) if data_path.exists() else None
        board = json.loads((run_dir / "c2000" / "hardware_run.json").read_text())
        manifest = json.loads((run_dir / "c2000" / "manifest.json").read_text())
        steps = manifest["input_shape"][0]
        fs = rate["fs_hz_values"][0] if rate and len(rate["fs_hz_values"]) == 1 else None
        window_ms = steps / fs * 1e3 if fs else None
        rows.append({
            "run": rel.split("/", 2)[2], "samples": steps, "dataset": meta["dataset_path"], "dataset_sha256_matches_run": sha_ok, "dataset_rate": rate,
            "worst_latency_ms": board["worst_latency_ms"], "pre_registered_budget_ms": board["latency_budget_ms"],
            "pre_registered_C5": board["verdicts"]["C5_real_time"], "assumed_fs_hz": ASSUMED_FS_HZ,
            "window_ms_at_dataset_rate": window_ms, "post_hoc_C5_at_dataset_rate": (board["worst_latency_ms"] < window_ms) if window_ms else None,
            "latency_over_window": board["worst_latency_ms"] / window_ms if window_ms else None,
        })
    result = {"finding": "AI-N26", "label": "post-hoc analysis, not pre-registered; pre-registered verdicts unchanged", "command": "python experiments/report_c2000_sample_rate.py", "runs": rows}
    OUT.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# AI-N26: the C2000 real-time budget assumed 5 kHz; the training data are 10 kHz",
        "",
        "**Post-hoc analysis, not pre-registered** (2026-11-02). The pre-registered verdicts in [`report.md`](report.md) stay as recorded. "
        "Nothing was re-measured on the board: the latencies are the board's own reports (`hardware_run.json`).",
        "",
        f"- **Command:** `{result['command']}`. Raw values: [`ai_n26_sample_rate.json`](ai_n26_sample_rate.json).",
        "- **Origin:** `scripts/edge_benchmark/c2000_collect.py` fixed `SAMPLE_RATE_HZ = 5000.0`, citing `windowing.py: fs_hz = 1 / TAU`. "
        "But `TAU = 1e-4 s` (`datagen/runner.py`), so `1 / TAU` = 10 000 Hz. The 5000 came from the `windowing.py` docstring, where it was the number of samples in a 0.5 s window, not a rate. "
        "No run recorded its sample rate, so the exporter could not read it and the value was typed by hand.",
        "",
        "| Run | Samples | Dataset fs_hz (column / from Δt) | Board worst latency | Pre-registered window (5 kHz) → C5 | Window at the dataset rate → C5 (post-hoc) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        d = r["dataset_rate"]
        fs_txt = f"{', '.join(f'{v:g}' for v in d['fs_hz_values'])} / {d['rate_from_dt_hz']:.0f} Hz" if d else "dataset missing"
        pre = f"{r['pre_registered_budget_ms']:.1f} ms → {'pass' if r['pre_registered_C5'] else 'FAIL'}"
        post = f"{r['window_ms_at_dataset_rate']:.1f} ms → **{'pass' if r['post_hoc_C5_at_dataset_rate'] else 'FAIL'}** ({r['latency_over_window']:.2f}× the window)" if r["window_ms_at_dataset_rate"] else "n/a"
        lines.append(f"| `{r['run']}` | {r['samples']} | {fs_txt} | {r['worst_latency_ms']:.1f} ms | {pre} | {post} |")
    lines += [
        "",
        "The dataset of each run is the one in its `run_metadata.json`; its SHA-256 matches the recorded one: "
        + ", ".join(f"`{r['run']}` {'yes' if r['dataset_sha256_matches_run'] else 'NO'}" for r in rows) + ".",
        "",
        "## Reading",
        "",
        "- `esp32_dscnn/2026-10-18_run01` met the pre-registered C5, but at the data's real rate its window lasts 51.2 ms and the board needs 68.5 ms: it does **not** keep up with non-overlapping windows.",
        "- `esp32_tiny/2026-10-19_run01` keeps up at either rate (19.7 ms against 25.6 ms).",
        "- `rpi5_tcn/2026-10-20_run01` failed C5 already (AI-N24); at the real rate it takes about 20 times its window, not 10.",
        "- The edge examples generated by SPADE (ESP32 and C2000) also sampled at 5 kHz (a 200 µs delay). A model trained at 10 kHz fed 5 kHz data sees every frequency doubled in samples, so its fault orders move.",
        "",
        "## What changed (2026-11-02)",
        "",
        "- Training records the dataset's `fs_hz` in `metrics.json`; `windowing.run_sample_rate_hz()` reads it, with the simulator's 10 kHz for runs trained before (all of them on simulator data).",
        "- The C2000 manifest carries `fs_hz` and `window_ms`; `c2000_collect.py` takes the budget from there and refuses a kit without it.",
        "- The ESP32/C2000 examples and the Edge Deployment tutorials take the rate from the run (100 µs, 10 kHz for every shipped run).",
        "- The Model Registry and Edge Deployment pages show the pre-registered C2000 verdict as recorded, followed by this post-hoc correction.",
        "",
        "## Not executed",
        "",
        "- Re-measuring on the board: **no ejecutado / not executed**. Not needed for this correction, because the latency does not depend on the assumed rate; only the budget does.",
        "- Re-exporting the shipped C2000 kits with `fs_hz` in their manifest: **no ejecutado / not executed**, so the measured kits stay exactly as they were on the board.",
    ]
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
