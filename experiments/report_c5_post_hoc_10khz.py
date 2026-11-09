"""AI-N26 post-hoc C5 at the data's real rate (10 kHz). Reads the board results already recorded in
docs/audit/c2000/ai_n26_sample_rate.json; measures nothing. Not pre-registered."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs/audit/c2000/ai_n26_sample_rate.json"
OUT = ROOT / "docs/audit/c2000/c5_post_hoc_10khz"
FS_HZ = 10_000.0

runs = []
for r in json.loads(SRC.read_text())["runs"]:
    window_ms = r["samples"] / FS_HZ * 1e3
    runs.append({"run": r["run"], "samples": r["samples"], "worst_latency_ms": round(r["worst_latency_ms"], 2),
                 "pre_registered_window_ms_5khz": r["pre_registered_budget_ms"], "pre_registered_C5": r["pre_registered_C5"],
                 "window_ms_10khz": window_ms, "post_hoc_C5_10khz": r["worst_latency_ms"] < window_ms,
                 "latency_over_window": round(r["worst_latency_ms"] / window_ms, 2)})
doc = {"finding": "AI-N26", "label": "POST HOC, not pre-registered; the pre-registered C5 verdicts in criteria.md / report.md are unchanged",
       "command": "python experiments/report_c5_post_hoc_10khz.py", "source": str(SRC.relative_to(ROOT)), "fs_hz": FS_HZ,
       "windows_ms": {"512": 51.2, "256": 25.6}, "runs": runs}
OUT.with_suffix(".json").write_text(json.dumps(doc, indent=2) + "\n")

rows = "\n".join(f"| `{r['run']}` | {r['samples']} | {r['worst_latency_ms']:.1f} ms | {r['pre_registered_window_ms_5khz']} ms → {'pass' if r['pre_registered_C5'] else 'FAIL'} "
                 f"| {r['window_ms_10khz']:.1f} ms → **{'pass' if r['post_hoc_C5_10khz'] else 'FAIL'}** | {r['latency_over_window']:.2f}× |" for r in runs)
OUT.with_suffix(".md").write_text(f"""# C5 at 10 kHz (AI-N26) · **POST HOC**

**Post-hoc analysis, not pre-registered** (2026-11-09). The pre-registered C5 verdicts in [`criteria.md`](criteria.md) and [`report.md`](report.md) are **not changed**. Nothing was re-measured: the latencies are the board's own reports, as already recorded in [`ai_n26_sample_rate.json`](ai_n26_sample_rate.json).

- **Command:** `python experiments/report_c5_post_hoc_10khz.py`. Raw values: [`c5_post_hoc_10khz.json`](c5_post_hoc_10khz.json).
- **Rate:** every training dataset has `fs_hz = 1 / TAU = 10 000 Hz` (TAU = 1e-4 s). C5 assumed 5 kHz.
- **Windows at 10 kHz:** 512 samples = **51.2 ms**; 256 samples = **25.6 ms**.

| Run | Samples | Worst latency (board) | Pre-registered window (5 kHz) → C5 | Window at 10 kHz → C5 (post hoc) | Latency / window |
|---|---|---|---|---|---|
{rows}

## Reading

- `esp32_dscnn/2026-10-18_run01` passed the pre-registered C5, but at the data's real rate it does not keep up with non-overlapping windows.
- `esp32_tiny/2026-10-19_run01` keeps up at either rate.
- `rpi5_tcn/2026-10-20_run01` failed C5 already (AI-N24); at 10 kHz it takes about 20× its window.

## Not executed

- Re-measuring on the boards: **no ejecutado / not executed**; it belongs to B4. The latency does not depend on the assumed rate, only the budget does.
""")
print(OUT.with_suffix(".md").read_text())
