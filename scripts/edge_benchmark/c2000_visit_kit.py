"""B4 C2000 visit (D-B4-4 option B): exports a run to a fresh C kit in --out, never into the run's own c2000/
directory, so the kits measured in October stay exactly as they were. The new manifest carries fs_hz and
window_ms (AI-N26), so scripts/edge_benchmark/c2000_collect.py takes C5's budget at the data's 10 kHz.

    python scripts/edge_benchmark/c2000_visit_kit.py --domain vsc_dpc --block regressor \
        --run-dir configs/regressors/esp32_tiny/2026-10-19_run01 --out docs/audit/b4/c2000_kit/esp32_tiny
    python scripts/edge_benchmark/c2000_visit_kit.py --check-only --out docs/audit/b4/c2000_kit/esp32_tiny

It also builds the host self-test with gcc (on this PC, not the board) and, if the TI compiler is installed,
the board image with `make` (C1). Nothing here touches the board: flashing and timing are c2000_collect.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

EXPECTED_FS_HZ = 10000.0


def preflight(kit: Path) -> dict:
    m = json.loads((kit / "manifest.json").read_text())
    fs = m.get("fs_hz")
    out = {"kit": str(kit), "source_run": m.get("source_run"), "model_id": m.get("model_id"), "input_samples": m["input_shape"][0],
           "fs_hz": fs, "window_ms": m.get("window_ms"), "memory": m.get("memory"),
           "fs_ok": fs == EXPECTED_FS_HZ, "c5_budget_ms": (m["input_shape"][0] / fs * 1e3) if fs else None}
    if not out["fs_ok"]:
        out["error"] = f"manifest fs_hz is {fs}, expected {EXPECTED_FS_HZ} (AI-N26)"
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain")
    ap.add_argument("--block", choices=["classifier", "regressor"])
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reference-windows", type=int, default=6)
    ap.add_argument("--check-only", action="store_true", help="only the manifest preflight of an existing kit")
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args(argv)
    if not args.check_only:
        from driveflow.ai.c2000_export import export_c2000, host_parity, plan_model, simulated_raw_windows
        from driveflow.ai.registry import load_run_model

        if args.out.resolve() == (args.run_dir / "c2000").resolve():
            raise SystemExit("refusing to overwrite the run's measured c2000/ kit")
        model, _, metrics = load_run_model(args.run_dir, args.block)
        plan = plan_model(model)
        raw = simulated_raw_windows(args.domain, metrics["channels"], plan.input_shape[0], args.reference_windows)
        export_c2000(model, metrics, raw, args.out, source_run=f"{args.run_dir.parent.name}/{args.run_dir.name}")
        print("host parity (gcc, this PC):", host_parity(args.out))
        if not args.no_build:
            r = subprocess.run(["make", "-C", str(args.out)], capture_output=True, text=True)
            print("C1 build (cl2000):", "exit 0" if r.returncode == 0 else f"exit {r.returncode}\n{r.stderr[-1500:]}")
    print(json.dumps(preflight(args.out), indent=2))


if __name__ == "__main__":
    main()
