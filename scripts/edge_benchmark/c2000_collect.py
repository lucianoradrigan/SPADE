"""Flashes a C2000 kit onto the LAUNCHXL-F28379D and records what the board itself reports.
Standard library + pyserial only, like rpi5_benchmark.py. Needs the board on --port.

Writes into the kit directory:
- hardware_run.json: every window of the last complete report, plus the pre-registered verdicts of
  docs/audit/c2000/criteria.md;
- deployment_evidence.json: the deployment stages this run supports.

bench_verified means parity and timing measured on the board, replaying simulated windows from
flash. hardware_verified stays unset: no ADC, sensor or real signal is involved.

Usage:

    python3 scripts/edge_benchmark/c2000_collect.py configs/classifiers/esp32_dscnn/<run>/c2000
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import struct
import subprocess
import time
from pathlib import Path

import serial

PARITY_TOLERANCE = 1e-3  # C3
SAMPLE_RATE_HZ = 5000.0  # windowing.py: fs_hz = 1 / TAU
BEGIN = re.compile(r"SPADE-BEGIN model=(\w+) vectors=(\d+) sysclk=(\d+) arena_bytes=(\d+)")
LINE = re.compile(r"V (\d+) cycles=(\d+) err_bits=(\d+) top=(\d+) exp=(\d+)")


def read_report(port: str, timeout_s: float) -> tuple:
    """The first report that starts after the port opens, so a partial one is never parsed."""
    with serial.Serial(port, 115200, timeout=1) as tty:
        deadline = time.monotonic() + timeout_s
        header, windows = None, []
        while time.monotonic() < deadline:
            line = tty.readline().decode("ascii", errors="replace").strip()
            begin = BEGIN.search(line)
            if begin:
                header, windows = begin, []
            elif header is not None and (m := LINE.search(line)):
                windows.append([int(g) for g in m.groups()])
            elif header is not None and "SPADE-END" in line:
                return header, windows
    raise SystemExit(f"no complete report on {port} within {timeout_s} s")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("kit", type=Path)
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--no-flash", action="store_true", help="only read a board that already runs this kit")
    args = ap.parse_args(argv)

    manifest = json.loads((args.kit / "manifest.json").read_text())
    if not args.no_flash:
        subprocess.run(["make", "-C", str(args.kit), "flash"], check=True)
    header, windows = read_report(args.port, args.timeout)
    model_id, vectors, sysclk, arena = header.group(1), int(header.group(2)), int(header.group(3)), int(header.group(4))
    if model_id != manifest["model_id"]:
        raise SystemExit(f"the board runs model {model_id}, the kit is {manifest['model_id']}")
    if len(windows) != vectors:
        raise SystemExit(f"{len(windows)} windows reported of {vectors}")

    rows = [{"window": v, "cycles": c, "latency_ms": c / sysclk * 1e3, "max_abs_err": struct.unpack("<f", struct.pack("<I", bits))[0], "top": t - 1, "expected_top": e - 1} for v, c, bits, t, e in windows]
    worst_err = max(r["max_abs_err"] for r in rows)
    worst_ms = max(r["latency_ms"] for r in rows)
    budget_ms = manifest["input_shape"][0] / SAMPLE_RATE_HZ * 1e3
    is_classifier = bool(manifest["classes"])
    verdicts = {
        "C2_memory": manifest["memory"]["fits"] and arena == manifest["memory"]["arena_bytes"],
        "C3_parity": worst_err <= PARITY_TOLERANCE,
        "C4_top1": all(r["top"] == r["expected_top"] for r in rows) if is_classifier else "not_applicable",
        "C5_real_time": worst_ms < budget_ms,
    }
    when = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    run = {"measured_on": manifest["target"], "when": when, "source_run": manifest["source_run"], "model_id": model_id, "sysclk_hz": sysclk,
           "input": "simulated windows replayed from flash", "worst_max_abs_err": worst_err, "worst_latency_ms": worst_ms, "latency_budget_ms": budget_ms, "verdicts": verdicts, "windows": rows}
    (args.kit / "hardware_run.json").write_text(json.dumps(run, indent=2) + "\n")
    passed = all(v is True or v == "not_applicable" for v in verdicts.values())
    evidence = {"static_compatibility_passed": True, "bench_verified": passed, "board": manifest["target"], "runtime": "SPADE float32 C kernels, CGT 25.11.1.LTS",
                "latency_ms_worst": worst_ms, "arena_bytes": arena, "max_abs_err_vs_keras": worst_err, "when": when, "note": "Simulated windows replayed from flash; no ADC or real signal."}
    (args.kit / "deployment_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({k: v for k, v in run.items() if k != "windows"}, indent=2))


if __name__ == "__main__":
    main()
