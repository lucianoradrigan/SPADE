"""Extract the physical and controller parameters of DPC4PowerElectronics' Simulink models
without MATLAB -- TODO_CLAUDE_CODE.md V1.

A .slx file is a zip archive. This script reads:
- the model InitFcn callback (simulink/blockdiagram.xml), where the workspace variables live;
- every block's dialog parameters (simulink/systems/*.xml), with its "Commented" state;
- the MATLAB Function blocks' source (simulink/stateflow/chart_*.xml);
- the solver settings (simulink/configSet0.xml);
- Data4train.mat's load and reference columns.

It prints nothing it cannot point to: each value is reported with the XML file and block it came
from. The comparison against SPADE is in docs/audit/upstream/dpc4pe_parameters.md.

Usage (the upstream repository is MIT-licensed and is not committed here):

    git clone https://github.com/aipoweraau/DPC4PowerElectronics.git data/upstream/dpc4pe
    python experiments/extract_dpc4pe_slx_params.py --upstream data/upstream/dpc4pe \
        --out docs/audit/upstream/dpc4pe_parameters.json
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import subprocess
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

_BLOCK = re.compile(r'<Block BlockType="([^"]*)"[^>]*?Name="([^"]*)"[^>]*?SID="(\d+)"[^>]*>(.*?)</Block>', re.S)
_PARAM = re.compile(r'<P Name="([^"]+)"[^>]*>([^<]*)</P>')
_SKIP_PARAMS = {"Position", "ZOrder", "Ports", "SID", "LibraryVersion", "SourceBlock", "PortProperties", "BlockMirror", "NameLocation", "ShowName", "FontName", "BackgroundColor", "ContentPreviewEnabled", "AttributesFormatString", "SourceProductBaseCode", "LConnTagsString", "RConnTagsString"}
#: Block kinds whose dialog values carry physical or controller parameters.
_PARAM_SOURCE_TYPES = {"Three-Phase Series RLC Branch", "DC Voltage Source", "Three-Phase Breaker", "IGBT/Diode", "Repeating table", "On/Off Delay ", "Predict"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _params(body: str) -> dict:
    return {k: html.unescape(v) for k, v in _PARAM.findall(body) if k not in _SKIP_PARAMS and not k.startswith("RTW")}


def _init_assignments(init_fcn: str) -> dict:
    """``name=value;`` lines of the InitFcn that are not commented out."""
    out = {}
    for line in init_fcn.splitlines():
        m = re.match(r"^\s*([A-Za-z]\w*)\s*=\s*([^;%]+);", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def _commented_arrays(init_fcn: str) -> list[str]:
    return sorted(set(re.findall(r"^%\s*(\w+)\s*=\s*\[", init_fcn, re.M)))


def _system_paths(z: zipfile.ZipFile) -> dict:
    """system file -> (parent block path, parent commented) so a block's full path and whether any
    ancestor is commented out can be reported."""
    parent = {}
    for name in z.namelist():
        if not re.match(r"simulink/systems/system_\w+\.xml$", name):
            continue
        for bt, bname, _sid, body in _BLOCK.findall(z.read(name).decode()):
            ref = re.search(r'<System Ref="([^"]+)"', body)
            if ref:
                p = _params(body)
                parent[f"simulink/systems/{ref.group(1)}.xml"] = (name, html.unescape(bname).replace("\n", " ").strip(), p.get("Commented", "off") == "on")
    return parent


def _ancestry(system_file: str, parent: dict) -> tuple[str, bool]:
    names, commented, cur = [], False, system_file
    while cur in parent:
        cur, bname, c = parent[cur]
        names.append(bname)
        commented |= c
    return "/".join(reversed(names)), commented


def _mpc_summary(script: str) -> dict:
    """Facts read directly from the active MPC function's source."""

    def num(pattern):
        m = re.search(pattern, script, re.M)
        return m.group(1) if m else None

    adf = re.search(r"^Adf\s*=\s*\[([^\]]*)\]", script, re.M)
    bdf = re.search(r"^Bdf\s*=\s*\[([^\]]*)\]", script, re.M)
    return {
        "Adf": adf.group(1).strip() if adf else None,
        "Bdf": bdf.group(1).strip() if bdf else None,
        "prediction_horizon_N": num(r"^N\s*=\s*(\d+)\s*;"),
        "state_weight_Q": num(r"^Q\s*=\s*(diag\([^)]*\))"),
        "Qx": num(r"^Qx\s*=\s*([\d.eE+-]+)\s*;"),
        "C": num(r"^\s*C\s*=\s*([\d.eE+-]+)\s*;"),
        "wref": num(r"^\s*wref\s*=\s*([^;]+);"),
        "closed_form_solution": bool(re.search(r"=\s*\(-H\s*\\\s*f'\)\s*/\s*2", script)),
        "explicit_constraints": bool(re.search(r"quadprog|constraint\s*=\s*inf|lb\s*=|ub\s*=", script, re.I)),
        "one_step_delay_compensation": "iftd" in script and "a_old" in script,
        "reference_extrapolation": "3*v_ref_alpha - 3*v_ref_oldr" in script.replace(" ", " "),
        "targets_used": re.search(r"^c_targets\s*=\s*\{([^}]*)\}", script, re.M).group(1) if re.search(r"^c_targets\s*=\s*\{", script, re.M) else None,
    }


def extract_slx(path: Path) -> dict:
    z = zipfile.ZipFile(path)
    bd = z.read("simulink/blockdiagram.xml").decode()
    init = re.search(r'<P Name="InitFcn">(.*?)</P>', bd, re.S)
    init_fcn = html.unescape(init.group(1)) if init else ""
    cfg = z.read("simulink/configSet0.xml").decode()
    solver = {k: v for k, v in re.findall(r'<P Name="(SolverName|SolverType|FixedStep|StopTime)">([^<]*)</P>', cfg)}
    parent = _system_paths(z)

    blocks, functions = [], []
    for name in sorted(z.namelist()):
        if not re.match(r"simulink/systems/system_\w+\.xml$", name):
            continue
        path_prefix, ancestor_commented = _ancestry(name, parent)
        for bt, bname, sid, body in _BLOCK.findall(z.read(name).decode()):
            p = _params(body)
            st = p.get("SourceType", "")
            keep = bt in ("Saturate", "Step") or st in _PARAM_SOURCE_TYPES or (bt == "Constant" and len(p.get("Value", "")) < 40 and p.get("Value"))
            if not keep:
                continue
            blocks.append({
                "file": name, "sid": sid, "block_type": bt, "source_type": st or None,
                "path": f"{path_prefix}/{html.unescape(bname)}".replace("\n", " "),
                "active": not (ancestor_commented or p.get("Commented", "off") == "on"),
                "params": {k: v for k, v in p.items() if k not in ("SourceType", "Commented")},
            })

    for name in sorted(z.namelist()):
        if not re.match(r"simulink/stateflow/chart_\d+\.xml$", name):
            continue
        c = z.read(name).decode()
        fname = re.search(r'<P Name="name">([^<]*)', c)
        script = "\n".join(html.unescape(s) for s in re.findall(r'<P Name="script">(.*?)</P>', c, re.S))
        functions.append({"file": name, "path": html.unescape(fname.group(1)).replace("\n", " ").strip() if fname else None, "script_chars": len(script), "_script": script})

    # A MATLAB Function block is active when its SubSystem is; match by trailing path.
    active_paths = {b["path"]: b["active"] for b in blocks}
    sub_active = {}
    for name in z.namelist():
        if re.match(r"simulink/systems/system_\w+\.xml$", name):
            prefix, anc = _ancestry(name, parent)
            for bt, bname, _sid, body in _BLOCK.findall(z.read(name).decode()):
                if bt == "SubSystem":
                    p = _params(body)
                    sub_active[f"{prefix}/{html.unescape(bname)}".replace("\n", " ").strip()] = not (anc or p.get("Commented", "off") == "on")
    for f in functions:
        f["active"] = sub_active.get(f["path"], None)
    del active_paths

    return {
        "file": path.name, "sha256": _sha256(path),
        "init_fcn_assignments": _init_assignments(init_fcn),
        "init_fcn_commented_arrays": _commented_arrays(init_fcn),
        "solver": solver, "blocks": blocks, "matlab_functions": functions,
    }


def data4train_summary(path: Path) -> dict:
    from scipy.io import loadmat

    a = loadmat(path)["Data4train"]
    r = a[:, 6]
    loads, counts = np.unique(np.round(r, 4), return_counts=True)
    split = {}
    for name, sl in (("train_rows_1_6000", slice(0, 6000)), ("holdout_rows_6001_10000", slice(6000, 10000))):
        lr, lc = np.unique(np.round(r[sl], 4), return_counts=True)
        split[name] = {f"{v:g}": int(c) for v, c in zip(lr, lc)}
    vref = np.hypot(a[:, 4], a[:, 5])
    step = np.mod(np.angle(a[:, 7] + 1j * a[:, 8]) - np.angle(a[:, 4] + 1j * a[:, 5]), 2 * np.pi)
    return {
        "file": path.name, "sha256": _sha256(path), "shape": list(a.shape),
        "load_resistance_ohm_counts": {f"{v:g}": int(c) for v, c in zip(loads, counts)},
        "load_by_split": split,
        "v_ref_magnitude_v": [float(vref.min()), float(vref.max())],
        "reference_phase_step_rad_median": float(np.median(step)),
        "v_c_magnitude_v": [float(np.hypot(a[:, 2], a[:, 3]).min()), float(np.hypot(a[:, 2], a[:, 3]).max())],
        "i_f_magnitude_max_a": float(np.hypot(a[:, 0], a[:, 1]).max()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--upstream", type=Path, default=ROOT / "data" / "upstream" / "dpc4pe")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "audit" / "upstream" / "dpc4pe_parameters.json")
    args = ap.parse_args(argv)
    up = args.upstream.resolve()

    try:
        commit = subprocess.run(["git", "-C", str(up), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None

    models = {name: extract_slx(up / name) for name in ("CCS_MPC.slx", "DPCvalidation.slx")}
    active_mpc = [f for f in models["CCS_MPC.slx"]["matlab_functions"] if f["active"] and "Adf" in f["_script"] and "H \\ f'" in f["_script"].replace("H\\", "H \\")]
    result = {
        "upstream": "https://github.com/aipoweraau/DPC4PowerElectronics",
        "upstream_commit": commit,
        "text_files_sha256": {n: _sha256(up / n) for n in ("Main.txt", "Custom_Loss.txt", "Model_gradients.txt")},
        "models": models,
        "active_data_collection_mpc": [{"path": f["path"], "file": f["file"], **_mpc_summary(f["_script"])} for f in active_mpc],
        "data4train": data4train_summary(up / "Data4train.mat"),
    }
    for m in models.values():
        for f in m["matlab_functions"]:
            f.pop("_script")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}")
    for name, m in models.items():
        print(f"{name}: InitFcn {m['init_fcn_assignments'] or 'none'}; solver {m['solver']}")
    for mpc in result["active_data_collection_mpc"]:
        print("active MPC:", {k: v for k, v in mpc.items() if k not in ("Adf", "Bdf")})
    print("Data4train loads:", result["data4train"]["load_by_split"])


if __name__ == "__main__":
    main()
