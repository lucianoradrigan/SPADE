"""Compare a MATLAB-trained DPC network recovered from the upstream repository with SPADE's DPC
checkpoints -- TODO_CLAUDE_CODE.md V3.

CCS_MPC.slx's InitFcn holds, commented out, a complete 15 -> 16 -> 16 -> 10 ReLU network
(``weights_fc1..3``, ``bias_fc1..3``). That is the architecture Main.txt builds, so it is the only
MATLAB-trained weight set in the repository that can be loaded into SPADE's network
(control/dpc/network.py). Its provenance (which Main.txt run, how many epochs) is not stated
upstream.

The trained network file the upstream history did contain, trainedNet1nDPCv4_1.mat (deleted in
upstream commit b307cad), is a MAT v5 file with an opaque MCOS ``dlnetwork`` object. Its embedded
array dimensions (80x7, 80x80, 2x80) show a 7 -> 80 -> 80 -> 2 network, a different architecture
from Main.txt, so it cannot be compared with SPADE's 15-input network.

Reported, on Data4train.mat's holdout rows 6001-10000 (all at R = 8.0064 ohm):
- the DPC training loss (control/dpc/loss.py, the port of Custom_Loss.txt);
- the RMS difference between each SPADE checkpoint's outputs and the MATLAB network's;
and in SPADE's closed loop at R = 8.0064 ohm, |v_ref| = 50 V, 2000 steps (0.2 s):
- the settled tracking RMSE of v_c, as in experiments/measure_dpc_rmse_by_r_range.py.

Usage (after extract_dpc4pe_slx_params.py's clone step):

    python experiments/compare_matlab_dpc_network.py --upstream data/upstream/dpc4pe \
        --out docs/audit/upstream/matlab_network_comparison.json
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tensorflow as tf  # noqa: E402

from driveflow.control.dpc.loss import dpc_loss  # noqa: E402
from driveflow.control.dpc.network import N_INPUTS, build_dpc_network  # noqa: E402
from driveflow.control.dpc.train import load_training_data  # noqa: E402
from experiments.measure_dpc_rmse_by_r_range import CHECKPOINTS, settled_rmse  # noqa: E402

_NAMES = ("weights_fc1", "bias_fc1", "weights_fc2", "bias_fc2", "weights_fc3", "bias_fc3")


def initfcn_network(slx: Path) -> list[np.ndarray]:
    """Keras-ordered weights [W1, b1, W2, b2, W3, b3] from the commented-out InitFcn arrays.
    The first assignment of each name is used (a later ``% bias_fc3`` is a 2-element variant)."""
    bd = zipfile.ZipFile(slx).read("simulink/blockdiagram.xml").decode()
    init = html.unescape(re.search(r'<P Name="InitFcn">(.*?)</P>', bd, re.S).group(1))
    out = []
    for name in _NAMES:
        body = re.search(rf"^%\s*{name}\s*=\s*\[([^\]]*)\]", init, re.M).group(1)
        a = np.array([[float(x) for x in row.replace(",", " ").split()] for row in body.split(";") if row.strip()])
        out.append(a.T.astype(np.float32) if name.startswith("weights") else a.ravel().astype(np.float32))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--upstream", type=Path, default=ROOT / "data" / "upstream" / "dpc4pe")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "audit" / "upstream" / "matlab_network_comparison.json")
    args = ap.parse_args(argv)

    _, holdout = load_training_data(args.upstream / "Data4train.mat")
    x = tf.constant(holdout)

    matlab = build_dpc_network()
    matlab(tf.zeros((1, N_INPUTS)))
    matlab.set_weights(initfcn_network(args.upstream / "CCS_MPC.slx"))
    y_matlab = matlab(x).numpy()

    with tempfile.TemporaryDirectory() as tmp:
        matlab_path = Path(tmp) / "matlab_initfcn.weights.h5"
        matlab.save_weights(matlab_path)
        nets = {"MATLAB InitFcn network": matlab_path, **CHECKPOINTS}
        rows = {}
        for label, path in nets.items():
            m = build_dpc_network()
            m(tf.zeros((1, N_INPUTS)))
            m.load_weights(path)
            y = m(x).numpy()
            rows[label] = {
                "holdout_dpc_loss": float(dpc_loss(x, tf.constant(y))),
                "rms_output_difference_vs_matlab_v": float(np.sqrt(np.mean((y - y_matlab) ** 2))),
                "closed_loop_settled_rmse_v_at_8ohm_50v": settled_rmse(path, 8.0064),
            }
            print(f"{label:34s} " + "  ".join(f"{k}={v:.4g}" for k, v in rows[label].items()), flush=True)

    result = {
        "holdout_rows": "Data4train.mat rows 6001-10000 (R = 8.0064 ohm)",
        "closed_loop": "SPADE averaged plant, R = 8.0064 ohm, |v_ref| = 50 V, 2000 steps, RMSE after 50 warm-up steps",
        "matlab_network_source": "CCS_MPC.slx InitFcn, commented-out weights_fc1..3 / bias_fc1..3 (provenance not stated upstream)",
        "results": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
