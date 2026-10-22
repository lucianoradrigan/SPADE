"""One-off conversion of NASA's IGBT "Device Aging log.xls" to CSV (E11a, round 2026-10-22).

Why this is not a dependency: the legacy .xls format needs ``xlrd``, which SPADE does not use
anywhere else. Decision D3 (2026-10-22) forbids adding it to pyproject.toml, so this script runs
once in a throwaway environment and its output (a CSV under data/, not committed) is what the audit
reads:

    uv run --with xlrd --with pandas python scripts/oneoff/convert_igbt_aging_log.py \
        "<archive>/Data/Thermal Overstress Aging with Square Signal at gate and SMU data/Device Aging log.xls" \
        data/upstream/fund/igbt/device_aging_log

One CSV per sheet is written next to the given output stem (<stem>__<sheet>.csv).
"""

import sys
from pathlib import Path

import pandas as pd


def main(src: str, out_stem: str) -> None:
    sheets = pd.read_excel(src, sheet_name=None, header=None, engine="xlrd")
    out = Path(out_stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    for name, df in sheets.items():
        path = out.parent / f"{out.name}__{str(name).strip().replace(' ', '_')}.csv"
        df.to_csv(path, index=False, header=False)
        print(f"{path}: {df.shape[0]} rows x {df.shape[1]} columns")


if __name__ == "__main__":
    main(*sys.argv[1:3])
