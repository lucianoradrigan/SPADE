"""Data merger: combine simulated and externally-provided datasets for a transfer-learning run.

Operates on already-loaded pandas DataFrames -- the same shape experiments/train_model.py's
_load_domain_dataframe produces -- for ONE domain at a time. The domain-isolation rule the rest of
the AI layer enforces (docs/design_ai_layer_transversal.md Sec. 1) applies here too: this merges
simulated+external data WITHIN a domain, never across dc_motor/vsc_dpc -- validate_schema() is the
guard against accidentally handing it the wrong domain's external data.

Mixing unit (AUDIT AI-01, AUDIT_ANSWERS.md Q01): the merger moves WHOLE ACQUISITIONS (every row of
one `source_file`), never individual rows. The previous implementation drew `.sample(n=...)` rows
independently from each side, which destroyed the temporal order windowing relies on -- a window
built from those rows was a random bag of instants, not 0.5 s of signal. Rows inside an
acquisition are kept in their original order; groups whose timestamps are not strictly increasing
or whose sampling rate is wrong are EXCLUDED and reported, never sorted or resampled.

`mix_ratio` is the target fraction of the merged dataset's *units* that come from simulation,
where a unit is one non-overlapping training window when `window_samples` is given (the
recommended setting -- it is what the model actually trains on) or one row otherwise. Because
whole acquisitions are selected greedily, the realized ratio is close to, not exactly, the target;
it is reported in get_statistics()["realized_mix_ratio"].
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from driveflow.models.common.acquisitions import GROUP_COL, TIME_COL, check_acquisitions, window_starts
from driveflow.models.common.windowing import CANDIDATE_CHANNELS, VSC_DPC_CANDIDATE_CHANNELS

_CANDIDATE_CHANNELS_BY_DOMAIN = {"dc_motor": CANDIDATE_CHANNELS, "vsc_dpc": VSC_DPC_CANDIDATE_CHANNELS}
#: Origin prefixes making acquisition IDs unique across sources (Q05): a simulated "run_a" and an
#: external "run_a" are different recordings and must never be windowed or split as one group.
ORIGIN_PREFIX = {"simulated": "sim::", "external": "ext::"}


class InsufficientDataError(ValueError):
    """The requested mix cannot be built from the valid acquisitions available."""


def _with_origin(df: pd.DataFrame, origin: str, group_col: str) -> pd.DataFrame:
    prefix = ORIGIN_PREFIX[origin]
    out = df.copy()
    ids = out[group_col].astype(str)
    out[group_col] = ids.where(ids.str.startswith(prefix), prefix + ids)
    out["_data_source"] = origin
    return out


class DataMerger:
    """Combina un DataFrame simulado (de driveflow.datagen.run_scenario/export_parquet) con un
    DataFrame externo (ya validado por driveflow.ai.transfer.dataset_validation) en la proporción
    `mix_ratio` (fracción de unidades -- ventanas o filas -- que viene de simulated_data),
    seleccionando adquisiciones completas."""

    def __init__(
        self,
        simulated_data: pd.DataFrame | None = None,
        external_data: pd.DataFrame | None = None,
        mix_ratio: float = 0.7,
        window_samples: int | None = None,
        expected_fs_hz: float | None = None,
        group_col: str = GROUP_COL,
        time_col: str = TIME_COL,
    ):
        if not 0.0 <= mix_ratio <= 1.0:
            raise ValueError(f"mix_ratio must be in [0, 1], got {mix_ratio}")
        if window_samples is not None and window_samples <= 0:
            raise ValueError("window_samples must be positive")
        self.simulated_data = simulated_data
        self.external_data = external_data
        self.mix_ratio = mix_ratio
        self.window_samples = window_samples
        self.expected_fs_hz = expected_fs_hz
        self.group_col = group_col
        self.time_col = time_col
        self.merged_data: pd.DataFrame | None = None
        self.excluded_groups: list = []
        self._stats: dict = {}

    def validate_schema(self, domain: str) -> bool:
        """True si external_data tiene al menos uno de los canales candidatos de `domain` (mismo
        criterio que validate_dc_motor_upload/validate_vsc_dpc_forecast_upload -- "at least one
        known channel", no un esquema fijo de columnas)."""
        if domain not in _CANDIDATE_CHANNELS_BY_DOMAIN:
            raise ValueError(f"unknown domain {domain!r} -- expected one of {sorted(_CANDIDATE_CHANNELS_BY_DOMAIN)}")
        if self.external_data is None:
            return False
        candidates = _CANDIDATE_CHANNELS_BY_DOMAIN[domain]
        return any(c in self.external_data.columns for c in candidates)

    # ------------------------------------------------------------------ internals
    def _units(self, sub: pd.DataFrame) -> int:
        if self.window_samples is None:
            return len(sub)
        t = sub[self.time_col].to_numpy(dtype=float)
        return len(window_starts(len(sub), self.window_samples, t))

    def _valid_groups(self, df: pd.DataFrame, origin: str) -> list:
        """[(group_id, sub_df, n_units)] for every acquisition that passes the integrity checks,
        in first-appearance order; failures go to self.excluded_groups with their reason."""
        checks = check_acquisitions(
            df, self.group_col, self.time_col, expected_fs_hz=self.expected_fs_hz, min_rows=self.window_samples or 1
        )
        by_group = {c.group: c for c in checks}
        out = []
        for group, sub in df.groupby(self.group_col, sort=False):
            check = by_group[str(group)]
            if not check.ok:
                self.excluded_groups.append({"group": str(group), "origin": origin, "reason": check.reason})
                continue
            units = self._units(sub)
            if units == 0:
                self.excluded_groups.append({"group": str(group), "origin": origin, "reason": "no complete window inside one contiguous segment"})
                continue
            out.append((str(group), sub, units))
        return out

    @staticmethod
    def _take(groups: list, target_units: float, rng: np.random.Generator) -> list:
        """Whole groups in a seeded random order until target_units is reached (overshoot is at
        most one group). Always at least one group when target_units > 0."""
        order = rng.permutation(len(groups))
        chosen, total = [], 0
        for i in order:
            if total >= target_units and chosen:
                break
            chosen.append(groups[i])
            total += groups[i][2]
        return chosen

    # ------------------------------------------------------------------ public API
    def merge(self, seed: int = 42) -> pd.DataFrame:
        """Builds the merged dataset from whole acquisitions (see module docstring). If only one
        side is given, every valid acquisition of that side is used."""
        if self.simulated_data is None and self.external_data is None:
            raise ValueError("need at least one of simulated_data/external_data")
        self.excluded_groups = []
        rng = np.random.default_rng(seed)

        sim_groups = self._valid_groups(_with_origin(self.simulated_data, "simulated", self.group_col), "simulated") if self.simulated_data is not None else []
        ext_groups = self._valid_groups(_with_origin(self.external_data, "external", self.group_col), "external") if self.external_data is not None else []

        if self.external_data is None or self.mix_ratio == 1.0:
            chosen_sim, chosen_ext = sim_groups, []
        elif self.simulated_data is None or self.mix_ratio == 0.0:
            chosen_sim, chosen_ext = [], ext_groups
        else:
            s_units = sum(g[2] for g in sim_groups)
            e_units = sum(g[2] for g in ext_groups)
            if e_units == 0:
                raise InsufficientDataError(f"external dataset has no valid acquisition for mix_ratio={self.mix_ratio} (excluded: {self.excluded_groups})")
            if s_units == 0:
                raise InsufficientDataError("simulated dataset has no valid acquisition")
            # Largest total that both sides can supply at the requested proportion.
            total = min(s_units / self.mix_ratio, e_units / (1.0 - self.mix_ratio))
            chosen_sim = self._take(sim_groups, self.mix_ratio * total, rng)
            chosen_ext = self._take(ext_groups, (1.0 - self.mix_ratio) * total, rng)

        if not chosen_sim and not chosen_ext:
            raise InsufficientDataError(f"no valid acquisition left after integrity checks (excluded: {self.excluded_groups})")

        parts = [g[1] for g in chosen_sim + chosen_ext]
        self.merged_data = pd.concat(parts, ignore_index=True)
        units_sim, units_ext = sum(g[2] for g in chosen_sim), sum(g[2] for g in chosen_ext)
        self._stats = {
            "mix_unit": "windows" if self.window_samples is not None else "rows",
            "window_samples": self.window_samples,
            "target_mix_ratio": self.mix_ratio,
            "realized_mix_ratio": units_sim / (units_sim + units_ext) if units_sim + units_ext else None,
            "units": {"simulated": units_sim, "external": units_ext},
            "groups": {"simulated": [g[0] for g in chosen_sim], "external": [g[0] for g in chosen_ext]},
            "excluded_groups": list(self.excluded_groups),
        }
        return self.merged_data

    def stratify_by_fault(self, label_column: str = "label", seed: int = 0) -> pd.DataFrame:
        """Balances classes by keeping whole acquisitions per class until each class reaches the
        minority class's row count (overshoot <= one acquisition). Rows are never sub-sampled
        individually -- that would break temporal continuity exactly like the old merge() did.
        Requires merge() already run; not meaningful for vsc_dpc (regression, no class column)."""
        if self.merged_data is None:
            raise ValueError("call merge() first")
        if label_column not in self.merged_data.columns:
            raise ValueError(f"{label_column!r} not in merged_data -- nothing to stratify by")
        rng = np.random.default_rng(seed)
        df = self.merged_data
        group_key = self.group_col if self.group_col in df.columns else None
        per_class = {}
        for label, sub in df.groupby(label_column, sort=True):
            if group_key is None:
                per_class[label] = [(str(label), sub, len(sub))]
            else:
                per_class[label] = [(str(g), gsub, len(gsub)) for g, gsub in sub.groupby(group_key, sort=False)]
        n_min = min(sum(g[2] for g in groups) for groups in per_class.values())
        parts = []
        for label in per_class:
            parts.extend(g[1] for g in self._take(per_class[label], n_min, rng))
        return pd.concat(parts, ignore_index=True)

    def get_statistics(self) -> dict:
        """Estadísticas descriptivas de merged_data (requiere merge() ya corrido)."""
        if self.merged_data is None:
            raise ValueError("call merge() first")
        stats = {"n_rows": int(len(self.merged_data)), **self._stats}
        if "_data_source" in self.merged_data.columns:
            stats["by_source"] = {k: int(v) for k, v in self.merged_data["_data_source"].value_counts().items()}
        if "label" in self.merged_data.columns:
            stats["by_label"] = {k: int(v) for k, v in self.merged_data["label"].value_counts().items()}
        return stats
