"""Data merger: combine simulated and externally-provided datasets for a transfer-learning run.

Phase 2: operates on already-loaded pandas DataFrames -- the same shape
experiments/train_model.py's _load_domain_dataframe produces -- for ONE domain at a time. The
domain-isolation rule the rest of the AI layer enforces (docs/design_ai_layer_transversal.md
Sec. 1) applies here too: this merges simulated+external data WITHIN a domain, never across
dc_motor/vsc_dpc -- validate_schema() is the guard against accidentally handing it the wrong
domain's external data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from driveflow.models.common.windowing import CANDIDATE_CHANNELS, VSC_DPC_CANDIDATE_CHANNELS

_CANDIDATE_CHANNELS_BY_DOMAIN = {"dc_motor": CANDIDATE_CHANNELS, "vsc_dpc": VSC_DPC_CANDIDATE_CHANNELS}


class DataMerger:
    """Combina un DataFrame simulado (de driveflow.datagen.run_scenario/export_parquet) con un
    DataFrame externo (ej. ya validado por viz.dpc_upload_validation) en la proporción
    `mix_ratio` (fracción del total que viene de simulated_data)."""

    def __init__(self, simulated_data: pd.DataFrame | None = None, external_data: pd.DataFrame | None = None, mix_ratio: float = 0.7):
        if not 0.0 <= mix_ratio <= 1.0:
            raise ValueError(f"mix_ratio must be in [0, 1], got {mix_ratio}")
        self.simulated_data = simulated_data
        self.external_data = external_data
        self.mix_ratio = mix_ratio
        self.merged_data: pd.DataFrame | None = None

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

    def merge(self, seed: int = 42) -> pd.DataFrame:
        """Muestrea mix_ratio del total desde simulated_data y el resto desde external_data
        (acotado por lo que cada uno realmente tiene), y los concatena. Si falta uno de los dos,
        devuelve una copia del otro sin muestrear."""
        if self.simulated_data is None and self.external_data is None:
            raise ValueError("need at least one of simulated_data/external_data")
        if self.external_data is None:
            self.merged_data = self.simulated_data.copy()
            return self.merged_data
        if self.simulated_data is None:
            self.merged_data = self.external_data.copy()
            return self.merged_data

        rng = np.random.default_rng(seed)
        # Target size = len(simulated_data), not len(simulated_data) + len(external_data) -- the
        # latter was a real bug (found by testing, not by inspection): with mix_ratio=0.7 and 100
        # rows on each side, it produced 200 rows (140 sim + 60 ext) instead of the 100 rows
        # (70/30) mix_ratio=0.7 actually promises. The merged set keeps the simulated dataset's
        # own size, with mix_ratio of it swapped out for external rows.
        n_total = len(self.simulated_data)
        n_sim = min(int(round(n_total * self.mix_ratio)), len(self.simulated_data))
        n_ext = min(n_total - n_sim, len(self.external_data))

        sim_sample = self.simulated_data.sample(n=n_sim, random_state=int(rng.integers(2**31))) if n_sim > 0 else self.simulated_data.iloc[0:0]
        ext_sample = self.external_data.sample(n=n_ext, random_state=int(rng.integers(2**31))) if n_ext > 0 else self.external_data.iloc[0:0]

        sim_sample = sim_sample.assign(_data_source="simulated")
        ext_sample = ext_sample.assign(_data_source="external")
        self.merged_data = pd.concat([sim_sample, ext_sample], ignore_index=True)
        return self.merged_data

    def stratify_by_fault(self, label_column: str = "label") -> pd.DataFrame:
        """Sub-muestrea merged_data para que cada clase de `label_column` quede con el mismo
        número de filas (el de la clase minoritaria). Requiere merge() ya corrido; no aplica al
        dominio vsc_dpc (regresión, sin columna de clase discreta) -- llamarlo ahí es un error."""
        if self.merged_data is None:
            raise ValueError("call merge() first")
        if label_column not in self.merged_data.columns:
            raise ValueError(f"{label_column!r} not in merged_data -- nothing to stratify by")
        n_min = int(self.merged_data[label_column].value_counts().min())
        parts = [group.sample(n=n_min, random_state=0) for _, group in self.merged_data.groupby(label_column)]
        return pd.concat(parts, ignore_index=True)

    def get_statistics(self) -> dict:
        """Estadísticas descriptivas de merged_data (requiere merge() ya corrido)."""
        if self.merged_data is None:
            raise ValueError("call merge() first")
        stats = {"n_rows": int(len(self.merged_data))}
        if "_data_source" in self.merged_data.columns:
            stats["by_source"] = {k: int(v) for k, v in self.merged_data["_data_source"].value_counts().items()}
        if "label" in self.merged_data.columns:
            stats["by_label"] = {k: int(v) for k, v in self.merged_data["label"].value_counts().items()}
        return stats
