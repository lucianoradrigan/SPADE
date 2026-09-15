"""Data merger: combine simulated and externally-provided datasets for a transfer-learning run.

Phase 1 scaffolding only -- every method below raises NotImplementedError. Phase 2 must respect
the same domain-isolation rule the rest of the AI layer already does (docs/design_ai_layer_transversal.md
Sec. 1: dc_motor and vsc_dpc data are never combined) -- merging here is simulated+external WITHIN
one domain, not across domains.
"""

from __future__ import annotations


class DataMerger:
    """Combina datos simulados (de Scenario/run_scenario) con datos externos subidos por el
    usuario, en la proporción `mix_ratio`, para un único dominio."""

    def __init__(self, simulated_data: dict | None = None, external_data: dict | None = None, mix_ratio: float = 0.7):
        self.simulated_data = simulated_data
        self.external_data = external_data
        self.mix_ratio = mix_ratio
        self.merged_data = None

    def validate_schema(self, domain: str) -> bool:
        """Valida que external_data tenga las columnas que ese dominio espera."""
        raise NotImplementedError("Phase 2")

    def merge(self) -> dict:
        """Combina simulated_data + external_data según mix_ratio."""
        raise NotImplementedError("Phase 2")

    def stratify_by_fault(self) -> dict:
        """Estratifica el dataset combinado por tipo de falla."""
        raise NotImplementedError("Phase 2")

    def get_statistics(self) -> dict:
        """Estadísticas descriptivas del dataset combinado (medias, conteos por clase, etc.)."""
        raise NotImplementedError("Phase 2")
