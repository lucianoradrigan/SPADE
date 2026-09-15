"""Model loader: retrieve pre-trained models from the registry for transfer learning.

Phase 1 scaffolding only -- every method below raises NotImplementedError; Phase 2 implements
them against the real ai/registry.py (resolve()/load_promoted_model()), not a parallel YAML
schema. See docs/design_ai_layer_transversal.md Sec. 7 for the registry's existing contract.
"""

from __future__ import annotations

from pathlib import Path


class ModelLoader:
    """Carga modelos ya promovidos desde el registry, para usarlos como punto de partida de un
    fine-tune (Phase 2). domain/block/tier siguen el mismo vocabulario que ai/registry.py."""

    def __init__(self, registry_path: str | Path, domain: str, block: str, tier: str = "pc"):
        self.registry_path = registry_path
        self.domain = domain
        self.block = block
        self.tier = tier

    def load_latest(self) -> dict:
        """Carga el modelo actualmente promovido para (domain, block, tier)."""
        raise NotImplementedError("Phase 2")

    def load_by_generation(self, generation: str) -> dict:
        """Carga una generación específica del linaje de transfer learning (no solo la última)."""
        raise NotImplementedError("Phase 2")

    def get_metadata(self) -> dict:
        """Metadatos del modelo (config + metrics) sin cargar los pesos."""
        raise NotImplementedError("Phase 2")
