"""Model loader: retrieve pre-trained models from the registry for transfer learning.

Phase 2: wraps driveflow.ai.registry's real API (resolve()/load_promoted_model()) -- no parallel
YAML schema, no "transfer_lineage" registry extension (docs/design_ai_layer_transversal.md Sec. 7
documents the registry's existing, single-current-run-per-key contract; adding lineage tracking
there is a bigger, separate decision than this loader needs). "generation" in load_by_generation()
is a run directory -- the same <config_dir>/<config_stem>/<date>_runNN/ layout
experiments/train_model.py already produces -- not a name tracked anywhere new: any past run,
promoted or not, can be loaded directly by path.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from driveflow.ai.registry import REPO_ROOT, RegistryError, load_promoted_model, resolve
from driveflow.models.classifiers.builder import build_classifier
from driveflow.models.classifiers.schemas import load_classifier_config
from driveflow.models.regressors.builder import build_forecaster
from driveflow.models.regressors.schemas import load_forecaster_config


class ModelLoader:
    """Carga modelos para usarlos como punto de partida de un fine-tune
    (transfer.pipeline.TransferLearningPipeline). domain/block/tier siguen el mismo vocabulario
    que ai/registry.py ("classifier"/"regressor" para block, no "classifier"/"forecaster" --
    distinto de experiments/train_model.py's propia nomenclatura interna, ver ese módulo)."""

    def __init__(self, registry_path: str | Path, domain: str, block: str, tier: str = "pc"):
        self.registry_path = Path(registry_path)
        self.domain = domain
        self.block = block
        self.tier = tier

    def load_latest(self) -> dict:
        """Carga el modelo actualmente promovido para (domain, block, tier). Levanta
        RegistryError si no hay nada promovido todavía."""
        model, config, metrics = load_promoted_model(self.domain, self.tier, self.block, registry_path=self.registry_path)
        run_dir = resolve(self.domain, self.tier, self.block, registry_path=self.registry_path)
        return {"model": model, "config": config, "metrics": metrics, "run_dir": run_dir}

    def load_by_generation(self, generation: str | Path) -> dict:
        """Carga un run específico por su directorio -- no necesariamente el promovido -- para
        encadenar un fine-tune sobre un fine-tune anterior sin tener que promoverlo primero."""
        run_dir = Path(generation)
        if not run_dir.is_absolute():
            run_dir = REPO_ROOT / run_dir
        return self._load_from_run_dir(run_dir)

    def get_metadata(self) -> dict:
        """Metadatos (config + metrics) del modelo actualmente promovido, sin construir ni
        cargar los pesos -- para mostrar en UI antes de decidir si vale la pena cargarlo entero."""
        run_dir = resolve(self.domain, self.tier, self.block, registry_path=self.registry_path)
        with (run_dir / "metrics.json").open() as f:
            metrics = json.load(f)
        with (run_dir / "config.yaml").open() as f:
            config = yaml.safe_load(f)
        return {"run_dir": run_dir, "config": config, "metrics": metrics}

    def _load_from_run_dir(self, run_dir: Path) -> dict:
        if not run_dir.exists():
            raise RegistryError(f"{run_dir}: no such run directory")
        with (run_dir / "metrics.json").open() as f:
            metrics = json.load(f)
        n_channels = len(metrics["channels"])
        if self.block == "classifier":
            config = load_classifier_config(run_dir / "config.yaml")
            model = build_classifier(config, n_channels=n_channels)
        elif self.block == "regressor":
            config = load_forecaster_config(run_dir / "config.yaml")
            model = build_forecaster(config, n_channels=n_channels)
        else:
            raise ValueError(f"block must be 'classifier' or 'regressor', got {self.block!r}")
        model.load_weights(run_dir / "model.weights.h5")
        return {"model": model, "config": config, "metrics": metrics, "run_dir": run_dir}
