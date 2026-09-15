"""Transfer Learning Pipeline: orchestrates fine-tuning a model already loaded via ModelLoader.

Phase 2: FEATURE_EXTRACTOR freezes the base model's first N layers (freeze_backbone) then runs a
normal model.fit(). DISCRIMINATIVE_LR runs a manual per-layer-group training loop -- Keras's
built-in fit() has no per-layer learning rate, so this uses one tf.keras.optimizers.Adam per
layer group, applied inside a plain tf.GradientTape loop. ADAPTER freezes everything except the
last `adapter_layers` trainable layers (default 1 -- the output head) -- a deliberately
simplified, generically-architecture-safe reading of "parameter-efficient fine-tuning": not a
LoRA low-rank-matrix implementation, which would need bespoke adapter modules per layer type
(Conv1D/LSTM/Dense) this project doesn't have.

train_data/val_data: {"X": np.ndarray, "y": np.ndarray} -- already-windowed, the same shape
models.common.windowing.build_classification_windows/build_direct_forecast_windows produce.
Windowing itself is the caller's job (DataMerger only merges rows; a real training script would
window the merged DataFrame the same way experiments/train_model.py already does), not this
pipeline's.
"""

from __future__ import annotations

from enum import Enum

import keras
import numpy as np
import tensorflow as tf


class FineTuneStrategy(Enum):
    FEATURE_EXTRACTOR = "feature_extractor"
    DISCRIMINATIVE_LR = "discriminative_lr"
    ADAPTER = "adapter"


def _is_classifier_target(y: np.ndarray) -> bool:
    """Sparse integer class labels (classifier) vs. continuous targets (regressor) -- same
    duality experiments/train_model.py's own kind detection resolves, inferred here from the
    target array itself since this pipeline isn't handed a ClassifierConfig/ForecasterConfig."""
    return np.issubdtype(np.asarray(y).dtype, np.integer)


class TransferLearningPipeline:
    """Orquesta el fine-tune de un modelo ya cargado. Cargar el modelo base
    (transfer.loader.ModelLoader) y fusionar los datos (transfer.data_merger.DataMerger) son
    pasos previos y separados -- este pipeline solo entrena y guarda."""

    def __init__(self, base_model_path: str, strategy: FineTuneStrategy = FineTuneStrategy.FEATURE_EXTRACTOR, train_config: dict | None = None):
        self.base_model_path = base_model_path
        self.strategy = strategy
        self.train_config = train_config or {}
        self.model: keras.Model | None = None
        self.history: dict | None = None
        self._discriminative_lr: dict | None = None

    @classmethod
    def from_loaded_model(cls, model: keras.Model, strategy: FineTuneStrategy = FineTuneStrategy.FEATURE_EXTRACTOR, train_config: dict | None = None, base_model_path: str = "") -> "TransferLearningPipeline":
        """Normal entry point: wrap a keras.Model a ModelLoader.load_*() call already built and
        loaded weights into. base_model_path is only used as save_checkpoint()'s implicit
        source-of-truth label, not re-loaded here."""
        pipeline = cls(base_model_path=base_model_path, strategy=strategy, train_config=train_config)
        pipeline.model = model
        return pipeline

    def load_base_model(self) -> None:
        """Carga los pesos guardados en base_model_path sobre self.model. self.model debe tener
        ya la arquitectura correcta (ver from_loaded_model, el camino normal para construir este
        pipeline) -- este método es para el caso de reusar el mismo pipeline con pesos distintos
        de la MISMA arquitectura."""
        if self.model is None:
            raise ValueError("self.model must be set (e.g. via from_loaded_model()) before load_base_model()")
        self.model.load_weights(self.base_model_path)

    def freeze_backbone(self, n_layers: int) -> None:
        """Estrategia FEATURE_EXTRACTOR: congela las primeras n_layers capas CON PESOS de
        self.model (Input/Dropout/BatchNorm sin gamma/beta entrenable no cuentan aparte de sus
        propios pesos). El resto queda entrenable."""
        if self.model is None:
            raise ValueError("no model loaded -- call from_loaded_model() first")
        weighted_layers = [layer for layer in self.model.layers if layer.weights]
        for layer in weighted_layers[:n_layers]:
            layer.trainable = False

    def apply_discriminative_lr(self, base_lr: float, layer_multipliers: dict) -> None:
        """Estrategia DISCRIMINATIVE_LR: guarda, por nombre de capa, el multiplicador de
        learning_rate a aplicar sobre base_lr. Capas sin entrada en layer_multipliers usan
        multiplicador 1.0. Consumido por train() -- Keras no soporta learning rate por capa en
        un fit() normal, así que train() corre un loop manual cuando self.strategy es
        DISCRIMINATIVE_LR."""
        if self.model is None:
            raise ValueError("no model loaded -- call from_loaded_model() first")
        self._discriminative_lr = {"base_lr": base_lr, "layer_multipliers": dict(layer_multipliers)}

    def train(self, train_data: dict, val_data: dict, epochs: int = 50, batch_size: int = 32) -> dict:
        if self.model is None:
            raise ValueError("no model loaded -- call from_loaded_model() first")

        X_train, y_train = np.asarray(train_data["X"]), np.asarray(train_data["y"])
        X_val = np.asarray(val_data["X"]) if val_data.get("X") is not None else None
        y_val = np.asarray(val_data["y"]) if val_data.get("y") is not None else None
        is_clf = _is_classifier_target(y_train)

        if self.strategy is FineTuneStrategy.ADAPTER:
            weighted_layers = [layer for layer in self.model.layers if layer.weights]
            n_adapter = self.train_config.get("adapter_layers", 1)
            for layer in weighted_layers[:-n_adapter] if n_adapter > 0 else weighted_layers:
                layer.trainable = False
            for layer in weighted_layers[-n_adapter:]:
                layer.trainable = True

        if self.strategy is FineTuneStrategy.DISCRIMINATIVE_LR:
            history = self._train_discriminative_lr(X_train, y_train, X_val, y_val, epochs, batch_size, is_clf)
        else:
            lr = self.train_config.get("learning_rate", 1e-3)
            loss = "sparse_categorical_crossentropy" if is_clf else "mse"
            metrics = ["accuracy"] if is_clf else None
            self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=loss, metrics=metrics)
            validation_data = (X_val, y_val) if X_val is not None and len(X_val) else None
            fit_history = self.model.fit(X_train, y_train, validation_data=validation_data, epochs=epochs, batch_size=batch_size, verbose=0)
            history = {k: [float(v) for v in vals] for k, vals in fit_history.history.items()}

        self.history = history
        return history

    def _train_discriminative_lr(self, X_train, y_train, X_val, y_val, epochs: int, batch_size: int, is_clf: bool) -> dict:
        if self._discriminative_lr is None:
            raise ValueError("call apply_discriminative_lr() before train() with strategy=DISCRIMINATIVE_LR")
        base_lr = self._discriminative_lr["base_lr"]
        multipliers = self._discriminative_lr["layer_multipliers"]
        loss_fn = keras.losses.SparseCategoricalCrossentropy() if is_clf else keras.losses.MeanSquaredError()

        groups = []
        for layer in self.model.layers:
            if not layer.trainable or not layer.weights:
                continue
            lr = base_lr * multipliers.get(layer.name, 1.0)
            groups.append((keras.optimizers.Adam(learning_rate=lr), layer.trainable_weights))
        if not groups:
            raise ValueError("no trainable layers -- freeze_backbone() left nothing to train")

        n = len(X_train)
        history = {"loss": [], "val_loss": []}
        for _ in range(epochs):
            epoch_losses = []
            for start in range(0, n, batch_size):
                xb = X_train[start : start + batch_size]
                yb = y_train[start : start + batch_size]
                with tf.GradientTape() as tape:
                    preds = self.model(xb, training=True)
                    loss = loss_fn(yb, preds)
                all_vars = [v for _, vs in groups for v in vs]
                grads = tape.gradient(loss, all_vars)
                offset = 0
                for opt, vs in groups:
                    opt.apply_gradients(zip(grads[offset : offset + len(vs)], vs))
                    offset += len(vs)
                epoch_losses.append(float(loss))
            history["loss"].append(float(np.mean(epoch_losses)))
            if X_val is not None and len(X_val):
                val_preds = self.model(X_val, training=False)
                history["val_loss"].append(float(loss_fn(y_val, val_preds)))
        return history

    def save_checkpoint(self, path: str) -> None:
        if self.model is None:
            raise ValueError("no model loaded -- call from_loaded_model() first")
        self.model.save_weights(path)
