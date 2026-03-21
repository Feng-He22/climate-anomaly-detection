from __future__ import annotations

from pathlib import Path

import numpy as np


def _import_tensorflow():
    try:
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
        from tensorflow.keras.layers import Dense, Input, LSTM, RepeatVector, TimeDistributed
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required to train the LSTM autoencoder. "
            "Install dependencies from requirements.txt to enable this model."
        ) from exc

    return {
        "Adam": Adam,
        "Dense": Dense,
        "EarlyStopping": EarlyStopping,
        "Input": Input,
        "LSTM": LSTM,
        "Model": Model,
        "ModelCheckpoint": ModelCheckpoint,
        "ReduceLROnPlateau": ReduceLROnPlateau,
        "RepeatVector": RepeatVector,
        "TimeDistributed": TimeDistributed,
    }


class LSTMAutoencoder:
    """Sequence reconstruction model used for unsupervised anomaly detection."""

    def __init__(self, config) -> None:
        self.config = config
        self.model = None
        self.history = None
        self.output_prefix = "default"

    def build_model(self, input_shape: tuple[int, int]):
        layers = _import_tensorflow()
        encoder_units = self.config.LSTM_UNITS[: len(self.config.LSTM_UNITS) // 2]
        decoder_units = self.config.LSTM_UNITS[len(self.config.LSTM_UNITS) // 2 :]

        inputs = layers["Input"](shape=input_shape)
        x = inputs

        for index, units in enumerate(encoder_units):
            x = layers["LSTM"](
                units,
                activation="tanh",
                dropout=self.config.DROPOUT_RATE,
                recurrent_dropout=self.config.RECURRENT_DROPOUT,
                return_sequences=index < len(encoder_units) - 1,
            )(x)

        x = layers["RepeatVector"](input_shape[0])(x)

        for units in decoder_units:
            x = layers["LSTM"](
                units,
                activation="tanh",
                dropout=self.config.DROPOUT_RATE,
                recurrent_dropout=self.config.RECURRENT_DROPOUT,
                return_sequences=True,
            )(x)

        outputs = layers["TimeDistributed"](layers["Dense"](input_shape[1]))(x)
        self.model = layers["Model"](inputs, outputs, name="lstm_autoencoder")
        self.model.compile(
            optimizer=layers["Adam"](learning_rate=self.config.LEARNING_RATE),
            loss=self.config.LOSS_FUNCTION,
            metrics=["mae"],
        )
        return self.model

    def train(self, X_train: np.ndarray, X_val: np.ndarray):
        if self.model is None:
            self.build_model((X_train.shape[1], X_train.shape[2]))

        layers = _import_tensorflow()
        callbacks = [
            layers["EarlyStopping"](
                monitor="val_loss",
                patience=self.config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1,
            ),
            layers["ModelCheckpoint"](
                filepath=str(self.config.get_output_path("models", f"{self.output_prefix}_best_lstm_autoencoder.h5")),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
            layers["ReduceLROnPlateau"](
                monitor="val_loss",
                factor=0.5,
                patience=max(2, self.config.EARLY_STOPPING_PATIENCE // 3),
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        self.history = self.model.fit(
            X_train,
            X_train,
            validation_data=(X_val, X_val),
            epochs=self.config.EPOCHS,
            batch_size=self.config.BATCH_SIZE,
            callbacks=callbacks,
            shuffle=False,
            verbose=1,
        )
        return self.history

    def reconstruction_errors(self, X: np.ndarray) -> dict[str, np.ndarray]:
        if self.model is None:
            raise RuntimeError("LSTM autoencoder has not been built or trained yet.")

        predictions = self.model.predict(X, verbose=0)
        mse = np.mean(np.square(predictions - X), axis=(1, 2))
        return {"mse": mse, "predictions": predictions}

    def detect_anomalies(self, X: np.ndarray, threshold_percentile: int = 95) -> dict[str, np.ndarray | float]:
        errors = self.reconstruction_errors(X)
        mse = errors["mse"]
        threshold = float(np.percentile(mse, threshold_percentile))
        anomalies = mse > threshold

        error_range = mse.max() - mse.min()
        if error_range > 0:
            anomaly_scores = (mse - mse.min()) / error_range
        else:
            anomaly_scores = np.zeros_like(mse)

        return {
            "anomalies": anomalies,
            "anomaly_scores": anomaly_scores,
            "reconstruction_error": mse,
            "threshold": threshold,
        }

    def save(self, output_path: Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save an untrained LSTM autoencoder.")
        self.model.save(output_path)
