from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class IsolationForestDetector:
    """Isolation Forest anomaly detector for windowed climate sequences."""

    def __init__(self, config) -> None:
        self.config = config
        self.model: IsolationForest | None = None
        self.scaler = StandardScaler()
        self.used_n_jobs = config.N_JOBS
        self.feature_mode = "flatten_only"

    @staticmethod
    def _flatten_windows(X: np.ndarray) -> np.ndarray:
        return X.reshape(X.shape[0], -1)

    @staticmethod
    def _rolling_features(X: np.ndarray) -> np.ndarray:
        flattened = X.reshape(X.shape[0], X.shape[1])
        means = flattened.mean(axis=1, keepdims=True)
        stds = flattened.std(axis=1, keepdims=True)
        mins = flattened.min(axis=1, keepdims=True)
        maxs = flattened.max(axis=1, keepdims=True)
        medians = np.median(flattened, axis=1, keepdims=True)
        first_values = flattened[:, :1]
        last_values = flattened[:, -1:]
        ranges = maxs - mins
        trends = last_values - first_values
        return np.concatenate(
            [means, stds, mins, maxs, medians, first_values, last_values, ranges, trends],
            axis=1,
        )

    @staticmethod
    def _seasonal_features(dates) -> np.ndarray:
        if dates is None:
            raise ValueError("Dates are required when using seasonal feature modes.")
        day_of_year = pd.to_datetime(dates).dayofyear.to_numpy(dtype=np.float32).reshape(-1, 1)
        radians = (2.0 * np.pi * day_of_year) / 365.25
        return np.concatenate([np.sin(radians), np.cos(radians)], axis=1)

    def build_features(self, X: np.ndarray, dates=None, feature_mode: str = "flatten_only") -> np.ndarray:
        flattened = self._flatten_windows(X)
        if feature_mode == "flatten_only":
            return flattened
        if feature_mode == "flatten_rolling_seasonal":
            rolling = self._rolling_features(X)
            seasonal = self._seasonal_features(dates)
            return np.concatenate([flattened, rolling, seasonal], axis=1)
        raise ValueError(f"Unsupported feature mode: {feature_mode}")

    def fit(self, X_train: np.ndarray, dates=None, feature_mode: str = "flatten_only") -> "IsolationForestDetector":
        self.feature_mode = feature_mode
        X_train_features = self.build_features(X_train, dates=dates, feature_mode=feature_mode)
        X_train_scaled = self.scaler.fit_transform(X_train_features)
        max_samples = min(self.config.MAX_SAMPLES, len(X_train_scaled))

        def build_model(n_jobs: int) -> IsolationForest:
            return IsolationForest(
                n_estimators=self.config.N_ESTIMATORS,
                contamination=self.config.CONTAMINATION,
                max_samples=max_samples,
                max_features=self.config.MAX_FEATURES,
                bootstrap=self.config.BOOTSTRAP,
                random_state=self.config.RANDOM_STATE,
                n_jobs=n_jobs,
            )

        self.used_n_jobs = self.config.N_JOBS
        self.model = build_model(self.used_n_jobs)
        try:
            self.model.fit(X_train_scaled)
        except PermissionError:
            self.used_n_jobs = 1
            self.model = build_model(self.used_n_jobs)
            self.model.fit(X_train_scaled)
        return self

    def detect(self, X: np.ndarray, dates=None) -> dict[str, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Isolation Forest model has not been fitted yet.")

        X_features = self.build_features(X, dates=dates, feature_mode=self.feature_mode)
        X_scaled = self.scaler.transform(X_features)
        predictions = self.model.predict(X_scaled)
        raw_scores = -self.model.decision_function(X_scaled)

        score_range = raw_scores.max() - raw_scores.min()
        if score_range > 0:
            anomaly_scores = (raw_scores - raw_scores.min()) / score_range
        else:
            anomaly_scores = np.zeros_like(raw_scores)

        return {
            "anomalies": predictions == -1,
            "anomaly_scores": anomaly_scores,
            "raw_scores": raw_scores,
        }

    def fit_and_detect(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        train_dates=None,
        test_dates=None,
        feature_mode: str = "flatten_only",
    ) -> dict[str, np.ndarray]:
        self.fit(X_train, dates=train_dates, feature_mode=feature_mode)
        return self.detect(X_test, dates=test_dates)

    def save(self, output_path: Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save an unfitted Isolation Forest model.")

        with output_path.open("wb") as handle:
            pickle.dump({"model": self.model, "scaler": self.scaler}, handle)
