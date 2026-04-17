from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.preprocessing import StandardScaler


class ClimateDataLoader:
    """Data loading and preprocessing utilities for climate anomaly detection."""

    def __init__(self, config) -> None:
        self.config = config
        self.scalers: dict[str, StandardScaler] = {}

    def _get_data_backend(self) -> str:
        backend = getattr(self.config, "DATA_BACKEND", "standard")
        if backend not in {"standard", "dask"}:
            raise ValueError("DATA_BACKEND must be either 'standard' or 'dask'.")
        return backend

    @staticmethod
    def _is_dask_backed(data_array: xr.DataArray) -> bool:
        module_name = type(data_array.data).__module__.lower()
        return "dask" in module_name

    @staticmethod
    def _open_dataset(file_path: Path, *, chunks: Optional[dict[str, int]] = None) -> xr.Dataset:
        engines = (None, "h5netcdf", "scipy")
        errors: list[str] = []

        for engine in engines:
            kwargs = {} if engine is None else {"engine": engine}
            if chunks is not None:
                kwargs["chunks"] = chunks
            engine_name = engine or "default"
            try:
                return xr.open_dataset(file_path, **kwargs)
            except Exception as exc:  # pragma: no cover
                errors.append(f"{engine_name}: {exc}")

        raise OSError(f"Unable to open {file_path.name}. Tried backends: {' | '.join(errors)}")

    def _open_mfdataset(self, nc_files: list[Path], *, chunks: dict[str, int]) -> xr.Dataset:
        engines = ("h5netcdf", None, "scipy")
        errors: list[str] = []

        for engine in engines:
            kwargs = {
                "combine": "by_coords",
                "parallel": getattr(self.config, "DASK_PARALLEL", False),
                "chunks": chunks,
                "data_vars": "minimal",
                "coords": "minimal",
                "compat": "override",
            }
            if engine is not None:
                kwargs["engine"] = engine
            engine_name = engine or "default"
            try:
                return xr.open_mfdataset(nc_files, **kwargs)
            except Exception as exc:  # pragma: no cover
                errors.append(f"{engine_name}: {exc}")

        raise OSError(f"Unable to load NetCDF files with Dask. Tried backends: {' | '.join(errors)}")

    def load_variable_dataset(self, variable_name: str) -> xr.Dataset:
        variable_path = self.config.get_variable_path(variable_name)
        nc_files = sorted(variable_path.glob("*.nc"))
        backend = self._get_data_backend()

        if not nc_files:
            raise FileNotFoundError(f"No NetCDF files found in {variable_path}")

        if self.config.MAX_FILES_TO_LOAD is not None:
            nc_files = nc_files[: self.config.MAX_FILES_TO_LOAD]

        if backend == "dask":
            chunks = dict(getattr(self.config, "DASK_CHUNKS", {"time": 365}))
            if len(nc_files) == 1:
                return self._open_dataset(nc_files[0], chunks=chunks).sortby("time")
            return self._open_mfdataset(nc_files, chunks=chunks).sortby("time")

        datasets: list[xr.Dataset] = []
        for file_path in nc_files:
            try:
                datasets.append(self._open_dataset(file_path))
            except OSError:
                continue

        if not datasets:
            raise OSError(f"Unable to load any NetCDF files for {variable_name}.")

        if len(datasets) == 1:
            return datasets[0].sortby("time")

        combined = xr.concat(datasets, dim="time", data_vars="minimal", coords="minimal", compat="override")
        return combined.sortby("time")

    def load_dataset(self, variable_name: str, allow_synthetic: bool = True) -> tuple[xr.Dataset, str]:
        try:
            return self.load_variable_dataset(variable_name), "raw"
        except Exception as exc:
            if not allow_synthetic:
                raise

            dataset = self.create_synthetic_dataset(variable_name)
            dataset.attrs["fallback_reason"] = str(exc)
            self.save_synthetic_snapshot(dataset, variable_name)
            return dataset, "synthetic"

    def create_synthetic_dataset(self, variable_name: str) -> xr.Dataset:
        rng = np.random.default_rng(self.config.RANDOM_STATE)
        dates = pd.date_range("2014-01-01", "2023-12-31", freq="D")
        t = np.arange(len(dates), dtype=np.float32)

        seasonal = 10.0 * np.sin(2 * np.pi * t / 365.25)
        trend = 0.015 * t / 365.25
        noise = rng.normal(0, 1.75, len(dates))
        values = 15 + seasonal + trend + noise

        anomaly_label = np.zeros(len(dates), dtype=np.int8)
        anomaly_indices = rng.choice(len(dates), size=max(24, len(dates) // 90), replace=False)
        anomaly_label[anomaly_indices] = 1
        values[anomaly_indices] += rng.normal(8.0, 2.0, size=len(anomaly_indices)) * rng.choice(
            np.array([-1.0, 1.0]),
            size=len(anomaly_indices),
        )

        return xr.Dataset(
            data_vars={
                variable_name: ("time", values.astype(np.float32)),
                "anomaly_label": ("time", anomaly_label),
            },
            coords={"time": dates},
            attrs={"source": "synthetic"},
        )

    def save_synthetic_snapshot(self, dataset: xr.Dataset, variable_name: str) -> Path:
        output_path = self.config.SYNTHETIC_DATA_DIR / f"{variable_name}_synthetic.csv"
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(dataset["time"].values),
                "value": dataset[variable_name].values,
                "anomaly_label": dataset["anomaly_label"].values,
            }
        )
        frame.to_csv(output_path, index=False)
        return output_path

    @staticmethod
    def extract_time_series(dataset: xr.Dataset, variable_name: str) -> pd.Series:
        if variable_name not in dataset.data_vars:
            variable_name = next(iter(dataset.data_vars))

        data_array = dataset[variable_name]
        spatial_dims = [dim for dim in data_array.dims if dim != "time"]
        if spatial_dims:
            data_array = data_array.mean(dim=spatial_dims, skipna=True)

        # Materialise lazily loaded arrays before converting to pandas.
        if ClimateDataLoader._is_dask_backed(data_array):
            data_array = data_array.compute()

        series = data_array.to_series()
        series.name = variable_name
        return series.sort_index()

    @staticmethod
    def extract_labels(dataset: xr.Dataset) -> Optional[pd.Series]:
        if "anomaly_label" not in dataset.data_vars:
            return None

        series = dataset["anomaly_label"].to_series().sort_index().astype(int)
        series.name = "anomaly_label"
        return series

    @staticmethod
    def handle_missing_values(series: pd.Series) -> pd.Series:
        if series.isna().sum() == 0:
            return series

        filled = series.interpolate(method="linear", limit_direction="both")
        if filled.isna().sum() > 0:
            filled = filled.ffill().bfill()
        return filled

    @staticmethod
    def create_sequences(values: np.ndarray, sequence_length: int) -> np.ndarray:
        if len(values) < sequence_length:
            raise ValueError(
                f"Not enough time steps to build sequences: got {len(values)}, need at least {sequence_length}."
            )

        return np.array(
            [values[index : index + sequence_length] for index in range(len(values) - sequence_length + 1)],
            dtype=np.float32,
        )

    @staticmethod
    def create_window_labels(labels: np.ndarray, sequence_length: int) -> np.ndarray:
        return np.array(
            [int(labels[index : index + sequence_length].max()) for index in range(len(labels) - sequence_length + 1)],
            dtype=np.int8,
        )

    def save_processed_series(
        self,
        series: pd.Series,
        variable_name: str,
        labels: Optional[pd.Series] = None,
    ) -> Path:
        output_path = self.config.PROCESSED_DATA_DIR / f"{variable_name}_series.csv"
        frame = pd.DataFrame({"date": pd.to_datetime(series.index), "value": series.values})
        if labels is not None:
            aligned_labels = labels.reindex(series.index).fillna(0).astype(int).values
            frame["anomaly_label"] = aligned_labels
        frame.to_csv(output_path, index=False)
        return output_path

    def prepare_lstm_data(
        self,
        series: pd.Series,
        labels: Optional[pd.Series] = None,
        sequence_length: Optional[int] = None,
    ) -> dict[str, np.ndarray | pd.Index | StandardScaler]:
        series = self.handle_missing_values(series).sort_index()
        raw_values = series.to_numpy(dtype=np.float32)
        window_length = sequence_length or self.config.SEQUENCE_LENGTH

        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(raw_values.reshape(-1, 1)).astype(np.float32).flatten()
        self.scalers[series.name or "series"] = scaler

        windows = self.create_sequences(scaled_values, window_length)
        dates = pd.Index(series.index[window_length - 1 :])
        end_values = raw_values[window_length - 1 :]

        window_labels = None
        if labels is not None:
            aligned_labels = labels.reindex(series.index).fillna(0).astype(int).to_numpy()
            window_labels = self.create_window_labels(aligned_labels, window_length)

        sample_count = len(windows)
        if sample_count < 8:
            raise ValueError("At least 8 sequences are required to create train/validation/test splits.")

        test_count = max(1, int(round(sample_count * self.config.TEST_SIZE)))
        train_val_count = sample_count - test_count
        validation_count = max(1, int(round(train_val_count * self.config.VALIDATION_SPLIT)))
        train_count = train_val_count - validation_count

        if train_count < 1:
            raise ValueError("Training split is empty after applying the configured test and validation ratios.")

        train_slice = slice(0, train_count)
        val_slice = slice(train_count, train_count + validation_count)
        test_slice = slice(train_count + validation_count, sample_count)

        X_train = windows[train_slice].reshape(-1, window_length, 1)
        X_val = windows[val_slice].reshape(-1, window_length, 1)
        X_test = windows[test_slice].reshape(-1, window_length, 1)

        if len(X_val) == 0:
            raise ValueError("Validation split produced zero samples.")

        output: dict[str, np.ndarray | pd.Index | StandardScaler] = {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": end_values[train_slice],
            "y_val": end_values[val_slice],
            "y_test": end_values[test_slice],
            "train_dates": dates[train_slice],
            "val_dates": dates[val_slice],
            "test_dates": dates[test_slice],
            "scaler": scaler,
            "sequence_length": window_length,
        }

        if window_labels is not None:
            output["train_labels"] = window_labels[train_slice]
            output["val_labels"] = window_labels[val_slice]
            output["test_labels"] = window_labels[test_slice]

        return output
