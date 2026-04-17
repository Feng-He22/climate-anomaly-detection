import xarray as xr

from config.config import AppConfig
from src.data_loader import ClimateDataLoader


def test_synthetic_dataset_contains_signal_and_labels(tmp_path):
    config = AppConfig(project_root=tmp_path)
    loader = ClimateDataLoader(config)

    dataset = loader.create_synthetic_dataset("tasmax")

    assert "tasmax" in dataset.data_vars
    assert "anomaly_label" in dataset.data_vars
    assert int(dataset["anomaly_label"].sum()) > 0


def test_prepare_lstm_data_returns_expected_shapes(tmp_path):
    config = AppConfig(project_root=tmp_path)
    loader = ClimateDataLoader(config)
    dataset = loader.create_synthetic_dataset("tasmax")
    series = loader.extract_time_series(dataset, "tasmax")
    labels = loader.extract_labels(dataset)

    prepared = loader.prepare_lstm_data(series, labels=labels)

    assert prepared["X_train"].shape[1:] == (config.SEQUENCE_LENGTH, 1)
    assert prepared["X_val"].shape[1:] == (config.SEQUENCE_LENGTH, 1)
    assert prepared["X_test"].shape[1:] == (config.SEQUENCE_LENGTH, 1)
    assert len(prepared["test_dates"]) == prepared["X_test"].shape[0]
    assert prepared["test_labels"].shape[0] == prepared["X_test"].shape[0]


def test_prepare_lstm_data_respects_custom_sequence_length(tmp_path):
    config = AppConfig(project_root=tmp_path)
    loader = ClimateDataLoader(config)
    dataset = loader.create_synthetic_dataset("tasmax")
    series = loader.extract_time_series(dataset, "tasmax")
    labels = loader.extract_labels(dataset)

    prepared = loader.prepare_lstm_data(series, labels=labels, sequence_length=14)

    assert prepared["sequence_length"] == 14
    assert prepared["X_train"].shape[1:] == (14, 1)
    assert prepared["X_test"].shape[1:] == (14, 1)


def test_dask_backend_loads_multi_file_dataset(tmp_path):
    config = AppConfig(project_root=tmp_path)
    variable_dir = config.RAW_DATA_DIR / config.DATASET_DIRNAME / "tasmax"
    variable_dir.mkdir(parents=True)

    ds1 = xr.Dataset({"tasmax": (("time",), [1.0, 2.0])}, coords={"time": ["2023-01-01", "2023-01-02"]})
    ds2 = xr.Dataset({"tasmax": (("time",), [3.0, 4.0])}, coords={"time": ["2023-01-03", "2023-01-04"]})
    ds1.to_netcdf(variable_dir / "tasmax_part1.nc")
    ds2.to_netcdf(variable_dir / "tasmax_part2.nc")

    config.DATA_BACKEND = "dask"
    loader = ClimateDataLoader(config)
    dataset = loader.load_variable_dataset("tasmax")
    series = loader.extract_time_series(dataset, "tasmax")

    assert len(series) == 4
    assert float(series.iloc[-1]) == 4.0
