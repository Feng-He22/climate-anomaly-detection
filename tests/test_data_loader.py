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
