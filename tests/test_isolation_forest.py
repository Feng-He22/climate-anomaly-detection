import numpy as np

from config.config import AppConfig
from models.isolation_forest import IsolationForestDetector


def test_isolation_forest_fit_and_detect_preserves_test_length(tmp_path):
    rng = np.random.default_rng(42)
    config = AppConfig(project_root=tmp_path)
    detector = IsolationForestDetector(config)

    X_train = rng.normal(size=(64, config.SEQUENCE_LENGTH, 1)).astype("float32")
    X_test = rng.normal(size=(24, config.SEQUENCE_LENGTH, 1)).astype("float32")

    results = detector.fit_and_detect(X_train, X_test)

    assert results["anomalies"].shape == (24,)
    assert results["anomaly_scores"].shape == (24,)
    assert np.logical_or(results["anomalies"], ~results["anomalies"]).all()
