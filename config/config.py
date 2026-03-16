from __future__ import annotations

from pathlib import Path
from typing import Optional


class AppConfig:
    """Application configuration for the climate anomaly detection project."""

    def __init__(self, project_root: Optional[Path] = None) -> None:
        self.PROJECT_ROOT = Path(project_root) if project_root else Path(__file__).resolve().parents[1]

        self.DATA_DIR = self.PROJECT_ROOT / "data"
        self.RAW_DATA_DIR = self.DATA_DIR / "raw"
        self.PROCESSED_DATA_DIR = self.DATA_DIR / "processed"
        self.SYNTHETIC_DATA_DIR = self.DATA_DIR / "synthetic"

        self.RESULTS_DIR = self.PROJECT_ROOT / "results"
        self.FIGURES_DIR = self.RESULTS_DIR / "figures"
        self.METRICS_DIR = self.RESULTS_DIR / "metrics"
        self.MODELS_DIR = self.RESULTS_DIR / "models"

        self.NOTEBOOKS_DIR = self.PROJECT_ROOT / "notebooks"
        self.DATASET_DIRNAME = "hadukgrid_60km_last10y"
        self.SUPPORTED_VARIABLES = ("rainfall", "tasmax", "tasmin")
        self.DEFAULT_VARIABLE = "tasmax"

        self.SEQUENCE_LENGTH = 30
        self.TEST_SIZE = 0.2
        self.VALIDATION_SPLIT = 0.2
        self.RANDOM_STATE = 42
        self.MAX_FILES_TO_LOAD = None

        self.LSTM_UNITS = [128, 64, 64, 128]
        self.LEARNING_RATE = 0.001
        self.BATCH_SIZE = 32
        self.EPOCHS = 100
        self.EARLY_STOPPING_PATIENCE = 10
        self.DROPOUT_RATE = 0.2
        self.RECURRENT_DROPOUT = 0.1
        self.OPTIMIZER = "adam"
        self.LOSS_FUNCTION = "mse"
        self.ANOMALY_THRESHOLD_PERCENTILE = 95

        self.N_ESTIMATORS = 100
        self.CONTAMINATION = 0.1
        self.MAX_SAMPLES = 256
        self.MAX_FEATURES = 1.0
        self.BOOTSTRAP = False
        self.N_JOBS = -1

    def ensure_directories(self) -> None:
        for directory in (
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.SYNTHETIC_DATA_DIR,
            self.RESULTS_DIR,
            self.FIGURES_DIR,
            self.METRICS_DIR,
            self.MODELS_DIR,
            self.NOTEBOOKS_DIR,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def resolve_data_root(self) -> Path:
        standard_root = self.RAW_DATA_DIR / self.DATASET_DIRNAME
        if standard_root.exists():
            return standard_root
        return self.RAW_DATA_DIR

    def get_variable_path(self, variable_name: str) -> Path:
        if variable_name not in self.SUPPORTED_VARIABLES:
            raise ValueError(f"Unsupported variable '{variable_name}'. Expected one of {self.SUPPORTED_VARIABLES}.")
        return self.resolve_data_root() / variable_name

    def get_output_path(self, category: str, filename: str) -> Path:
        category_map = {
            "figures": self.FIGURES_DIR,
            "metrics": self.METRICS_DIR,
            "models": self.MODELS_DIR,
        }
        if category not in category_map:
            raise ValueError(f"Unknown output category: {category}")
        return category_map[category] / filename
