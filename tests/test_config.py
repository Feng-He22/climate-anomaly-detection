from config.config import AppConfig


def test_ensure_directories_creates_project_layout(tmp_path):
    config = AppConfig(project_root=tmp_path)
    config.ensure_directories()

    assert config.DATA_DIR.exists()
    assert config.RAW_DATA_DIR.exists()
    assert config.PROCESSED_DATA_DIR.exists()
    assert config.SYNTHETIC_DATA_DIR.exists()
    assert config.FIGURES_DIR.exists()
    assert config.METRICS_DIR.exists()
    assert config.MODELS_DIR.exists()


def test_resolve_data_root_prefers_standard_directory(tmp_path):
    config = AppConfig(project_root=tmp_path)
    standard_root = config.RAW_DATA_DIR / config.DATASET_DIRNAME / "tasmax"
    standard_root.mkdir(parents=True)

    assert config.resolve_data_root() == config.RAW_DATA_DIR / config.DATASET_DIRNAME
