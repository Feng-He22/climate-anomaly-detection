import pandas as pd

from config.config import AppConfig
from src.anomaly_detector import ClimateAnomalyDetectionSystem


def test_event_alignment_reports_hit_rate_and_overlap(tmp_path):
    config = AppConfig(project_root=tmp_path)
    system = ClimateAnomalyDetectionSystem(config)

    results = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=10, freq="D"),
            "value": range(10),
            "lstm_anomaly": [False, False, True, False, False, False, False, False, False, False],
            "if_anomaly": [False, True, True, False, False, False, False, True, False, False],
        }
    )
    results_path = config.get_output_path("metrics", "tasmax_anomaly_results.csv")
    results.to_csv(results_path, index=False)

    events = pd.DataFrame(
        {
            "event_id": ["event_a", "event_b"],
            "event_name": ["Event A", "Event B"],
            "start_date": ["2023-01-02", "2023-01-08"],
            "end_date": ["2023-01-03", "2023-01-09"],
            "window_basis": ["meteorological_event", "operational_alert_proxy"],
            "source_org": ["Source", "Source"],
            "source_title": ["Title A", "Title B"],
            "source_url": ["https://example.com/a", "https://example.com/b"],
            "label_strength": ["event_summary", "weak_label_proxy"],
            "notes": ["A", "B"],
        }
    )
    events_path = config.EXTERNAL_DATA_DIR / "tasmax_event_windows.csv"
    events.to_csv(events_path, index=False)

    outputs = system.run_event_alignment_analysis("tasmax", results_path=results_path, events_path=events_path)

    summary = outputs["summary"].set_index("model")
    detail = outputs["detail"]

    assert summary.loc["LSTM Autoencoder", "events_hit"] == 1
    assert summary.loc["Isolation Forest", "events_hit"] == 2
    assert summary.loc["Isolation Forest", "overlapping_flagged_windows"] == 3
    assert set(detail["reference_flag_relation"]) <= {"in_window", "lead", "lag", "none"}
