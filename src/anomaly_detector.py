from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from config.config import AppConfig
from models.isolation_forest import IsolationForestDetector
from models.lstm_autoencoder import LSTMAutoencoder
from src.data_loader import ClimateDataLoader
from src.visualization import ClimateVisualizer


class ClimateAnomalyDetectionSystem:
    """End-to-end anomaly detection pipeline for climate time series."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self.config.ensure_directories()

        self.loader = ClimateDataLoader(self.config)
        self.visualizer = ClimateVisualizer(self.config)
        self.if_model = IsolationForestDetector(self.config)
        self.lstm_model: LSTMAutoencoder | None = None

    @staticmethod
    def _metric_row(model_name: str, y_true, y_pred) -> dict[str, float | int | str]:
        anomaly_count = int(np.asarray(y_pred, dtype=int).sum())
        anomaly_rate = float(np.mean(y_pred))
        row: dict[str, float | int | str] = {
            "model": model_name,
            "anomaly_count": anomaly_count,
            "anomaly_rate": anomaly_rate,
            "precision": np.nan,
            "recall": np.nan,
            "f1_score": np.nan,
            "accuracy": np.nan,
            "true_positive": np.nan,
            "false_positive": np.nan,
            "true_negative": np.nan,
            "false_negative": np.nan,
        }

        if y_true is None:
            return row

        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        row.update(
            {
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1_score": f1_score(y_true, y_pred, zero_division=0),
                "accuracy": accuracy_score(y_true, y_pred),
                "true_positive": int(tp),
                "false_positive": int(fp),
                "true_negative": int(tn),
                "false_negative": int(fn),
            }
        )
        return row

    def run_pipeline(
        self,
        variable_name: str | None = None,
        allow_synthetic: bool = True,
        make_plots: bool = True,
        run_lstm: bool = True,
    ) -> dict[str, object]:
        variable = variable_name or self.config.DEFAULT_VARIABLE
        dataset, data_source = self.loader.load_dataset(variable, allow_synthetic=allow_synthetic)
        labels = self.loader.extract_labels(dataset)
        series = self.loader.extract_time_series(dataset, variable)
        series = self.loader.handle_missing_values(series)
        self.loader.save_processed_series(series, variable, labels)

        if make_plots:
            self.visualizer.plot_time_series(series, f"{variable} daily time series", f"{variable}_time_series.png")

        prepared = self.loader.prepare_lstm_data(series, labels=labels)
        test_labels = prepared.get("test_labels")
        notes: list[str] = []

        lstm_results = None
        history = None
        if run_lstm:
            try:
                self.lstm_model = LSTMAutoencoder(self.config)
                self.lstm_model.output_prefix = variable
                self.lstm_model.build_model((prepared["X_train"].shape[1], prepared["X_train"].shape[2]))
                history = self.lstm_model.train(prepared["X_train"], prepared["X_val"])
                lstm_results = self.lstm_model.detect_anomalies(
                    prepared["X_test"],
                    threshold_percentile=self.config.ANOMALY_THRESHOLD_PERCENTILE,
                )
                self.lstm_model.save(self.config.get_output_path("models", f"{variable}_lstm_autoencoder.h5"))
            except ImportError as exc:
                notes.append(str(exc))

        if_results = self.if_model.fit_and_detect(prepared["X_train"], prepared["X_test"])
        self.if_model.save(self.config.get_output_path("models", f"{variable}_isolation_forest.pkl"))

        results_df = pd.DataFrame(
            {
                "date": pd.to_datetime(prepared["test_dates"]),
                "value": prepared["y_test"],
                "lstm_anomaly": lstm_results["anomalies"] if lstm_results is not None else np.nan,
                "lstm_anomaly_score": lstm_results["anomaly_scores"] if lstm_results is not None else np.nan,
                "lstm_reconstruction_error": (
                    lstm_results["reconstruction_error"] if lstm_results is not None else np.nan
                ),
                "if_anomaly": if_results["anomalies"],
                "if_anomaly_score": if_results["anomaly_scores"],
                "ground_truth": test_labels if test_labels is not None else np.nan,
                "data_source": data_source,
            }
        )
        results_path = self.config.get_output_path("metrics", f"{variable}_anomaly_results.csv")
        results_df.to_csv(results_path, index=False)

        metrics_rows = []
        if lstm_results is not None:
            metrics_rows.append(self._metric_row("LSTM Autoencoder", test_labels, lstm_results["anomalies"]))
        metrics_rows.append(self._metric_row("Isolation Forest", test_labels, if_results["anomalies"]))
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_path = self.config.get_output_path("metrics", f"{variable}_model_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)

        if make_plots:
            if history is not None:
                self.visualizer.plot_training_history(history, f"{variable}_training_history.png")

            preferred_anomalies = (
                lstm_results["anomalies"] if lstm_results is not None else if_results["anomalies"]
            )
            preferred_scores = (
                lstm_results["anomaly_scores"] if lstm_results is not None else if_results["anomaly_scores"]
            )
            preferred_title = (
                "LSTM Autoencoder Anomaly Detection" if lstm_results is not None else "Isolation Forest Anomaly Detection"
            )
            preferred_filename = (
                f"{variable}_lstm_anomaly_detection.png"
                if lstm_results is not None
                else f"{variable}_if_anomaly_detection.png"
            )
            self.visualizer.plot_anomaly_detection(
                prepared["test_dates"],
                prepared["y_test"],
                preferred_anomalies,
                preferred_scores,
                preferred_title,
                preferred_filename,
            )
            self.visualizer.plot_metric_summary(metrics_df, f"{variable}_metric_summary.png")

        summary_lines = [
            "Climate Anomaly Detection Summary",
            "================================",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Variable: {variable}",
            f"Data source: {data_source}",
            f"Samples evaluated: {len(results_df)}",
            f"LSTM enabled: {'yes' if lstm_results is not None else 'no'}",
            f"Isolation Forest anomalies: {int(results_df['if_anomaly'].sum())}",
        ]
        if lstm_results is not None:
            summary_lines.append(f"LSTM anomalies: {int(np.sum(lstm_results['anomalies']))}")
            summary_lines.append(f"LSTM threshold: {lstm_results['threshold']:.6f}")
        if notes:
            summary_lines.extend(["", "Notes:"] + notes)

        summary_path = self.config.get_output_path("metrics", f"{variable}_summary.txt")
        summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

        return {
            "variable": variable,
            "results": results_df,
            "metrics": metrics_df,
            "results_path": results_path,
            "metrics_path": metrics_path,
            "summary_path": summary_path,
            "notes": notes,
            "data_source": data_source,
        }

    def run_all_variables(
        self,
        allow_synthetic: bool = True,
        make_plots: bool = True,
        run_lstm: bool = True,
    ) -> dict[str, object]:
        outputs = []
        for variable in self.config.SUPPORTED_VARIABLES:
            outputs.append(
                self.run_pipeline(
                    variable_name=variable,
                    allow_synthetic=allow_synthetic,
                    make_plots=make_plots,
                    run_lstm=run_lstm,
                )
            )

        aggregate_rows = []
        for output in outputs:
            metrics_df = output["metrics"].copy()
            metrics_df.insert(0, "variable", output["variable"])
            aggregate_rows.append(metrics_df)

        aggregate_metrics = pd.concat(aggregate_rows, ignore_index=True)
        aggregate_metrics_path = self.config.get_output_path("metrics", "all_variables_model_metrics.csv")
        aggregate_metrics.to_csv(aggregate_metrics_path, index=False)

        aggregate_summary_lines = [
            "Climate Anomaly Detection Summary",
            "================================",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "Variables: rainfall, tasmax, tasmin",
            "",
        ]
        for output in outputs:
            metrics_df = output["metrics"]
            lstm_rows = metrics_df[metrics_df["model"] == "LSTM Autoencoder"]
            if_rows = metrics_df[metrics_df["model"] == "Isolation Forest"]
            lstm_count = int(lstm_rows["anomaly_count"].iloc[0]) if not lstm_rows.empty else 0
            if_count = int(if_rows["anomaly_count"].iloc[0]) if not if_rows.empty else 0
            aggregate_summary_lines.extend(
                [
                    f"[{output['variable']}]",
                    f"Data source: {output['data_source']}",
                    f"LSTM anomalies: {lstm_count}",
                    f"Isolation Forest anomalies: {if_count}",
                    "",
                ]
            )

        aggregate_summary_path = self.config.get_output_path("metrics", "all_variables_summary.txt")
        aggregate_summary_path.write_text("\n".join(aggregate_summary_lines), encoding="utf-8")

        return {
            "outputs": outputs,
            "aggregate_metrics_path": aggregate_metrics_path,
            "aggregate_summary_path": aggregate_summary_path,
        }
