from __future__ import annotations

import copy
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from config.config import AppConfig
from models.isolation_forest import IsolationForestDetector
from models.lstm_autoencoder import LSTMAutoencoder
from src.data_loader import ClimateDataLoader
from src.visualization import ClimateVisualizer


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass


def _clear_tensorflow_session() -> None:
    try:
        from tensorflow.keras import backend as keras_backend

        keras_backend.clear_session()
    except ImportError:
        pass


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

    def _copy_config(self, **overrides) -> AppConfig:
        config_copy = copy.deepcopy(self.config)
        for key, value in overrides.items():
            setattr(config_copy, key, value)
        config_copy.ensure_directories()
        return config_copy

    @staticmethod
    def _select_event_reference_flag(
        flagged_dates: pd.Series,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        buffer_days: int,
    ) -> tuple[pd.Timestamp | pd.NaT, str]:
        in_window = flagged_dates[(flagged_dates >= start_date) & (flagged_dates <= end_date)]
        if not in_window.empty:
            return in_window.min(), "in_window"

        search_delta = pd.Timedelta(days=buffer_days)
        pre_window = flagged_dates[(flagged_dates >= start_date - search_delta) & (flagged_dates < start_date)]
        if not pre_window.empty:
            return pre_window.max(), "lead"

        post_window = flagged_dates[(flagged_dates > end_date) & (flagged_dates <= end_date + search_delta)]
        if not post_window.empty:
            return post_window.min(), "lag"

        return pd.NaT, "none"

    def _run_lstm_experiment(
        self,
        prepared: dict[str, object],
        *,
        variable_name: str,
        seed: int,
        output_prefix: str,
        save_checkpoint: bool = False,
        save_model: bool = False,
    ) -> dict[str, object]:
        config = self._copy_config(RANDOM_STATE=seed)
        _clear_tensorflow_session()
        _set_global_seed(seed)

        model = LSTMAutoencoder(config)
        model.output_prefix = output_prefix
        model.build_model((prepared["X_train"].shape[1], prepared["X_train"].shape[2]))
        history = model.train(
            prepared["X_train"],
            prepared["X_val"],
            verbose=0,
            save_checkpoint=save_checkpoint,
        )
        results = model.detect_anomalies(
            prepared["X_test"],
            threshold_percentile=config.ANOMALY_THRESHOLD_PERCENTILE,
        )
        if save_model:
            model.save(config.get_output_path("models", f"{output_prefix}_{variable_name}_lstm_autoencoder.h5"))

        metrics = self._metric_row("LSTM Autoencoder", prepared.get("test_labels"), results["anomalies"])
        _clear_tensorflow_session()
        return {
            "history": history,
            "results": results,
            "metrics": metrics,
            "seed": seed,
            "sequence_length": prepared["sequence_length"],
        }

    def _run_if_experiment(
        self,
        prepared: dict[str, object],
        *,
        feature_mode: str = "flatten_only",
        random_state: int | None = None,
        save_model: bool = False,
        output_prefix: str | None = None,
    ) -> dict[str, object]:
        config = self._copy_config(RANDOM_STATE=random_state or self.config.RANDOM_STATE)
        detector = IsolationForestDetector(config)
        results = detector.fit_and_detect(
            prepared["X_train"],
            prepared["X_test"],
            train_dates=prepared.get("train_dates"),
            test_dates=prepared.get("test_dates"),
            feature_mode=feature_mode,
        )
        if save_model and output_prefix is not None:
            detector.save(config.get_output_path("models", f"{output_prefix}_isolation_forest.pkl"))

        metrics = self._metric_row("Isolation Forest", prepared.get("test_labels"), results["anomalies"])
        return {
            "results": results,
            "metrics": metrics,
            "feature_mode": feature_mode,
            "sequence_length": prepared["sequence_length"],
        }

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

    def run_fairness_ablation(self, variable_name: str = "tasmax") -> dict[str, object]:
        dataset = self.loader.create_synthetic_dataset(variable_name)
        self.loader.save_synthetic_snapshot(dataset, variable_name)
        labels = self.loader.extract_labels(dataset)
        series = self.loader.extract_time_series(dataset, variable_name)
        series = self.loader.handle_missing_values(series)

        window_rows: list[dict[str, object]] = []
        for window_length in self.config.FAIRNESS_WINDOW_LENGTHS:
            prepared = self.loader.prepare_lstm_data(series, labels=labels, sequence_length=window_length)
            lstm_run = self._run_lstm_experiment(
                prepared,
                variable_name=variable_name,
                seed=self.config.RANDOM_STATE,
                output_prefix=f"fairness_seq{window_length}_seed{self.config.RANDOM_STATE}",
            )
            if_run = self._run_if_experiment(
                prepared,
                feature_mode="flatten_only",
                random_state=self.config.RANDOM_STATE,
            )

            for run in (lstm_run, if_run):
                metrics_row = dict(run["metrics"])
                metrics_row.update(
                    {
                        "variable": variable_name,
                        "dataset": "synthetic_tasmax_binary",
                        "window_length": window_length,
                        "seed": run.get("seed", self.config.RANDOM_STATE),
                        "feature_mode": run.get("feature_mode", "sequence_reconstruction"),
                    }
                )
                window_rows.append(metrics_row)

        window_df = pd.DataFrame(window_rows)
        window_path = self.config.get_output_path("metrics", f"{variable_name}_fairness_window_sweep.csv")
        window_df.to_csv(window_path, index=False)

        baseline_prepared = self.loader.prepare_lstm_data(
            series,
            labels=labels,
            sequence_length=self.config.SEQUENCE_LENGTH,
        )

        feature_rows: list[dict[str, object]] = []
        for feature_mode in self.config.IF_FEATURE_MODES:
            if_run = self._run_if_experiment(
                baseline_prepared,
                feature_mode=feature_mode,
                random_state=self.config.RANDOM_STATE,
            )
            metrics_row = dict(if_run["metrics"])
            metrics_row.update(
                {
                    "variable": variable_name,
                    "dataset": "synthetic_tasmax_binary",
                    "window_length": self.config.SEQUENCE_LENGTH,
                    "feature_mode": feature_mode,
                }
            )
            feature_rows.append(metrics_row)

        feature_df = pd.DataFrame(feature_rows)
        feature_path = self.config.get_output_path("metrics", f"{variable_name}_fairness_if_feature_ablation.csv")
        feature_df.to_csv(feature_path, index=False)

        seed_rows: list[dict[str, object]] = []
        for seed in self.config.LSTM_ABLATION_SEEDS:
            lstm_run = self._run_lstm_experiment(
                baseline_prepared,
                variable_name=variable_name,
                seed=seed,
                output_prefix=f"fairness_seed{seed}",
            )
            metrics_row = dict(lstm_run["metrics"])
            metrics_row.update(
                {
                    "variable": variable_name,
                    "dataset": "synthetic_tasmax_binary",
                    "window_length": self.config.SEQUENCE_LENGTH,
                    "seed": seed,
                }
            )
            seed_rows.append(metrics_row)

        seed_df = pd.DataFrame(seed_rows)
        seed_path = self.config.get_output_path("metrics", f"{variable_name}_fairness_lstm_seed_runs.csv")
        seed_df.to_csv(seed_path, index=False)

        numeric_metrics = ["precision", "recall", "f1_score", "accuracy", "anomaly_count", "anomaly_rate"]
        summary_values: dict[str, object] = {
            "variable": variable_name,
            "dataset": "synthetic_tasmax_binary",
            "window_length": self.config.SEQUENCE_LENGTH,
            "seed_runs": len(seed_df),
        }
        for metric_name in numeric_metrics:
            mean_value = float(seed_df[metric_name].mean())
            std_value = float(seed_df[metric_name].std(ddof=0))
            summary_values[f"{metric_name}_mean"] = mean_value
            summary_values[f"{metric_name}_std"] = std_value
            summary_values[f"{metric_name}_mean_std"] = f"{mean_value:.3f} ± {std_value:.3f}"

        seed_summary_df = pd.DataFrame([summary_values])
        seed_summary_path = self.config.get_output_path("metrics", f"{variable_name}_fairness_lstm_seed_summary.csv")
        seed_summary_df.to_csv(seed_summary_path, index=False)

        summary_lines = [
            "Fairness Ablation Summary",
            "========================",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Variable: {variable_name}",
            "Dataset: synthetic tasmax benchmark",
            f"Window lengths tested: {', '.join(map(str, self.config.FAIRNESS_WINDOW_LENGTHS))}",
            f"Isolation Forest feature modes: {', '.join(self.config.IF_FEATURE_MODES)}",
            f"LSTM seeds: {', '.join(map(str, self.config.LSTM_ABLATION_SEEDS))}",
        ]
        summary_path = self.config.get_output_path("metrics", f"{variable_name}_fairness_ablation_summary.txt")
        summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

        return {
            "window_sweep": window_df,
            "window_sweep_path": window_path,
            "if_feature_ablation": feature_df,
            "if_feature_ablation_path": feature_path,
            "lstm_seed_runs": seed_df,
            "lstm_seed_runs_path": seed_path,
            "lstm_seed_summary": seed_summary_df,
            "lstm_seed_summary_path": seed_summary_path,
            "summary_path": summary_path,
        }

    def run_event_alignment_analysis(
        self,
        variable_name: str = "tasmax",
        *,
        results_path: Path | None = None,
        events_path: Path | None = None,
    ) -> dict[str, object]:
        results_path = Path(results_path) if results_path is not None else self.config.get_output_path(
            "metrics",
            f"{variable_name}_anomaly_results.csv",
        )
        events_path = Path(events_path) if events_path is not None else self.config.EXTERNAL_DATA_DIR / (
            f"{variable_name}_event_windows.csv"
        )

        if not results_path.exists():
            raise FileNotFoundError(f"Could not find anomaly results at {results_path}")
        if not events_path.exists():
            raise FileNotFoundError(f"Could not find external event window table at {events_path}")

        results_df = pd.read_csv(results_path, parse_dates=["date"])
        results_df["date"] = pd.to_datetime(results_df["date"]).dt.normalize()

        events_df = pd.read_csv(events_path, parse_dates=["start_date", "end_date"])
        events_df["start_date"] = pd.to_datetime(events_df["start_date"]).dt.normalize()
        events_df["end_date"] = pd.to_datetime(events_df["end_date"]).dt.normalize()

        evaluation_start = results_df["date"].min()
        evaluation_end = results_df["date"].max()
        events_df = events_df[
            (events_df["end_date"] >= evaluation_start) & (events_df["start_date"] <= evaluation_end)
        ].copy()
        if events_df.empty:
            raise ValueError("No external event windows overlap the current evaluation dates.")

        events_df["analysis_start"] = events_df["start_date"].clip(lower=evaluation_start)
        events_df["analysis_end"] = events_df["end_date"].clip(upper=evaluation_end)

        model_columns = {
            "LSTM Autoencoder": "lstm_anomaly",
            "Isolation Forest": "if_anomaly",
        }

        detail_rows: list[dict[str, object]] = []
        summary_rows: list[dict[str, object]] = []

        for model_name, column_name in model_columns.items():
            if column_name not in results_df.columns:
                continue

            model_flags = results_df[column_name].fillna(False).astype(bool)
            if results_df[column_name].dropna().empty:
                continue

            flagged_dates = results_df.loc[model_flags, "date"]
            union_mask = pd.Series(False, index=results_df.index)
            model_detail_rows: list[dict[str, object]] = []

            for event in events_df.itertuples(index=False):
                in_window_mask = (results_df["date"] >= event.analysis_start) & (results_df["date"] <= event.analysis_end)
                union_mask |= in_window_mask

                available_rows = results_df.loc[in_window_mask, ["date", column_name]].copy()
                available_rows[column_name] = available_rows[column_name].fillna(False).astype(bool)
                flagged_in_window = available_rows.loc[available_rows[column_name], "date"]
                reference_flag_date, reference_relation = self._select_event_reference_flag(
                    flagged_dates,
                    event.analysis_start,
                    event.analysis_end,
                    self.config.EVENT_ALIGNMENT_BUFFER_DAYS,
                )

                event_days = int(len(available_rows))
                flagged_days = int(len(flagged_in_window))
                event_hit = int(flagged_days > 0)
                event_coverage = float(flagged_days / event_days) if event_days else np.nan
                lead_lag_days = (
                    int((reference_flag_date - event.analysis_start).days)
                    if pd.notna(reference_flag_date)
                    else np.nan
                )

                detail_row = {
                    "model": model_name,
                    "event_id": event.event_id,
                    "event_name": event.event_name,
                    "window_basis": event.window_basis,
                    "label_strength": event.label_strength,
                    "source_org": event.source_org,
                    "source_title": event.source_title,
                    "source_url": event.source_url,
                    "analysis_start": event.analysis_start.date().isoformat(),
                    "analysis_end": event.analysis_end.date().isoformat(),
                    "event_days": event_days,
                    "flagged_days_in_window": flagged_days,
                    "event_hit": event_hit,
                    "event_coverage": event_coverage,
                    "overlap_count": flagged_days,
                    "reference_flag_date": (
                        reference_flag_date.date().isoformat() if pd.notna(reference_flag_date) else ""
                    ),
                    "reference_flag_relation": reference_relation,
                    "lead_lag_days": lead_lag_days,
                    "notes": event.notes,
                }
                detail_rows.append(detail_row)
                model_detail_rows.append(detail_row)

            overlapping_flagged_windows = int(model_flags[union_mask].sum())
            total_flagged_windows = int(model_flags.sum())
            model_detail_df = pd.DataFrame(model_detail_rows)
            lead_lag_series = model_detail_df["lead_lag_days"].dropna()

            summary_rows.append(
                {
                    "model": model_name,
                    "events_total": int(len(model_detail_df)),
                    "events_hit": int(model_detail_df["event_hit"].sum()),
                    "event_hit_rate": float(model_detail_df["event_hit"].mean()),
                    "mean_event_coverage": float(model_detail_df["event_coverage"].mean()),
                    "median_event_coverage": float(model_detail_df["event_coverage"].median()),
                    "median_lead_lag_days": float(lead_lag_series.median()) if not lead_lag_series.empty else np.nan,
                    "overlapping_flagged_windows": overlapping_flagged_windows,
                    "total_flagged_windows": total_flagged_windows,
                    "overlap_rate_with_event_windows": (
                        float(overlapping_flagged_windows / total_flagged_windows)
                        if total_flagged_windows
                        else np.nan
                    ),
                }
            )

        detail_df = pd.DataFrame(detail_rows)
        summary_df = pd.DataFrame(summary_rows)
        if detail_df.empty or summary_df.empty:
            raise ValueError("Event alignment could not be computed because no model outputs were available.")

        detail_path = self.config.get_output_path("metrics", f"{variable_name}_event_alignment_by_event.csv")
        summary_path = self.config.get_output_path("metrics", f"{variable_name}_event_alignment_summary.csv")
        detail_df.to_csv(detail_path, index=False)
        summary_df.to_csv(summary_path, index=False)

        notes_path = self.config.get_output_path("metrics", f"{variable_name}_event_alignment_summary.txt")
        notes_lines = [
            "External Event Alignment Summary",
            "===============================",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Variable: {variable_name}",
            f"Results file: {results_path}",
            f"External windows file: {events_path}",
            "Interpretation note: these windows are external event references and weak-label proxies, not point-wise ground truth labels.",
        ]
        for row in summary_df.itertuples(index=False):
            notes_lines.extend(
                [
                    "",
                    f"[{row.model}]",
                    f"Events hit: {row.events_hit}/{row.events_total} ({row.event_hit_rate:.3f})",
                    f"Mean event coverage: {row.mean_event_coverage:.3f}",
                    f"Overlap rate with event windows: {row.overlap_rate_with_event_windows:.3f}",
                ]
            )
        notes_path.write_text("\n".join(notes_lines) + "\n", encoding="utf-8")

        return {
            "detail": detail_df,
            "detail_path": detail_path,
            "summary": summary_df,
            "summary_path": summary_path,
            "notes_path": notes_path,
            "events_path": events_path,
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
