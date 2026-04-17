from __future__ import annotations

import argparse
import sys

from config.config import AppConfig
from src.anomaly_detector import ClimateAnomalyDetectionSystem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Climate anomaly detection pipeline")
    parser.add_argument(
        "--variable",
        default="tasmax",
        choices=("all", "rainfall", "tasmax", "tasmin"),
        help="Climate variable to analyse, or 'all' to process rainfall, tasmax, and tasmin.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of NetCDF files to load per variable.",
    )
    parser.add_argument(
        "--data-backend",
        default="standard",
        choices=("standard", "dask"),
        help="Dataset loading backend. 'dask' enables chunked lazy NetCDF loading.",
    )
    parser.add_argument(
        "--skip-lstm",
        action="store_true",
        help="Skip the LSTM autoencoder and run Isolation Forest only.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip figure generation.",
    )
    parser.add_argument(
        "--no-synthetic-fallback",
        action="store_true",
        help="Fail instead of generating synthetic data when raw data cannot be loaded.",
    )
    parser.add_argument(
        "--run-fairness-ablation",
        action="store_true",
        help="Run the synthetic tasmax fairness ablation package and save the result tables.",
    )
    parser.add_argument(
        "--run-event-alignment",
        action="store_true",
        help="Run external event-window alignment for the real tasmax results and save event-aligned metrics.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    config = AppConfig()
    if args.max_files is not None:
        config.MAX_FILES_TO_LOAD = args.max_files
    config.DATA_BACKEND = args.data_backend

    system = ClimateAnomalyDetectionSystem(config)
    if args.run_fairness_ablation:
        outputs = system.run_fairness_ablation("tasmax")
        print(f"Window sweep saved to: {outputs['window_sweep_path']}")
        print(f"IF feature ablation saved to: {outputs['if_feature_ablation_path']}")
        print(f"LSTM seed runs saved to: {outputs['lstm_seed_runs_path']}")
        print(f"LSTM seed summary saved to: {outputs['lstm_seed_summary_path']}")
        print(f"Ablation summary saved to: {outputs['summary_path']}")
        return 0
    if args.run_event_alignment:
        outputs = system.run_event_alignment_analysis("tasmax")
        print(f"Event detail saved to: {outputs['detail_path']}")
        print(f"Event summary saved to: {outputs['summary_path']}")
        print(f"Event notes saved to: {outputs['notes_path']}")
        return 0

    if args.variable == "all":
        outputs = system.run_all_variables(
            allow_synthetic=not args.no_synthetic_fallback,
            make_plots=not args.no_plots,
            run_lstm=not args.skip_lstm,
        )
        for output in outputs["outputs"]:
            print(f"[{output['variable']}] Results saved to: {output['results_path']}")
            print(f"[{output['variable']}] Metrics saved to: {output['metrics_path']}")
            print(f"[{output['variable']}] Summary saved to: {output['summary_path']}")
            for note in output["notes"]:
                print(f"[{output['variable']}] Note: {note}")
        print(f"Aggregate metrics saved to: {outputs['aggregate_metrics_path']}")
        print(f"Aggregate summary saved to: {outputs['aggregate_summary_path']}")
    else:
        outputs = system.run_pipeline(
            variable_name=args.variable,
            allow_synthetic=not args.no_synthetic_fallback,
            make_plots=not args.no_plots,
            run_lstm=not args.skip_lstm,
        )

        print(f"Results saved to: {outputs['results_path']}")
        print(f"Metrics saved to: {outputs['metrics_path']}")
        print(f"Summary saved to: {outputs['summary_path']}")

        for note in outputs["notes"]:
            print(f"Note: {note}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
