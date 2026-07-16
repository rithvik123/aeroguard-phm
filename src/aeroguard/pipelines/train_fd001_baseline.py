"""Train and evaluate the first AeroGuard FD001 classical RUL baseline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from aeroguard.data.columns import (
    CYCLE_COLUMN,
    SENSOR_COLUMNS,
    TEST_TARGET_COLUMN,
    UNIT_COLUMN,
)
from aeroguard.data.loader import load_cmapss_dataset
from aeroguard.data.targets import add_training_rul_targets, final_observed_test_rows
from aeroguard.evaluation.metrics import (
    per_engine_prediction_frame,
    regression_metrics,
)
from aeroguard.features.preprocessing import AeroGuardPreprocessor
from aeroguard.models.baseline_rul import (
    build_baseline_models,
    fit_models,
    predict_models,
)


REQUIRED_CONFIG_KEYS = {
    "dataset_dir",
    "subset",
    "random_seed",
    "validation_fraction",
    "target_column",
    "rul_cap",
    "include_cycle_as_feature",
    "features_to_exclude",
    "near_constant_threshold",
    "correlation_threshold",
    "rolling_features_enabled",
    "rolling_window_sizes",
    "ridge",
    "random_forest",
    "output_dir",
    "plot_sample_engines",
}


def project_root() -> Path:
    """Resolve the repository root from the src-layout module location."""
    return Path(__file__).resolve().parents[3]


def resolve_project_path(value: str | Path, root: Path) -> Path:
    """Resolve relative paths against the project root."""
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate YAML configuration."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    validate_config(config, project_root())
    return config


def validate_config(config: dict[str, Any], root: Path) -> None:
    """Validate the small baseline configuration surface."""
    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"Missing required configuration keys: {missing}")
    if str(config["subset"]).upper() != "FD001":
        raise ValueError("This phase supports only subset FD001.")
    validation_fraction = float(config["validation_fraction"])
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if float(config["rul_cap"]) <= 0:
        raise ValueError("rul_cap must be positive.")
    if float(config["near_constant_threshold"]) < 0:
        raise ValueError("near_constant_threshold must be non-negative.")
    if not 0.0 < float(config["correlation_threshold"]) <= 1.0:
        raise ValueError("correlation_threshold must be in (0, 1].")
    if config["target_column"] not in {"rul_capped", "rul_uncapped"}:
        raise ValueError("target_column must be 'rul_capped' or 'rul_uncapped'.")
    dataset_dir = resolve_project_path(config["dataset_dir"], root)
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    forest = dict(config["random_forest"])
    if int(forest.get("n_estimators", 0)) <= 0:
        raise ValueError("random_forest.n_estimators must be positive.")
    if "min_samples_leaf" in forest and int(forest["min_samples_leaf"]) <= 0:
        raise ValueError("random_forest.min_samples_leaf must be positive.")
    if "max_depth" in forest and forest["max_depth"] is not None and int(forest["max_depth"]) <= 0:
        raise ValueError("random_forest.max_depth must be positive or null.")
    if not isinstance(config["features_to_exclude"], list):
        raise ValueError("features_to_exclude must be a list.")
    if not isinstance(config["rolling_window_sizes"], list):
        raise ValueError("rolling_window_sizes must be a list.")
    if int(config["plot_sample_engines"]) <= 0:
        raise ValueError("plot_sample_engines must be positive.")


def split_train_validation_by_engine(
    frame: pd.DataFrame,
    validation_fraction: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int], list[int]]:
    """Create a deterministic engine-level train/validation split."""
    engine_ids = np.array(sorted(frame[UNIT_COLUMN].unique()))
    if len(engine_ids) < 2:
        raise ValueError("At least two training engines are required for validation.")
    rng = np.random.default_rng(int(random_seed))
    shuffled = engine_ids.copy()
    rng.shuffle(shuffled)
    n_validation = max(1, int(round(len(shuffled) * validation_fraction)))
    n_validation = min(n_validation, len(shuffled) - 1)
    validation_ids = sorted(int(value) for value in shuffled[:n_validation])
    train_ids = sorted(int(value) for value in shuffled[n_validation:])
    train_split = frame[frame[UNIT_COLUMN].isin(train_ids)].copy()
    validation_split = frame[frame[UNIT_COLUMN].isin(validation_ids)].copy()
    return train_split, validation_split, train_ids, validation_ids


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_ready(payload), handle, indent=2, allow_nan=True)


def evaluate_prediction_dict(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    return {name: regression_metrics(y_true, pred) for name, pred in predictions.items()}


def _save_predicted_vs_actual(
    predictions_frame: pd.DataFrame,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for model, group in predictions_frame.groupby("model"):
        ax.scatter(group["true_rul"], group["predicted_rul"], s=24, alpha=0.75, label=model)
    lower = min(predictions_frame["true_rul"].min(), predictions_frame["predicted_rul"].min())
    upper = max(predictions_frame["true_rul"].max(), predictions_frame["predicted_rul"].max())
    ax.plot([lower, upper], [lower, upper], color="black", linewidth=1, linestyle="--", label="perfect")
    ax.set_title("FD001 Test RUL: Predicted Versus Actual")
    ax.set_xlabel("Actual final-cycle RUL (cycles)")
    ax.set_ylabel("Predicted final-cycle RUL (cycles)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_residual_plot(predictions_frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for model, group in predictions_frame.groupby("model"):
        ax.scatter(group["true_rul"], group["residual"], s=24, alpha=0.75, label=model)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_title("FD001 Test Residuals")
    ax.set_xlabel("Actual final-cycle RUL (cycles)")
    ax.set_ylabel("Residual: predicted - actual (cycles)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_error_distribution(
    predictions_frame: pd.DataFrame,
    path: Path,
    column: str,
    title: str,
    xlabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for model, group in predictions_frame.groupby("model"):
        ax.hist(group[column], bins=20, alpha=0.45, label=model)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Engine count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_sensor_trajectories(
    train_frame: pd.DataFrame,
    sample_engines: list[int],
    sensor_columns: list[str],
    path: Path,
) -> None:
    sensors = [sensor for sensor in sensor_columns if sensor in train_frame.columns][:5]
    fig, axes = plt.subplots(len(sensors), 1, figsize=(9, max(3, 2.4 * len(sensors))), sharex=True)
    if len(sensors) == 1:
        axes = [axes]
    for ax, sensor in zip(axes, sensors):
        for unit_id in sample_engines:
            group = train_frame[train_frame[UNIT_COLUMN] == unit_id]
            ax.plot(group[CYCLE_COLUMN], group[sensor], linewidth=1.2, label=f"engine {unit_id}")
        ax.set_ylabel(sensor)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Cycle")
    fig.suptitle("Selected Training Sensor Trajectories")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_rul_trajectories(
    train_frame: pd.DataFrame,
    sample_engines: list[int],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for unit_id in sample_engines:
        group = train_frame[train_frame[UNIT_COLUMN] == unit_id]
        ax.plot(group[CYCLE_COLUMN], group["rul_uncapped"], linewidth=1.2, label=f"engine {unit_id} uncapped")
        ax.plot(group[CYCLE_COLUMN], group["rul_capped"], linewidth=1.2, linestyle="--", label=f"engine {unit_id} capped")
    ax.set_title("Training RUL Target Trajectories")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Remaining useful life (cycles)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_model_comparison(
    metrics: dict[str, dict[str, dict[str, float]]],
    path: Path,
) -> None:
    models = list(metrics["validation"].keys())
    x_positions = np.arange(len(models))
    width = 0.35
    validation_rmse = [metrics["validation"][name]["rmse"] for name in models]
    test_rmse = [metrics["test"][name]["rmse"] for name in models]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x_positions - width / 2, validation_rmse, width, label="validation RMSE")
    ax.bar(x_positions + width / 2, test_rmse, width, label="test RMSE")
    ax.set_title("FD001 Baseline Model Comparison")
    ax.set_ylabel("RMSE (cycles)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_figures(
    output_dir: Path,
    train_frame: pd.DataFrame,
    predictions_frame: pd.DataFrame,
    metrics: dict[str, dict[str, dict[str, float]]],
    config: dict[str, Any],
) -> list[str]:
    """Generate the requested non-interactive visual outputs."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    sample_count = int(config["plot_sample_engines"])
    sample_engines = sorted(train_frame[UNIT_COLUMN].unique())[:sample_count]
    sensor_columns = list(config.get("sample_sensor_columns", SENSOR_COLUMNS[:3]))

    figure_paths = {
        "predicted_vs_actual_test_rul.png": lambda path: _save_predicted_vs_actual(predictions_frame, path),
        "residual_plot.png": lambda path: _save_residual_plot(predictions_frame, path),
        "absolute_error_distribution.png": lambda path: _save_error_distribution(
            predictions_frame,
            path,
            "absolute_error",
            "FD001 Test Absolute Error Distribution",
            "Absolute error (cycles)",
        ),
        "prediction_error_distribution.png": lambda path: _save_error_distribution(
            predictions_frame,
            path,
            "residual",
            "FD001 Test Prediction Error Distribution",
            "Residual: predicted - actual (cycles)",
        ),
        "selected_sensor_trajectories.png": lambda path: _save_sensor_trajectories(
            train_frame,
            sample_engines,
            sensor_columns,
            path,
        ),
        "rul_target_trajectories.png": lambda path: _save_rul_trajectories(
            train_frame,
            sample_engines,
            path,
        ),
        "model_comparison.png": lambda path: _save_model_comparison(metrics, path),
    }
    written: list[str] = []
    for filename, writer in figure_paths.items():
        path = figures_dir / filename
        writer(path)
        written.append(str(path))
    return written


def write_results_note(
    path: Path,
    result: dict[str, Any],
    config_path: Path,
) -> None:
    """Write the human-readable FD001 baseline result note."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = result["metrics"]
    feature_reasons = result["features_excluded"]

    def metric_lines(split: str) -> str:
        rows = []
        for model, values in metrics[split].items():
            rows.append(
                f"| {model} | {values['mae']:.4f} | {values['rmse']:.4f} | "
                f"{values['r2']:.6f} | {values['nasa_score']:.4f} |"
            )
        return "\n".join(rows)

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# FD001 Baseline Results\n\n")
        handle.write(f"- Python interpreter used: `{result['python_executable']}`\n")
        handle.write(f"- Python version: `{result['python_version']}`\n")
        handle.write(f"- Dataset path: `{result['dataset_dir']}`\n")
        handle.write(f"- Train dataset dimensions: `{result['train_shape']}`\n")
        handle.write(f"- Test dataset dimensions: `{result['test_shape']}`\n")
        handle.write(f"- Training engines: `{result['train_engine_count']}`\n")
        handle.write(f"- Model-training engines: `{result['model_train_engine_count']}`\n")
        handle.write(f"- Validation engines: `{result['validation_engine_count']}`\n")
        handle.write(f"- Test engines: `{result['test_engine_count']}`\n")
        handle.write(f"- Retained feature count: `{len(result['features_retained'])}`\n")
        handle.write(f"- Target used for training: `{result['target_column']}`\n")
        handle.write(f"- Best validation model: `{result['best_validation_model']}`\n")
        handle.write(f"- Best test model, reported for final evaluation only: `{result['best_test_model']}`\n")
        handle.write(f"- Runtime seconds: `{result['runtime_seconds']:.3f}`\n\n")
        handle.write("## Features Retained\n\n")
        handle.write(", ".join(result["features_retained"]) + "\n\n")
        handle.write("## Features Excluded\n\n")
        if feature_reasons:
            for feature, reason in feature_reasons.items():
                handle.write(f"- `{feature}`: {reason}\n")
        else:
            handle.write("- None\n")
        handle.write("\n## Validation Metrics\n\n")
        handle.write("| model | MAE | RMSE | R2 | NASA score |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: |\n")
        handle.write(metric_lines("validation") + "\n\n")
        handle.write("## Test Metrics\n\n")
        handle.write("| model | MAE | RMSE | R2 | NASA score |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: |\n")
        handle.write(metric_lines("test") + "\n\n")
        handle.write("## Warnings\n\n")
        for warning in result["warnings"]:
            handle.write(f"- {warning}\n")
        handle.write("\n## Limitations\n\n")
        handle.write("- Model selection is based on validation performance only; test metrics are final evaluation.\n")
        handle.write("- The initial baseline uses row-level classical regressors and final-cycle test evaluation only.\n")
        handle.write("- Rolling features are disabled by default and no hyperparameter sweep was run.\n\n")
        handle.write("## Exact Reproduction Command\n\n")
        handle.write("```powershell\n")
        handle.write('$env:PYTHONPATH = ".\\src"\n')
        handle.write(
            "python -m aeroguard.pipelines.train_fd001_baseline "
            f'--config "{config_path.as_posix()}"\n'
        )
        handle.write("```\n")


def run_pipeline(config_path: str | Path) -> dict[str, Any]:
    """Run the complete FD001 baseline and return a result summary."""
    start = time.perf_counter()
    root = project_root()
    config_path = Path(config_path)
    config = load_config(config_path)
    dataset_dir = resolve_project_path(config["dataset_dir"], root)
    output_dir = resolve_project_path(config["output_dir"], root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    dataset = load_cmapss_dataset(dataset_dir, subset=str(config["subset"]))
    train_with_targets = add_training_rul_targets(dataset.train, rul_cap=float(config["rul_cap"]))
    test_final = final_observed_test_rows(dataset.test, dataset.test_rul)

    model_train, validation, train_ids, validation_ids = split_train_validation_by_engine(
        train_with_targets,
        validation_fraction=float(config["validation_fraction"]),
        random_seed=int(config["random_seed"]),
    )

    preprocessor = AeroGuardPreprocessor(
        include_cycle_as_feature=bool(config["include_cycle_as_feature"]),
        features_to_exclude=list(config["features_to_exclude"]),
        near_constant_threshold=float(config["near_constant_threshold"]),
        rolling_features_enabled=bool(config["rolling_features_enabled"]),
        rolling_window_sizes=list(config["rolling_window_sizes"]),
    )
    x_train = preprocessor.fit_transform(
        model_train,
        correlation_threshold=float(config["correlation_threshold"]),
    )
    x_validation = preprocessor.transform(validation)
    x_test = preprocessor.transform(test_final)

    feature_audit_path = output_dir / "feature_audit.csv"
    correlation_audit_path = output_dir / "correlation_audit.csv"
    preprocessor.feature_audit_.to_csv(feature_audit_path, index=False)
    preprocessor.correlation_audit_.to_csv(correlation_audit_path, index=False)

    target_column = str(config["target_column"])
    y_train = model_train[target_column].to_numpy(dtype=float)
    y_validation = validation[target_column].to_numpy(dtype=float)
    y_test = test_final[TEST_TARGET_COLUMN].to_numpy(dtype=float)

    models = build_baseline_models(config)
    fitted_models = fit_models(models, x_train, y_train)
    validation_predictions = predict_models(
        fitted_models,
        x_validation,
        clip_non_negative=bool(config.get("clip_predictions_non_negative", True)),
    )
    test_predictions = predict_models(
        fitted_models,
        x_test,
        clip_non_negative=bool(config.get("clip_predictions_non_negative", True)),
    )

    metrics = {
        "validation": evaluate_prediction_dict(y_validation, validation_predictions),
        "test": evaluate_prediction_dict(y_test, test_predictions),
    }
    selection_metric = str(config.get("model_selection_metric", "rmse"))
    best_validation_model = min(
        metrics["validation"],
        key=lambda name: metrics["validation"][name][selection_metric],
    )
    best_test_model = min(
        metrics["test"],
        key=lambda name: metrics["test"][name][selection_metric],
    )

    prediction_frames = [
        per_engine_prediction_frame(
            test_final[UNIT_COLUMN].to_numpy(),
            y_test,
            prediction,
            model_name=name,
        )
        for name, prediction in test_predictions.items()
    ]
    predictions_frame = pd.concat(prediction_frames, ignore_index=True)
    predictions_path = output_dir / "test_predictions.csv"
    predictions_frame.to_csv(predictions_path, index=False)

    metrics_path = output_dir / "metrics.json"
    write_json(
        metrics_path,
        {
            "selection_metric": selection_metric,
            "best_validation_model": best_validation_model,
            "best_test_model_final_evaluation_only": best_test_model,
            "metrics": metrics,
        },
    )

    figure_paths = write_figures(
        output_dir,
        train_with_targets,
        predictions_frame,
        metrics,
        config,
    )

    features_excluded = dict(preprocessor.feature_exclusion_reasons_)
    if not bool(config["include_cycle_as_feature"]):
        features_excluded[CYCLE_COLUMN] = "not configured as a model feature"

    warnings = []
    if best_validation_model != best_test_model:
        warnings.append(
            "Best validation model differs from best test model; model selection remains validation-based."
        )
    if features_excluded:
        warnings.append(
            "Features were excluded using model-training engines only; see feature_audit.csv."
        )
    if bool(config["rolling_features_enabled"]):
        warnings.append("Rolling features were enabled for this run.")
    else:
        warnings.append("Rolling features were disabled for this first baseline.")

    runtime_seconds = time.perf_counter() - start
    result = {
        "python_executable": str(Path(__import__("sys").executable)),
        "python_version": __import__("sys").version.replace("\n", " "),
        "dataset_dir": str(dataset_dir),
        "subset": str(config["subset"]),
        "train_shape": list(dataset.train.shape),
        "test_shape": list(dataset.test.shape),
        "train_engine_count": int(dataset.train[UNIT_COLUMN].nunique()),
        "model_train_engine_count": len(train_ids),
        "validation_engine_count": len(validation_ids),
        "test_engine_count": int(dataset.test[UNIT_COLUMN].nunique()),
        "model_train_ids": train_ids,
        "validation_ids": validation_ids,
        "features_retained": preprocessor.retained_feature_names,
        "features_excluded": features_excluded,
        "target_column": target_column,
        "metrics": metrics,
        "best_validation_model": best_validation_model,
        "best_test_model": best_test_model,
        "output_files": {
            "feature_audit": str(feature_audit_path),
            "correlation_audit": str(correlation_audit_path),
            "test_predictions": str(predictions_path),
            "metrics": str(metrics_path),
        },
        "figures": figure_paths,
        "warnings": warnings,
        "runtime_seconds": runtime_seconds,
    }

    summary_path = output_dir / "run_summary.json"
    write_json(summary_path, result)
    result["output_files"]["run_summary"] = str(summary_path)

    results_note_path = resolve_project_path(
        config.get("results_note_path", "notes/fd001_baseline_results.md"),
        root,
    )
    write_results_note(results_note_path, result, config_path)
    result["output_files"]["results_note"] = str(results_note_path)
    print(json.dumps(_json_ready(result), indent=2, allow_nan=True))
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the AeroGuard FD001 classical RUL baseline."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the FD001 baseline YAML configuration.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
