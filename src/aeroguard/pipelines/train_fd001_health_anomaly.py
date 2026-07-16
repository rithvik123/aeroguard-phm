"""Train and evaluate FD001 health-index and anomaly-detection baselines."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from sklearn.preprocessing import StandardScaler

from aeroguard.anomaly.isolation_forest import IsolationForestAnomalyDetector
from aeroguard.anomaly.one_class_svm import OneClassSVMAnomalyDetector
from aeroguard.anomaly.pca_reconstruction import PCAReconstructionAnomalyDetector
from aeroguard.anomaly.persistence import apply_persistent_alarms
from aeroguard.data.columns import (
    BASE_FEATURE_COLUMNS,
    CYCLE_COLUMN,
    EXCLUDED_MODEL_INPUT_COLUMNS,
    TEST_TARGET_COLUMN,
    UNIT_COLUMN,
)
from aeroguard.data.loader import load_cmapss_dataset
from aeroguard.data.targets import add_training_rul_targets
from aeroguard.evaluation.anomaly_metrics import (
    row_level_anomaly_metrics,
    summarize_engine_onsets,
)
from aeroguard.evaluation.health_metrics import health_index_metrics
from aeroguard.features.preprocessing import audit_features
from aeroguard.health.pca_health_index import PCAHealthIndex
from aeroguard.health.smoothing import smooth_by_engine
from aeroguard.onset.onset_detection import (
    add_proxy_labels,
    apply_page_hinkley_by_engine,
    derive_test_true_rul_trajectory,
)
from aeroguard.pipelines.train_fd001_baseline import (
    project_root,
    resolve_project_path,
    split_train_validation_by_engine,
)


REQUIRED_CONFIG_KEYS = {
    "dataset_dir",
    "subset",
    "random_seed",
    "validation_fraction",
    "baseline_config_path",
    "include_cycle_as_feature",
    "features_to_exclude",
    "healthy_rul_threshold",
    "critical_rul_threshold",
    "near_constant_threshold",
    "correlation_threshold",
    "health_index",
    "smoothing",
    "pca_reconstruction",
    "isolation_forest",
    "one_class_svm",
    "persistence",
    "page_hinkley",
    "primary_validation_selection_metric",
    "output_dir",
    "representative_engine_count",
}

DETECTORS = {
    "pca_reconstruction": {
        "score": "pca_anomaly_score",
        "flag": "pca_anomaly_flag",
        "persistent_prefix": "pca",
        "persistent_flag": "pca_persistent_alarm_flag",
    },
    "isolation_forest": {
        "score": "isolation_forest_score",
        "flag": "isolation_forest_flag",
        "persistent_prefix": "isolation_forest",
        "persistent_flag": "isolation_forest_persistent_alarm_flag",
    },
    "one_class_svm": {
        "score": "one_class_svm_score",
        "flag": "one_class_svm_flag",
        "persistent_prefix": "one_class_svm",
        "persistent_flag": "one_class_svm_persistent_alarm_flag",
    },
}


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    validate_config(config, project_root())
    return config


def validate_config(config: dict[str, Any], root: Path) -> None:
    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"Missing required configuration keys: {missing}")
    if str(config["subset"]).upper() != "FD001":
        raise ValueError("This phase supports only FD001.")
    dataset_dir = resolve_project_path(config["dataset_dir"], root)
    baseline_config = resolve_project_path(config["baseline_config_path"], root)
    output_dir = resolve_project_path(config["output_dir"], root)
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not baseline_config.exists():
        raise FileNotFoundError(f"Baseline config not found: {baseline_config}")
    if str(output_dir).lower().find("\\references\\") >= 0 or str(output_dir).lower().find("\\extracted-code\\") >= 0:
        raise ValueError("Output directory must not be inside read-only reference directories.")

    validation_fraction = float(config["validation_fraction"])
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    healthy = float(config["healthy_rul_threshold"])
    critical = float(config["critical_rul_threshold"])
    if healthy <= critical:
        raise ValueError("healthy_rul_threshold must be greater than critical_rul_threshold.")
    if critical < 0:
        raise ValueError("critical_rul_threshold must be non-negative.")
    if float(config["near_constant_threshold"]) < 0:
        raise ValueError("near_constant_threshold must be non-negative.")
    if not 0.0 < float(config["correlation_threshold"]) <= 1.0:
        raise ValueError("correlation_threshold must be in (0, 1].")

    pca_percentile = float(config["pca_reconstruction"]["threshold_percentile"])
    if not 0.0 < pca_percentile < 100.0:
        raise ValueError("PCA anomaly threshold percentile must be in (0, 100).")
    contamination = config["isolation_forest"]["contamination"]
    if contamination != "auto" and not 0.0 < float(contamination) < 0.5:
        raise ValueError("Isolation Forest contamination must be 'auto' or in (0, 0.5).")
    nu = float(config["one_class_svm"]["nu"])
    if not 0.0 < nu < 1.0:
        raise ValueError("One-Class SVM nu must be in (0, 1).")
    if int(config["persistence"]["window"]) <= 0:
        raise ValueError("persistence.window must be positive.")
    if float(config["page_hinkley"]["threshold"]) <= 0:
        raise ValueError("Page-Hinkley threshold must be positive.")
    if float(config["page_hinkley"]["delta"]) < 0:
        raise ValueError("Page-Hinkley delta must be non-negative.")
    if int(config["page_hinkley"]["min_observations"]) <= 0:
        raise ValueError("Page-Hinkley min_observations must be positive.")
    if int(config["representative_engine_count"]) <= 0:
        raise ValueError("representative_engine_count must be positive.")


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
        return None if np.isnan(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_ready(payload), handle, indent=2, allow_nan=False)


def _candidate_features(frame: pd.DataFrame, include_cycle: bool) -> list[str]:
    features = list(BASE_FEATURE_COLUMNS)
    if include_cycle:
        features.insert(0, CYCLE_COLUMN)
    return [feature for feature in features if feature in frame.columns]


def select_anomaly_features(
    model_train: pd.DataFrame,
    include_cycle: bool,
    configured_exclusions: list[str],
    near_constant_threshold: float,
    correlation_threshold: float,
) -> tuple[list[str], list[str], dict[str, str], pd.DataFrame, pd.DataFrame]:
    """Select retained features using only model-training engines."""
    candidates = _candidate_features(model_train, include_cycle)
    reasons: dict[str, str] = {}
    configured = set(configured_exclusions)
    forbidden = set(EXCLUDED_MODEL_INPUT_COLUMNS)
    for feature in candidates:
        if feature in configured:
            reasons[feature] = "configured exclusion"
        elif feature in forbidden:
            reasons[feature] = "identifier, cycle, target, or prediction column"
        else:
            values = pd.to_numeric(model_train[feature], errors="coerce")
            finite = values.replace([np.inf, -np.inf], np.nan).dropna()
            unique = int(finite.nunique(dropna=True))
            variance = float(finite.var(ddof=0)) if len(finite) else np.nan
            if unique <= 1:
                reasons[feature] = "constant in model-training engines"
            elif np.isfinite(variance) and variance <= near_constant_threshold:
                reasons[feature] = (
                    "near-constant in model-training engines "
                    f"(variance <= {near_constant_threshold})"
                )
    retained = [feature for feature in candidates if feature not in reasons]
    if not retained:
        raise ValueError("No features retained for anomaly models.")
    feature_audit, correlation_audit = audit_features(
        model_train,
        candidate_features=candidates,
        retained_features=retained,
        exclusion_reasons=reasons,
        near_constant_threshold=near_constant_threshold,
        correlation_threshold=correlation_threshold,
    )
    return candidates, retained, reasons, feature_audit, correlation_audit


def prepare_datasets(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    root = project_root()
    dataset_dir = resolve_project_path(config["dataset_dir"], root)
    dataset = load_cmapss_dataset(dataset_dir, str(config["subset"]))
    train = add_training_rul_targets(dataset.train, rul_cap=float(config["healthy_rul_threshold"]))
    train["true_rul_uncapped"] = train["rul_uncapped"]
    test = derive_test_true_rul_trajectory(dataset.test, dataset.test_rul)

    train = add_proxy_labels(
        train,
        healthy_rul_threshold=float(config["healthy_rul_threshold"]),
        critical_rul_threshold=float(config["critical_rul_threshold"]),
    )
    test = add_proxy_labels(
        test,
        healthy_rul_threshold=float(config["healthy_rul_threshold"]),
        critical_rul_threshold=float(config["critical_rul_threshold"]),
    )
    model_train, validation, train_ids, validation_ids = split_train_validation_by_engine(
        train,
        validation_fraction=float(config["validation_fraction"]),
        random_seed=int(config["random_seed"]),
    )
    metadata = {
        "dataset_dir": str(dataset_dir),
        "train_shape": list(dataset.train.shape),
        "test_shape": list(dataset.test.shape),
        "train_engine_count": int(dataset.train[UNIT_COLUMN].nunique()),
        "model_train_engine_count": len(train_ids),
        "validation_engine_count": len(validation_ids),
        "test_engine_count": int(dataset.test[UNIT_COLUMN].nunique()),
        "model_train_ids": train_ids,
        "validation_ids": validation_ids,
    }
    return model_train.copy(), validation.copy(), test.copy(), metadata


def apply_health_and_detectors(
    frames: dict[str, pd.DataFrame],
    retained_features: list[str],
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    model_train = frames["model_train"]
    healthy_mask = model_train["true_rul_uncapped"] > float(config["healthy_rul_threshold"])
    healthy_train = model_train.loc[healthy_mask]
    if healthy_train.empty:
        raise ValueError("No healthy model-training rows are available for anomaly fitting.")

    scaler = StandardScaler()
    scaler.fit(healthy_train[retained_features])
    transformed = {
        split: scaler.transform(frame[retained_features])
        for split, frame in frames.items()
    }
    healthy_x = scaler.transform(healthy_train[retained_features])

    health_cfg = dict(config["health_index"])
    health_model = PCAHealthIndex(
        n_components=health_cfg["n_components"],
        lower_quantile=float(health_cfg["lower_quantile"]),
        upper_quantile=float(health_cfg["upper_quantile"]),
        clip_scaled=bool(health_cfg["clip_scaled"]),
    )
    health_model.fit(
        transformed["model_train"],
        model_train["true_rul_uncapped"].to_numpy(dtype=float),
    )
    for split, frame in frames.items():
        raw, scaled = health_model.transform(transformed[split])
        frame["health_index_raw"] = raw
        frame["health_index_scaled"] = scaled
        frames[split] = smooth_by_engine(
            frame,
            value_column="health_index_scaled",
            output_column="smoothed_health_index",
            method=str(config["smoothing"]["method"]),
            window=int(config["smoothing"]["window"]),
            causal=bool(config["smoothing"]["causal"]),
        )

    pca_cfg = dict(config["pca_reconstruction"])
    pca_detector = PCAReconstructionAnomalyDetector(
        n_components=pca_cfg["n_components"],
        threshold_percentile=float(pca_cfg["threshold_percentile"]),
    ).fit(healthy_x)
    iso_cfg = dict(config["isolation_forest"])
    iso_detector = IsolationForestAnomalyDetector(**iso_cfg).fit(healthy_x)
    svm_cfg = dict(config["one_class_svm"])
    svm_detector = OneClassSVMAnomalyDetector(
        kernel=svm_cfg["kernel"],
        nu=float(svm_cfg["nu"]),
        gamma=svm_cfg["gamma"],
        max_training_rows=svm_cfg["max_healthy_training_rows"],
        random_state=int(config["random_seed"]),
    ).fit(healthy_x)

    for split, frame in frames.items():
        pca_error, pca_score, pca_flag = pca_detector.score(transformed[split])
        iso_score, iso_flag = iso_detector.score(transformed[split])
        svm_score, svm_flag = svm_detector.score(transformed[split])
        frame["pca_reconstruction_error"] = pca_error
        frame["pca_anomaly_score"] = pca_score
        frame["pca_anomaly_flag"] = pca_flag
        frame["isolation_forest_score"] = iso_score
        frame["isolation_forest_flag"] = iso_flag
        frame["one_class_svm_score"] = svm_score
        frame["one_class_svm_flag"] = svm_flag
        frames[split] = frame

    persistence_cfg = dict(config["persistence"])
    persistence_summaries = {}
    for split, frame in frames.items():
        for detector_name, columns in DETECTORS.items():
            frame, summary = apply_persistent_alarms(
                frame,
                flag_column=columns["flag"],
                output_prefix=columns["persistent_prefix"],
                persistence_window=int(persistence_cfg["window"]),
                alarm_state_from_onset=bool(persistence_cfg["alarm_state_from_onset"]),
                require_consecutive_cycles=bool(persistence_cfg["require_consecutive_cycles"]),
            )
            persistence_summaries[f"{split}_{detector_name}"] = summary
        frames[split] = frame

    ph_cfg = dict(config["page_hinkley"])
    for split, frame in frames.items():
        frame, _ = apply_page_hinkley_by_engine(
            frame,
            signal_column=ph_cfg["primary_signal"],
            output_prefix="page_hinkley",
            delta=float(ph_cfg["delta"]),
            threshold=float(ph_cfg["threshold"]),
            min_observations=int(ph_cfg["min_observations"]),
            direction=ph_cfg["direction"],
            reset_after_detection=bool(ph_cfg["reset_after_detection"]),
        )
        frames[split] = frame

    model_info = {
        "healthy_training_row_count": int(len(healthy_train)),
        "scaler_fit_population": "healthy rows from model-training engines only",
        "health_index_explained_variance_ratio": health_model.explained_variance_ratio_,
        "health_index_orientation": health_model.orientation_,
        "health_index_scaling_quantiles": [
            float(health_model.lower_quantile),
            float(health_model.upper_quantile),
        ],
        "pca_reconstruction_threshold": pca_detector.threshold_,
        "pca_reconstruction_explained_variance_ratio": pca_detector.explained_variance_ratio_,
        "isolation_forest_parameters": iso_cfg,
        "one_class_svm_parameters": svm_cfg,
        "one_class_svm_subsampling_applied": svm_detector.subsampling_applied_,
        "one_class_svm_fit_row_count": svm_detector.fit_row_count_,
        "persistence": persistence_cfg,
        "page_hinkley": ph_cfg,
    }
    return frames, model_info


def compute_metrics(
    frames: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, dict[str, Any]]:
    row_metrics: dict[str, Any] = {
        "proxy_label_notice": (
            "Proxy labels are based on true uncapped RUL thresholds and are not "
            "certified physical anomaly-onset labels."
        ),
        "health_index": {},
        "anomaly_detectors": {},
    }
    for split in ["validation", "test"]:
        row_metrics["health_index"][split] = health_index_metrics(frames[split])
        row_metrics["anomaly_detectors"][split] = {}
        for detector_name, columns in DETECTORS.items():
            row_metrics["anomaly_detectors"][split][detector_name] = row_level_anomaly_metrics(
                frames[split]["proxy_degradation_label"],
                frames[split][columns["flag"]],
                frames[split][columns["score"]],
            )

    engine_summary_frames = []
    engine_metrics: dict[str, Any] = {}
    for split in ["validation", "test"]:
        engine_metrics[split] = {}
        for detector_name, columns in DETECTORS.items():
            summary, metrics = summarize_engine_onsets(
                frames[split],
                detection_flag_column=columns["persistent_flag"],
                method_name=f"{detector_name}_persistent",
                split_name=split,
                healthy_rul_threshold=float(config["healthy_rul_threshold"]),
                critical_rul_threshold=float(config["critical_rul_threshold"]),
            )
            engine_summary_frames.append(summary)
            engine_metrics[split][f"{detector_name}_persistent"] = metrics
        ph_summary, ph_metrics = summarize_engine_onsets(
            frames[split],
            detection_flag_column="page_hinkley_change_flag",
            method_name="page_hinkley",
            split_name=split,
            healthy_rul_threshold=float(config["healthy_rul_threshold"]),
            critical_rul_threshold=float(config["critical_rul_threshold"]),
        )
        engine_summary_frames.append(ph_summary)
        engine_metrics[split]["page_hinkley"] = ph_metrics

    selection = summarize_validation_selection(row_metrics, engine_metrics)
    engine_summary = pd.concat(
        [frame.astype(object) for frame in engine_summary_frames if not frame.empty],
        ignore_index=True,
    )
    return row_metrics, engine_metrics, engine_summary, selection


def summarize_validation_selection(row_metrics: dict[str, Any], engine_metrics: dict[str, Any]) -> dict[str, Any]:
    validation = row_metrics["anomaly_detectors"]["validation"]

    def best_by(metric: str, higher: bool = True) -> str | None:
        candidates = {
            name: values.get(metric)
            for name, values in validation.items()
            if values.get(metric) is not None
        }
        if not candidates:
            return None
        return max(candidates, key=candidates.get) if higher else min(candidates, key=candidates.get)

    persistent_validation = engine_metrics["validation"]

    def best_engine(metric: str, higher: bool = True) -> str | None:
        candidates = {
            name: values.get(metric)
            for name, values in persistent_validation.items()
            if values.get(metric) is not None
        }
        if not candidates:
            return None
        return max(candidates, key=candidates.get) if higher else min(candidates, key=candidates.get)

    return {
        "best_validation_detector_by_pr_auc": best_by("pr_auc", True),
        "best_validation_detector_by_f1": best_by("f1", True),
        "lowest_validation_false_positive_rate": best_by("false_positive_rate", False),
        "highest_validation_engine_detection_rate": best_engine("detection_rate", True),
        "longest_validation_median_warning_lead_time": best_engine("median_lead_time", True),
    }


def cycle_level_output(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    wanted = [
        "split",
        UNIT_COLUMN,
        CYCLE_COLUMN,
        "true_rul_uncapped",
        "proxy_degradation_label",
        "proxy_critical_label",
        "health_index_raw",
        "health_index_scaled",
        "smoothed_health_index",
        "pca_reconstruction_error",
        "pca_anomaly_score",
        "pca_anomaly_flag",
        "pca_persistent_alarm_flag",
        "isolation_forest_score",
        "isolation_forest_flag",
        "isolation_forest_persistent_alarm_flag",
        "one_class_svm_score",
        "one_class_svm_flag",
        "one_class_svm_persistent_alarm_flag",
        "page_hinkley_change_flag",
    ]
    pieces = []
    for split, frame in frames.items():
        out = frame.copy()
        out["split"] = "train" if split == "model_train" else split
        pieces.append(out[[column for column in wanted if column in out.columns]])
    return pd.concat(pieces, ignore_index=True)


def _region_labels(frame: pd.DataFrame) -> list[str]:
    return ["healthy_proxy", "degradation_proxy", "critical_proxy"]


def create_figures(
    frames: dict[str, pd.DataFrame],
    row_metrics: dict[str, Any],
    engine_summary: pd.DataFrame,
    model_info: dict[str, Any],
    output_dir: Path,
    config: dict[str, Any],
) -> list[str]:
    figures_dir = output_dir / "figures"
    timelines_dir = output_dir / "engine_timelines"
    figures_dir.mkdir(parents=True, exist_ok=True)
    timelines_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    combined = pd.concat(frames.values(), ignore_index=True)
    validation = frames["validation"]

    path = figures_dir / "pca_explained_variance.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.cumsum(model_info["health_index_explained_variance_ratio"]), marker="o")
    ax.set_title("PCA Health Index Explained Variance")
    ax.set_xlabel("Component count")
    ax.set_ylabel("Cumulative explained variance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    path = figures_dir / "health_index_distribution_by_region.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    for region in _region_labels(combined):
        values = combined.loc[combined["proxy_health_region"] == region, "smoothed_health_index"]
        if len(values):
            ax.hist(values, bins=30, alpha=0.45, label=region)
    ax.set_title("Health Index Distribution by Proxy Region")
    ax.set_xlabel("Smoothed health index")
    ax.set_ylabel("Cycle count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    path = figures_dir / "health_index_vs_true_rul.png"
    fig, ax = plt.subplots(figsize=(7, 5))
    sample = combined.sample(min(len(combined), 5000), random_state=int(config["random_seed"]))
    ax.scatter(sample["true_rul_uncapped"], sample["smoothed_health_index"], s=6, alpha=0.35)
    ax.set_title("Health Index Versus True Uncapped RUL")
    ax.set_xlabel("True uncapped RUL (cycles)")
    ax.set_ylabel("Smoothed health index")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    path = figures_dir / "mean_health_index_normalized_life.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    temp = combined.copy()
    max_cycle = temp.groupby(UNIT_COLUMN)[CYCLE_COLUMN].transform("max")
    temp["normalized_life"] = temp[CYCLE_COLUMN] / max_cycle
    temp["life_bin"] = pd.cut(temp["normalized_life"], bins=np.linspace(0, 1, 21), include_lowest=True)
    mean_hi = temp.groupby("life_bin", observed=False)["smoothed_health_index"].mean()
    centers = [interval.mid for interval in mean_hi.index]
    ax.plot(centers, mean_hi.to_numpy(), marker="o")
    ax.set_title("Mean Health Index by Normalized Observed Life")
    ax.set_xlabel("Normalized observed life")
    ax.set_ylabel("Mean smoothed health index")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    path = figures_dir / "anomaly_score_distribution_by_region.png"
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (detector, columns) in zip(axes, DETECTORS.items()):
        for region in _region_labels(combined):
            values = combined.loc[combined["proxy_health_region"] == region, columns["score"]]
            if len(values):
                ax.hist(values, bins=25, alpha=0.42, label=region)
        ax.set_title(detector)
        ax.set_xlabel("Anomaly score")
    axes[0].set_ylabel("Cycle count")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Anomaly Score Distribution by Proxy Region")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    path = figures_dir / "detector_comparison.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(DETECTORS)
    f1 = [row_metrics["anomaly_detectors"]["validation"][name]["f1"] for name in names]
    pr_auc = [
        row_metrics["anomaly_detectors"]["validation"][name]["pr_auc"] or 0.0
        for name in names
    ]
    x = np.arange(len(names))
    ax.bar(x - 0.18, f1, width=0.36, label="Validation F1")
    ax.bar(x + 0.18, pr_auc, width=0.36, label="Validation PR-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Validation Detector Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    path = figures_dir / "validation_precision_recall_curves.png"
    fig, ax = plt.subplots(figsize=(7, 5))
    for detector, columns in DETECTORS.items():
        if validation["proxy_degradation_label"].nunique() == 2:
            PrecisionRecallDisplay.from_predictions(
                validation["proxy_degradation_label"],
                validation[columns["score"]],
                name=detector,
                ax=ax,
            )
    ax.set_title("Validation Precision-Recall Curves")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    path = figures_dir / "validation_roc_curves.png"
    fig, ax = plt.subplots(figsize=(7, 5))
    for detector, columns in DETECTORS.items():
        if validation["proxy_degradation_label"].nunique() == 2:
            RocCurveDisplay.from_predictions(
                validation["proxy_degradation_label"],
                validation[columns["score"]],
                name=detector,
                ax=ax,
            )
    ax.set_title("Validation ROC Curves")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    path = figures_dir / "detection_delay_distribution.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    delays = engine_summary["detection_delay"].dropna()
    ax.hist(delays, bins=25)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Detection Delay Distribution")
    ax.set_xlabel("Detection delay: detection cycle - proxy onset cycle")
    ax.set_ylabel("Engine-method count")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    path = figures_dir / "detection_lead_time_distribution.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    lead = engine_summary["lead_time_before_failure"].dropna()
    ax.hist(lead, bins=25)
    ax.set_title("Detection Lead-Time Distribution")
    ax.set_xlabel("True RUL at detection (cycles)")
    ax.set_ylabel("Engine-method count")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    path = figures_dir / "false_alarm_comparison.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    false_counts = engine_summary.groupby("method")["false_alarm"].sum().sort_values()
    ax.bar(false_counts.index, false_counts.values)
    ax.set_title("False-Alarm Engine Count by Method")
    ax.set_ylabel("False-alarm engines")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    timeline_count = int(config["representative_engine_count"])
    timeline_split = str(config.get("representative_timeline_split", "test"))
    timeline_frame = frames[timeline_split]
    selected_units = sorted(timeline_frame[UNIT_COLUMN].unique())[:timeline_count]
    for unit_id in selected_units:
        group = timeline_frame[timeline_frame[UNIT_COLUMN] == unit_id].sort_values(CYCLE_COLUMN)
        path = timelines_dir / f"{timeline_split}_engine_{int(unit_id):03d}_timeline.png"
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        axes[0].plot(group[CYCLE_COLUMN], group["true_rul_uncapped"], label="true RUL")
        axes[0].axhline(float(config["healthy_rul_threshold"]), color="orange", linestyle="--", label="proxy onset threshold")
        axes[0].axhline(float(config["critical_rul_threshold"]), color="red", linestyle="--", label="critical threshold")
        axes[0].set_ylabel("RUL cycles")
        axes[0].legend(fontsize=8)

        axes[1].plot(group[CYCLE_COLUMN], group["smoothed_health_index"], color="tab:green", label="smoothed health index")
        axes[1].set_ylabel("Health index")
        axes[1].legend(fontsize=8)

        for detector, columns in DETECTORS.items():
            axes[2].plot(group[CYCLE_COLUMN], group[columns["score"]], linewidth=1, label=f"{detector} score")
            alarm_cycles = group.loc[group[columns["persistent_flag"]], CYCLE_COLUMN]
            if len(alarm_cycles):
                axes[2].axvline(alarm_cycles.iloc[0], linestyle="--", linewidth=1, label=f"{detector} persistent")
        ph_cycles = group.loc[group["page_hinkley_change_flag"], CYCLE_COLUMN]
        if len(ph_cycles):
            axes[2].axvline(ph_cycles.iloc[0], color="black", linestyle=":", linewidth=1.2, label="Page-Hinkley")
        onset = group.loc[group["proxy_degradation_label"] == 1, CYCLE_COLUMN]
        if len(onset):
            axes[2].axvline(onset.iloc[0], color="orange", linestyle="-.", linewidth=1.2, label="proxy onset")
        axes[2].set_ylabel("Scores / onsets")
        axes[2].set_xlabel("Cycle")
        axes[2].legend(fontsize=7, ncol=2)
        fig.suptitle(f"{timeline_split.title()} Engine {int(unit_id)} Health and Anomaly Timeline")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(str(path))

    return written


def write_results_note(
    path: Path,
    result: dict[str, Any],
    config_path: Path,
) -> None:
    row_metrics = result["row_level_metrics"]["anomaly_detectors"]
    engine_metrics = result["engine_level_metrics"]
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# FD001 Health-Anomaly Results\n\n")
        handle.write("Proxy labels are based on true uncapped RUL thresholds and are not certified physical anomaly truth.\n\n")
        handle.write(f"- Python interpreter: `{result['python_executable']}`\n")
        handle.write(f"- Python version: `{result['python_version']}`\n")
        handle.write(f"- Dataset path: `{result['dataset_dir']}`\n")
        handle.write(f"- Train dimensions: `{result['train_shape']}`\n")
        handle.write(f"- Test dimensions: `{result['test_shape']}`\n")
        handle.write(f"- Model-training engines: `{result['model_train_engine_count']}`\n")
        handle.write(f"- Validation engines: `{result['validation_engine_count']}`\n")
        handle.write(f"- Test engines: `{result['test_engine_count']}`\n")
        handle.write(f"- Healthy training rows: `{result['healthy_training_row_count']}`\n")
        handle.write(f"- Runtime seconds: `{result['runtime_seconds']:.3f}`\n\n")
        handle.write("## Retained Features\n\n")
        handle.write(", ".join(result["retained_features"]) + "\n\n")
        handle.write("## Excluded Features\n\n")
        for feature, reason in result["excluded_features"].items():
            handle.write(f"- `{feature}`: {reason}\n")
        handle.write("\n## PCA Settings and Explained Variance\n\n")
        handle.write(f"- Health-index explained variance ratio: `{result['health_index_explained_variance_ratio']}`\n")
        handle.write(f"- PCA reconstruction explained variance ratio: `{result['pca_reconstruction_explained_variance_ratio']}`\n")
        handle.write(f"- PCA reconstruction threshold: `{result['pca_reconstruction_threshold']:.6f}`\n\n")
        handle.write("## Health-Index Statistics\n\n")
        for split, metrics in result["row_level_metrics"]["health_index"].items():
            handle.write(f"- `{split}`: {metrics}\n")
        handle.write("\n## Validation Anomaly Metrics\n\n")
        handle.write("| detector | precision | recall | f1 | specificity | FPR | PR-AUC | ROC-AUC |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for detector, metrics in row_metrics["validation"].items():
            handle.write(
                f"| {detector} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                f"{metrics['f1']:.4f} | {metrics['specificity']:.4f} | "
                f"{metrics['false_positive_rate']:.4f} | {metrics['pr_auc']:.4f} | "
                f"{metrics['roc_auc']:.4f} |\n"
            )
        handle.write("\n## Test Anomaly Metrics\n\n")
        handle.write("| detector | precision | recall | f1 | specificity | FPR | PR-AUC | ROC-AUC |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for detector, metrics in row_metrics["test"].items():
            handle.write(
                f"| {detector} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                f"{metrics['f1']:.4f} | {metrics['specificity']:.4f} | "
                f"{metrics['false_positive_rate']:.4f} | {metrics['pr_auc']:.4f} | "
                f"{metrics['roc_auc']:.4f} |\n"
            )
        handle.write("\n## Engine-Level Detection Metrics\n\n")
        for split, methods in engine_metrics.items():
            handle.write(f"### {split.title()}\n\n")
            for method, metrics in methods.items():
                handle.write(f"- `{method}`: {metrics}\n")
            handle.write("\n")
        handle.write("## Best Validation Detector by Criterion\n\n")
        for criterion, detector in result["validation_selection"].items():
            handle.write(f"- {criterion}: `{detector}`\n")
        handle.write("\n## Generated Outputs\n\n")
        for output in result["generated_outputs"]:
            handle.write(f"- `{output}`\n")
        handle.write("\n## Warnings and Limitations\n\n")
        for warning in result["warnings"]:
            handle.write(f"- {warning}\n")
        handle.write("\n## Exact Reproduction Command\n\n")
        handle.write("```powershell\n")
        handle.write('$env:PYTHONPATH = ".\\src"\n')
        handle.write(
            "python -m aeroguard.pipelines.train_fd001_health_anomaly "
            f'--config "{config_path.as_posix()}"\n'
        )
        handle.write("```\n")


def run_pipeline(config_path: str | Path) -> dict[str, Any]:
    start = time.perf_counter()
    root = project_root()
    config_path = Path(config_path)
    config = load_config(config_path)
    output_dir = resolve_project_path(config["output_dir"], root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "engine_timelines").mkdir(parents=True, exist_ok=True)

    model_train, validation, test, metadata = prepare_datasets(config)
    candidates, retained, reasons, feature_audit, correlation_audit = select_anomaly_features(
        model_train,
        include_cycle=bool(config["include_cycle_as_feature"]),
        configured_exclusions=list(config["features_to_exclude"]),
        near_constant_threshold=float(config["near_constant_threshold"]),
        correlation_threshold=float(config["correlation_threshold"]),
    )
    if not bool(config["include_cycle_as_feature"]):
        reasons[CYCLE_COLUMN] = "not configured as an anomaly-model feature"

    feature_audit_path = output_dir / "feature_audit.csv"
    correlation_audit_path = output_dir / "correlation_audit.csv"
    feature_audit.to_csv(feature_audit_path, index=False)
    correlation_audit.to_csv(correlation_audit_path, index=False)

    frames = {"model_train": model_train, "validation": validation, "test": test}
    frames, model_info = apply_health_and_detectors(frames, retained, config)
    row_metrics, engine_metrics, engine_summary, selection = compute_metrics(frames, config)

    cycle_output = cycle_level_output(frames)
    cycle_scores_path = output_dir / "cycle_level_scores.csv"
    engine_summary_path = output_dir / "engine_onset_summary.csv"
    row_metrics_path = output_dir / "row_level_metrics.json"
    engine_metrics_path = output_dir / "engine_level_metrics.json"
    feature_set_path = output_dir / "feature_set.json"

    cycle_output.to_csv(cycle_scores_path, index=False)
    engine_summary.to_csv(engine_summary_path, index=False)
    write_json(row_metrics_path, row_metrics)
    write_json(engine_metrics_path, engine_metrics)
    feature_set = {
        "candidate_features": candidates,
        "retained_features": retained,
        "excluded_features": reasons,
        "cycle_included": bool(config["include_cycle_as_feature"]),
        "healthy_training_row_count": model_info["healthy_training_row_count"],
        "scaler_fit_population_description": model_info["scaler_fit_population"],
    }
    write_json(feature_set_path, feature_set)

    figure_paths = create_figures(
        frames,
        row_metrics,
        engine_summary,
        model_info,
        output_dir,
        config,
    )
    runtime_seconds = time.perf_counter() - start
    generated_outputs = [
        str(feature_audit_path),
        str(correlation_audit_path),
        str(feature_set_path),
        str(row_metrics_path),
        str(engine_metrics_path),
        str(engine_summary_path),
        str(cycle_scores_path),
        *figure_paths,
    ]
    warnings = [
        "Proxy degradation labels use RUL thresholds and are not certified physical onset labels.",
        "Validation is used for detector comparison; final test metrics are reported only after validation comparison.",
        "All anomaly detectors and scalers were fitted using healthy rows from model-training engines only.",
    ]
    result = {
        **metadata,
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "retained_features": retained,
        "excluded_features": reasons,
        "healthy_training_row_count": model_info["healthy_training_row_count"],
        "health_index_explained_variance_ratio": model_info["health_index_explained_variance_ratio"],
        "pca_reconstruction_threshold": model_info["pca_reconstruction_threshold"],
        "pca_reconstruction_explained_variance_ratio": model_info["pca_reconstruction_explained_variance_ratio"],
        "row_level_metrics": row_metrics,
        "engine_level_metrics": engine_metrics,
        "validation_selection": selection,
        "generated_outputs": generated_outputs,
        "runtime_seconds": runtime_seconds,
        "warnings": warnings,
        "one_class_svm_subsampling_applied": model_info["one_class_svm_subsampling_applied"],
        "one_class_svm_fit_row_count": model_info["one_class_svm_fit_row_count"],
    }
    results_note_path = resolve_project_path(
        config.get("results_note_path", "notes/fd001_health_anomaly_results.md"),
        root,
    )
    write_results_note(results_note_path, result, config_path)
    result["generated_outputs"].append(str(results_note_path))
    write_json(output_dir / "run_summary.json", result)
    result["generated_outputs"].append(str(output_dir / "run_summary.json"))
    print(json.dumps(_json_ready(result), indent=2, allow_nan=False))
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train FD001 PCA health-index and classical anomaly baselines."
    )
    parser.add_argument("--config", required=True, help="Path to the Phase 2 YAML config.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
