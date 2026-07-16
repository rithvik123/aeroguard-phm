"""Bounded alert-policy registry validation and application."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd

from aeroguard.anomaly.alerting import (
    apply_hysteresis_alert,
    apply_persistence_rule,
    assign_operational_alert_levels,
)
from aeroguard.anomaly.ensemble import DETECTOR_ORDER, fuse_scores, validate_weights, voting_flags, voting_score


CALIBRATED_SCORE_COLUMNS = {
    name: f"{name}_calibrated_score" for name in DETECTOR_ORDER
}

REQUIRED_POLICY_FIELDS = {
    "policy_id",
    "calibration_method",
    "fusion_method",
    "threshold",
    "persistence",
    "hysteresis",
    "operational_profile",
}

SUPPORTED_FUSION_METHODS = {
    "max",
    "mean",
    "median",
    "weighted_mean",
    "detector",
    "voting",
}


def persistence_name(rule: dict[str, Any]) -> str:
    kind = rule.get("type")
    if kind == "consecutive":
        return f"consecutive_{int(rule['k'])}"
    if kind == "k_of_n":
        return f"{int(rule['k'])}_of_{int(rule['n'])}"
    if kind == "score_duration":
        return f"score_duration_{int(rule['duration'])}_{float(rule['threshold'])}"
    raise ValueError("Unsupported persistence rule type.")


def validate_persistence_rule(rule: dict[str, Any]) -> dict[str, Any]:
    kind = rule.get("type")
    if kind == "consecutive":
        if int(rule.get("k", 0)) <= 0:
            raise ValueError("Consecutive persistence requires positive k.")
    elif kind == "k_of_n":
        k = int(rule.get("k", 0))
        n = int(rule.get("n", 0))
        if k <= 0 or n <= 0 or k > n:
            raise ValueError("K-of-N persistence requires 0 < k <= n.")
    elif kind == "score_duration":
        duration = int(rule.get("duration", 0))
        threshold = float(rule.get("threshold", -1))
        if duration <= 0 or not 0 <= threshold <= 1:
            raise ValueError("Score-duration persistence requires positive duration and threshold in [0, 1].")
    else:
        raise ValueError("Persistence type must be consecutive, k_of_n, or score_duration.")
    clean = deepcopy(rule)
    clean["name"] = str(rule.get("name") or persistence_name(rule))
    return clean


def validate_hysteresis(config: dict[str, Any]) -> dict[str, Any]:
    enter = float(config["enter_threshold"])
    exit_ = float(config["exit_threshold"])
    if not 0 <= exit_ < enter <= 1:
        raise ValueError("Hysteresis requires 0 <= exit_threshold < enter_threshold <= 1.")
    min_enter = int(config["min_enter_duration"])
    min_clear = int(config["min_clear_duration"])
    if min_enter <= 0 or min_clear <= 0:
        raise ValueError("Hysteresis minimum durations must be positive.")
    return {
        "enter_threshold": enter,
        "exit_threshold": exit_,
        "min_enter_duration": min_enter,
        "min_clear_duration": min_clear,
    }


def validate_policy_registry(
    policies: list[dict[str, Any]],
    maximum_policy_count: int,
    operational_profiles: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    """Validate and normalize a fixed alert-policy registry."""
    if maximum_policy_count <= 0:
        raise ValueError("maximum_policy_count must be positive.")
    if len(policies) > maximum_policy_count:
        raise ValueError("Candidate policy count exceeds maximum_policy_count.")
    seen: set[str] = set()
    resolved: list[dict[str, Any]] = []
    for policy in policies:
        missing = sorted(REQUIRED_POLICY_FIELDS - set(policy))
        if missing:
            raise ValueError(f"Policy is missing required fields: {missing}")
        policy_id = str(policy["policy_id"])
        if policy_id in seen:
            raise ValueError(f"Duplicate policy_id: {policy_id}")
        seen.add(policy_id)
        method = str(policy["fusion_method"])
        if method not in SUPPORTED_FUSION_METHODS:
            raise ValueError(f"Unsupported fusion_method for {policy_id}: {method}")
        threshold = float(policy["threshold"])
        if not 0 <= threshold <= 1:
            raise ValueError(f"Policy {policy_id} threshold must be in [0, 1].")
        profile = str(policy["operational_profile"])
        if profile not in operational_profiles:
            raise ValueError(f"Policy {policy_id} references missing operational profile {profile}.")
        clean = deepcopy(policy)
        clean["policy_id"] = policy_id
        clean["fusion_method"] = method
        clean["threshold"] = threshold
        clean["persistence"] = validate_persistence_rule(dict(policy["persistence"]))
        clean["hysteresis"] = validate_hysteresis(dict(policy["hysteresis"]))
        if method == "weighted_mean":
            clean["weights"] = validate_weights({name: policy["weights"][name] for name in DETECTOR_ORDER})
        elif method == "detector":
            detector = str(policy.get("detector", ""))
            if detector not in DETECTOR_ORDER:
                raise ValueError(f"Policy {policy_id} detector must be one of {DETECTOR_ORDER}.")
            clean["detector"] = detector
        elif method == "voting":
            rule = str(policy.get("voting_rule", ""))
            if rule not in {"any_one", "at_least_two", "all_three"}:
                raise ValueError(f"Policy {policy_id} has invalid voting_rule.")
            clean["voting_rule"] = rule
        elif "weights" in clean:
            validate_weights({name: clean["weights"][name] for name in DETECTOR_ORDER})
        resolved.append(clean)
    return resolved


def policy_score_and_flag(frame: pd.DataFrame, policy: dict[str, Any]) -> tuple[pd.Series, pd.Series]:
    """Return score and raw anomaly flag for one policy."""
    method = str(policy["fusion_method"])
    threshold = float(policy["threshold"])
    if method in {"max", "mean", "median"}:
        score = pd.Series(fuse_scores(frame, CALIBRATED_SCORE_COLUMNS, method), index=frame.index)
        flag = score >= threshold
    elif method == "weighted_mean":
        score = pd.Series(fuse_scores(frame, CALIBRATED_SCORE_COLUMNS, method, policy["weights"]), index=frame.index)
        flag = score >= threshold
    elif method == "detector":
        score = frame[CALIBRATED_SCORE_COLUMNS[str(policy["detector"])]].astype(float)
        flag = score >= threshold
    elif method == "voting":
        flag_columns = {}
        for detector in DETECTOR_ORDER:
            column = f"policy_vote_{detector}_flag"
            flag_columns[detector] = column
            frame[column] = frame[CALIBRATED_SCORE_COLUMNS[detector]].astype(float) >= threshold
        score = pd.Series(voting_score(frame, flag_columns), index=frame.index)
        flag = pd.Series(voting_flags(frame, flag_columns, str(policy["voting_rule"])), index=frame.index)
    else:
        raise ValueError(f"Unsupported policy fusion method: {method}")
    return score.astype(float), flag.astype(bool)


def apply_alert_policy(
    frame: pd.DataFrame,
    policy: dict[str, Any],
    operational_alert_thresholds: dict[str, float],
    output_prefix: str = "locked",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply one policy and produce raw, persistent, hysteresis, and level columns."""
    result = frame.copy()
    score_col = f"{output_prefix}_ensemble_score"
    flag_col = f"{output_prefix}_raw_anomaly_flag"
    result[score_col], result[flag_col] = policy_score_and_flag(result, policy)
    result, persistence_summary = apply_persistence_rule(
        result,
        flag_column=flag_col,
        score_column=score_col,
        output_prefix=output_prefix,
        rule=dict(policy["persistence"]),
    )
    hysteresis = dict(policy["hysteresis"])
    result, hysteresis_summary = apply_hysteresis_alert(
        result,
        score_column=score_col,
        output_prefix=output_prefix,
        enter_threshold=float(hysteresis["enter_threshold"]),
        exit_threshold=float(hysteresis["exit_threshold"]),
        min_enter_duration=int(hysteresis["min_enter_duration"]),
        min_clear_duration=int(hysteresis["min_clear_duration"]),
    )
    result = assign_operational_alert_levels(
        result,
        score_column=score_col,
        persistent_column=f"{output_prefix}_persistent_alarm_state",
        health_column="smoothed_health_index",
        output_column=f"{output_prefix}_operational_alert_level",
        thresholds=operational_alert_thresholds,
    )
    summary = persistence_summary.merge(hysteresis_summary, on="unit_id", how="outer")
    return result, summary
