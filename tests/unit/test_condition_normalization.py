import numpy as np
import pandas as pd

from aeroguard.features.condition_normalization import ConditionNormalizer


def _frame() -> pd.DataFrame:
    rows = []
    for idx in range(12):
        op1 = float(idx % 4)
        op2 = float(idx // 4)
        op3 = float((idx % 3) * 0.5)
        rows.append(
            {
                "operational_setting_1": op1,
                "operational_setting_2": op2,
                "operational_setting_3": op3,
                "sensor_2": 10.0 + 2.0 * op1 + 0.5 * op2 + 0.1 * idx,
                "sensor_3": 20.0 - op1 + 0.2 * op3 + 0.05 * idx,
                "proxy_degradation_label": 0 if idx < 8 else 1,
            }
        )
    return pd.DataFrame(rows)


def test_global_standardization_uses_healthy_training_rows() -> None:
    frame = _frame()
    normalizer = ConditionNormalizer(method="global_standardization").fit(frame, ["sensor_2", "sensor_3"])

    transformed = normalizer.transform(frame)
    healthy = transformed[transformed["proxy_degradation_label"] == 0]

    assert abs(float(healthy["sensor_2_global_z"].mean())) < 1.0e-12
    assert np.isfinite(transformed[["sensor_2_global_z", "sensor_3_global_z"]].to_numpy()).all()


def test_regime_standardization_creates_finite_features() -> None:
    frame = _frame()
    normalizer = ConditionNormalizer(method="regime_standardization", n_regimes=3, random_state=11).fit(frame, ["sensor_2"])

    transformed = normalizer.transform(frame)

    assert {"operating_regime", "sensor_2_regime_z"}.issubset(transformed.columns)
    assert np.isfinite(transformed["sensor_2_regime_z"]).all()


def test_residualization_removes_linear_operating_condition_signal() -> None:
    frame = _frame()
    normalizer = ConditionNormalizer(method="residualization", ridge_alpha=0.01).fit(frame, ["sensor_2"])

    transformed = normalizer.transform(frame)
    healthy_residuals = transformed.loc[transformed["proxy_degradation_label"] == 0, "sensor_2_condition_residual"]

    assert "sensor_2_condition_residual" in transformed.columns
    assert abs(float(healthy_residuals.mean())) < 0.2
