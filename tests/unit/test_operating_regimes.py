import pandas as pd
import pytest

from aeroguard.data.operating_regimes import OperatingRegimeModel, assign_operating_regimes, regime_counts


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "operational_setting_1": [0.0, 0.1, 0.2, 2.0, 2.1, 2.2],
            "operational_setting_2": [1.0, 1.1, 1.2, 3.0, 3.1, 3.2],
            "operational_setting_3": [0.5, 0.4, 0.6, 1.5, 1.6, 1.4],
        }
    )


def test_operating_regime_model_assigns_deterministic_regimes() -> None:
    frame = _frame()
    model = OperatingRegimeModel(n_regimes=3, random_state=7).fit(frame)

    first = model.predict(frame)
    second = OperatingRegimeModel(n_regimes=3, random_state=7).fit(frame).predict(frame)

    assert first.tolist() == second.tolist()
    assert set(first).issubset({0, 1, 2})
    assert model.metadata()["n_regimes_requested"] == 3


def test_assign_operating_regimes_and_counts() -> None:
    frame = _frame()
    model = OperatingRegimeModel(n_regimes=2, random_state=3).fit(frame)

    assigned = assign_operating_regimes(frame, model)
    counts = regime_counts(assigned)

    assert "operating_regime" in assigned.columns
    assert sum(counts.values()) == len(frame)


def test_operating_regime_model_rejects_invalid_regime_count() -> None:
    with pytest.raises(ValueError, match="n_regimes"):
        OperatingRegimeModel(n_regimes=0).fit(_frame())
