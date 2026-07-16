import pandas as pd
import pytest

from aeroguard.uncertainty.support import SupportModel


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "f1": [0.0, 0.1, 0.2, 0.3, 0.4],
            "f2": [1.0, 1.1, 1.2, 1.3, 1.4],
            "operating_regime": [0, 0, 1, 1, 1],
        }
    )


def test_support_model_scores_in_and_out_of_support_rows() -> None:
    model = SupportModel(["f1", "f2"], feature_exceedance_out=0.4).fit(_frame())
    scores = model.score(pd.DataFrame({"f1": [0.2, 9.0], "f2": [1.2, 9.0], "operating_regime": [1, 1]}))

    assert scores["support_status"].tolist()[0] in {"IN_SUPPORT", "LIMITED_SUPPORT"}
    assert scores["support_status"].tolist()[1] == "OUT_OF_SUPPORT"
    assert scores["support_score"].iloc[0] <= scores["support_score"].iloc[1]


def test_support_model_rejects_missing_feature() -> None:
    model = SupportModel(["f1", "f2"]).fit(_frame())

    with pytest.raises(ValueError):
        model.score(pd.DataFrame({"f1": [1.0]}))
