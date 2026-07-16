import numpy as np
import pandas as pd
import pytest

from aeroguard.anomaly.ensemble import fuse_scores, validate_weights, voting_flags


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pca": [0.1, 0.9, 0.5],
            "iso": [0.2, 0.8, 0.7],
            "svm": [0.3, 0.7, 0.9],
            "pca_flag": [False, True, False],
            "iso_flag": [False, True, True],
            "svm_flag": [False, False, True],
        }
    )


SCORES = {"pca_reconstruction": "pca", "isolation_forest": "iso", "one_class_svm": "svm"}
FLAGS = {"pca_reconstruction": "pca_flag", "isolation_forest": "iso_flag", "one_class_svm": "svm_flag"}


def test_mean_median_max_and_weighted_fusion() -> None:
    frame = _frame()

    assert np.allclose(fuse_scores(frame, SCORES, "mean"), [0.2, 0.8, 0.7])
    assert np.allclose(fuse_scores(frame, SCORES, "median"), [0.2, 0.8, 0.7])
    assert np.allclose(fuse_scores(frame, SCORES, "max"), [0.3, 0.9, 0.9])
    weighted = fuse_scores(
        frame,
        SCORES,
        "weighted_mean",
        {"pca_reconstruction": 0.0, "isolation_forest": 0.5, "one_class_svm": 0.5},
    )
    assert np.allclose(weighted, [0.25, 0.75, 0.8])


def test_invalid_weights_raise() -> None:
    with pytest.raises(ValueError, match="sum"):
        validate_weights({"pca_reconstruction": 0.5, "isolation_forest": 0.5, "one_class_svm": 0.5})


def test_voting_rules() -> None:
    frame = _frame()

    assert voting_flags(frame, FLAGS, "any_one").tolist() == [False, True, True]
    assert voting_flags(frame, FLAGS, "at_least_two").tolist() == [False, True, True]
    assert voting_flags(frame, FLAGS, "all_three").tolist() == [False, False, False]
