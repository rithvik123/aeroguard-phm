import pandas as pd

from aeroguard.evaluation.temporal_model_stability import locked_epoch_from_cv, summarize_model_stability


def test_stability_summary_and_locked_epoch() -> None:
    metrics = pd.DataFrame(
        {
            "model_id": ["a", "a", "b"],
            "fold": ["f1", "f2", "f1"],
            "seed": [1, 2, 1],
            "best_epoch": [4, 6, 3],
            "validation_rmse": [10.0, 12.0, 15.0],
            "validation_mae": [8.0, 9.0, 10.0],
            "validation_nasa_score": [100.0, 120.0, 200.0],
            "validation_optimistic_rate": [0.2, 0.3, 0.4],
        }
    )

    stability = summarize_model_stability(metrics)
    epoch = locked_epoch_from_cv(metrics, "a")

    assert stability.iloc[0]["model_id"] == "a"
    assert epoch["locked_epoch_count"] == 5
    assert epoch["iqr_best_epoch"] == 1.0
