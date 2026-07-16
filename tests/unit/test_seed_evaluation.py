import pandas as pd

from aeroguard.deep.seed_evaluation import aggregate_seed_metrics, prediction_disagreement


def test_seed_aggregation_records_best_and_worst_seed() -> None:
    metrics = pd.DataFrame(
        {
            "model_id": ["a", "a", "a"],
            "seed": [1, 2, 3],
            "validation_rmse": [3.0, 2.0, 4.0],
            "validation_mae": [2.0, 1.5, 2.5],
            "validation_nasa_score": [10.0, 9.0, 12.0],
            "validation_optimistic_rate": [0.2, 0.1, 0.3],
        }
    )

    summary = aggregate_seed_metrics(metrics)

    assert summary.loc[0, "best_seed"] == 2
    assert summary.loc[0, "worst_seed"] == 3
    assert summary.loc[0, "run_count"] == 3


def test_prediction_disagreement_aggregates_seed_predictions() -> None:
    predictions = pd.DataFrame(
        {
            "model_id": ["a", "a"],
            "fold": ["f1", "f1"],
            "global_engine_id": ["e1", "e1"],
            "cycle": [10, 10],
            "predicted_rul": [5.0, 7.0],
        }
    )

    disagreement = prediction_disagreement(predictions)

    assert disagreement.loc[0, "prediction_mean"] == 6.0
    assert disagreement.loc[0, "prediction_range"] == 2.0

