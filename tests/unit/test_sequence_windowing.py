import numpy as np
import pandas as pd
import pytest

from aeroguard.deep.windowing import (
    WindowSpec,
    build_inference_windows,
    build_training_windows,
    build_window_from_endpoint,
    build_windows,
    candidate_endpoint_indices,
    endpoints_for_normalized_positions,
    final_endpoint_table,
)


def _engine_frame(length: int = 5) -> pd.DataFrame:
    rows = []
    for cycle in range(1, length + 1):
        rows.append(
            {
                "subset": "FD001",
                "source_domain": "FD001",
                "global_engine_id": "FD001_0001",
                "local_unit_id": 1,
                "unit_id": 1,
                "cycle": cycle,
                "sensor_1": float(cycle),
                "sensor_2": float(cycle * 10),
                "rul_capped": float(length - cycle),
                "true_rul_uncapped": float(length - cycle),
                "operating_regime": 0,
                "proxy_health_region": "degradation_proxy",
            }
        )
    return pd.DataFrame(rows)


def test_candidate_endpoint_indices_are_past_only_and_include_final_cycle() -> None:
    spec = WindowSpec(window_length=4, stride=2, minimum_valid_history=2)

    assert candidate_endpoint_indices(_engine_frame(5), spec) == [1, 3, 4]
    assert candidate_endpoint_indices(_engine_frame(1), spec) == []


def test_build_window_left_pads_and_masks_history() -> None:
    spec = WindowSpec(window_length=4, stride=1, minimum_valid_history=1)
    frame = _engine_frame(5)

    window, valid_length, padded_count = build_window_from_endpoint(frame, ["sensor_1", "sensor_2"], 2, spec)

    assert window.shape == (4, 3)
    assert valid_length == 3
    assert padded_count == 1
    np.testing.assert_allclose(window[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(window[-1], [3.0, 30.0, 1.0])
    np.testing.assert_allclose(window[:, -1], [0.0, 1.0, 1.0, 1.0])


def test_build_windows_returns_metadata_at_endpoint_without_future_rows() -> None:
    spec = WindowSpec(window_length=4, stride=1, minimum_valid_history=1)
    frame = _engine_frame(5)
    endpoints = pd.DataFrame([{"global_engine_id": "FD001_0001", "endpoint_index": 2}])

    sequences, metadata = build_windows(frame, ["sensor_1", "sensor_2"], endpoints, spec)

    assert sequences.shape == (1, 4, 3)
    assert metadata.loc[0, "cycle"] == 3
    assert metadata.loc[0, "target_rul_uncapped"] == 2.0
    np.testing.assert_allclose(sequences[0, -1, :-1], [3.0, 30.0])


def test_training_windows_require_rul_capped() -> None:
    spec = WindowSpec(window_length=4, stride=1, minimum_valid_history=1)
    frame = _engine_frame(5).drop(columns=["rul_capped"])
    endpoints = pd.DataFrame([{"global_engine_id": "FD001_0001", "endpoint_index": 2}])

    with pytest.raises(ValueError, match="Training windows require target"):
        build_training_windows(frame, ["sensor_1", "sensor_2"], endpoints, spec)


def test_inference_windows_do_not_require_any_rul_column() -> None:
    spec = WindowSpec(window_length=4, stride=1, minimum_valid_history=1)
    frame = _engine_frame(3).drop(columns=["rul_capped", "true_rul_uncapped"])
    endpoints = pd.DataFrame([{"global_engine_id": "FD001_0001", "endpoint_index": 2}])

    sequences, metadata = build_inference_windows(frame, ["sensor_1", "sensor_2"], endpoints, spec)

    assert sequences.shape == (1, 4, 3)
    assert "target_rul_capped" not in metadata.columns
    assert metadata.loc[0, "cycle"] == 3
    assert metadata.loc[0, "padded_cycle_count"] == 1


def test_normalized_endpoint_positions_and_final_endpoint_table() -> None:
    frame = _engine_frame(5)

    assert endpoints_for_normalized_positions(frame, [0.25, 1.0]) == [1, 4]
    assert final_endpoint_table(frame).to_dict(orient="records") == [
        {"global_engine_id": "FD001_0001", "endpoint_index": 4}
    ]
    with pytest.raises(ValueError, match="Normalized-life"):
        endpoints_for_normalized_positions(frame, [0.0])
