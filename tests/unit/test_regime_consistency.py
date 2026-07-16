import inspect

import numpy as np
import pandas as pd
import pytest
import torch

from aeroguard.deep.physics.regime_consistency import (
    RegimePairingConfig,
    build_regime_pairs,
    latent_consistency_loss,
    prediction_consistency_loss,
)


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_index": [0, 1, 2, 3, 4, 5],
            "target_rul_capped": [10.0, 11.0, 30.0, 10.5, 11.2, 50.0],
            "operating_regime": [0, 1, 0, 2, 2, 1],
            "sequence_valid_length": [5, 5, 5, 5, 5, 5],
            "global_engine_id": ["FD001_1", "FD002_1", "FD001_2", "FD003_1", "FD004_1", "FD002_2"],
            "subset": ["FD001", "FD002", "FD001", "FD003", "FD004", "FD002"],
        }
    )


def test_valid_cross_regime_pairs_follow_contract_and_tolerance() -> None:
    pairs = build_regime_pairs(
        _metadata(),
        RegimePairingConfig(enabled=True, rul_tolerance=2.0, max_pairs=10, sampling_method="first", max_partners_per_anchor=2),
    )

    assert len(pairs) <= 10
    assert (pairs["left_operating_regime"] != pairs["right_operating_regime"]).all()
    assert (pairs["target_rul_difference"] <= 2.0).all()
    assert (pairs["left_valid_length"] > 0).all()
    assert (pairs["right_valid_length"] > 0).all()
    assert set(
        [
            "left_index",
            "right_index",
            "left_global_engine_id",
            "right_global_engine_id",
            "left_subset",
            "right_subset",
            "left_operating_regime",
            "right_operating_regime",
            "left_target_rul",
            "right_target_rul",
            "target_rul_difference",
            "left_valid_length",
            "right_valid_length",
        ]
    ).issubset(pairs.columns)


def test_deterministic_and_no_duplicate_unordered_pairs() -> None:
    config = RegimePairingConfig(enabled=True, rul_tolerance=2.0, max_pairs=20, seed=4, max_partners_per_anchor=2)

    first = build_regime_pairs(_metadata(), config)
    second = build_regime_pairs(_metadata(), config)
    unordered = first[["left_index", "right_index"]].apply(lambda row: tuple(sorted((int(row.iloc[0]), int(row.iloc[1])))), axis=1)

    pd.testing.assert_frame_equal(first, second)
    assert unordered.nunique() == len(first)
    assert not (first["left_index"] == first["right_index"]).any()


def test_caps_are_enforced_before_unbounded_expansion() -> None:
    rows = []
    for index in range(300):
        rows.append(
            {
                "sample_index": index,
                "target_rul_capped": float(index % 25),
                "operating_regime": index % 3,
                "sequence_valid_length": 10,
                "global_engine_id": f"engine_{index // 10}",
                "subset": "FD001",
            }
        )
    pairs = build_regime_pairs(
        pd.DataFrame(rows),
        RegimePairingConfig(
            enabled=True,
            rul_tolerance=5.0,
            max_pairs=7,
            max_anchors=12,
            max_partners_per_anchor=1,
            max_pairs_per_regime_pair=3,
            seed=7,
        ),
    )
    diagnostics = pairs.attrs["diagnostics"]

    assert len(pairs) <= 7
    assert diagnostics["anchor_count_considered"] <= 12
    assert all(count <= 3 for count in diagnostics["regime_pair_limit_counts"].values())
    assert diagnostics["limit_reached"] is True


def test_empty_sparse_cases_are_schema_valid_or_clear_errors() -> None:
    one_regime = _metadata().assign(operating_regime=0)
    optional = build_regime_pairs(one_regime, RegimePairingConfig(enabled=True, allow_empty_pairs=True))

    assert optional.empty
    assert list(optional.columns)
    assert optional.attrs["diagnostics"]["empty_reason"] == "only_one_regime"
    with pytest.raises(ValueError, match="at least two operating regimes"):
        build_regime_pairs(one_regime, RegimePairingConfig(enabled=True, allow_empty_pairs=False))

    sparse = build_regime_pairs(_metadata(), RegimePairingConfig(enabled=True, rul_tolerance=0.01, allow_empty_pairs=True))
    assert sparse.empty
    assert sparse.attrs["diagnostics"]["empty_reason"] == "no_cross_regime_rul_matches"


def test_compact_dtypes_and_no_metadata_mutation() -> None:
    metadata = _metadata()
    original = metadata.copy(deep=True)

    pairs = build_regime_pairs(metadata, RegimePairingConfig(enabled=True, rul_tolerance=2.0, max_pairs=10, sampling_method="first"))

    pd.testing.assert_frame_equal(metadata, original)
    assert str(pairs["left_index"].dtype) in {"int32", "int64"}
    assert str(pairs["left_operating_regime"].dtype) == "int16"
    assert str(pairs["target_rul_difference"].dtype) == "float32"
    assert str(pairs["left_subset"].dtype) == "category"


def test_benchmark_and_test_rows_are_rejected() -> None:
    with pytest.raises(ValueError, match="Benchmark/test"):
        build_regime_pairs(
            _metadata().assign(data_role=["training", "benchmark_test", "training", "training", "training", "training"]),
            RegimePairingConfig(enabled=True),
        )
    with pytest.raises(ValueError, match="Benchmark/test"):
        build_regime_pairs(_metadata().assign(subset=["FD001", "test_FD001", "FD001", "FD003", "FD004", "FD002"]), RegimePairingConfig(enabled=True))


def test_bounded_stress_100k_rows_is_capped_and_diagnostic() -> None:
    size = 100_000
    rng = np.random.default_rng(123)
    metadata = pd.DataFrame(
        {
            "sample_index": np.arange(size, dtype=np.int32),
            "target_rul_capped": rng.integers(0, 126, size=size).astype(np.float32),
            "operating_regime": rng.integers(0, 6, size=size).astype(np.int16),
            "sequence_valid_length": rng.integers(10, 51, size=size).astype(np.int16),
            "global_engine_id": np.asarray([f"engine_{idx // 100}" for idx in range(size)], dtype=object),
            "subset": np.asarray(["FD001"] * size, dtype=object),
        }
    )

    pairs = build_regime_pairs(
        metadata,
        RegimePairingConfig(
            enabled=True,
            rul_tolerance=1.0,
            max_pairs=1_000,
            max_anchors=600,
            max_partners_per_anchor=2,
            max_pairs_per_regime_pair=200,
            seed=42,
        ),
    )
    diagnostics = pairs.attrs["diagnostics"]

    assert len(pairs) <= 1_000
    assert diagnostics["metadata_rows"] == size
    assert diagnostics["anchor_count_considered"] <= 600
    assert diagnostics["candidate_matches_examined"] < size * 600
    assert diagnostics["pair_table_memory_mb"] < 5.0
    assert diagnostics["algorithm"] == "bounded_rul_searchsorted"


def test_source_does_not_contain_known_quadratic_generation_patterns() -> None:
    source = inspect.getsource(build_regime_pairs)

    for token in [".merge(", "how=\"cross\"", "how='cross'", "np.subtract.outer", "np.meshgrid", "itertools.product", "[:, None]", "[None, :]"]:
        assert token not in source


def test_latent_distance_and_prediction_consistency_losses() -> None:
    left = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    right = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    pred_left = torch.tensor([[10.0], [10.0]])
    pred_right = torch.tensor([[10.0], [13.0]])

    assert latent_consistency_loss(left[:1], right[:1]).item() == 0.0
    assert latent_consistency_loss(left, right).item() > 0.0
    assert prediction_consistency_loss(pred_left, pred_right, tolerance=1.0).item() > 0.0
