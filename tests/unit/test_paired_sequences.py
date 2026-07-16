import pandas as pd

from aeroguard.deep.physics.paired_sequences import TemporalPairingConfig, build_temporal_pairs, pair_indices, triplet_indices


def _metadata() -> pd.DataFrame:
    rows = []
    for engine in ["e1", "e2"]:
        for cycle in range(1, 7):
            rows.append(
                {
                    "sample_index": len(rows),
                    "subset": "train_FD001",
                    "global_engine_id": engine,
                    "cycle": cycle,
                    "target_rul_capped": 6 - cycle,
                    "operating_regime": cycle % 2,
                    "sequence_valid_length": cycle,
                }
            )
    return pd.DataFrame(rows)


def test_same_engine_enforcement_and_no_cross_engine_pairs() -> None:
    pairs = build_temporal_pairs(_metadata(), TemporalPairingConfig(allowed_cycle_gaps=(1, 2), sampling_method="first"))

    assert not pairs.empty
    assert (pairs["global_engine_id"].isin(["e1", "e2"])).all()
    assert (pairs["later_cycle"] > pairs["earlier_cycle"]).all()


def test_deterministic_pair_sampling_and_maximum_pairs() -> None:
    config = TemporalPairingConfig(allowed_cycle_gaps=(1, 2), max_adjacent_pairs_per_engine=2, max_fixed_gap_pairs_per_engine=1, max_triplets_per_engine=1, seed=10)

    first = build_temporal_pairs(_metadata(), config)
    second = build_temporal_pairs(_metadata(), config)

    pd.testing.assert_frame_equal(first, second)
    assert (first[first["pair_type"] == "adjacent"].groupby("global_engine_id").size() <= 2).all()


def test_pair_and_triplet_indices() -> None:
    pairs = build_temporal_pairs(_metadata(), TemporalPairingConfig(allowed_cycle_gaps=(1, 2), sampling_method="first"))

    assert pair_indices(pairs).shape[1] == 2
    assert triplet_indices(pairs).shape[1] == 3
    triplets = pairs[pairs["pair_type"] == "triplet"]
    assert (triplets["earlier_cycle"] < triplets["middle_cycle"]).all()
    assert (triplets["middle_cycle"] < triplets["later_cycle"]).all()


def test_no_padded_only_samples_are_paired() -> None:
    metadata = _metadata()
    metadata.loc[0, "sequence_valid_length"] = 0

    pairs = build_temporal_pairs(metadata, TemporalPairingConfig(allowed_cycle_gaps=(1,), sampling_method="first"))

    assert 0 not in set(pairs["earlier_index"]).union(set(pairs["later_index"]))
