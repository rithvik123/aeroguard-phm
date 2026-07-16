import copy

import pytest

from aeroguard.deep.physics.candidate_registry import default_candidate_registry, validate_candidate_registry


def test_default_registry_unique_ids_and_maximum_count() -> None:
    registry = default_candidate_registry()
    ids = [candidate["candidate_id"] for candidate in registry]

    assert len(registry) <= 10
    assert len(ids) == len(set(ids))
    assert "phase5b_reimplementation_baseline" in ids


def test_required_heads_for_active_losses() -> None:
    candidate = copy.deepcopy(default_candidate_registry()[4])
    candidate["active_output_heads"] = []

    with pytest.raises(ValueError, match="health"):
        validate_candidate_registry([default_candidate_registry()[0], candidate])


def test_required_pair_builders_for_active_losses() -> None:
    candidate = copy.deepcopy(default_candidate_registry()[1])
    candidate["pairing_requirements"] = []

    with pytest.raises(ValueError, match="monotonic"):
        validate_candidate_registry([default_candidate_registry()[0], candidate])


def test_duplicate_ids_and_too_many_candidates_rejected() -> None:
    registry = default_candidate_registry()
    duplicate = copy.deepcopy(registry[0])

    with pytest.raises(ValueError, match="unique"):
        validate_candidate_registry([registry[0], duplicate])
    with pytest.raises(ValueError, match="Too many"):
        validate_candidate_registry(registry + [copy.deepcopy(registry[0]), copy.deepcopy(registry[1])], max_candidates=10)
