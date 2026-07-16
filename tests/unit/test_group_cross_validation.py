import pytest

from aeroguard.evaluation.group_cross_validation import repeated_group_kfold_splits, validate_group_folds


def test_group_folds_have_no_overlap_and_full_repeat_coverage() -> None:
    groups = list(range(1, 11))
    folds = repeated_group_kfold_splits(groups, n_splits=5, n_repeats=2, seeds=[11, 12])

    assert len(folds) == 10
    for fold in folds:
        assert not set(fold.train_groups).intersection(fold.validation_groups)
    validate_group_folds(folds, groups, n_repeats=2)


def test_group_folds_are_deterministic() -> None:
    first = repeated_group_kfold_splits(range(1, 8), n_splits=3, n_repeats=2, seeds=[3, 4])
    second = repeated_group_kfold_splits(range(1, 8), n_splits=3, n_repeats=2, seeds=[3, 4])

    assert [fold.to_dict() for fold in first] == [fold.to_dict() for fold in second]


def test_repeated_splits_are_independent() -> None:
    folds = repeated_group_kfold_splits(range(1, 16), n_splits=5, n_repeats=2, seeds=[100, 200])

    repeat_one = [tuple(fold.validation_groups) for fold in folds if fold.repeat == 1]
    repeat_two = [tuple(fold.validation_groups) for fold in folds if fold.repeat == 2]
    assert repeat_one != repeat_two


def test_invalid_fold_count_raises() -> None:
    with pytest.raises(ValueError, match="n_splits"):
        repeated_group_kfold_splits([1, 2, 3], n_splits=1, n_repeats=1, seeds=[1])
