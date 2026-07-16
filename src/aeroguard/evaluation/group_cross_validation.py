"""Deterministic repeated group cross-validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class GroupFold:
    """One group-based train/validation split."""

    repeat: int
    fold: int
    seed: int
    train_groups: list[int]
    validation_groups: list[int]

    def to_dict(self) -> dict[str, object]:
        return {
            "repeat": self.repeat,
            "fold": self.fold,
            "seed": self.seed,
            "train_groups": self.train_groups,
            "validation_groups": self.validation_groups,
        }


def _clean_groups(groups: Iterable[object]) -> list[int]:
    values = sorted({int(group) for group in groups})
    if not values:
        raise ValueError("At least one group is required.")
    return values


def repeated_group_kfold_splits(
    groups: Iterable[object],
    n_splits: int,
    n_repeats: int,
    seeds: Iterable[int],
) -> list[GroupFold]:
    """Create deterministic repeated K-fold splits over complete groups."""
    unique_groups = _clean_groups(groups)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")
    if n_splits > len(unique_groups):
        raise ValueError("n_splits cannot exceed the number of groups.")
    seed_values = [int(seed) for seed in seeds]
    if len(seed_values) != n_repeats:
        raise ValueError("Number of seeds must equal n_repeats.")

    folds: list[GroupFold] = []
    for repeat_index, seed in enumerate(seed_values, start=1):
        rng = np.random.default_rng(seed)
        shuffled = np.array(unique_groups, dtype=int)
        rng.shuffle(shuffled)
        validation_parts = np.array_split(shuffled, n_splits)
        for fold_index, validation in enumerate(validation_parts, start=1):
            validation_set = sorted(int(group) for group in validation.tolist())
            validation_lookup = set(validation_set)
            train_set = [group for group in unique_groups if group not in validation_lookup]
            if validation_lookup.intersection(train_set):
                raise RuntimeError("Group overlap detected between train and validation split.")
            folds.append(
                GroupFold(
                    repeat=repeat_index,
                    fold=fold_index,
                    seed=seed,
                    train_groups=train_set,
                    validation_groups=validation_set,
                )
            )
    return folds


def validate_group_folds(folds: list[GroupFold], expected_groups: Iterable[object], n_repeats: int) -> None:
    """Validate non-overlap and per-repeat validation coverage."""
    expected = set(_clean_groups(expected_groups))
    for fold in folds:
        train = set(fold.train_groups)
        validation = set(fold.validation_groups)
        if train.intersection(validation):
            raise ValueError(f"Fold {fold.repeat}-{fold.fold} has train/validation group overlap.")
        if train.union(validation) != expected:
            raise ValueError(f"Fold {fold.repeat}-{fold.fold} does not cover the expected groups.")
    for repeat in range(1, int(n_repeats) + 1):
        seen: list[int] = []
        for fold in [item for item in folds if item.repeat == repeat]:
            seen.extend(fold.validation_groups)
        if sorted(seen) != sorted(expected):
            raise ValueError(f"Repeat {repeat} does not validate every group exactly once.")
