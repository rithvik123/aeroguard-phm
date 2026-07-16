"""Engine-group validation split helpers for multidomain experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DomainSplit:
    split_id: str
    train_engine_ids: list[str]
    validation_engine_ids: list[str]
    train_domains: list[str]
    validation_domains: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "split_id": self.split_id,
            "train_engine_ids": self.train_engine_ids,
            "validation_engine_ids": self.validation_engine_ids,
            "train_domains": self.train_domains,
            "validation_domains": self.validation_domains,
        }


def stratified_engine_group_splits(
    frame: pd.DataFrame,
    n_splits: int,
    n_repeats: int,
    seeds: Iterable[int],
    group_column: str = "global_engine_id",
    domain_column: str = "source_domain",
) -> list[DomainSplit]:
    """Create deterministic engine-group folds with each domain represented."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if n_repeats < 1:
        raise ValueError("n_repeats must be positive.")
    seeds = [int(seed) for seed in seeds]
    if len(seeds) != n_repeats:
        raise ValueError("Number of seeds must equal n_repeats.")
    engine_domains = frame[[group_column, domain_column]].drop_duplicates()
    all_engines = sorted(engine_domains[group_column].tolist())
    splits: list[DomainSplit] = []
    for repeat_idx, seed in enumerate(seeds, start=1):
        rng = np.random.default_rng(seed)
        parts = [[] for _ in range(n_splits)]
        for _, group in engine_domains.groupby(domain_column):
            ids = np.array(sorted(group[group_column].tolist()), dtype=object)
            rng.shuffle(ids)
            for idx, part in enumerate(np.array_split(ids, n_splits)):
                parts[idx].extend(str(item) for item in part.tolist())
        for fold_idx, validation in enumerate(parts, start=1):
            validation_ids = sorted(validation)
            validation_set = set(validation_ids)
            train_ids = [engine for engine in all_engines if engine not in validation_set]
            if validation_set.intersection(train_ids):
                raise RuntimeError("Engine leakage detected in stratified split.")
            splits.append(
                DomainSplit(
                    split_id=f"cv_r{repeat_idx}_f{fold_idx}",
                    train_engine_ids=train_ids,
                    validation_engine_ids=validation_ids,
                    train_domains=sorted(frame[frame[group_column].isin(train_ids)][domain_column].unique().tolist()),
                    validation_domains=sorted(frame[frame[group_column].isin(validation_ids)][domain_column].unique().tolist()),
                )
            )
    return splits


def leave_one_domain_out_splits(
    frame: pd.DataFrame,
    domains: Iterable[str],
    group_column: str = "global_engine_id",
    domain_column: str = "source_domain",
) -> list[DomainSplit]:
    """Create leave-one-training-domain-out splits."""
    splits = []
    for domain in [str(item).upper() for item in domains]:
        validation = sorted(frame.loc[frame[domain_column] == domain, group_column].unique().tolist())
        train = sorted(frame.loc[frame[domain_column] != domain, group_column].unique().tolist())
        if set(train).intersection(validation):
            raise RuntimeError("Engine leakage detected in leave-one-domain-out split.")
        splits.append(
            DomainSplit(
                split_id=f"lodo_{domain.lower()}",
                train_engine_ids=train,
                validation_engine_ids=validation,
                train_domains=sorted(frame.loc[frame[group_column].isin(train), domain_column].unique().tolist()),
                validation_domains=[domain],
            )
        )
    return splits


def validate_no_engine_leakage(splits: Iterable[DomainSplit]) -> None:
    for split in splits:
        overlap = set(split.train_engine_ids).intersection(split.validation_engine_ids)
        if overlap:
            raise ValueError(f"Split {split.split_id} has engine leakage: {sorted(overlap)[:3]}")
