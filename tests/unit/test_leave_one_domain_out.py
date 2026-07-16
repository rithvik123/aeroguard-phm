import pandas as pd
import pytest

from aeroguard.evaluation.leave_one_domain_out import (
    DomainSplit,
    leave_one_domain_out_splits,
    stratified_engine_group_splits,
    validate_no_engine_leakage,
)


def _frame() -> pd.DataFrame:
    rows = []
    for domain in ["FD001", "FD002", "FD003"]:
        for engine in [1, 2, 3]:
            for cycle in [1, 2]:
                rows.append(
                    {
                        "source_domain": domain,
                        "global_engine_id": f"{domain}_{engine:04d}",
                        "cycle": cycle,
                    }
                )
    return pd.DataFrame(rows)


def test_stratified_engine_group_splits_have_no_engine_leakage() -> None:
    splits = stratified_engine_group_splits(_frame(), n_splits=3, n_repeats=2, seeds=[1, 2])

    validate_no_engine_leakage(splits)
    assert len(splits) == 6
    assert all({"FD001", "FD002", "FD003"} == set(split.validation_domains) for split in splits)


def test_leave_one_domain_out_splits_hold_out_each_training_domain() -> None:
    splits = leave_one_domain_out_splits(_frame(), ["FD001", "FD002", "FD003"])

    assert [split.validation_domains for split in splits] == [["FD001"], ["FD002"], ["FD003"]]
    assert all(not set(split.train_engine_ids).intersection(split.validation_engine_ids) for split in splits)


def test_validate_no_engine_leakage_rejects_overlap() -> None:
    split = DomainSplit(
        split_id="bad",
        train_engine_ids=["FD001_0001"],
        validation_engine_ids=["FD001_0001"],
        train_domains=["FD001"],
        validation_domains=["FD001"],
    )

    with pytest.raises(ValueError, match="leakage"):
        validate_no_engine_leakage([split])
