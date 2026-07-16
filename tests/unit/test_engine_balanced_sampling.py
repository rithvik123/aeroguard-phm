import pandas as pd

from aeroguard.deep.sampling import build_endpoint_table, sample_engine_endpoints
from aeroguard.deep.windowing import WindowSpec


def _engine(engine: str, length: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "global_engine_id": engine,
            "cycle": list(range(1, length + 1)),
            "rul_capped": list(reversed(range(length))),
        }
    )


def test_engine_balanced_sampling_respects_cap_and_preserves_final_cycle() -> None:
    spec = WindowSpec(window_length=5, stride=1, minimum_valid_history=2)

    endpoints = sample_engine_endpoints(_engine("FD001_0001"), spec, maximum_windows=4, seed=11)

    assert len(endpoints) <= 4
    assert endpoints[-1] == 19
    assert endpoints == sample_engine_endpoints(_engine("FD001_0001"), spec, maximum_windows=4, seed=11)


def test_build_endpoint_table_applies_cap_per_engine() -> None:
    spec = WindowSpec(window_length=5, stride=1, minimum_valid_history=2)
    frame = pd.concat([_engine("FD001_0001"), _engine("FD001_0002")], ignore_index=True)

    table = build_endpoint_table(frame, spec, maximum_windows_per_engine=3, seed=5)

    assert set(table["global_engine_id"]) == {"FD001_0001", "FD001_0002"}
    assert table.groupby("global_engine_id").size().max() <= 3
    assert table.groupby("global_engine_id")["endpoint_index"].max().to_dict() == {
        "FD001_0001": 19,
        "FD001_0002": 19,
    }

