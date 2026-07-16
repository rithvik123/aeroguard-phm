import numpy as np

from aeroguard.health.pca_health_index import PCAHealthIndex


def test_pca_health_index_orients_higher_as_healthier() -> None:
    rng = np.random.default_rng(42)
    rul = np.linspace(120, 0, 60)
    x = np.column_stack([rul, rul * 0.5]) + rng.normal(0, 0.5, size=(60, 2))

    model = PCAHealthIndex(n_components=1, lower_quantile=0.05, upper_quantile=0.95)
    raw, scaled = model.fit_transform(x, rul)

    assert np.corrcoef(raw, rul)[0, 1] > 0.95
    assert scaled[0] > scaled[-1]
    assert np.all((scaled >= 0) & (scaled <= 1))


def test_pca_health_index_rejects_mismatched_rul_length() -> None:
    model = PCAHealthIndex(n_components=1)

    try:
        model.fit(np.ones((5, 2)), np.ones(4))
    except ValueError as exc:
        assert "length" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
