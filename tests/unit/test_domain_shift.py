import numpy as np
import pandas as pd

from aeroguard.evaluation.domain_shift import feature_shift_table, population_stability_index


def test_identical_distributions_have_near_zero_shift() -> None:
    reference = pd.DataFrame({"sensor_2": [1.0, 2.0, 3.0, 4.0]})
    table = feature_shift_table(reference, reference.copy(), reference.copy(), ["sensor_2"], psi_bins=4)

    assert abs(table["fd001_test_standardized_mean_difference"].iloc[0]) < 1e-12
    assert abs(table["fd003_standardized_mean_difference"].iloc[0]) < 1e-12
    assert population_stability_index(reference["sensor_2"], reference["sensor_2"], bins=4) < 1e-9


def test_shifted_distribution_and_out_of_range_fraction() -> None:
    reference = pd.DataFrame({"sensor_2": [1.0, 2.0, 3.0, 4.0]})
    shifted = pd.DataFrame({"sensor_2": [10.0, 11.0, 12.0, 13.0]})
    table = feature_shift_table(reference, reference.copy(), shifted, ["sensor_2"], psi_bins=4)

    assert table["fd003_standardized_mean_difference"].iloc[0] > 0
    assert table["fd003_outside_healthy_1_99_fraction"].iloc[0] == 1.0
    assert table["fd003_psi"].iloc[0] > 0


def test_missing_and_infinite_values_are_counted() -> None:
    reference = pd.DataFrame({"sensor_2": [1.0, 2.0, np.inf, None]})
    comparison = pd.DataFrame({"sensor_2": [1.0, np.inf, None, 4.0]})
    table = feature_shift_table(reference, comparison, comparison, ["sensor_2"], psi_bins=2)

    assert table["fd001_healthy_missing_count"].iloc[0] == 1
    assert table["fd001_healthy_infinite_count"].iloc[0] == 1
    assert table["fd003_missing_count"].iloc[0] == 1
    assert table["fd003_infinite_count"].iloc[0] == 1
