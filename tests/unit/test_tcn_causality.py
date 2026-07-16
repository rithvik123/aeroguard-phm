import torch

from aeroguard.deep.models.tcn import TCNRegressor


def test_tcn_sequence_features_do_not_depend_on_future_inputs() -> None:
    torch.manual_seed(7)
    model = TCNRegressor(input_dim=4, hidden_dim=5, dropout=0.0, kernel_size=3, dilations=[1, 2]).eval()
    x = torch.randn(1, 8, 4)
    x[:, :, -1] = 1.0
    altered = x.clone()
    altered[:, 5:, :-1] += 100.0

    y = model.sequence_features(x)
    y_altered = model.sequence_features(altered)

    torch.testing.assert_close(y[:, :5, :], y_altered[:, :5, :])

