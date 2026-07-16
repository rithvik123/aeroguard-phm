import pytest
import torch

from aeroguard.deep.models.patch_transformer import PatchTemporalTransformerRegressor
from aeroguard.deep.models.physics_guided_patch_transformer import PhysicsGuidedPatchTransformer


def _input(batch: int = 2) -> torch.Tensor:
    x = torch.randn(batch, 8, 5)
    x[:, :3, -1] = 0.0
    x[:, 3:, -1] = 1.0
    return x


def _model(**kwargs) -> PhysicsGuidedPatchTransformer:
    return PhysicsGuidedPatchTransformer(
        input_dim=5,
        window_length=8,
        patch_length=4,
        patch_stride=2,
        projection_dim=16,
        layers=1,
        heads=4,
        feedforward_dim=32,
        dropout=0.0,
        positional_encoding="sinusoidal",
        pooling="mean",
        **kwargs,
    )


def test_output_dictionary_and_shapes() -> None:
    model = _model()
    outputs = model(_input())

    assert set(outputs) == {"rul_raw", "rul_prediction", "health_score", "degradation_rate", "latent", "valid_token_count"}
    assert tuple(outputs["rul_prediction"].shape) == (2, 1)
    assert tuple(outputs["health_score"].shape) == (2, 1)
    assert tuple(outputs["degradation_rate"].shape) == (2, 1)


def test_batch_size_one_and_padded_sequence_invariance() -> None:
    torch.manual_seed(2)
    model = _model().eval()
    x = _input(batch=1)
    altered = x.clone()
    altered[:, :3, :-1] -= 999.0

    with torch.no_grad():
        torch.testing.assert_close(model(x)["rul_prediction"], model(altered)["rul_prediction"], rtol=1e-5, atol=1e-5)


def test_optional_heads_disabled() -> None:
    outputs = _model(health_head_enabled=False, rate_head_enabled=False)(_input())

    assert outputs["health_score"] is None
    assert outputs["degradation_rate"] is None


def test_rejects_all_padding_incorrect_feature_count_and_nonfinite() -> None:
    model = _model()
    x = _input()
    x[:, :, -1] = 0.0
    with pytest.raises(ValueError, match="All-padding"):
        model(x)
    with pytest.raises(ValueError, match="feature"):
        model(torch.randn(2, 8, 4))
    bad = _input()
    bad[0, 3, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(bad)


def test_warm_start_encoder_compatibility_and_state_reload(tmp_path) -> None:
    phase5b = PatchTemporalTransformerRegressor(
        input_dim=5,
        window_length=8,
        patch_length=4,
        patch_stride=2,
        projection_dim=16,
        layers=1,
        heads=4,
        feedforward_dim=32,
        dropout=0.0,
        positional_encoding="sinusoidal",
        pooling="mean",
    )
    checkpoint = tmp_path / "phase5b.pt"
    torch.save({"state_dict": phase5b.state_dict()}, checkpoint)
    model = _model()

    report = model.warm_start_from_checkpoint(checkpoint, load_encoder_only=True)
    payload = {"state_dict": model.state_dict()}
    reload_path = tmp_path / "physics.pt"
    torch.save(payload, reload_path)
    reloaded = _model()
    reloaded.load_state_dict(torch.load(reload_path, map_location="cpu")["state_dict"])
    x = _input()

    assert report["loaded_key_count"] > 0
    torch.testing.assert_close(model(x)["rul_prediction"], reloaded(x)["rul_prediction"], rtol=1e-4, atol=1e-4)


def test_parameter_budget_validation() -> None:
    with pytest.raises(ValueError, match="exceeding budget"):
        _model(parameter_budget=1)
