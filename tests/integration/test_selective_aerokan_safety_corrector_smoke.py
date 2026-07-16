from __future__ import annotations

from aeroguard.pipelines.train_selective_aerokan_safety_corrector import run_smoke_test


def test_selective_aerokan_safety_corrector_smoke() -> None:
    result = run_smoke_test("configs/selective_aerokan_safety_corrector.yaml")
    assert result["status"] == "smoke_complete"
    assert result["synthetic_only"] is True
    assert result["engine_overlap_count"] == 0
    assert result["benchmark_leakage"] is False
    assert result["backbone_training_called"] is False
