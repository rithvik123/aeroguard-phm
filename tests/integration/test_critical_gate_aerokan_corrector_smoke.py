from __future__ import annotations

from aeroguard.pipelines.train_critical_gate_aerokan_corrector import run_smoke_test


def test_critical_gate_aerokan_corrector_smoke() -> None:
    result = run_smoke_test("configs/critical_gate_aerokan_corrector.yaml")
    assert result["status"] == "smoke_complete"
    assert result["synthetic_only"] is True
    assert result["benchmark_leakage"] is False
    assert result["backbone_training_called"] is False
