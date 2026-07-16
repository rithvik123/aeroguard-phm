from __future__ import annotations

from aeroguard.pipelines.train_aerokan_rul_corrector import run_smoke_test


def test_aerokan_rul_corrector_smoke() -> None:
    result = run_smoke_test("configs/aerokan_rul_corrector.yaml")
    assert result["status"] == "smoke_complete"
    assert result["synthetic_only"] is True
    assert result["engine_overlap_count"] == 0
