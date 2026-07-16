from __future__ import annotations

from aeroguard.pipelines.refine_physics_guided_temporal_rul import run_smoke_test


def test_physics_guided_temporal_rul_refinement_smoke() -> None:
    result = run_smoke_test("configs/physics_guided_temporal_rul_refinement.yaml")
    assert result["status"] == "smoke_complete"
    assert result["synthetic_only"] is True
    assert result["neural_training_function_called"] is False
    assert result["uncertainty_candidate_count"] >= 2
    assert result["paired_bootstrap_rows"] > 0
