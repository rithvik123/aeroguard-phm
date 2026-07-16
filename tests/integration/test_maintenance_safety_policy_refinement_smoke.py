from __future__ import annotations

from aeroguard.pipelines.refine_maintenance_safety_policy import run_smoke_test


def test_maintenance_safety_policy_refinement_smoke() -> None:
    result = run_smoke_test("configs/maintenance_safety_policy_refinement.yaml")
    assert result["status"] == "smoke_complete"
    assert result["synthetic_only"] is True
    assert result["policy_family_count"] >= 3
    assert result["policy_locked_before_benchmark"] is True
    assert result["neural_training_function_called"] is False
