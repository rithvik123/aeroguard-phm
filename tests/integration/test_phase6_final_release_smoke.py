from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aeroguard.inference.cli import run_cli
from aeroguard.inference.predictor import AeroGuardPredictor
from aeroguard.pipelines.build_final_release import run_release


def test_phase6_final_release_smoke(tmp_path: Path) -> None:
    summary = run_release(ROOT)
    assert summary["status"] == "complete"
    assert summary["model_retrained"] is False
    assert summary["threshold_retuned"] is False

    manifest = ROOT / "artifacts" / "final_release" / "frozen_system_manifest.json"
    sample = ROOT / "examples" / "sample_engine_history.csv"
    predictor = AeroGuardPredictor.from_manifest(manifest)
    prediction = predictor.predict_engine(pd.read_csv(sample))

    assert prediction["valid"] is True
    assert prediction["base_rul"] >= 0
    assert prediction["safety_adjusted_rul"] >= 0
    assert "interval_width_90" in prediction
    assert "maintenance_action" in prediction
    assert "explanation" in prediction
    assert "monitoring_log" in prediction
    assert prediction["monitoring_log"]["model_version"] == "aeroguard-phm-safety-v1"

    cli_output = tmp_path / "prediction.json"
    assert run_cli(["--manifest", str(manifest), "--input", str(sample), "--output", str(cli_output)]) == 0
    assert json.loads(cli_output.read_text(encoding="utf-8"))["model_version"] == "aeroguard-phm-safety-v1"

    registry = pd.read_csv(ROOT / "reports" / "final_release" / "model_registry.csv")
    comparison = pd.read_csv(ROOT / "reports" / "final_release" / "point_prediction_comparison.csv")
    assert len(registry) >= 64
    assert set(comparison["engine_count"]) == {707}
