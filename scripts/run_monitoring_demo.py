from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from aeroguard.inference.monitoring import INFERENCE_LOG_FIELDS, monitoring_spec
from aeroguard.inference.predictor import AeroGuardPredictor


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts" / "final_release" / "frozen_system_manifest.json"
SAMPLE = ROOT / "examples" / "sample_engine_history.csv"
OUT = ROOT / "reports" / "release_readiness" / "monitoring_demo.json"


def run_demo() -> dict[str, object]:
    predictor = AeroGuardPredictor.from_manifest(MANIFEST)
    frame = pd.read_csv(SAMPLE)
    prediction = predictor.predict_engine(frame)
    monitoring_log = prediction.get("monitoring_log") or {}
    missing_fields = [field for field in INFERENCE_LOG_FIELDS if field not in monitoring_log]
    payload = {
        "status": "pass" if not missing_fields else "fail",
        "system_name": prediction.get("system_name"),
        "model_version": prediction.get("model_version"),
        "sample_input": SAMPLE.relative_to(ROOT).as_posix(),
        "manifest": MANIFEST.relative_to(ROOT).as_posix(),
        "monitoring_spec": monitoring_spec(),
        "monitoring_log": monitoring_log,
        "missing_monitoring_fields": missing_fields,
        "raw_sensor_data_logged": any(str(key).startswith("sensor_") for key in monitoring_log),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")
    return payload


def main() -> int:
    payload = run_demo()
    print(json.dumps({"status": payload["status"], "output": OUT.relative_to(ROOT).as_posix()}, indent=2))
    return 0 if payload["status"] == "pass" and not payload["raw_sensor_data_logged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
