from pathlib import Path

import pandas as pd

from aeroguard.inference import AeroGuardPredictor


ROOT = Path(__file__).resolve().parents[1]
predictor = AeroGuardPredictor.from_manifest(ROOT / "artifacts/final_release/frozen_system_manifest.json")
engine_history = pd.read_csv(ROOT / "examples/sample_engine_history.csv")
results = predictor.predict_batch([engine_history])
print(results[0]["maintenance_action"])
