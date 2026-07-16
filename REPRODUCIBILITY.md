# Reproducibility

## Python Version
`3.11.9`

## PyTorch Version
`2.12.1+cu126`

## CUDA Version
`12.6`

## Operating System Assumptions
Generated on `Windows-10-10.0.26200-SP0` using the existing `aerostat-ai` environment.

## Data Directory Structure
C-MAPSS files are expected under `data/raw/cmapss`.

## Installation

Core installation:

```powershell
python -m pip install -e .
```

Reproducible release-validation installation:

```powershell
python -m pip install -e ".[api,dashboard,dev]" -c requirements/constraints.txt
```

Docker and containerization are intentionally deferred to a future release.

## Dataset-File Hashes
Dataset hashes are not duplicated in this final release unless present in earlier source manifests.

## Configuration-File Hashes
The final configuration hash is `555728247323b6b37dddc32bf8e9b9090d03bb8e0cde82eda548e2ecd6b367dd`.

## Artifact Hashes
See `artifacts/final_release/frozen_system_manifest.json`.

## Exact Evaluation Commands
```powershell
$env:PYTHONPATH = ".\src"
python -m aeroguard.pipelines.build_final_release
```

## Expected Benchmark Engine Counts
Aligned final comparison engine count: `707`.

## Expected Final Metrics
| display_name | mae | rmse | nasa_score | severe_optimistic_prediction_rate |
| --- | --- | --- | --- | --- |
| Critical-Boundary Safety-Guarded Physics-Guided Transformer | 14.5632 | 20.9603 | 7584.7765 | 0.0566 |

## Deterministic Seeds
Final paired bootstrap uses seed `20260715`. The release builder does not train models.

## Hardware Notes
No model training is performed in Phase 6. Inference can run on CPU by default.

## Known Nondeterministic Operations
CUDA neural inference can vary slightly across hardware. This release does not claim bitwise reproducibility for CUDA operations.

## Verification Commands
```powershell
$env:PYTHONPATH = ".\src"
python -m pytest tests\unit\test_final_release.py -q
python -m pytest tests\integration\test_phase6_final_release_smoke.py -q
python scripts\run_monitoring_demo.py
python scripts\release_integrity_check.py
```
