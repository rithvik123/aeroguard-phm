# AeroGuard-PHM Safety-Guarded RUL System

## Project Title

AeroGuard-PHM Safety-Guarded RUL System

## One-Sentence Value Proposition

A reproducible turbofan RUL decision-support project that combines a physics-guided temporal predictor, calibrated uncertainty, and an auditable critical-boundary safety guard.

## Problem Statement

Remaining useful life prediction estimates how many cycles a turbofan engine can continue operating before failure under observed sensor history. The project studies prediction accuracy, optimistic error, uncertainty, and maintenance-review behavior separately.

## Why RUL Prediction Matters

Late RUL estimates can delay inspection, while overly conservative estimates can increase unnecessary maintenance. This release therefore reports both point-prediction metrics and policy-level safety metrics.

## C-MAPSS Data

The experiments use NASA C-MAPSS simulated turbofan degradation subsets FD001-FD004. These are benchmark simulations and are not certified aircraft-maintenance records.

## Complete Development Journey

### Patch Transformer — 10×5 Patches with Mean Pooling

Established a strong temporal RUL benchmark using patch-based sensor-sequence modelling.

### Regime-Consistent Physics-Guided Patch Transformer

Added operating-regime consistency guidance, improving generalization and reducing optimistic errors.

### AeroKAN-PHM Compact Residual Corrector

Tested whether named engineering features and KAN edge functions could correct residual RUL errors. It reduced some critical misses but worsened overall error and was not selected.

### Selective One-Sided AeroKAN Safety Corrector

Restricted KAN intervention to downward corrections on risk-selected engines. It preserved point accuracy but the learned gate did not provide sufficient critical coverage.

### Critical-Boundary Safety-Guarded Physics-Guided Transformer

Used the frozen physics-guided predictor with an auditable critical-boundary safeguard. This became the selected final safety system.

## Final Architecture

The selected architecture is the Critical-Boundary Safety-Guarded Physics-Guided Patch Transformer. The complete release pipeline adds regime-aware preprocessing, conformal uncertainty, support/review logic, and the frozen maintenance policy.

## Model Comparison

| Model | Family | Overall MAE | Overall RMSE | NASA score | Severe optimism | Critical misses | Operational recall | Review workload | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Patch Transformer — 10×5 Patches with Mean Pooling | Patch Transformer | 14.9550 | 21.2671 | 8246.1682 | 0.0651 | 26 | 0.7451 | 0.1188 | Candidate |
| Regime-Consistent Physics-Guided Patch Transformer | Physics-guided Transformer | 14.4885 | 20.9134 | 7577.1771 | 0.0566 | 25 | 0.7549 | 0.1259 | Candidate |
| AeroKAN-PHM Compact Residual Corrector | AeroKAN residual correction | 15.4658 | 21.4713 | 9516.8317 | 0.0863 | 5 | 0.9510 | 0.1641 | Candidate |
| Selective One-Sided AeroKAN Safety Corrector | Selective AeroKAN safety correction | 14.4485 | 20.9017 | 7570.6490 | 0.0566 | 24 | 0.7647 | 0.1273 | Candidate |
| Critical-Boundary Safety-Guarded Physics-Guided Transformer | Safety-guarded Transformer | 14.5632 | 20.9603 | 7584.7765 | 0.0566 | 1 | 0.9902 | 0.1924 | Final selected system |

Point-prediction metrics are separated from fixed-policy and native-policy metrics in `reports/final_release`.

## Physics-Guided Modelling

The selected predictive backbone is the Regime-Consistent Physics-Guided Patch Transformer. Physics-guided ablations are reported separately in `physics_ablation_comparison.csv`.

## KAN Experimental Branch

KAN models are documented as experimental residual-correction candidates. The deployed final system is not a KAN model.

## Critical-Boundary Safety Guard

The final guard is deterministic: it activates when `15 < base_rul <= 25`, applies a downward correction capped at 10 cycles, and uses a 0.5-cycle margin.

## Uncertainty Quantification

The final system uses global split conformal uncertainty with frozen radii from the selected release manifest.

## Maintenance Recommendations

The frozen maintenance policy uses urgent, schedule, and inspection thresholds at 15, 30, and 60 cycles respectively. Human engineering review remains required.

## Installation

Use the existing project environment. No additional packages are required for the generated release in this workspace.

## Quick Start

```powershell
python -m aeroguard.inference.cli --manifest artifacts/final_release/frozen_system_manifest.json --input examples/sample_engine_history.csv --output reports/inference/sample_prediction.json
```

## Python Inference

```python
import pandas as pd
from aeroguard.inference import AeroGuardPredictor

predictor = AeroGuardPredictor.from_manifest("artifacts/final_release/frozen_system_manifest.json")
prediction = predictor.predict_engine(pd.read_csv("examples/sample_engine_history.csv"))
```

## CLI Inference

Use `python -m aeroguard.inference.cli` with `--input`, `--batch-dir`, `--output`, `--output-csv`, and `--validation-only`.

## API Usage

```powershell
python -m uvicorn aeroguard.api.app:app --host 127.0.0.1 --port 8000
```

## Dashboard Usage

```powershell
python -m streamlit run dashboard/app.py
```

## Testing

Phase 6 tests cover registry generation, hash validation, inference, CLI, API import safety, dashboard import safety, monitoring, and smoke execution.

## Reproducibility

See `REPRODUCIBILITY.md` for hashes, commands, environment metadata, and expected metrics.

## Repository Structure

Final release outputs live in `reports/final_release` and `artifacts/final_release`. Production wrappers live under `src/aeroguard/inference`, the API under `src/aeroguard/api`, and the dashboard under `dashboard`.

## Limitations

The system is evaluated on simulated benchmark data, is not certified for aircraft maintenance, and cannot guarantee failure prevention.

## Responsible-Use Statement

Use this system as research decision support only. Maintenance decisions require qualified human engineering review and independent validation.

## Future Work

Validate on independent fleets, expand monitoring with real operational envelopes, and reassess learned safety gates only with sufficient critical-miss support.

## Citation

Reference NASA C-MAPSS for the dataset and this repository for the AeroGuard-PHM implementation.

## License

Use the repository license file if present. Otherwise treat this release as project-specific research material until licensing is clarified.
