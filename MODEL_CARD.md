# Model Card: AeroGuard-PHM Safety-Guarded RUL System

## Model Name
AeroGuard-PHM Safety-Guarded RUL System

## Version
Release 1.0.0; model version `aeroguard-phm-safety-v1`.

## Intended Use
Research decision support for simulated turbofan RUL benchmarking, uncertainty analysis, and maintenance-policy experimentation.

## Out-of-Scope Use
The system is not certified for real aircraft maintenance, dispatch, or safety-critical operational control.

## Architecture
Critical-Boundary Safety-Guarded Physics-Guided Patch Transformer: regime-aware preprocessing, Regime-Consistent Physics-Guided Patch Transformer, deterministic critical-boundary guard, conformal uncertainty, and maintenance policy.

## Input Requirements
Engine-history CSVs require `cycle`, three operational settings, and sensors `sensor_1` through `sensor_21`.

## Outputs
The predictor returns base RUL, safety-adjusted RUL, conformal intervals, support status, guard activation, review requirement, maintenance action, warnings, and explanations.

## Training Datasets
NASA C-MAPSS simulated turbofan subsets FD001-FD004.

## Evaluation Datasets
Final benchmark evaluation uses aligned FD001-FD004 benchmark engine keys with uncapped true RUL.

## Point-Performance Metrics
| display_name | mae | rmse | nasa_score | severe_optimistic_prediction_rate |
| --- | --- | --- | --- | --- |
| Patch Transformer — 10×5 Patches with Mean Pooling | 14.9550 | 21.2671 | 8246.1682 | 0.0651 |
| Regime-Consistent Physics-Guided Patch Transformer | 14.4885 | 20.9134 | 7577.1771 | 0.0566 |
| AeroKAN-PHM Compact Residual Corrector | 15.4658 | 21.4713 | 9516.8317 | 0.0863 |
| Selective One-Sided AeroKAN Safety Corrector | 14.4485 | 20.9017 | 7570.6490 | 0.0566 |
| Critical-Boundary Safety-Guarded Physics-Guided Transformer | 14.5632 | 20.9603 | 7584.7765 | 0.0566 |

## Safety Metrics
| display_name | critical_miss_count | operational_recall | urgent_precision | review_workload |
| --- | --- | --- | --- | --- |
| Patch Transformer — 10×5 Patches with Mean Pooling | 26 | 0.7451 | 0.9048 | 0.1188 |
| Regime-Consistent Physics-Guided Patch Transformer | 25 | 0.7549 | 0.8652 | 0.1259 |
| AeroKAN-PHM Compact Residual Corrector | 5 | 0.9510 | 0.8362 | 0.1641 |
| Selective One-Sided AeroKAN Safety Corrector | 24 | 0.7647 | 0.8667 | 0.1273 |
| Critical-Boundary Safety-Guarded Physics-Guided Transformer | 1 | 0.9902 | 0.7426 | 0.1924 |

## Uncertainty Metrics
The final release uses global split conformal radii recorded in `artifacts/final_release/frozen_system_manifest.json`.

## Comparison With Prior Models
The final guard materially reduced critical misses under the fixed policy while preserving RMSE non-inferiority relative to the selected backbone.

## Safety-Guard Behaviour
The final guard is deterministic. It activates when `15 < base_rul <= 25` and applies a downward correction capped at 10 cycles. The final deployed system is not a KAN model.

## Known Limitations
C-MAPSS is simulated. Benchmark results do not guarantee real-world safety. The compatibility inference path is for packaging smoke tests when direct neural reconstruction is unavailable.

## Ethical and Operational Considerations
Human engineering review remains required. The model must not be used as an autonomous aviation safety authority.

## Monitoring Recommendations
Monitor input-schema drift, missing sensors, feature range violations, regime distribution, support score, prediction distribution, interval width, guard activation rate, review rate, maintenance-action distribution, RUL trajectory instability, inference latency, and runtime failures.

## Retraining Triggers
Consider retraining only after independent validation shows material drift, degraded calibration, changed sensor definitions, or new fleet operating regimes.

## Independent-Validation Requirements
Validate with external simulated or real-world-like data before any operational interpretation. KAN models were experimental candidates and are not the deployed final system.

## License
AeroGuard-PHM is released under Apache License 2.0. Dataset and third-party provenance notes are maintained in `THIRD_PARTY_NOTICES.md`.
