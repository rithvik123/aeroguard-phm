# AeroGuard-PHM: Physics-Guided Remaining Useful Life Prediction with Uncertainty Quantification and Safety-Constrained Maintenance Decisions

## Abstract
This report summarizes the final AeroGuard-PHM release, a safety-guarded RUL decision-support system for simulated C-MAPSS turbofan degradation data.

## Problem Definition
Estimate engine-level remaining useful life while reporting optimistic-error and maintenance-review behavior.

## Dataset
The project uses NASA C-MAPSS FD001-FD004 simulated turbofan data.

## Data Preprocessing
The selected backbone uses regime-aware normalization and fixed-length historical windows.

## Evaluation Protocol
Final comparisons use identical benchmark engine keys, uncapped true RUL, residual `predicted - true`, and a 30-cycle severe optimism threshold.

## Patch Transformer Benchmark
The Patch Transformer with 10x5 mean-pooled patches is the principal temporal benchmark.

## Physics-Guided Modelling
The regime-consistent physics-guided candidate was selected as the predictive backbone.

## Constraint Diagnostics
Physics ablations are separated in `physics_ablation_comparison.csv`.

## Uncertainty Calibration
The final system uses frozen global split conformal radii.

## Abstention Analysis
The final release uses no abstention after the previous abstention policy failed enrichment checks.

## Maintenance-Policy Analysis
Fixed-policy and native-policy tables are separated to avoid mixing model effects with downstream policy effects.

## KAN Residual-Correction Experiments
KAN residual models are retained as experimental research candidates.

## Selective One-Sided KAN Experiment
The selective one-sided AeroKAN preserved point accuracy but did not provide enough critical coverage for final selection.

## Invariant Audit
The prior one-sided invariant audit passed, and metric-definition inconsistencies were documented.

## Critical-Boundary Safety Guard
The final deterministic guard reduced fixed-policy critical misses while preserving configured RMSE non-inferiority.

## Final Comparison
| Model | Family | Overall MAE | Overall RMSE | NASA score | Severe optimism | Critical misses | Operational recall | Review workload | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Patch Transformer — 10×5 Patches with Mean Pooling | Patch Transformer | 14.9550 | 21.2671 | 8246.1682 | 0.0651 | 26 | 0.7451 | 0.1188 | Candidate |
| Regime-Consistent Physics-Guided Patch Transformer | Physics-guided Transformer | 14.4885 | 20.9134 | 7577.1771 | 0.0566 | 25 | 0.7549 | 0.1259 | Candidate |
| AeroKAN-PHM Compact Residual Corrector | AeroKAN residual correction | 15.4658 | 21.4713 | 9516.8317 | 0.0863 | 5 | 0.9510 | 0.1641 | Candidate |
| Selective One-Sided AeroKAN Safety Corrector | Selective AeroKAN safety correction | 14.4485 | 20.9017 | 7570.6490 | 0.0566 | 24 | 0.7647 | 0.1273 | Candidate |
| Critical-Boundary Safety-Guarded Physics-Guided Transformer | Safety-guarded Transformer | 14.5632 | 20.9603 | 7584.7765 | 0.0566 | 1 | 0.9902 | 0.1924 | Final selected system |

## Statistical Testing
| comparator_display_name | metric | point_difference_comparator_minus_final | ci_lower | ci_upper | probability_final_improves | statistical_interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| Patch Transformer — 10×5 Patches with Mean Pooling | absolute_error | 0.3918 | 0.0434 | 0.7603 | 0.9860 | final lower/better |
| Patch Transformer — 10×5 Patches with Mean Pooling | squared_error | 12.9523 | -3.4872 | 29.0908 | 0.9320 | uncertain |
| Patch Transformer — 10×5 Patches with Mean Pooling | nasa_contribution | 0.9355 | -0.0629 | 2.0681 | 0.9610 | uncertain |
| Patch Transformer — 10×5 Patches with Mean Pooling | optimistic_indicator | 0.0580 | 0.0368 | 0.0806 | 1.0000 | final lower/better |
| Patch Transformer — 10×5 Patches with Mean Pooling | severe_optimistic_indicator | 0.0085 | -0.0028 | 0.0198 | 0.9340 | uncertain |
| Patch Transformer — 10×5 Patches with Mean Pooling | fixed_policy_critical_miss_indicator | 0.0354 | 0.0212 | 0.0495 | 1.0000 | final lower/better |
| Patch Transformer — 10×5 Patches with Mean Pooling | mandatory_review_indicator | -0.0736 | -0.0934 | -0.0551 | 0.0000 | comparator lower/better |
| Regime-Consistent Physics-Guided Patch Transformer | absolute_error | -0.0747 | -0.1710 | 0.0121 | 0.0500 | uncertain |
| Regime-Consistent Physics-Guided Patch Transformer | squared_error | -1.9655 | -3.6769 | -0.3736 | 0.0060 | comparator lower/better |
| Regime-Consistent Physics-Guided Patch Transformer | nasa_contribution | -0.0107 | -0.0267 | 0.0051 | 0.0930 | uncertain |
| Regime-Consistent Physics-Guided Patch Transformer | optimistic_indicator | 0.0226 | 0.0127 | 0.0339 | 1.0000 | final lower/better |
| Regime-Consistent Physics-Guided Patch Transformer | severe_optimistic_indicator | 0.0000 | 0.0000 | 0.0000 | 0.0000 | uncertain |

## Interpretability
The safety layer is auditable as a threshold rule. KAN explanations are available only for experimental branches.

## Deployment Architecture
The release includes Python, CLI, API, and Streamlit interfaces around the frozen manifest.

## Limitations
The benchmark is simulated, the guard is deterministic, and the system is not certified for real aircraft maintenance.

## Conclusions
The selected final system is the Critical-Boundary Safety-Guarded Physics-Guided Transformer inside the complete AeroGuard-PHM Safety-Guarded RUL System.

## Future Independent Validation
Validate on independent datasets, fleet-specific operating regimes, and external maintenance-review workflows before operational use.
