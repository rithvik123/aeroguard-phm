# AeroGuard-PHM

AeroGuard-PHM is an end-to-end predictive-maintenance system for turbofan engines. It studies an engine's recent operating conditions and sensor measurements, estimates how many operating cycles remain before failure, quantifies uncertainty, identifies risky predictions near the maintenance boundary, and converts the result into a clear maintenance decision.

`Frozen v1.0.0 release` | `Python >=3.11` | `Apache-2.0` | `aeroguard-phm-safety-v1`

<p align="center">
  <img
    src="docs/assets/readme/hero/aeroguard_phm_hero.png"
    alt="AeroGuard-PHM transforms turbofan sensor history into Remaining Useful Life estimates, uncertainty intervals and safety-aware maintenance recommendations"
    width="100%">
</p>

> AeroGuard-PHM converts multivariate turbofan sensor histories into an estimated Remaining Useful Life, measures prediction uncertainty, applies a transparent safety safeguard near the critical maintenance boundary, and produces an actionable maintenance recommendation.

The project is not just a neural-network benchmark. It covers data validation, operating-regime-aware preprocessing, classical machine-learning baselines, deep-learning sequence models, Transformer-based temporal modelling, physics-guided learning, KAN-based residual-correction experiments, conformal uncertainty estimation, safety-aware decision logic, maintenance-policy evaluation, Python inference, a CLI, FastAPI, Streamlit, monitoring, testing, reproducibility and release integrity.

## Release Snapshot

| Item | Value |
| --- | --- |
| Complete system | AeroGuard-PHM Safety-Guarded RUL System |
| Final selected system | Critical-Boundary Safety-Guarded Physics-Guided Transformer |
| Selected predictive backbone | Regime-Consistent Physics-Guided Patch Transformer |
| Model version | aeroguard-phm-safety-v1 |
| Release state | Frozen v1.0.0 release |
| Model/system candidates evaluated | 64 |
| Window length | 50-cycle past-only engine history |
| Minimum inference history | 10 cycles, with short-history warnings |
| Safety rule | `15 < base_rul <= 25` |
| Critical misses | 1 |
| Final MAE | 14.5632 |
| Final RMSE | 20.9603 |
| Severe optimism | 0.0566 |
| Operational recall | 0.9902 |
| Review workload | 0.1924 |

The final system is a safety-guarded Transformer release pipeline. It is not a learned KAN component. KAN experiments were not selected for deployment, although they were useful development evidence.

## Quick Navigation

- [What AeroGuard-PHM Solves](#what-aeroguard-phm-solves)
- [Dataset: NASA C-MAPSS](#dataset-nasa-c-mapss)
- [End-to-End Data Pipeline](#end-to-end-data-pipeline)
- [Models Evaluated](#models-evaluated)
- [Development Journey](#development-journey)
- [Final Architecture](#final-architecture)
- [Final Results](#final-results)
- [Quick Start](#quick-start)
- [FastAPI Inference Service](#fastapi-inference-service)
- [Interactive Streamlit Dashboard](#interactive-streamlit-dashboard)
- [Testing](#testing)
- [Reproducibility and Frozen Artifacts](#reproducibility-and-frozen-artifacts)

## What AeroGuard-PHM Solves

A turbofan engine produces many sensor readings during each operating cycle. These readings include operating settings and sensor channels related to temperature, pressure, rotational speed and flow behavior. As an engine degrades, some readings drift gradually. The difficulty is that the same readings also move when the engine enters a different operating regime, when measurements are noisy, or when sensors respond to normal operating-condition changes.

Remaining Useful Life, usually shortened to RUL, is the estimated number of operating cycles between the last observed cycle and eventual failure. At inference time the engine has not failed yet. The future failure point is hidden. The model receives only the past engine history and must infer how much useful life remains.

Healthy operation, gradual degradation and near-failure behavior are not directly labeled at every cycle. The system therefore learns from historical run-to-failure examples, then applies that learned relationship to a test engine whose sequence stops before failure. AeroGuard-PHM treats this as a decision-support problem: a prediction is useful only if it can be validated, explained, bounded by uncertainty and translated into maintenance action.

## The Problem in Simple Terms

Imagine watching an engine for 50 recent operating cycles. Each row has the cycle number, three operating settings and 21 sensor measurements. The model does not see the future. It sees the final observed cycle and must answer: "How many cycles are likely left before failure?"

The answer is not a direct sensor. It is inferred from patterns. A healthy engine may show stable sensor behavior. A degrading engine may show slow drift, sharper changes near the end of life, or regime-dependent signals that look different under different operating settings. The final observed cycle is only a snapshot of the known past; the failure point sits somewhere beyond the observed sequence.

This is why the project separates several ideas. The base model estimates RUL. The uncertainty layer describes how wide a plausible error range may be. The safety guard corrects a narrow risky region near the urgent-maintenance boundary. The maintenance policy converts the safety-adjusted RUL into one of the frozen actions used by the release.

### Why Operating Regimes Matter

A temperature increase does not automatically mean the engine has degraded. The increase may have occurred because the engine entered a different operating regime. A model that ignores this distinction can confuse normal operating-condition changes with health deterioration.

AeroGuard-PHM therefore uses operating settings, regime-aware normalization, leakage-safe fitting, temporal history and physics-consistency checks. The preprocessor transforms operating settings and sensors into regime-normalized features. The learned model then consumes the ordered past sequence rather than a single isolated row. This design is especially important on FD002 and FD004, where multiple operating conditions make raw sensor values harder to interpret.

## Why Accuracy Alone Is Not Enough

A single point prediction can look good by MAE or RMSE while still being unsafe near a maintenance boundary. An error of 15 cycles is not equally dangerous in every direction. Predicting 25 cycles when the actual future life is 40 is conservative. Predicting 40 cycles when the actual future life is 25 is optimistic and may delay inspection.

The project reports point-prediction metrics and operational metrics separately:

| Metric | Plain-language meaning | Direction |
| --- | --- | --- |
| MAE | Average absolute cycle error. | Lower is better |
| RMSE | Like MAE, but larger errors receive more weight. | Lower is better |
| NASA score | Asymmetric prognostics score used in the release comparisons. | Lower is better |
| Optimistic error | Prediction is later than the benchmark RUL label. | Lower is safer |
| Severe optimism | Optimistic error at or beyond the project severe threshold. | Lower is safer |
| Critical miss | A truly critical engine is not treated as critical by the policy. | Lower is safer |
| Operational recall | Share of critical engines captured by urgent/review behavior. | Higher is safer |
| Review workload | Share of engines routed into review or urgent action. | Trade-off |
| Uncertainty coverage | How often intervals contain benchmark labels under the calibration protocol. | Target-dependent |

### Why RMSE Alone Is Not Enough

RMSE is useful because it highlights large numerical errors, but it does not know whether a large error is operationally conservative or dangerously optimistic. It also does not know whether an error occurs far from maintenance thresholds or exactly near an urgent boundary. AeroGuard-PHM therefore reports severe optimism, critical misses, operational recall, review workload and uncertainty coverage alongside MAE and RMSE.

## Dataset: NASA C-MAPSS

NASA C-MAPSS is a simulated turbofan degradation benchmark. Each row represents one engine at one operating cycle. Every training engine is observed until failure, and every test engine stops before failure. The task is to estimate the missing future life for each test engine from the observed history.

The four public subsets differ by fault-mode and operating-regime complexity:

| Subset | Fault modes | Operating regimes | Relative difficulty | Role in this project |
| --- | --- | --- | --- | --- |
| FD001 | One | One | Easier single-regime baseline | Included in development, validation, benchmark evaluation and final reporting |
| FD002 | One | Multiple operating conditions | Harder because regime shifts can mask degradation | Included in development, validation, benchmark evaluation and final reporting |
| FD003 | Multiple fault modes | One | Harder fault diversity with stable operating regime | Included in development, validation, benchmark evaluation and final reporting |
| FD004 | Multiple fault modes | Multiple operating conditions | Hardest combined regime and fault variation | Included in development, validation, benchmark evaluation and final reporting |

FD002 and FD004 are more challenging because the same health state can produce different raw readings under different operating conditions. The release protocol evaluates all four subsets and reports overall metrics over 707 aligned benchmark engines. Earlier project phases also used development and external-style checks, but the final release tables are drawn from the frozen files under `reports/final_release`.

## End-to-End Data Pipeline

The implemented pipeline follows a past-only RUL workflow:

1. Raw engine-cycle records are read with canonical columns.
2. Schema validation checks required operating settings and sensor channels.
3. Operating-regime preprocessing assigns regime-aware normalized features.
4. RUL targets are constructed from run-to-failure training engines.
5. A 50-cycle temporal history is created for sequence models.
6. Rows are split by engine, not by individual cycle.
7. Model-ready tensors are sent to baselines, sequence models and release inference.

The frozen inference schema requires `cycle`, three `operational_setting_*` columns and `sensor_1` through `sensor_21`. Optional identifiers include `engine_id`, `global_engine_id`, `unit_id` and `subset`. For inference, histories shorter than 50 cycles are left-padded inside the model path and accompanied by a validity mask; histories shorter than the 10-cycle minimum remain valid only with warnings when no hard schema error is present. Histories longer than the manifest maximum of 500 cycles are warned about by validation.

Engine-level splitting is essential. Adjacent rows from the same engine are highly correlated, so random row splitting would leak future-like information into validation. The project uses grouped splitting and locked benchmark files to prevent that leakage.

Input validation is intentionally plain and conservative. The inference wrapper checks that required columns are present, that cycle values are numeric, unique and monotonically increasing, that required sensor and operating-setting columns are numeric and finite, and that unexpected columns are treated as warnings rather than silent assumptions. These checks do not prove that an input engine belongs to the training distribution, but they catch common integration mistakes before a prediction is treated as maintenance evidence.

Regime-aware preprocessing is also fitted as a release artifact rather than recomputed at inference time. That matters because normalization statistics must come from the training protocol, not from the engine being predicted. Re-fitting scalers on a test engine would leak information from the input distribution and make predictions difficult to reproduce. The frozen preprocessor keeps feature construction, operating-regime handling and model input order consistent across CLI, API and dashboard use.

### Temporal Patching

The selected Patch Transformer configuration receives a 50-cycle window. It uses 10-cycle patches with stride 5, producing 10 patch tokens over the window. Each patch summarizes a local segment of sensor behavior, and the Transformer relates those patch tokens before mean pooling and RUL regression. Patching reduces the effective sequence length while preserving local degradation structure.

## Models Evaluated

The final registry contains 64 model and system candidates. The table below groups the verified families rather than listing every candidate row.

| Model family | Representative models | What the model learns | Why evaluated | Main strength | Main limitation | Final role |
| --- | --- | --- | --- | --- | --- | --- |
| Classical tree baseline | Random Forest RUL Baseline | Nonlinear tabular mapping from engineered/final-row features to RUL | Establish a fast non-deep sanity check | Robust, quick, interpretable feature importance style diagnostics | Does not naturally model ordered temporal history | Classical benchmark |
| Deep vector baseline | Sequence MLP RUL Baseline | Fixed-window feature mapping | Tests whether a simple neural baseline is enough | Simple and fast | Weak sequence-order inductive bias | Deep-learning benchmark |
| Convolutional temporal models | Temporal CNN, TCN | Local temporal degradation motifs and dilated patterns | Evaluate local temporal filters | Efficient local pattern extraction | May miss global history relationships | Deep-learning benchmark |
| Recurrent sequence models | LSTM, GRU, CNN-LSTM | Sequential state over cycles | Evaluate standard temporal-memory models | Natural ordered-history processing | Can be harder to tune and parallelize | Deep-learning benchmark |
| Temporal Transformer models | Sinusoidal mean-pooling Transformer, learned attention Transformer | Attention over cycle-level history | Test long-range temporal interactions | Flexible sequence representation | Full sequence attention can be heavier than patched representation | Not selected |
| Patch Transformer models | 5x5 attention, Patch Transformer — 10×5 Patches with Mean Pooling | Attention over local temporal patches | Build a strong temporal benchmark | Efficient compact temporal representation | Still purely data-driven before physics guidance | Selected benchmark |
| Physics-guided Transformers | Monotonicity, cycle-rate, smoothness, health, regime, full physics variants | RUL plus consistency under selected guidance losses | Improve temporal model behavior using project-specific constraints | Better aligned with degradation/regime structure | Guidance must be validated and can trade off metrics | Selected predictive backbone |
| Residual correction controls | Linear residual corrector, MLP residual corrector, constant downward controls | Post-hoc correction magnitudes | Compare learned residual correction against simpler controls | Clear ablation evidence | Not complete deployed systems | Experimental controls |
| KAN correction systems | AeroKAN-PHM Compact Residual Corrector, Selective One-Sided AeroKAN Safety Corrector | Flexible nonlinear residual/safety corrections | Test interpretable nonlinear correction | Useful critical-case evidence | Did not provide the best final safety-accuracy-workload balance | Experimental correction models |
| Safety-guarded systems | Critical-Boundary Safety-Guarded Physics-Guided Transformer, AeroGuard-PHM Safety-Guarded RUL System | Deterministic boundary correction plus policy output | Convert RUL prediction into auditable decision support | Transparent, bounded, high critical recall | Benchmark-derived and needs external validation | Final safety system |

## Classical Machine-Learning Baselines

The verified classical baseline in the final registry is the Random Forest RUL Baseline. It is a tree-based machine-learning model rather than a deep sequence model. It uses tabular RUL features from earlier multidomain baseline work and predicts point RUL without learning an ordered sequence representation. This makes it valuable as a sanity check: if a classical model performs competitively, then the added complexity of deep temporal learning must justify itself.

Tree models can capture nonlinear feature interactions, tolerate mixed feature scales after preprocessing and run quickly. Their limitation is temporal structure. Unless the input includes engineered history summaries, a random forest does not naturally understand that cycle 48 follows cycle 47 or that a trend over 50 cycles matters differently from a single final reading. AeroGuard-PHM keeps this baseline because speed, interpretability and lower-complexity comparison are useful engineering evidence.

The registry also includes non-deployed residual controls such as the Linear Residual Corrector and constant downward one-sided controls. These are not standalone classical baselines for the full RUL task, but they are important comparison points for the KAN experiments. They test whether simple correction rules or regularized residual maps can provide the same safety benefit as a more flexible learned correction.

Classical baselines also keep the evaluation honest. They are fast enough to rerun, easy to inspect and hard to dismiss when a complex model only produces small gains. In this repository they provide a lower-complexity reference for the later sequence models and a reminder that better final decision support requires more than lowering an aggregate error metric.

The final registry does not present XGBoost, Support Vector Regression or k-nearest-neighbour RUL systems as release candidates. They are therefore not claimed here.

## Deep-Learning Sequence Models

The deep-learning family was broader than a single Transformer. The registry includes a Sequence MLP, Temporal CNN, LSTM, GRU, TCN, CNN-LSTM, temporal Transformers and Patch Transformers.

The MLP treats each prepared sequence as a fixed vector. It is useful because it shows whether nonlinear feature mixing alone can produce competitive RUL estimates. Its weakness is that it does not naturally preserve cycle order.

The Temporal CNN and TCN learn local and dilated temporal filters. They can detect degradation motifs over nearby cycles and broader temporal neighborhoods. They are efficient, but the learned filters may be less flexible than attention when the relevant evidence appears at separated parts of the engine history.

The LSTM and GRU process cycles sequentially and maintain hidden state. They are natural baselines for degradation sequences because they can accumulate evidence over time. The CNN-LSTM combines local convolutional feature extraction with recurrent temporal memory.

Transformer candidates use attention to relate different parts of the engine history. The Patch Transformer groups nearby cycles into patches, then applies attention to those compact tokens. This was the strongest temporal benchmark and the starting point for the physics-guided selected backbone.

The deep-learning comparison therefore answers several different questions. The MLP asks whether a fixed vector is enough. CNN and TCN candidates ask whether local temporal patterns are enough. LSTM and GRU candidates ask whether recurrent memory is enough. Transformer candidates ask whether attention over history helps, and Patch Transformers ask whether attention over compact local history segments is a better representation for this release.

## Patch Transformer

The strongest temporal benchmark is **Patch Transformer — 10×5 Patches with Mean Pooling**.

It receives a 50-cycle past-only history. The sequence is divided into 10-cycle patches with stride 5, producing 10 patch tokens. Each patch becomes a compact representation of local operating and sensor behavior. Transformer layers model relationships among patches, mean pooling combines temporal information, and a regression head predicts the base RUL.

Patching helps because it preserves short-term degradation structure while reducing the number of tokens the Transformer must compare. The release does not claim unsupported computational complexity advantages; it simply uses the architecture that was registered, evaluated and selected as the strong temporal benchmark.

## Physics-Guided Patch Transformer

The selected predictive backbone is the **Regime-Consistent Physics-Guided Patch Transformer**. In this project, "physics-guided" does not mean the model solves complete thermodynamic equations. It means the training and evaluation pipeline tested explicit consistency ideas that are relevant to turbofan degradation.

The physics-guided ablations include monotonicity-guided, cycle-rate-guided, smoothness-guided, health-guided, regime-consistent, temporally combined, full physics-guided and full physics-and-safety variants. The selected public backbone is the regime-consistent variant, whose active losses are recorded as data plus regime consistency. Its frozen architecture uses 50-cycle windows, 10-cycle patches, stride 5, 64-dimensional projection, two Transformer layers, four attention heads, 192 feedforward dimensions, dropout 0.15, learnable positional encoding and mean pooling.

Regime consistency matters because FD002 and FD004 contain multiple operating conditions. The model should not treat every operating-condition shift as degradation. The selected backbone improved overall MAE from 14.9550 to 14.4885 against the Patch Transformer benchmark and reduced severe optimism from 0.0651 to 0.0566. It was selected as the learned predictive layer, not as the final decision layer.

## KAN-Based Correction Experiments

Kolmogorov-Arnold Networks can be understood as networks that learn flexible one-dimensional nonlinear functions along connections rather than relying only on fixed activation functions. AeroGuard-PHM used them as experimental correction systems after the physics-guided Transformer, not as the final deployed model.

### AeroKAN-PHM Compact Residual Corrector

The AeroKAN-PHM Compact Residual Corrector receives engineering features and the base model output, then learns a residual correction. It was designed to test whether interpretable nonlinear residual functions could improve risky RUL predictions. It reduced critical misses to 5 and achieved operational recall of 0.9510, which is scientifically useful. However, it worsened overall point accuracy to MAE 15.4658 and RMSE 21.4713, increased severe optimism to 0.0863 and increased review workload to 0.1641. It was not selected because the global correction could damage calibration and overall accuracy.

### Selective One-Sided AeroKAN Safety Corrector

The Selective One-Sided AeroKAN Safety Corrector applies corrections only to selected risky cases and permits downward RUL corrections only. It preserved strong point accuracy with MAE 14.4485 and RMSE 20.9017, but it still produced 24 critical misses and operational recall of 0.7647 under the fixed policy. It remained an experimental branch because its learned risk selection did not identify enough critical engines to provide the desired safety coverage.

These KAN systems were not failures. They showed that global correction can harm calibration, that selective correction needs reliable risk detection, and that a simpler deterministic rule may be more auditable and robust for this release.

## Development Journey

The project moved through a deliberately broad model-development path:

1. Establish classical and simple baselines.
2. Evaluate deep temporal models.
3. Select a strong Patch Transformer benchmark.
4. Add operating-regime and physics consistency.
5. Explore interpretable KAN residual corrections.
6. Explore selective one-sided safety corrections.
7. Audit metric definitions and decision invariants.
8. Select a deterministic critical-boundary safeguard.
9. Freeze the complete release system.

![AeroGuard-PHM model development journey](docs/assets/readme/architecture/model_development_journey.png)

The journey is important because the final system was not chosen in advance. The evidence showed modest point-prediction gains from physics guidance, useful but imperfect KAN correction behavior, and a large critical-miss reduction from the deterministic boundary guard.

## Final Architecture

The final release pipeline is the **Critical-Boundary Safety-Guarded Physics-Guided Transformer** inside the complete **AeroGuard-PHM Safety-Guarded RUL System**.

```text
Engine sensor history
-> input validation
-> operating-regime-aware preprocessing
-> 50-cycle temporal history
-> temporal patches
-> Regime-Consistent Physics-Guided Patch Transformer
-> base_rul
-> Critical-Boundary Safety Guard
-> safety_adjusted_rul
-> global split conformal intervals
-> support and review logic
-> maintenance action
-> explanation
-> monitoring record
```

The learned predictive layer produces the base RUL. The deterministic safety and decision layer applies the guard, uncertainty intervals, support status and maintenance policy. The monitoring layer records structured inference information such as schema hash, row count, base RUL, adjusted RUL, correction amount, interval width, support score, guard activation, review status, maintenance action, latency and warning count.

## Final Safety Guard

The final safety guard is deterministic and auditable. It is active when:

```text
15 < base_rul <= 25
```

The frozen correction is:

```text
min(10, base_rul - 15 + 0.5)
```

The implemented final safety-adjusted RUL can be written as:

```text
base_rul - min(correction_bound, base_rul - (urgent_threshold - margin))
```

with `correction_bound: 10.0 cycles` and a margin of `0.5 cycles`.

Examples:

| Base RUL | Guard behavior | Result |
| ---: | --- | --- |
| 40 | Outside the boundary band | Unchanged |
| 23 | Guard applies a bounded downward correction | Moved closer to urgent review |
| 12 | Already critical | No additional boundary correction required |

The guard never increases RUL, is bounded, is deterministic, is auditable, does not modify the Transformer and only acts near the critical decision boundary. It makes no positive RUL corrections. It was selected over learned KAN correction because it gave better critical-case coverage, reduced critical misses substantially, avoided hidden learned behavior in the final safety layer and preserved overall accuracy reasonably well. It is not universally optimal; it is benchmark-derived and requires external validation before any operational use.

![Critical-boundary safety guard](docs/assets/readme/architecture/critical_boundary_safety_guard.png)

## Evaluation Protocol

The release separates validation experiments, final benchmark metrics, fixed-policy safety metrics and native end-to-end system metrics. Individual cycle rows are not randomly split because cycles from the same engine are temporally correlated. Engine-grouped splitting prevents a model from seeing nearly identical neighboring cycles in both training and validation.

Candidate selection was done through locked experiment outputs and release registries. The final reporting reads from `reports/final_release/headline_model_comparison.csv`, `model_registry.csv`, `point_prediction_comparison.csv`, `fixed_policy_safety_comparison.csv`, `native_system_comparison.csv`, `subset_metrics_comparison.csv` and the frozen manifest. Hash verification protects byte-sensitive release artifacts. Bootstrap and subset analyses are used where generated by the project reports, but the README does not mix those diagnostics with headline fixed-policy metrics.

The key distinction is this: validation metrics help choose candidates; benchmark metrics compare point prediction; fixed-policy safety metrics compare each candidate under the same urgent/schedule/inspection policy; native system metrics describe each system with its own locked release behavior. The selected release is frozen after those comparisons.

This separation prevents an easy mistake: selecting a model because one table looks good while another table exposes unacceptable operational behavior. AeroGuard-PHM records the point predictor, the correction layer, the uncertainty method and the policy as separate pieces, then evaluates their combined effect. That is why the final selection can prefer one critical miss and higher review workload over the lowest MAE row.

## Final Results

Lower is better for MAE, RMSE, NASA score, severe optimism and critical misses. Higher is better for operational recall. Review workload is a trade-off: lower workload is easier operationally, but too low can miss risky engines.

| Model | MAE | RMSE | NASA score | Severe optimism | Critical misses | Operational recall | Review workload | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Patch Transformer — 10×5 Patches with Mean Pooling | 14.9550 | 21.2671 | 8246.1682 | 0.0651 | 26 | 0.7451 | 0.1188 | Temporal benchmark |
| Regime-Consistent Physics-Guided Patch Transformer | 14.4885 | 20.9134 | 7577.1771 | **0.0566** | 25 | 0.7549 | 0.1259 | Selected predictive backbone |
| AeroKAN-PHM Compact Residual Corrector | 15.4658 | 21.4713 | 9516.8317 | 0.0863 | 5 | 0.9510 | 0.1641 | Experimental correction model |
| Selective One-Sided AeroKAN Safety Corrector | **14.4485** | **20.9017** | **7570.6490** | **0.0566** | 24 | 0.7647 | 0.1273 | Experimental selective correction |
| Critical-Boundary Safety-Guarded Physics-Guided Transformer | 14.5632 | 20.9603 | 7584.7765 | **0.0566** | **1** | **0.9902** | 0.1924 | Final selected system |

The final system does not dominate every metric. The selective one-sided KAN experiment has slightly lower MAE and RMSE. The final system was selected because it trades additional review workload for a much larger reduction in critical misses.

## Improvement Over the Patch Transformer

Compared with **Patch Transformer — 10×5 Patches with Mean Pooling**:

| Metric | Patch Transformer | Final system | Change |
| --- | ---: | ---: | ---: |
| MAE | 14.9550 | 14.5632 | -0.3918 cycles |
| RMSE | 21.2671 | 20.9603 | -0.3067 cycles |
| Critical misses | 26 | 1 | -25 |
| Operational recall | 0.7451 | 0.9902 | +0.2451 |
| Review workload | 0.1188 | 0.1924 | +0.0736 |

The point-prediction improvement is modest. The critical-case protection improvement is substantial. The price is a higher review workload, moving from 84 reviewed/urgent engines under the Patch Transformer fixed policy to 136 for the final system.

## Subset Results

The final release subset file reports subset rows for the Patch Transformer, physics-guided backbone and KAN experiment families. It does not provide a separate subset row for the final deterministic guard, so this section reports only verified subset-level rows.

| System | FD001 MAE | FD002 MAE | FD003 MAE | FD004 MAE | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Patch Transformer benchmark | 11.5597 | 16.5261 | 11.7657 | 15.9693 | Strong temporal benchmark, harder on multi-regime subsets |
| Regime-Consistent Physics-Guided Patch Transformer | 11.1245 | 15.8810 | 11.3794 | 15.6444 | Improves MAE on all four subset rows in the release table |
| AeroKAN-PHM Compact Residual Corrector | 12.8155 | 16.5130 | 13.1621 | 16.3698 | Improves some safety behavior but worsens subset MAE |
| Selective One-Sided AeroKAN Safety Corrector | 11.1245 | 15.8404 | 11.3794 | 15.5729 | Preserves point accuracy but does not capture enough critical engines |

FD002 and FD004 remain the most difficult because operating-regime variation makes raw sensor trajectories less directly comparable.

## Safety and Operational Metrics

The fixed-policy safety comparison uses the same policy for every headline system:

| System | Critical engines | Critical misses | Operational recall | Urgent precision | Reviews | Review workload |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Patch Transformer — 10×5 Patches with Mean Pooling | 102 | 26 | 0.7451 | 0.9048 | 84 | 0.1188 |
| Regime-Consistent Physics-Guided Patch Transformer | 102 | 25 | 0.7549 | 0.8652 | 89 | 0.1259 |
| AeroKAN-PHM Compact Residual Corrector | 102 | 5 | 0.9510 | 0.8362 | 116 | 0.1641 |
| Selective One-Sided AeroKAN Safety Corrector | 102 | 24 | 0.7647 | 0.8667 | 90 | 0.1273 |
| Critical-Boundary Safety-Guarded Physics-Guided Transformer | 102 | 1 | 0.9902 | 0.7426 | 136 | 0.1924 |

This table is the clearest view of the safety-accuracy-workload trade-off. The final system routes more engines to review, reducing urgent precision, but it captures nearly all critical engines under the fixed policy.

## Uncertainty Quantification

A point prediction gives one number, but the model may be uncertain. AeroGuard-PHM uses global split conformal uncertainty. Conformal calibration adds intervals around the safety-adjusted prediction using held-out calibration errors from the release protocol. The frozen nominal levels are 80%, 90% and 95%.

| Nominal level | Frozen radius |
| ---: | ---: |
| 80% | 32.1584 |
| 90% | 53.5571 |
| 95% | 86.5394 |

For a safety-adjusted RUL of 50, the 90% interval is approximately 50 +/- 53.5571 cycles before lower-bound clipping. The lower and upper bounds are not probabilities of failure for a single engine. They are calibrated error intervals under the held-out distribution used by the release. If deployed data drift away from that distribution, coverage must be rechecked.

## Review and Maintenance Policy

The frozen maintenance policy uses `policy_id: point_u15_s30_i60`. It converts the safety-adjusted RUL into a recommendation and review status. The thresholds are:

| Predicted condition | Guard/support context | Recommended action | Explanation |
| --- | --- | --- | --- |
| `RUL <= 15` | Critical or already inside urgent band | `URGENT_ENGINEERING_REVIEW` | Immediate qualified review is required |
| `15 < RUL <= 30` | Near-term maintenance band | `SCHEDULE_MAINTENANCE` | Schedule maintenance planning |
| `30 < RUL <= 60` | Inspection planning band | `PLAN_INSPECTION` | Plan inspection and continue tracking |
| `RUL > 60` | Supported and outside near-term bands | `CONTINUE_MONITORING` | Continue routine monitoring |

The policy uses the point estimate after the safety guard. Support status and warnings still matter: unsupported inputs should not be trusted as normal predictions, and limited-support outputs should be reviewed before action.

## Monitoring

The implemented monitoring layer is local and structured rather than a live fleet platform. Each prediction can include a monitoring record with timestamp, engine ID, model version, input schema hash, input row count, operating regime, base RUL, adjusted RUL, correction amount, 90% interval width, support score, guard activation, review status, maintenance action, latency and warning count.

The monitoring specification also names input-schema drift, missing-sensor rate, feature-range violations, regime-distribution drift, support-score drift, prediction-distribution drift, interval-width drift, safety-guard activation rate, review rate, maintenance-action distribution, RUL trajectory instability, inference latency and runtime failures.

In practical terms, those signals answer questions an engineer would ask after deployment. Are incoming files still shaped like the release schema? Are operating regimes appearing in proportions similar to the benchmark data? Is the safety guard activating far more often than expected? Are interval widths widening? Are warnings becoming common? A real production system would place these answers on persistent dashboards and alerting channels; this repository provides the structured fields and drift vocabulary needed to build that layer.

What is not implemented yet: real fleet telemetry ingestion, a central monitoring platform, automated alert routing, on-call procedures, automated recalibration and real aircraft maintenance integration.

## Quick Start

Clone and install:

```powershell
git clone https://github.com/rithvik123/aeroguard-phm.git
cd aeroguard-phm
python -m pip install -e ".[all]"
```

Verify the frozen release:

```powershell
python scripts/release_integrity_check.py
python -m pytest tests/unit -q
```

Run one inference call through the CLI:

```powershell
aeroguard-infer --manifest artifacts/final_release/frozen_system_manifest.json --input examples/sample_engine_history.csv --output examples/sample_prediction.json
```

The same CLI can also be called as a module:

```powershell
python -m aeroguard.inference.cli --manifest artifacts/final_release/frozen_system_manifest.json --input examples/sample_engine_history.csv --output examples/sample_prediction.json
```

## Python Usage

```python
import pandas as pd

from aeroguard.inference import AeroGuardPredictor

predictor = AeroGuardPredictor.from_manifest(
    "artifacts/final_release/frozen_system_manifest.json",
    device="cpu",
)
history = pd.read_csv("examples/sample_engine_history.csv")
prediction = predictor.predict_engine(history)

print(prediction["base_rul"])
print(prediction["safety_adjusted_rul"])
print(prediction["lower_90"], prediction["upper_90"])
print(prediction["maintenance_action"])
print(prediction["explanation"])
```

The API returns the base RUL, safety-adjusted RUL, conformal intervals, support status, safety-guard status, maintenance action, warnings, explanation and monitoring record.

## CLI Usage

The console entry point is `aeroguard-infer`. Required arguments are the frozen manifest plus either `--input` for one engine-history CSV or `--batch-dir` for a directory of CSVs. Useful optional arguments include `--output`, `--output-json`, `--output-csv`, `--device`, `--explanation-level` and `--validation-only`.

```powershell
aeroguard-infer --manifest artifacts/final_release/frozen_system_manifest.json --input examples/sample_engine_history.csv
```

Validation-only mode checks the input schema without running prediction:

```powershell
aeroguard-infer --manifest artifacts/final_release/frozen_system_manifest.json --input examples/sample_engine_history.csv --validation-only
```

## FastAPI Inference Service

Run the service:

```powershell
python -m uvicorn aeroguard.api.app:app --host 127.0.0.1 --port 8000
```

Implemented endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Reports manifest availability and model-load state |
| `GET /model` | Returns release metadata, required columns and hash-mismatch count |
| `POST /validate-input` | Validates an engine-history payload |
| `POST /predict` | Runs one engine-history prediction |
| `POST /predict-batch` | Runs multiple engine-history predictions |

The request body uses records shaped like the CSV schema: one engine with a list of cycle records for `/predict`, or multiple engines for `/predict-batch`.

## Interactive Streamlit Dashboard

Run the dashboard:

```powershell
python -m streamlit run dashboard/app.py
```

The Streamlit interface supports the bundled example or an uploaded engine-history CSV. It validates schema, displays sensor trajectories, shows base RUL, safety-adjusted RUL, uncertainty interval width, support status, guard status, review status, maintenance recommendation, explanations, warnings and a downloadable prediction JSON.

Screenshots are not rendered in this README until real captures exist. See the commented screenshot plan near the end of this document, `docs/assets/readme/README_IMAGE_PLAN.md`, and `docs/assets/README_IMAGE_MAPPING.md`.

## Production Inference Output

The frozen output schema is designed for decision support and monitoring:

```json
{
  "engine_id": "engine_0",
  "model_version": "aeroguard-phm-safety-v1",
  "base_rul": 24.2,
  "safety_adjusted_rul": 14.5,
  "correction_cycles": 9.7,
  "lower_90": 0.0,
  "upper_90": 68.1,
  "interval_width_90": 107.1,
  "operating_regime": 3,
  "support_status": "supported",
  "support_score": 1.0,
  "safety_guard_activated": true,
  "review_required": true,
  "maintenance_action": "URGENT_ENGINEERING_REVIEW",
  "warnings": [],
  "explanation": [
    "Critical-boundary guard activated near the urgent threshold.",
    "Maintenance policy selected urgent engineering review."
  ]
}
```

Benchmark labels are intentionally not part of production inference output. The service returns model and policy fields that can be audited without exposing evaluation-only labels.

## Repository Structure

```text
.
├── src/aeroguard/              # Package code: data, features, models, inference, API
├── configs/                    # Experiment and pipeline configuration
├── tests/                      # Unit and integration tests
├── dashboard/                  # Streamlit dashboard
├── examples/                   # Sample engine-history and prediction files
├── scripts/                    # Validation and release-integrity scripts
├── docs/                       # References, archives and README assets
├── reports/final_release/      # Frozen comparison tables and release reports
├── artifacts/final_release/    # Frozen manifest and release bundle metadata
├── README.md                   # Public project overview
├── MODEL_CARD.md               # Model-card style release notes
├── REPRODUCIBILITY.md          # Reproducibility instructions
├── LICENSE                     # Apache-2.0 license
├── CITATION.cff                # Citation metadata
└── THIRD_PARTY_NOTICES.md      # Third-party notices
```

## Testing

The release test suite covers README validation, frozen source hashes, model registry consistency, final-release metric tables, inference schema checks, CLI smoke behavior, API import and endpoint metadata, Streamlit import safety, monitoring output and the Phase 6 integration smoke path. CI runs on pushes and pull requests.

Common local checks:

```powershell
python scripts/validate_readme.py
python scripts/release_integrity_check.py
python -m pytest tests/unit/test_readme_release.py -q
python -m pytest tests/unit/test_final_release.py -q
python -m pytest tests/integration/test_phase6_final_release_smoke.py -q
git diff --check
```

## Reproducibility and Frozen Artifacts

The release is frozen by manifest, file paths and SHA-256 hashes. The frozen manifest identifies the physics-guided checkpoint, final preprocessor, critical-boundary guard configuration, conformal uncertainty model, maintenance policy and metric-definition registry. Release-integrity validation checks that protected files have not been rewritten or line-ending-normalized.

The byte-sensitive files include the critical-boundary guard metadata and metric-definition audit. Do not regenerate the frozen manifest, alter benchmark CSVs or rewrite protected artifacts unless a new release is intentionally prepared.

## Limitations

C-MAPSS is simulated. Real aircraft degradation may differ. The safety guard is benchmark-derived. External datasets are still required. Conformal calibration depends on calibration-distribution similarity. The project is not aviation certified. Maintenance decisions require qualified human oversight. Reported results are protocol-specific. Published results from other papers may not be directly comparable unless the preprocessing, split protocol and metric definitions match. Real deployment monitoring has not yet been performed.

## Future Work

Future work includes external evaluation on N-CMAPSS or another independent dataset, protocol-matched reproduction of strong public baselines, real-world transfer and domain adaptation, improved risk-selection models, cost-sensitive maintenance-policy optimisation, better uncertainty under domain shift, Docker/containerization, a live hosted demonstration, persistent operational monitoring, model-card automation and additional explainability studies.

## Responsible Use

AeroGuard-PHM is research decision-support software. It should not be used as an autonomous maintenance authority. Any real deployment would require independent validation, qualified engineering review, operational monitoring, documented escalation procedures and compliance with applicable aviation safety processes.

## Citation and License

Please cite the NASA C-MAPSS Turbofan Engine Degradation Simulation Data Set for the benchmark data and this repository for the AeroGuard-PHM implementation. AeroGuard-PHM is released under the Apache License 2.0. See `LICENSE`, `THIRD_PARTY_NOTICES.md` and `CITATION.cff`.

<!--
README IMAGE PLACEHOLDERS AND SCREENSHOT GALLERY TO ENABLE AFTER REAL SCREENSHOTS ARE ADDED

Visible, approved images:
- docs/assets/readme/hero/aeroguard_phm_hero.png
- docs/assets/readme/architecture/model_development_journey.png
- docs/assets/readme/architecture/critical_boundary_safety_guard.png

Approved or planned assets not currently rendered:
- docs/assets/readme/hero/rul_problem_statement.png
- docs/assets/readme/architecture/final_system_design.png
- docs/assets/readme/architecture/physics_guided_patch_transformer.png
- docs/assets/readme/architecture/uncertainty_maintenance_flow.png
- docs/assets/readme/architecture/deployment_monitoring.png
- docs/assets/readme/screenshots/streamlit_01_home.png
- docs/assets/readme/screenshots/streamlit_02_input_validation.png
- docs/assets/readme/screenshots/streamlit_03_sensor_history.png
- docs/assets/readme/screenshots/streamlit_04_prediction_result.png
- docs/assets/readme/screenshots/streamlit_05_maintenance_explanation.png
- docs/assets/readme/screenshots/fastapi_swagger.png
-->
