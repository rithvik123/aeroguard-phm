# README Image Mapping

This mapping records the five generated README images found in the original `images/` folder and their public release treatment. Images were visually inspected before README placement.

| Current filename | Recommended public filename | Dimensions | Aspect ratio | Intended README section | Alt text | Caption | Validation status | Spelling or visual issues | Public-release suitability |
| --- | --- | ---: | ---: | --- | --- | --- | --- | --- | --- |
| `IMAGE 1 — AEROGUARD-PHM HERO BANNER.png` | `docs/assets/readme/hero/aeroguard_phm_hero.png` | 2752 x 1536 | 1.79:1 | Hero banner | AeroGuard-PHM converts turbofan sensor histories into uncertainty-aware and safety-adjusted maintenance decisions | Main AeroGuard-PHM workflow from sensor history to safety-adjusted maintenance decision. | Excluded from visible README rendering | Contains visible generated-text typo: `Concluse maintenance recommendation`. | Not suitable for visible public README until regenerated or corrected. Asset retained only for provenance/replacement planning. |
| `IMAGE 2 — RUL PROBLEM STATEMENT.png` | `docs/assets/readme/hero/rul_problem_statement.png` | 2752 x 1536 | 1.79:1 | What AeroGuard-PHM Solves | RUL prediction problem showing past-only sensor history, hidden degradation, uncertainty and optimistic prediction risk | RUL prediction requires separating true degradation from sensor changes caused by operating conditions, then estimating the unknown time to failure from past-only sensor history. | Excluded from visible README rendering | Contains visible generated-text errors: `Layout layout` and `future failure prnnt point`. | Not suitable for visible public README until regenerated or corrected. Asset retained only for provenance/replacement planning. |
| `IMAGE 3 — FINAL AEROGUARD-PHM SYSTEM DESIGN.png` | `docs/assets/readme/architecture/final_system_design.png` | 2752 x 1536 | 1.79:1 | Final Architecture | Final AeroGuard-PHM inference architecture from preprocessing through monitoring | Production inference separates the frozen learned predictor from the deterministic safety, uncertainty, maintenance and monitoring layers. | Excluded from visible README rendering | Main boxes are usable, but the subtitle line contains heavily garbled generated text including `maintenrrance` and other nonsensical words. | Not suitable for visible public README until regenerated or corrected. Asset retained only for provenance/replacement planning. |
| `Image 4 — Critical-Boundary Safety Guard.png` | `docs/assets/readme/architecture/critical_boundary_safety_guard.png` | 2752 x 1536 | 1.79:1 | Final Safety Guard | Critical-boundary safety guard showing unchanged, bounded-correction and critical RUL regions | The safety guard applies only a bounded downward adjustment near the critical RUL boundary; it never increases the base prediction. | Approved for visible README rendering | No obvious spelling errors, private paths or incorrect KAN deployment claim observed. Safety ranges and correction direction match the frozen rule. | Suitable for public README. |
| `Image 5 — Model Development Journey.png` | `docs/assets/readme/architecture/model_development_journey.png` | 2749 x 1390 | 1.98:1 | Development Journey | AeroGuard-PHM model development journey from Patch Transformer to physics guidance, KAN experiments and final safety guard | AeroGuard-PHM progressed from temporal modelling to physics guidance, interpretable correction experiments and a final auditable safety safeguard. | Approved for visible README rendering | No serious generated-text errors observed. KAN stages are labeled experimental/not selected and final system is safety-guarded, not KAN-deployed. | Suitable for public README. |

## Visible README Images

- `docs/assets/readme/architecture/model_development_journey.png`
- `docs/assets/readme/architecture/critical_boundary_safety_guard.png`

## Non-Rendered Assets Pending Regeneration

- `docs/assets/readme/hero/aeroguard_phm_hero.png`
- `docs/assets/readme/hero/rul_problem_statement.png`
- `docs/assets/readme/architecture/final_system_design.png`

These non-rendered assets must not be enabled in the README until the documented generated-text errors are corrected.

