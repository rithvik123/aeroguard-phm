# README Image Plan

This plan lists the visual assets intended for the AeroGuard-PHM README. No images or screenshots have been generated as part of the README rewrite. Missing assets are represented in `README.md` only as HTML comments, so the public README does not render broken images.

## Generation and Capture Rules

- Generated visuals should be created outside this repository workflow and then copied into the exact paths below.
- Streamlit and FastAPI screenshots should be captured from real local runs of the frozen v1.0.0 application.
- Do not fabricate UI screenshots.
- Do not replace benchmark tables with images; tables remain text for auditability.
- After adding assets, uncomment only the corresponding README image block or screenshot-gallery rows that now point to existing files.

## Gemini-Generated Visuals

| File | Intended README section | Purpose | Suggested size | Alt text | Source type | Insertion location |
| --- | --- | --- | --- | --- | --- | --- |
| `docs/assets/readme/hero/aeroguard_phm_hero.png` | Top of README | Main hero banner showing sensor history, safety-guarded RUL prediction and review action. | 1600 x 700 | AeroGuard-PHM sensor-to-maintenance workflow | Gemini-generated conceptual visual | Directly below the frozen v1.0.0 status line |
| `docs/assets/readme/hero/rul_problem_statement.png` | What AeroGuard-PHM Solves | Explain observed engine history, unknown failure point and RUL target. | 1400 x 800 | Remaining useful life prediction problem | Gemini-generated explanatory visual | End of the problem-statement section |
| `docs/assets/readme/architecture/model_development_journey.png` | Development Journey | Summarize the path from patch Transformer to physics-guided backbone, KAN experiments and final safety guard. | 1600 x 900 | AeroGuard-PHM model development journey | Gemini-generated architecture timeline | End of Development Journey |
| `docs/assets/readme/architecture/final_system_design.png` | Final Architecture | Show production flow from CSV validation to adjusted RUL, uncertainty, policy and monitoring. | 1800 x 1100 | AeroGuard-PHM final system architecture | Gemini-generated architecture diagram | After the Mermaid architecture diagram |
| `docs/assets/readme/architecture/physics_guided_patch_transformer.png` | Physics-Guided Patch Transformer | Explain 50-cycle windows, 10-cycle patches, Transformer encoder and RUL output. | 1600 x 900 | Physics-guided patch Transformer inference flow | Gemini-generated model diagram | End of backbone section |
| `docs/assets/readme/architecture/critical_boundary_safety_guard.png` | Final Safety Guard | Visualize the `15 < base_rul <= 25` boundary rule and downward-only correction. | 1400 x 900 | Critical-boundary safety guard rule | Gemini-generated flowchart | End of safety-guard section |
| `docs/assets/readme/architecture/uncertainty_maintenance_flow.png` | Uncertainty Quantification | Show conformal intervals feeding review and maintenance thresholds. | 1500 x 900 | Uncertainty and maintenance policy flow | Gemini-generated explanatory diagram | End of uncertainty section |
| `docs/assets/readme/architecture/deployment_monitoring.png` | Monitoring | Show request logs, drift checks, guard rate, review workload and latency tracking. | 1600 x 900 | Deployment monitoring flow for AeroGuard-PHM | Gemini-generated monitoring diagram | End of monitoring section |

## Real Streamlit Screenshots

| File | Intended README section | Purpose | Suggested size | Alt text | Source type | Capture instruction |
| --- | --- | --- | --- | --- | --- | --- |
| `docs/assets/readme/screenshots/streamlit_01_home.png` | Interactive Streamlit Dashboard | Show app landing state and bundled example selection. | 1440 x 1000 | Streamlit home view | Real screenshot | Run `python -m streamlit run dashboard/app.py` and capture the initial page |
| `docs/assets/readme/screenshots/streamlit_02_input_validation.png` | Interactive Streamlit Dashboard | Show schema validation and warnings area. | 1440 x 1000 | Streamlit input validation view | Real screenshot | Upload or load the bundled sample, then capture validation output |
| `docs/assets/readme/screenshots/streamlit_03_sensor_history.png` | Interactive Streamlit Dashboard | Show sensor trajectory visualization. | 1440 x 1000 | Streamlit sensor-history visualization | Real screenshot | Scroll to or select the sensor-history view |
| `docs/assets/readme/screenshots/streamlit_04_prediction_result.png` | Interactive Streamlit Dashboard | Show base RUL, adjusted RUL and uncertainty intervals. | 1440 x 1000 | Streamlit prediction result view | Real screenshot | Run the bundled sample prediction and capture the result area |
| `docs/assets/readme/screenshots/streamlit_05_maintenance_explanation.png` | Interactive Streamlit Dashboard | Show maintenance action and explanation output. | 1440 x 1000 | Streamlit maintenance explanation view | Real screenshot | Capture the decision-support/explanation section |
## Real FastAPI Screenshot

| File | Intended README section | Purpose | Suggested size | Alt text | Source type | Capture instruction |
| --- | --- | --- | --- | --- | --- | --- |
| `docs/assets/readme/screenshots/fastapi_swagger.png` | FastAPI Inference Service | Show the live OpenAPI/Swagger page with the frozen endpoints. | 1440 x 1000 | FastAPI Swagger UI for AeroGuard-PHM | Real screenshot | Run `python -m uvicorn aeroguard.api.app:app --host 127.0.0.1 --port 8000` and capture `http://127.0.0.1:8000/docs` |

## Optional Future Chart Assets

The README currently uses markdown tables for release metrics. If charts are later added, place them in `docs/assets/readme/charts/` and keep the source CSV table references next to the chart.

Suggested optional chart filenames:

- `docs/assets/readme/charts/headline_metric_comparison.png`
- `docs/assets/readme/charts/critical_miss_reduction.png`
- `docs/assets/readme/charts/subset_performance.png`
- `docs/assets/readme/charts/uncertainty_interval_widths.png`
