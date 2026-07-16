# README Screenshot Checklist

The generated conceptual images are not substitutes for real application screenshots. Capture the following only from real local runs of the frozen AeroGuard-PHM v1.0.0 application.

## Streamlit

Run:

```powershell
$env:PYTHONPATH = ".\src"
python -m streamlit run dashboard/app.py
```

Expected screenshot files:

| File | Capture |
| --- | --- |
| `docs/assets/readme/screenshots/streamlit_01_home.png` | Initial dashboard home state and bundled example selector. |
| `docs/assets/readme/screenshots/streamlit_02_input_validation.png` | Input validation status after loading the example or an uploaded CSV. |
| `docs/assets/readme/screenshots/streamlit_03_sensor_history.png` | Sensor-history visualization. |
| `docs/assets/readme/screenshots/streamlit_04_prediction_result.png` | Base RUL, safety-adjusted RUL and interval results. |
| `docs/assets/readme/screenshots/streamlit_05_maintenance_explanation.png` | Maintenance action and explanation text. |

## FastAPI

Run:

```powershell
$env:PYTHONPATH = ".\src"
python -m uvicorn aeroguard.api.app:app --host 127.0.0.1 --port 8000
```

Expected screenshot file:

| File | Capture |
| --- | --- |
| `docs/assets/readme/screenshots/fastapi_swagger.png` | `http://127.0.0.1:8000/docs` with the frozen endpoints visible. |

## Rules

- Do not add broken image links.
- Do not use generated conceptual images as fake UI screenshots.
- Add screenshots only after confirming labels and endpoint names are readable.

