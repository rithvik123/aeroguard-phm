"""Streamlit dashboard for the AeroGuard-PHM final release."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

try:  # pragma: no cover - optional UI dependency
    import streamlit as st
except Exception:  # pragma: no cover
    st = None

from aeroguard.inference.predictor import AeroGuardPredictor


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "artifacts" / "final_release" / "frozen_system_manifest.json"
DEFAULT_EXAMPLE = ROOT / "examples" / "sample_engine_history.csv"


def load_predictor(manifest_path: Path = DEFAULT_MANIFEST) -> AeroGuardPredictor:
    return AeroGuardPredictor.from_manifest(manifest_path)


def run_dashboard() -> None:
    if st is None:
        raise RuntimeError("Streamlit is not installed. Install streamlit to run the dashboard.")
    st.set_page_config(page_title="AeroGuard-PHM", layout="wide")
    st.title("AeroGuard-PHM Safety-Guarded RUL System")

    manifest_path = Path(st.sidebar.text_input("Manifest", str(DEFAULT_MANIFEST)))
    source = st.sidebar.radio("Input source", ["Bundled example", "Upload CSV"])
    uploaded = st.sidebar.file_uploader("Engine-history CSV", type=["csv"]) if source == "Upload CSV" else None

    if source == "Bundled example":
        frame = pd.read_csv(DEFAULT_EXAMPLE)
    elif uploaded is not None:
        frame = pd.read_csv(uploaded)
    else:
        st.info("Upload an engine-history CSV to run inference.")
        return

    predictor = load_predictor(manifest_path)
    result = predictor.predict_engine(frame)

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Base RUL", f"{result.get('base_rul', 0):.1f}")
    col_b.metric("Safety-Adjusted RUL", f"{result.get('safety_adjusted_rul', 0):.1f}")
    col_c.metric("90% Interval Width", f"{result.get('interval_width_90', 0):.1f}")
    col_d.metric("Support", str(result.get("support_status", "unknown")))

    st.subheader("Sensor Trajectories")
    sensor_columns = [column for column in frame.columns if column.startswith("sensor_")]
    if sensor_columns:
        selected = st.multiselect("Sensors", sensor_columns, default=sensor_columns[:4])
        if selected:
            st.line_chart(frame.set_index("cycle")[selected])

    st.subheader("Decision")
    st.write(
        {
            "operating_regime": result.get("operating_regime"),
            "safety_guard_activated": result.get("safety_guard_activated"),
            "review_required": result.get("review_required"),
            "maintenance_action": result.get("maintenance_action"),
        }
    )
    st.subheader("Explanation")
    for line in result.get("explanation", []):
        st.write(f"- {line}")

    warnings = result.get("warnings", [])
    if warnings:
        st.subheader("Validation Warnings")
        st.dataframe(pd.DataFrame(warnings), use_container_width=True)

    st.download_button(
        "Download prediction JSON",
        data=json.dumps(result, indent=2, default=str),
        file_name="aeroguard_prediction.json",
        mime="application/json",
    )


def main() -> None:
    run_dashboard()


if __name__ == "__main__":
    main()
