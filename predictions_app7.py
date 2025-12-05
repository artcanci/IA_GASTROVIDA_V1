

# STREAMLIT APP WITH AUTO-BMI, BMI ZONES, WEIGHT PREDICTION, BMI PROGRESSION, AND %TWL

import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import joblib
import plotly.graph_objects as go
from plotly.colors import qualitative


# ==========================================
# FIX FOR REMOVED NUMPY ALIASES
# ==========================================
if not hasattr(np, "float"): np.float = float
if not hasattr(np, "int"): np.int = int
if not hasattr(np, "bool"): np.bool = bool
if not hasattr(np, "object"): np.object = object
if not hasattr(np, "str"): np.str = str

# ==========================================
# ABSOLUTE PATH FIXES  (IMPORTANT!)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FOLDER = os.path.join(BASE_DIR, "models")
PERFORMANCE_FILE = os.path.join(BASE_DIR, "Performance.xlsx")
LOGO_PATH = os.path.join(BASE_DIR, "logo.jpeg")

# ==========================================
# TARGET ORDER
# ==========================================
TARGETS_ORDER = [
    "1M", "3M", "6M", "9M",
    "1A", "2A", "3A", "4A", "5A",
    "6A", "7A", "8A", "9A", "10A"
]

# ==========================================
# SCAN AND LOAD MODELS
# ==========================================
def scan_model_filenames():
    # SAFE: ensures folder exists
    if not os.path.exists(MODEL_FOLDER):
        st.error(f"Models folder not found: {MODEL_FOLDER}")
        return {}

    files = [
        f for f in os.listdir(MODEL_FOLDER)
        if f.lower().endswith(".pkl")
    ]

    pattern = r"^(1M|3M|6M|9M|1A|2A|3A|4A|5A|6A|7A|8A|9A|10A)_(.+)\.pkl$"

    models_by_type = {}
    for fname in files:
        match = re.match(pattern, fname)
        if not match:
            continue
        target, mtype = match.group(1), match.group(2)
        if mtype not in models_by_type:
            models_by_type[mtype] = {}
        models_by_type[mtype][target] = fname

    return models_by_type


@st.cache_resource
def load_all_models(models_by_type):
    loaded = {}

    for mtype, targets in models_by_type.items():
        loaded[mtype] = {}
        for target, fname in targets.items():
            fullpath = os.path.join(MODEL_FOLDER, fname)

            try:
                loaded[mtype][target] = joblib.load(fullpath)
            except Exception as e:
                loaded[mtype][target] = None
                st.error(f"❌ Failed to load {fname}: {e}")

    return loaded


@st.cache_data
def load_performance():
    if not os.path.exists(PERFORMANCE_FILE):
        st.error(f"Performance.xlsx not found at {PERFORMANCE_FILE}")
        return pd.DataFrame()

    return pd.read_excel(PERFORMANCE_FILE)

# ==========================================
# STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Weight Loss Prediction", layout="wide")

# SAFETY CHECK FOR LOGO FILE
if os.path.exists(LOGO_PATH):
    st.image(LOGO_PATH, width=600)
else:
    st.error(f"Logo file not found: {LOGO_PATH}")

st.title("📈 Multi-Model Bariatric Surgery Weight Prediction")

# LOAD MODELS
models_by_type = scan_model_filenames()
available_models = sorted(models_by_type.keys())
LOADED = load_all_models(models_by_type)
perf_df = load_performance()

# ------------------------------------------
# SIDEBAR
# ------------------------------------------
st.sidebar.header("⚙️ Configuration")
default_model = ["XGBoost"] if "XGBoost" in available_models else available_models[:1]

selected_models = st.sidebar.multiselect(
    "Select model types:", available_models, default=default_model
)

st.sidebar.subheader("🔢 Input variables")

IDADE = st.sidebar.number_input("Age", 1, 120, 30)
SEXO = 1 if st.sidebar.selectbox("Gender", ["Female", "Male"]) == "Female" else 0

ALTURA = st.sidebar.number_input("Height (m)", 1.2, 2.2, 1.70)
P_INICIAL = st.sidebar.number_input("Initial Weight (kg)", 30.0, 300.0, 120.0)

IMC = P_INICIAL / (ALTURA ** 2) if ALTURA > 0 else 0.0

st.sidebar.number_input(
    "BMI (auto-calculated: weight / height²)",
    value=float(round(IMC, 2)),
    disabled=True
)

# INPUT DATAFRAME
input_df = pd.DataFrame(
    [[IDADE, IMC, ALTURA, P_INICIAL, SEXO]],
    columns=["IDADE", "IMC", "ALTURA", "P INICIAL", "S"]
)

# ==========================================
# SESSION STATE
# ==========================================
if "run_pred" not in st.session_state:
    st.session_state.run_pred = False

if st.button("🔮 Run predictions"):
    st.session_state.run_pred = True

color_map = {
    m: qualitative.Plotly[i % len(qualitative.Plotly)]
    for i, m in enumerate(available_models)
}

# ==========================================
# RUN PREDICTIONS
# ==========================================
if st.session_state.run_pred:

    predictions = {m: [] for m in selected_models}

    for mtype in selected_models:
        for tgt in TARGETS_ORDER:
            model = LOADED.get(mtype, {}).get(tgt)
            if model is None:
                predictions[mtype].append(np.nan)
            else:
                predictions[mtype].append(float(model.predict(input_df)[0]))

    results_df = pd.DataFrame(predictions, index=TARGETS_ORDER)
    st.subheader("📋 Predictions Table")
    st.dataframe(results_df)

    x_vals = ["0M"] + TARGETS_ORDER

    # ======================================================
    # GRAPH 1 — WEIGHT PREDICTION
    # ======================================================
    st.subheader("📉 Weight Prediction after Bariatric Surgery")
    fig_weight = go.Figure()

    for mtype in selected_models:
        base_color = color_map[mtype]
        y_main = [P_INICIAL] + predictions[mtype]

        upper = [P_INICIAL]
        lower = [P_INICIAL]

        for t, pred in zip(TARGETS_ORDER, predictions[mtype]):
            row = perf_df[(perf_df['TARGET'] == t) & (perf_df['MODEL'] == mtype)]
            if len(row) > 0:
                mae = float(row['MAE'])
                std = float(row['MAE_STD'])
                err = mae + 3 * std
                upper.append(pred + err)
                lower.append(pred - err)
            else:
                upper.append(pred)
                lower.append(pred)

        r = int(base_color[1:3], 16)
        g = int(base_color[3:5], 16)
        b = int(base_color[5:7], 16)

        rgba_line = f"rgba({r},{g},{b},0.7)"
        rgba_light = f"rgba({r},{g},{b},0.18)"

        # MAIN LINE
        fig_weight.add_trace(go.Scatter(
            x=x_vals, y=y_main, mode="lines+markers", name=mtype,
            legendgroup=mtype,
            line=dict(width=3, color=rgba_line, shape="spline", smoothing=1.2),
            marker=dict(size=8, color=rgba_line)
        ))

        # UPPER ERROR
        fig_weight.add_trace(go.Scatter(
            x=x_vals, y=upper, mode="lines",
            legendgroup=mtype, showlegend=False,
            line=dict(width=0.1, color=rgba_line, shape="spline", smoothing=1.2),
            opacity=0.3
        ))

        # LOWER ERROR
        fig_weight.add_trace(go.Scatter(
            x=x_vals, y=lower, mode="lines",
            legendgroup=mtype, showlegend=False,
            line=dict(width=0.1, color=rgba_line, shape="spline", smoothing=1.2),
            fill="tonexty", fillcolor=rgba_light, opacity=0.3
        ))

    fig_weight.update_layout(
        title="Weight Prediction after Bariatric Surgery",
        hovermode="x unified", template="plotly_white",
        width=1200, height=600,
        xaxis=dict(title="Time", tickmode='array', tickvals=x_vals),
        yaxis=dict(title="Weight (Kg)", dtick=5)
    )

    st.plotly_chart(fig_weight, use_container_width=True)

    # ======================================================
    # GRAPH 2 — BMI PROGRESSION
    # ======================================================
    st.subheader("📊 BMI Progression Over Time")

    fig_bmi = go.Figure()
    altura_sq = ALTURA ** 2
    bmi_initial = IMC

    # BMI ZONES
    fig_bmi.add_shape(type="rect", x0=0, x1=1, y0=20, y1=25,
                      xref="paper", yref="y",
                      fillcolor="rgba(0,200,0,0.15)", line=dict(width=0))
    fig_bmi.add_shape(type="rect", x0=0, x1=1, y0=25, y1=30,
                      xref="paper", yref="y",
                      fillcolor="rgba(255,215,0,0.20)", line=dict(width=0))
    fig_bmi.add_shape(type="rect", x0=0, x1=1, y0=30, y1=60,
                      xref="paper", yref="y",
                      fillcolor="rgba(255,0,0,0.15)", line=dict(width=0))

    for mtype in selected_models:
        preds = predictions[mtype]
        y_bmi = [bmi_initial] + [pred / altura_sq for pred in preds]

        base_color = color_map[mtype]
        r = int(base_color[1:3], 16)
        g = int(base_color[3:5], 16)
        b = int(base_color[5:7], 16)
        rgba_line = f"rgba({r},{g},{b},0.7)"

        fig_bmi.add_trace(go.Scatter(
            x=x_vals, y=y_bmi, mode="lines+markers", name=mtype,
            line=dict(width=3, color=rgba_line, shape="spline", smoothing=1.2),
            marker=dict(size=8, color=rgba_line)
        ))

    fig_bmi.update_layout(
        title="BMI Progression after Bariatric Surgery",
        hovermode="x unified", template="plotly_white",
        width=1200, height=500,
        xaxis=dict(title="Time", tickmode="array", tickvals=x_vals),
        yaxis=dict(title="BMI (kg/m²)", range=[18, 60])
    )

    st.plotly_chart(fig_bmi, use_container_width=True)

    # ======================================================
    # GRAPH 3 — % TOTAL WEIGHT LOSS
    # ======================================================
    st.subheader("📉 % Total Weight Loss (TWL%) Over Time")

    fig_twl = go.Figure()

    for mtype in selected_models:
        preds = predictions[mtype]
        y_twl = [0.0] + [100 * (P_INICIAL - p) / P_INICIAL for p in preds]

        base_color = color_map[mtype]
        r = int(base_color[1:3], 16)
        g = int(base_color[3:5], 16)
        b = int(base_color[5:7], 16)
        rgba_line = f"rgba({r},{g},{b},0.7)"

        fig_twl.add_trace(go.Scatter(
            x=x_vals, y=y_twl, mode="lines+markers", name=mtype,
            line=dict(width=3, color=rgba_line, shape="spline", smoothing=1.2),
            marker=dict(size=8, color=rgba_line)
        ))

    fig_twl.update_layout(
        title="% Total Weight Loss (TWL%) Over Time",
        hovermode="x unified", template="plotly_white",
        width=1200, height=500,
        xaxis=dict(title="Time", tickmode="array", tickvals=x_vals),
        yaxis=dict(title="TWL (%)")
    )

    st.plotly_chart(fig_twl, use_container_width=True)
# ==========================================
# FOOTER
# ==========================================
st.markdown(
    """
    <div style="position: fixed; bottom: 10px; width: 100%; text-align: center;">
        <p style="font-size:12px; font-style:italic; color:gray;">
            Developed by Arthur Canciglieri
        </p>
    </div>
    """,
    unsafe_allow_html=True
)