# STREAMLIT APP WITH AUTO-BMI, BMI ZONES, WEIGHT PREDICTION, BMI PROGRESSION, AND %TWL

import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import joblib
import io
import plotly.graph_objects as go
import plotly.io as pio
from plotly.colors import qualitative
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


# ==========================================
# FIX FOR REMOVED NUMPY ALIASES
# ==========================================
if not hasattr(np, "float"): np.float = float
if not hasattr(np, "int"): np.int = int
if not hasattr(np, "bool"): np.bool = bool
if not hasattr(np, "object"): np.object = object
if not hasattr(np, "str"): np.str = str

# ==========================================
# ABSOLUTE PATH FIXES
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
# MODEL SCAN / LOAD
# ==========================================
def scan_model_filenames():
    if not os.path.exists(MODEL_FOLDER):
        st.error(f"Models folder not found: {MODEL_FOLDER}")
        return {}

    pattern = r"^(1M|3M|6M|9M|1A|2A|3A|4A|5A|6A|7A|8A|9A|10A)_(.+)\.pkl$"
    models_by_type = {}

    for f in os.listdir(MODEL_FOLDER):
        if not f.lower().endswith(".pkl"):
            continue
        m = re.match(pattern, f)
        if not m:
            continue
        target, mtype = m.groups()
        models_by_type.setdefault(mtype, {})[target] = f

    return models_by_type


@st.cache_resource
def load_all_models(models_by_type):
    loaded = {}
    for mtype, targets in models_by_type.items():
        loaded[mtype] = {}
        for tgt, fname in targets.items():
            try:
                loaded[mtype][tgt] = joblib.load(os.path.join(MODEL_FOLDER, fname))
            except Exception as e:
                loaded[mtype][tgt] = None
                st.error(f"Failed to load {fname}: {e}")
    return loaded


@st.cache_data
def load_performance():
    if not os.path.exists(PERFORMANCE_FILE):
        return pd.DataFrame()
    return pd.read_excel(PERFORMANCE_FILE)

# ==========================================
# PDF EXPORT FUNCTION (WITH INPUT HEADER)
# ==========================================
def export_figures_to_pdf(figures, patient_info):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawCentredString(width / 2, height - 40, "Bariatric Surgery Weight Prediction Report")

    pdf.setFont("Helvetica", 10)
    y = height - 70
    for label, value in patient_info.items():
        pdf.drawString(50, y, f"{label}: {value}")
        y -= 14

    pdf.showPage()

    for title, fig in figures:
        img_bytes = pio.to_image(fig, format="png", scale=2)
        image = ImageReader(io.BytesIO(img_bytes))

        img_width = width - 80
        img_height = img_width * 0.6

        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(40, height - 40, title)

        pdf.drawImage(
            image,
            40,
            height - img_height - 70,
            width=img_width,
            height=img_height,
            preserveAspectRatio=True,
            mask="auto"
        )

        pdf.showPage()

    pdf.save()
    buffer.seek(0)
    return buffer

# ==========================================
# STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Weight Loss Prediction", layout="wide")

# DISCLAIMER
if "disclaimer_accepted" not in st.session_state:
    st.session_state.disclaimer_accepted = False

if not st.session_state.disclaimer_accepted:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            ### ⚠️ Important Disclaimer
            This application is **for testing purposes only**.
            It is **not a medical device**.
            Always consult your physician.
            """,
            unsafe_allow_html=True
        )
        if st.button("I understand and agree", use_container_width=True):
            st.session_state.disclaimer_accepted = True
            st.rerun()
    st.stop()

if os.path.exists(LOGO_PATH):
    st.image(LOGO_PATH, width=600)

st.title("📈 Multi-Model Bariatric Surgery Weight Prediction")

models_by_type = scan_model_filenames()
available_models = sorted(models_by_type.keys())
LOADED = load_all_models(models_by_type)
perf_df = load_performance()

# ==========================================
# SIDEBAR INPUTS
# ==========================================
st.sidebar.header("⚙️ Configuration")

selected_models = st.sidebar.multiselect(
    "Select model types",
    available_models,
    default=available_models[:1]
)

IDADE = st.sidebar.number_input("Age", 1, 120, 30)
SEXO_LABEL = st.sidebar.selectbox("Gender", ["Female", "Male"])
SEXO = 1 if SEXO_LABEL == "Female" else 0
ALTURA = st.sidebar.number_input("Height (m)", 1.2, 2.2, 1.70)
P_INICIAL = st.sidebar.number_input("Initial Weight (kg)", 30.0, 300.0, 120.0)

IMC = round(P_INICIAL / (ALTURA ** 2), 2)

st.sidebar.number_input("BMI", value=IMC, disabled=True)

input_df = pd.DataFrame(
    [[IDADE, IMC, ALTURA, P_INICIAL, SEXO]],
    columns=["IDADE", "IMC", "ALTURA", "P INICIAL", "S"]
)

if st.button("Run predictions"):
    st.session_state.run_pred = True

color_map = {m: qualitative.Plotly[i % len(qualitative.Plotly)]
             for i, m in enumerate(available_models)}

# ==========================================
# RUN PREDICTIONS
# ==========================================
if st.session_state.get("run_pred", False):

    predictions = {m: [] for m in selected_models}

    for mtype in selected_models:
        for tgt in TARGETS_ORDER:
            model = LOADED.get(mtype, {}).get(tgt)
            predictions[mtype].append(
                float(model.predict(input_df)[0]) if model else np.nan
            )

    results_df = pd.DataFrame(predictions, index=TARGETS_ORDER)
    st.subheader("📋 Predictions Table")
    st.dataframe(results_df)

    x_vals = ["0M"] + TARGETS_ORDER

    # ===== GRAPH 1 =====
    fig_weight = go.Figure()
    for mtype in selected_models:
        fig_weight.add_trace(go.Scatter(
            x=x_vals,
            y=[P_INICIAL] + predictions[mtype],
            mode="lines+markers",
            name=mtype
        ))

    st.plotly_chart(fig_weight, use_container_width=True)

    # ===== GRAPH 2 =====
    fig_bmi = go.Figure()
    for mtype in selected_models:
        fig_bmi.add_trace(go.Scatter(
            x=x_vals,
            y=[IMC] + [p / (ALTURA ** 2) for p in predictions[mtype]],
            mode="lines+markers",
            name=mtype
        ))

    st.plotly_chart(fig_bmi, use_container_width=True)

    # ===== GRAPH 3 =====
    fig_twl = go.Figure()
    for mtype in selected_models:
        fig_twl.add_trace(go.Scatter(
            x=x_vals,
            y=[0] + [100 * (P_INICIAL - p) / P_INICIAL for p in predictions[mtype]],
            mode="lines+markers",
            name=mtype
        ))

    st.plotly_chart(fig_twl, use_container_width=True)

    # ==========================================
    # PDF EXPORT BUTTON
    # ==========================================
    if st.button("📄 Export results to PDF"):
        patient_info = {
            "Age": IDADE,
            "Gender": SEXO_LABEL,
            "Height (m)": ALTURA,
            "Initial Weight (kg)": P_INICIAL,
            "BMI": IMC
        }

        pdf = export_figures_to_pdf(
            [
                ("Weight Prediction", fig_weight),
                ("BMI Progression", fig_bmi),
                ("% Total Weight Loss", fig_twl),
            ],
            patient_info
        )

        st.download_button(
            "⬇️ Download PDF",
            data=pdf,
            file_name="bariatric_prediction_report.pdf",
            mime="application/pdf"
        )

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
