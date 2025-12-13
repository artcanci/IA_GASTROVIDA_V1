# STREAMLIT APP WITH AUTO-BMI, BMI ZONES, WEIGHT PREDICTION, BMI PROGRESSION, %TWL
# + PDF EXPORT WITH HOSPITAL LOGO

import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import joblib
import plotly.graph_objects as go
from plotly.colors import qualitative
from io import BytesIO

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
# PATHS
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
# MODEL LOADING
# ==========================================
def scan_model_filenames():
    if not os.path.exists(MODEL_FOLDER):
        return {}

    files = [f for f in os.listdir(MODEL_FOLDER) if f.endswith(".pkl")]
    pattern = r"^(1M|3M|6M|9M|1A|2A|3A|4A|5A|6A|7A|8A|9A|10A)_(.+)\.pkl$"

    models = {}
    for f in files:
        m = re.match(pattern, f)
        if m:
            tgt, mtype = m.groups()
            models.setdefault(mtype, {})[tgt] = f
    return models


@st.cache_resource
def load_all_models(models_by_type):
    loaded = {}
    for mtype, tgts in models_by_type.items():
        loaded[mtype] = {}
        for tgt, fname in tgts.items():
            try:
                loaded[mtype][tgt] = joblib.load(os.path.join(MODEL_FOLDER, fname))
            except:
                loaded[mtype][tgt] = None
    return loaded


@st.cache_data
def load_performance():
    if not os.path.exists(PERFORMANCE_FILE):
        return pd.DataFrame()
    return pd.read_excel(PERFORMANCE_FILE)


# ==========================================
# PDF GENERATION (WITH LOGO)
# ==========================================
def generate_pdf(input_df, results_df, fig_weight, fig_bmi, fig_twl):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 40

    # ===== LOGO =====
    if os.path.exists(LOGO_PATH):
        logo = ImageReader(LOGO_PATH)
        c.drawImage(logo, 40, y - 80, width=200, height=60, preserveAspectRatio=True)
        y -= 90

    # ===== TITLE =====
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "Bariatric Surgery Weight Prediction Report")
    y -= 30

    # ===== INPUTS =====
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Patient Inputs")
    y -= 18
    c.setFont("Helvetica", 10)

    for col, val in input_df.iloc[0].items():
        c.drawString(50, y, f"{col}: {val}")
        y -= 14

    y -= 10

    # ===== TABLE =====
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Predictions")
    y -= 18
    c.setFont("Helvetica", 9)

    for idx, row in results_df.iterrows():
        c.drawString(50, y, f"{idx}: {row.to_dict()}")
        y -= 12
        if y < 120:
            c.showPage()
            y = height - 40

    # ===== PLOTS =====
    def add_plot(fig, title):
        nonlocal y
        img = ImageReader(BytesIO(fig.to_image(format="png", scale=2)))

        if y < 350:
            c.showPage()
            y = height - 40

        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y, title)
        y -= 15
        c.drawImage(img, 40, y - 300, width=520, height=300)
        y -= 320

    add_plot(fig_weight, "Weight Prediction")
    add_plot(fig_bmi, "BMI Progression")
    add_plot(fig_twl, "% Total Weight Loss")

    # ===== FOOTER =====
    c.setFont("Helvetica-Oblique", 8)
    c.drawCentredString(width / 2, 30, "Generated for clinical reference only – Not a medical device")

    c.save()
    buffer.seek(0)
    return buffer


# ==========================================
# STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Weight Loss Prediction", layout="wide")

if os.path.exists(LOGO_PATH):
    st.image(LOGO_PATH, width=500)

st.title("📈 Multi-Model Bariatric Surgery Weight Prediction")

models_by_type = scan_model_filenames()
available_models = sorted(models_by_type.keys())
LOADED = load_all_models(models_by_type)
perf_df = load_performance()

# ===== SIDEBAR =====
st.sidebar.header("⚙️ Configuration")
selected_models = st.sidebar.multiselect(
    "Select models", available_models,
    default=available_models[:1]
)

IDADE = st.sidebar.number_input("Age", 1, 120, 30)
SEXO = 1 if st.sidebar.selectbox("Gender", ["Female", "Male"]) == "Female" else 0
ALTURA = st.sidebar.number_input("Height (m)", 1.2, 2.2, 1.7)
P_INICIAL = st.sidebar.number_input("Initial Weight (kg)", 30.0, 300.0, 120.0)

IMC = P_INICIAL / (ALTURA ** 2)

input_df = pd.DataFrame(
    [[IDADE, IMC, ALTURA, P_INICIAL, SEXO]],
    columns=["IDADE", "IMC", "ALTURA", "P INICIAL", "S"]
)

if st.button("Run predictions"):
    predictions = {m: [] for m in selected_models}

    for mtype in selected_models:
        for tgt in TARGETS_ORDER:
            model = LOADED.get(mtype, {}).get(tgt)
            predictions[mtype].append(
                float(model.predict(input_df)[0]) if model else np.nan
            )

    results_df = pd.DataFrame(predictions, index=TARGETS_ORDER)
    st.dataframe(results_df)

    x_vals = ["0M"] + TARGETS_ORDER

    # ===== WEIGHT GRAPH =====
    fig_weight = go.Figure()
    for mtype in selected_models:
        fig_weight.add_trace(go.Scatter(
            x=x_vals,
            y=[P_INICIAL] + predictions[mtype],
            mode="lines+markers",
            name=mtype
        ))
    st.plotly_chart(fig_weight, use_container_width=True)

    # ===== BMI GRAPH =====
    fig_bmi = go.Figure()
    for mtype in selected_models:
        fig_bmi.add_trace(go.Scatter(
            x=x_vals,
            y=[IMC] + [p / (ALTURA ** 2) for p in predictions[mtype]],
            mode="lines+markers",
            name=mtype
        ))
    st.plotly_chart(fig_bmi, use_container_width=True)

    # ===== TWL GRAPH =====
    fig_twl = go.Figure()
    for mtype in selected_models:
        fig_twl.add_trace(go.Scatter(
            x=x_vals,
            y=[0] + [100 * (P_INICIAL - p) / P_INICIAL for p in predictions[mtype]],
            mode="lines+markers",
            name=mtype
        ))
    st.plotly_chart(fig_twl, use_container_width=True)

    # ===== PDF EXPORT =====
    st.divider()
    pdf = generate_pdf(input_df, results_df, fig_weight, fig_bmi, fig_twl)

    st.download_button(
        "📄 Export PDF Report",
        data=pdf,
        file_name="bariatric_prediction_report.pdf",
        mime="application/pdf"
    )


# ===== FOOTER =====
st.markdown(
    "<p style='text-align:center;color:gray;font-size:12px;'>Developed by Arthur Canciglieri</p>",
    unsafe_allow_html=True
)
