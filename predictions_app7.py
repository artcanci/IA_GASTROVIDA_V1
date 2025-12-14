# STREAMLIT APP WITH AUTO-BMI, BMI ZONES, WEIGHT PREDICTION, BMI PROGRESSION, AND %TWL

import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import joblib
import plotly.graph_objects as go
from plotly.colors import qualitative
from translations import TRANSLATIONS

# ===================== PDF EXPORT (ADD) =====================
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
import tempfile

# ============================================================
# FIX FOR REMOVED NUMPY ALIASES
# ============================================================
if not hasattr(np, "float"): np.float = float
if not hasattr(np, "int"): np.int = int
if not hasattr(np, "bool"): np.bool = bool
if not hasattr(np, "object"): np.object = object
if not hasattr(np, "str"): np.str = str

# ============================================================
# ABSOLUTE PATH FIXES
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
PERFORMANCE_FILE = os.path.join(BASE_DIR, "Performance.xlsx")
LOGO_PATH = os.path.join(BASE_DIR, "logo.jpeg")

# ============================================================
# LANGUAGE
# ============================================================
LANG = st.sidebar.selectbox(
    TRANSLATIONS["English"]["language"],
    list(TRANSLATIONS.keys())
)

def t(key):
    return TRANSLATIONS[LANG][key]

# ============================================================
# TARGET ORDER
# ============================================================
TARGETS_ORDER = [
    "1M", "3M", "6M", "9M",
    "1A", "2A", "3A", "4A", "5A",
    "6A", "7A", "8A", "9A", "10A"
]

# ============================================================
# SCAN AND LOAD MODELS
# ============================================================
def scan_model_filenames():
    if not os.path.exists(MODEL_FOLDER):
        st.error(f"Models folder not found: {MODEL_FOLDER}")
        return {}

    files = [f for f in os.listdir(MODEL_FOLDER) if f.lower().endswith(".pkl")]

    pattern = r"^(1M|3M|6M|9M|1A|2A|3A|4A|5A|6A|7A|8A|9A|10A)_(.+)\.pkl$"

    models_by_type = {}
    for fname in files:
        match = re.match(pattern, fname)
        if not match:
            continue
        target, mtype = match.group(1), match.group(2)
        models_by_type.setdefault(mtype, {})[target] = fname

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

# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title=t("page_title"), layout="wide")

# ============================================================
# DISCLAIMER GATE
# ============================================================
if "disclaimer_accepted" not in st.session_state:
    st.session_state.disclaimer_accepted = False

if not st.session_state.disclaimer_accepted:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.markdown(f"### {t('disclaimer_title')}")
        st.markdown(t("disclaimer_text"), unsafe_allow_html=True)
        st.markdown("---")
        if st.button(t("agree"), use_container_width=True):
            st.session_state.disclaimer_accepted = True
            st.rerun()
    st.stop()

# ============================================================
# LOGO & TITLE
# ============================================================
if os.path.exists(LOGO_PATH):
    st.image(LOGO_PATH, width=600)

st.title(t("main_title"))

# ============================================================
# LOAD MODELS
# ============================================================
models_by_type = scan_model_filenames()
available_models = sorted(models_by_type.keys())
LOADED = load_all_models(models_by_type)
perf_df = load_performance()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header(t("config"))

default_model = ["XGBoost"] if "XGBoost" in available_models else available_models[:1]
selected_models = st.sidebar.multiselect(
    t("select_models"), available_models, default=default_model
)

st.sidebar.subheader(t("input_vars"))

IDADE = st.sidebar.number_input(t("age"), 1, 120, 30)
SEXO = 1 if st.sidebar.selectbox(
    t("gender"), [t("female"), t("male")]
) == t("female") else 0

ALTURA = st.sidebar.number_input(t("height"), 1.2, 2.2, 1.70)
P_INICIAL = st.sidebar.number_input(t("initial_weight"), 30.0, 300.0, 120.0)

IMC = P_INICIAL / (ALTURA ** 2)

st.sidebar.number_input(
    t("bmi_auto"),
    value=float(round(IMC, 2)),
    disabled=True
)

input_df = pd.DataFrame(
    [[IDADE, IMC, ALTURA, P_INICIAL, SEXO]],
    columns=["IDADE", "IMC", "ALTURA", "P INICIAL", "S"]
)

# ============================================================
# RUN PREDICTIONS
# ============================================================
if st.button(t("run_predictions")):

    predictions = {m: [] for m in selected_models}

    for mtype in selected_models:
        for tgt in TARGETS_ORDER:
            model = LOADED.get(mtype, {}).get(tgt)
            predictions[mtype].append(
                float(model.predict(input_df)[0]) if model else np.nan
            )

    results_df = pd.DataFrame(predictions, index=TARGETS_ORDER)
    st.subheader(t("predictions_table"))
    st.dataframe(results_df)

    x_vals = ["0M"] + TARGETS_ORDER

    # ================= WEIGHT GRAPH =================
    fig_weight = go.Figure()
    for mtype in selected_models:
        fig_weight.add_trace(go.Scatter(
            x=x_vals, y=[P_INICIAL] + predictions[mtype],
            mode="lines+markers", name=mtype
        ))
    st.plotly_chart(fig_weight, use_container_width=True)

    # ================= BMI GRAPH =================
    fig_bmi = go.Figure()
    for mtype in selected_models:
        fig_bmi.add_trace(go.Scatter(
            x=x_vals,
            y=[IMC] + [p / (ALTURA ** 2) for p in predictions[mtype]],
            mode="lines+markers", name=mtype
        ))
    st.plotly_chart(fig_bmi, use_container_width=True)

    # ================= TWL GRAPH =================
    fig_twl = go.Figure()
    for mtype in selected_models:
        fig_twl.add_trace(go.Scatter(
            x=x_vals,
            y=[0] + [100 * (P_INICIAL - p) / P_INICIAL for p in predictions[mtype]],
            mode="lines+markers", name=mtype
        ))
    st.plotly_chart(fig_twl, use_container_width=True)

    # ================= PDF EXPORT =================
    st.markdown("### 📄 Export Report")

    if st.button("📥 Generate PDF"):
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph(t("main_title"), styles["Title"]))
        story.append(Spacer(1, 20))

        table_data = [results_df.columns.tolist()] + results_df.reset_index().values.tolist()
        table = Table(table_data)
        table.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ]))
        story.append(table)
        story.append(Spacer(1, 20))

        for title, fig in [
            ("Weight Prediction", fig_weight),
            ("BMI Progression", fig_bmi),
            ("% Total Weight Loss", fig_twl)
        ]:
            img_bytes = fig.to_image(format="png", width=1000, height=500, scale=2)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                tmp_img.write(img_bytes)
                img_path = tmp_img.name

            story.append(Paragraph(title, styles["Heading2"]))
            story.append(Image(img_path, width=17*cm, height=8*cm))
            story.append(Spacer(1, 20))

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            doc = SimpleDocTemplate(tmp_pdf.name, pagesize=A4)
            doc.build(story)

            with open(tmp_pdf.name, "rb") as f:
                st.download_button(
                    "⬇️ Download PDF",
                    f,
                    file_name="prediction_report.pdf",
                    mime="application/pdf"
                )
