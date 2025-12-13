# STREAMLIT APP WITH AUTO-BMI, BMI ZONES, WEIGHT PREDICTION, BMI PROGRESSION, %TWL
# + DISCLAIMER GATE + EN/PT-BR + MOBILE STATIC PLOTS + PDF EXPORT WITH WATERMARK

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import os
import re
import joblib
import io
from datetime import datetime
import plotly.graph_objects as go
from plotly.colors import qualitative
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors

# =====================================================
# NUMPY ALIAS FIX
# =====================================================
for a, b in [("float", float), ("int", int), ("bool", bool), ("object", object), ("str", str)]:
    if not hasattr(np, a):
        setattr(np, a, b)

# =====================================================
# PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
PERFORMANCE_FILE = os.path.join(BASE_DIR, "Performance.xlsx")
LOGO_PATH = os.path.join(BASE_DIR, "logo.jpeg")

# =====================================================
# TARGET ORDER
# =====================================================
TARGETS_ORDER = [
    "1M", "3M", "6M", "9M",
    "1A", "2A", "3A", "4A", "5A",
    "6A", "7A", "8A", "9A", "10A"
]

# =====================================================
# TRANSLATIONS
# =====================================================
TEXT = {
    "en": {
        "lang": "Language",
        "nav": "Navigation",
        "disclaimer_title": "🧾 Regulatory Disclaimer",
        "disclaimer": """
This application is **for testing and evaluation purposes only** and is **still under development**.

• This tool is **not a medical device**  
• Predictions may be inaccurate  
• Results are **for reference only**  
• **Always consult your bariatric surgeon or doctor**
""",
        "accept": "I understand and accept the disclaimer",
        "continue": "Continue",
        "pdf_watermark": "FOR TESTING ONLY – UNDER DEVELOPMENT – REFERENCE USE ONLY – CONSULT YOUR BARIATRIC SURGEON OR DOCTOR",
        "run": "🔮 Run predictions",
        "report": "Bariatric Surgery Prediction Report",
    },
    "pt": {
        "lang": "Idioma",
        "nav": "Navegação",
        "disclaimer_title": "🧾 Aviso Regulatório",
        "disclaimer": """
Este aplicativo é **apenas para fins de teste e avaliação** e **ainda está em desenvolvimento**.

• Esta ferramenta **não é um dispositivo médico**  
• As previsões podem estar incorretas  
• Os resultados são **apenas para referência**  
• **Consulte sempre seu cirurgião bariátrico ou médico**
""",
        "accept": "Li e aceito o aviso",
        "continue": "Continuar",
        "pdf_watermark": "APENAS PARA TESTES – EM DESENVOLVIMENTO – USO COMO REFERÊNCIA – CONSULTE SEU MÉDICO OU CIRURGIÃO BARIÁTRICO",
        "run": "🔮 Executar previsões",
        "report": "Relatório de Predição – Cirurgia Bariátrica",
    }
}

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Weight Loss Prediction", layout="wide")

# =====================================================
# SESSION STATE
# =====================================================
st.session_state.setdefault("accepted", False)
st.session_state.setdefault("is_mobile", False)
st.session_state.setdefault("run_pred", False)

# =====================================================
# SIDEBAR — LANGUAGE + NAV
# =====================================================
lang_choice = st.sidebar.selectbox(TEXT["en"]["lang"], ["English 🇺🇸", "Português 🇧🇷"])
LANG = "pt" if "Português" in lang_choice else "en"
T = TEXT[LANG]

st.sidebar.header(T["nav"])
page = st.sidebar.radio("", ["Disclaimer", "App"])

# =====================================================
# DISCLAIMER PAGE
# =====================================================
if page == "Disclaimer":
    st.title(T["disclaimer_title"])
    st.markdown(T["disclaimer"])

    if st.checkbox(T["accept"]):
        if st.button(T["continue"]):
            st.session_state.accepted = True
            st.rerun()
    st.stop()

if page == "App" and not st.session_state.accepted:
    st.warning("Please accept the disclaimer first.")
    st.stop()

# =====================================================
# MOBILE DETECTION
# =====================================================
components.html(
    """
    <script>
    const isMobile = /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent);
    window.parent.postMessage({type:"MOBILE", value:isMobile}, "*");
    </script>
    """,
    height=0,
)

# =====================================================
# MODEL LOADING
# =====================================================
def scan_model_filenames():
    if not os.path.exists(MODEL_FOLDER):
        return {}
    pattern = r"^(1M|3M|6M|9M|1A|2A|3A|4A|5A|6A|7A|8A|9A|10A)_(.+)\.pkl$"
    models = {}
    for f in os.listdir(MODEL_FOLDER):
        m = re.match(pattern, f)
        if m:
            tgt, mtype = m.groups()
            models.setdefault(mtype, {})[tgt] = f
    return models

@st.cache_resource
def load_all_models(models):
    out = {}
    for mtype, tgts in models.items():
        out[mtype] = {}
        for tgt, f in tgts.items():
            try:
                out[mtype][tgt] = joblib.load(os.path.join(MODEL_FOLDER, f))
            except:
                out[mtype][tgt] = None
    return out

@st.cache_data
def load_perf():
    return pd.read_excel(PERFORMANCE_FILE) if os.path.exists(PERFORMANCE_FILE) else pd.DataFrame()

models_by_type = scan_model_filenames()
available_models = sorted(models_by_type.keys())
LOADED = load_all_models(models_by_type)
perf_df = load_perf()

# =====================================================
# HEADER
# =====================================================
if os.path.exists(LOGO_PATH):
    st.image(LOGO_PATH, width=500)

st.title("📈 Multi-Model Bariatric Surgery Weight Prediction")

# =====================================================
# SIDEBAR INPUTS
# =====================================================
st.sidebar.header("⚙️ Configuration")

selected_models = st.sidebar.multiselect(
    "Select model types",
    available_models,
    default=["XGBoost"] if "XGBoost" in available_models else available_models[:1]
)

IDADE = st.sidebar.number_input("Age", 1, 120, 30)
SEXO = 1 if st.sidebar.selectbox("Gender", ["Female", "Male"]) == "Female" else 0
ALTURA = st.sidebar.number_input("Height (m)", 1.2, 2.2, 1.70)
P_INICIAL = st.sidebar.number_input("Initial Weight (kg)", 30.0, 300.0, 120.0)
IMC = P_INICIAL / (ALTURA ** 2)

st.sidebar.number_input("BMI", value=round(IMC, 2), disabled=True)

input_df = pd.DataFrame([[IDADE, IMC, ALTURA, P_INICIAL, SEXO]],
                        columns=["IDADE", "IMC", "ALTURA", "P INICIAL", "S"])

if st.sidebar.button(T["run"]):
    st.session_state.run_pred = True

# =====================================================
# PLOTLY CONFIG
# =====================================================
def plotly_config():
    return {"staticPlot": st.session_state.is_mobile}

# =====================================================
# PDF EXPORT
# =====================================================
def generate_pdf(predictions):
    buffer = io.BytesIO()
    styles = getSampleStyleSheet()

    watermark = ParagraphStyle(
        "wm", fontSize=8, textColor=colors.grey, alignment=1, italic=True
    )

    doc = SimpleDocTemplate(buffer, pagesize=A4)
    story = []

    story.append(Paragraph(T["pdf_watermark"], watermark))
    story.append(Spacer(1, 12))
    story.append(Paragraph(T["report"], styles["Title"]))
    story.append(Spacer(1, 12))

    table = Table([
        ["Age", IDADE],
        ["Gender", "Female" if SEXO else "Male"],
        ["Height (m)", ALTURA],
        ["Initial Weight (kg)", P_INICIAL],
        ["BMI", round(IMC, 2)],
        ["Models", ", ".join(selected_models)],
        ["Generated", datetime.now().strftime("%Y-%m-%d %H:%M")]
    ])

    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 1, colors.grey)]))
    story.append(table)

    doc.build(story)
    buffer.seek(0)
    return buffer

# =====================================================
# RUN PREDICTIONS
# =====================================================
if st.session_state.run_pred:

    predictions = {m: [] for m in selected_models}

    for m in selected_models:
        for t in TARGETS_ORDER:
            model = LOADED.get(m, {}).get(t)
            predictions[m].append(float(model.predict(input_df)[0]) if model else np.nan)

    results_df = pd.DataFrame(predictions, index=TARGETS_ORDER)
    st.subheader("📋 Predictions Table")
    st.dataframe(results_df)

    pdf = generate_pdf(predictions)
    st.download_button("📄 Download PDF Report", pdf, "bariatric_report.pdf")

    x_vals = ["0M"] + TARGETS_ORDER
    color_map = {m: qualitative.Plotly[i % len(qualitative.Plotly)] for i, m in enumerate(available_models)}

    # ================= GRAPH 1 =================
    st.subheader("📉 Weight Prediction")
    fig = go.Figure()
    for m in selected_models:
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=[P_INICIAL] + predictions[m],
            mode="lines+markers",
            name=m
        ))
    st.plotly_chart(fig, use_container_width=True, config=plotly_config())

    # ================= GRAPH 2 =================
    st.subheader("📊 BMI Progression")
    fig = go.Figure()
    for m in selected_models:
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=[IMC] + [p / (ALTURA ** 2) for p in predictions[m]],
            mode="lines+markers",
            name=m
        ))
    st.plotly_chart(fig, use_container_width=True, config=plotly_config())

    # ================= GRAPH 3 =================
    st.subheader("📉 % Total Weight Loss")
    fig = go.Figure()
    for m in selected_models:
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=[0] + [100 * (P_INICIAL - p) / P_INICIAL for p in predictions[m]],
            mode="lines+markers",
            name=m
        ))
    st.plotly_chart(fig, use_container_width=True, config=plotly_config())

# =====================================================
# FOOTER
# =====================================================
st.markdown(
    "<p style='text-align:center;font-size:12px;color:gray;'>Developed by Arthur Canciglieri</p>",
    unsafe_allow_html=True
)
