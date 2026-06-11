# =============================================================
# Dashboard de Análisis Bibliométrico - Scopus
# Tema: Mantenimiento predictivo y detección de fallas con IA
# Curso: Fundamentos de Machine Learning
# =============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
import random

# ----------------- Configuración de la página -----------------

st.set_page_config(
    page_title="Dashboard Scopus - Mantenimiento Predictivo con IA",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

URL_GITHUB = "https://raw.githubusercontent.com/mmarquez88/dashboard_scopus_pa3_Grupo2/main/scopus.csv"

# ----------------- Estilos personalizados (CSS) -----------------

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(160deg, #0d1b2a 0%, #1b263b 100%);
    }

    .tarjeta-pregunta {
        background: linear-gradient(135deg, rgba(30,58,95,0.6), rgba(27,38,59,0.6));
        border: 1px solid #2e6da4;
        border-left: 5px solid #ff9505;
        border-radius: 14px;
        padding: 22px 28px;
        margin: 10px 0 18px 0;
    }

    .label-pregunta {
        color: #ff9505;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 2px;
        text-align: center;
        margin-bottom: 8px;
    }

    .texto-pregunta {
        color: #f1f5f9;
        font-size: 20px;
        font-weight: 500;
        text-align: center;
        line-height: 1.5;
    }

    .chip {
        display: inline-block;
        background: rgba(46,109,164,0.25);
        color: #dceaf7;
        border: 1px solid #4d8cc7;
        border-radius: 20px;
        padding: 6px 16px;
        margin: 4px 6px 4px 0;
        font-family: monospace;
        font-size: 14px;
        font-weight: 600;
    }

    h1, h2, h3 {
        color: #f1f5f9 !important;
    }

    p, label, div {
        color: #e0e1dd;
    }

    [data-testid="stMetric"] {
        background: rgba(46,109,164,0.15);
        border: 1px solid #2e6da4;
        border-radius: 12px;
        padding: 16px;
    }

    [data-testid="stMetricValue"] {
        color: #ff9505 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #dceaf7 !important;
        font-weight: 600;
    }

    [data-testid="stSidebar"] {
        background: #0a1420;
        border-right: 1px solid #2e6da4;
    }

    .stCaption {
        color: #d6e6f2 !important;
        font-size: 14px !important;
    }

    [data-testid="stExpander"] {
        background: rgba(46,109,164,0.10);
        border: 1px solid #2e6da4;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------

st.sidebar.markdown("## 📁 Fuente de datos")

fuente = st.sidebar.radio(
    "Elige cómo cargar el CSV:",
    ["Dataset incluido", "Cargar archivo local", "URL pública de GitHub"]
)


@st.cache_data
def cargar(url):
    return pd.read_csv(url)


df = None

if fuente == "Dataset incluido":
    try:
        df = cargar(URL_GITHUB)
        st.sidebar.success(f"✅ Dataset oficial cargado — {len(df)} artículos")
    except Exception as e:
        st.sidebar.error("No se pudo cargar desde GitHub.")
        st.sidebar.caption(str(e))

elif fuente == "Cargar archivo local":
    up = st.sidebar.file_uploader("Sube tu scopus.csv", type=["csv"])

    if up is not None:
        df = pd.read_csv(up)
        st.sidebar.success(f"✅ Archivo cargado — {len(df)} artículos")

else:
    url = st.sidebar.text_input("Pega la URL RAW del CSV")

    if url:
        try:
            df = cargar(url)
            st.sidebar.success(f"✅ Cargado — {len(df)} artículos")
        except Exception as e:
            st.sidebar.error(str(e))

# ----------------- Encabezado -----------------

st.markdown("""
<div style='text-align:center; margin-bottom:6px;'>
    <span style='font-size:46px;'>🔧</span>
    <span style='font-size:42px; font-weight:800; color:#f1f5f9; vertical-align:middle;'>
        Mantenimiento Predictivo con IA
    </span>
</div>
<p style='text-align:center; color:#d6e6f2; font-size:17px; margin-top:0; font-weight:500;'>
    Análisis bibliométrico · Scopus · 2019–2026
</p>
""", unsafe_allow_html=True)

# ----------------- Pregunta de investigación -----------------

st.markdown("""
<div class='tarjeta-pregunta'>
    <div class='label-pregunta'>🔬 PREGUNTA DE INVESTIGACIÓN</div>
    <div class='texto-pregunta'>
        ¿Cómo contribuye el machine learning al mantenimiento predictivo y la
        detección temprana de fallas en equipos industriales?
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- Keywords -----------------

with st.expander("📌 Keywords utilizadas", expanded=True):
    st.markdown("""
    <span class='chip'>predictive maintenance</span>
    <span class='chip'>machine learning</span>
    <span class='chip'>fault detection</span>
    <span class='chip'>industrial</span>
    """, unsafe_allow_html=True)

if df is None:
    st.info("👉 Selecciona una fuente de datos en la barra lateral para comenzar.")
    st.stop()

# ----------------- Limpieza básica -----------------

df.columns = [c.strip() for c in df.columns]

df["Cited by"] = pd.to_numeric(df["Cited by"], errors="coerce").fillna(0)
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

# ----------------- Registro de Literatura Científica -----------------

with st.expander("📁 Desplegar Registro de Literatura Científica (Muestra de Scopus)", expanded=False):

    cols_registro = [
        c for c in [
            "Title",
            "Authors",
            "Year",
            "Source title",
            "Cited by",
            "Abstract"          # 👈 columna de abstracts a la derecha
        ]
        if c in df.columns
    ]

    registro_df = df[cols_registro].copy()

    st.dataframe(
        registro_df,
        use_container_width=True,
        height=380,
        column_config={
            "Title": st.column_config.TextColumn("Title", width="large"),
            "Authors": st.column_config.TextColumn("Authors", width="medium"),
            "Year": st.column_config.NumberColumn("Year", format="%d", width="small"),
            "Source title": st.column_config.TextColumn("Source title", width="medium"),
            "Cited by": st.column_config.NumberColumn("Cited by", format="%d", width="small"),
            "Abstract": st.column_config.TextColumn("Abstract", width="large"),
        }
    )

# ----------------- FILTROS -----------------

st.sidebar.markdown("---")
st.sidebar.markdown("## 🧰 Filtros")

anio_min = int(df["Year"].min())
anio_max = int(df["Year"].max())

if anio_min < anio_max:
    rango = st.sidebar.slider(
        "Rango de años",
        anio_min,
        anio_max,
        (anio_min, anio_max)
    )
else:
    rango = (anio_min, anio_max)

tipos_disp = sorted(df["Document Type"].dropna().unique().tolist())

tipos_sel = st.sidebar.multiselect(
    "Tipo de documento",
    tipos_disp,
    default=tipos_disp
)

df_f = df[
    (df["Year"] >= rango[0]) &
    (df["Year"] <= rango[1])
]

if tipos_sel:
    df_f = df_f[df_f["Document Type"].isin(tipos_sel)]

st.sidebar.markdown(f"**Mostrando:** {len(df_f)} artículos")

if len(df_f) == 0:
    st.warning("No hay artículos con los filtros seleccionados.")
    st.stop()

# ----------------- Plantilla gráfica -----------------

PLANTILLA = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#f1f5f9")
)

# ----------------- Resumen -----------------

st.markdown("---")
st.markdown("## 📊 Resumen del dataset")

c1, c2, c3, c4 = st.columns(4)

c1.metric("📄 Artículos", len(df_f))

c2.metric(
    "📅 Período",
    f"{int(df_f['Year'].min())}–{int(df_f['Year'].max())}"
)

c3.metric(
