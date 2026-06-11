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
    page_title="Dashboard Scopus - Detección de Fake News con IA",
    page_icon="📰",
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
    <span style='font-size:46px;'>📰🕵️</span>
    <span style='font-size:42px; font-weight:800; color:#f1f5f9; vertical-align:middle;'>
        Detección de Fake News con IA
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
        ¿Cómo contribuye el machine learning a la detección automática de
        fake news y desinformación en medios digitales y redes sociales?
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- Keywords -----------------

with st.expander("📌 Keywords utilizadas", expanded=True):
    st.markdown("""
    <span class='chip'>fake news</span>
    <span class='chip'>machine learning</span>
    <span class='chip'>detection</span>
    <span class='chip'>social media</span>
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

# ----------------- Acerca del proyecto (sidebar) -----------------

st.sidebar.markdown("---")
st.sidebar.markdown("## ℹ️ Acerca del proyecto")

st.sidebar.markdown("""
<div style='background: rgba(46,109,164,0.15); border: 1px solid #2e6da4;
border-radius: 12px; padding: 14px 16px; font-size: 13px; color: #dceaf7;'>
<b>📰 Tema:</b> Detección de fake news con Machine Learning<br><br>
<b>📚 Curso:</b> Fundamentos de Machine Learning<br><br>
<b>🗄️ Fuente:</b> Scopus (Elsevier)<br><br>
<b>🏫 Institución:</b> ISIL
</div>
""", unsafe_allow_html=True)

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
    "🏆 Máx. citas",
    int(df_f["Cited by"].max())
)

c4.metric(
    "📈 Citas promedio",
    round(df_f["Cited by"].mean(), 1)
)

# ----------------- Gráfico 1 -----------------

st.markdown("---")
st.markdown("### 📅 Evolución de publicaciones por año")

st.caption(
    "Refleja cómo ha crecido el interés académico en el tema a lo largo del tiempo."
)

por_anio = df_f["Year"].value_counts().sort_index().reset_index()
por_anio.columns = ["Año", "Cantidad"]

fig1 = px.area(
    por_anio,
    x="Año",
    y="Cantidad",
    markers=True
)

fig1.update_traces(
    line_color="#ff9505",
    fillcolor="rgba(255,149,5,0.2)",
    marker=dict(size=9, color="#ff9505")
)

fig1.update_layout(**PLANTILLA)

st.plotly_chart(fig1, use_container_width=True)

# ----------------- Gráficos 2 y 3 -----------------

colA, colB = st.columns(2)

with colA:
    st.markdown("### 🥧 Distribución por tipo de documento")

    st.caption("Proporción de artículos, revisiones y ponencias.")

    tipos_count = df_f["Document Type"].value_counts().reset_index()
    tipos_count.columns = ["Tipo", "Cantidad"]

    figT = px.pie(
        tipos_count,
        names="Tipo",
        values="Cantidad",
        hole=0.45,
        color_discrete_sequence=[
            "#2e6da4",
            "#ff9505",
            "#7cc4f0",
            "#48a9a6",
            "#c44536"
        ]
    )

    figT.update_traces(
        textposition="inside",
        textinfo="percent+label"
    )

    figT.update_layout(
        **PLANTILLA,
        showlegend=True,
        legend=dict(font=dict(color="#f1f5f9"))
    )

    st.plotly_chart(figT, use_container_width=True)

with colB:
    st.markdown("### 💰 Citas acumuladas por año")

    st.caption("Suma de citas que recibe la producción de cada año.")

    citas_anio = df_f.groupby("Year")["Cited by"].sum().reset_index()
    citas_anio.columns = ["Año", "Citas"]

    figC = px.bar(
        citas_anio,
        x="Año",
        y="Citas",
        text="Citas",
        color="Citas",
        color_continuous_scale=[
            "#1b263b",
            "#2e6da4",
            "#ff9505"
        ]
    )

    figC.update_traces(textposition="outside")
    figC.update_layout(**PLANTILLA, showlegend=False)

    st.plotly_chart(figC, use_container_width=True)

# ----------------- Gráfico 4 -----------------

st.markdown("### 🏆 Artículos más citados")

st.caption(
    "Los trabajos más influyentes según el número de citas recibidas."
)

top = df_f.nlargest(10, "Cited by")[["Title", "Cited by"]].copy()

top["Corto"] = top["Title"].apply(
    lambda t: t[:55] + "..." if len(t) > 55 else t
)

fig2 = px.bar(
    top,
    x="Cited by",
    y="Corto",
    orientation="h",
    color="Cited by",
    color_continuous_scale=[
        "#1b263b",
        "#2e6da4",
        "#7cc4f0"
    ],
    hover_data={
        "Title": True,
        "Corto": False
    }
)

fig2.update_layout(
    yaxis=dict(
        autorange="reversed",
        title=""
    ),
    **PLANTILLA
)

st.plotly_chart(fig2, use_container_width=True)

# ----------------- Gráfico 5 -----------------

st.markdown("### 👥 Autores más productivos")

st.caption(
    "Investigadores con mayor presencia en el conjunto analizado."
)

autores = []

for fila in df_f["Authors"].dropna():
    autores.extend(
        [a.strip() for a in str(fila).split(";")]
    )

conteo = Counter(autores).most_common(10)

df_aut = pd.DataFrame(
    conteo,
    columns=["Autor", "Publicaciones"]
)

fig3 = px.bar(
    df_aut,
    x="Publicaciones",
    y="Autor",
    orientation="h",
    color="Publicaciones",
    color_continuous_scale=[
        "#1b263b",
        "#ff9505"
    ]
)

fig3.update_layout(
    yaxis=dict(autorange="reversed"),
    **PLANTILLA
)

st.plotly_chart(fig3, use_container_width=True)

# ----------------- Gráfico 6: Mapa por país -----------------

st.markdown("### 🌍 Representación geográfica de las publicaciones")

st.caption(
    "Mapa mundial donde el tamaño de cada círculo indica la cantidad "
    "de publicaciones con autores afiliados a ese país."
)

# Buscar la columna de afiliaciones disponible en el dataset
col_afil = None
for c in ["Affiliations", "Authors with affiliations", "Correspondence Address"]:
    if c in df_f.columns:
        col_afil = c
        break

if col_afil is None:
    st.info(
        "ℹ️ El dataset no contiene columnas de afiliaciones "
        "(Affiliations / Authors with affiliations), por lo que no se puede "
        "generar el mapa geográfico."
    )
else:
    # Normalizar nombres de países al formato que reconoce Plotly
    correcciones = {
        "USA": "United States",
        "United States of America": "United States",
        "UK": "United Kingdom",
        "Russian Federation": "Russia",
        "Viet Nam": "Vietnam",
        "South Korea": "South Korea",
        "Republic of Korea": "South Korea",
        "Korea": "South Korea",
        "Iran": "Iran",
        "Islamic Republic of Iran": "Iran",
        "Czech Republic": "Czechia",
        "Hong Kong": "Hong Kong",
        "Taiwan": "Taiwan",
        "UAE": "United Arab Emirates",
    }

    def extraer_paises(celda):
        """En Scopus, cada afiliación termina con el país después de la última coma."""
        paises = set()
        for afil in str(celda).split(";"):
            partes = afil.strip().split(",")
            if partes:
                pais = partes[-1].strip()
                if pais and len(pais) > 2 and "@" not in pais:
                    paises.add(correcciones.get(pais, pais))
        return paises

    conteo_paises = Counter()

    for fila in df_f[col_afil].dropna():
        for pais in extraer_paises(fila):
            conteo_paises[pais] += 1

    if len(conteo_paises) == 0:
        st.info("No se pudieron identificar países en las afiliaciones.")
    else:
        df_paises = pd.DataFrame(
            conteo_paises.most_common(),
            columns=["País", "Publicaciones"]
        )

        fig_mapa = px.scatter_geo(
            df_paises,
            locations="País",
            locationmode="country names",
            size="Publicaciones",
            color="Publicaciones",
            hover_name="País",
            size_max=45,
            projection="natural earth",
            color_continuous_scale=[
                "#2e6da4",
                "#7cc4f0",
                "#ff9505"
            ]
        )

        fig_mapa.update_geos(
            bgcolor="rgba(0,0,0,0)",
            showland=True,
            landcolor="#1b263b",
            showcountries=True,
            countrycolor="#2e6da4",
            coastlinecolor="#2e6da4",
            showocean=True,
            oceancolor="#0d1b2a",
            showframe=False
        )

        fig_mapa.update_layout(
            **PLANTILLA,
            height=500,
            margin=dict(l=0, r=0, t=10, b=0)
        )

        st.plotly_chart(fig_mapa, use_container_width=True)

        # Ranking de países debajo del mapa
        colM1, colM2 = st.columns([2, 1])

        with colM2:
            st.markdown("**🏅 Top países**")
            for _, r in df_paises.head(5).iterrows():
                st.markdown(
                    f"- {r['País']}: **{int(r['Publicaciones'])}** publicaciones"
                )

# ----------------- Procesamiento de abstracts -----------------

texto = " ".join(
    df_f["Abstract"].dropna().astype(str)
).lower()

palabras = re.findall(r"[a-z]{4,}", texto)

stop = {
    "this", "that", "with", "from", "were", "have", "been",
    "their", "which", "these", "such", "also", "based",
    "using", "results", "study", "paper", "data", "more",
    "than", "other", "into", "between", "however", "used",
    "model", "models", "method", "methods", "approach",
    "proposed", "system", "systems", "both", "each", "well",
    "show", "shown", "high", "different", "while", "they",
    "when", "where", "could", "would", "first", "case",
    "various", "through", "provide", "provides",
    "including", "present", "application", "applications",
    "article", "research", "work", "analysis", "techniques"
}

palabras = [p for p in palabras if p not in stop]

frec = Counter(palabras).most_common(30)

# ----------------- Gráfico 6 -----------------

st.markdown("### 🔤 Palabras más frecuentes en los abstracts")

st.caption(
    "Los términos más repetidos revelan los temas centrales de la literatura."
)

df_pal = pd.DataFrame(
    frec[:15],
    columns=["Palabra", "Frecuencia"]
)

fig4 = px.bar(
    df_pal,
    x="Palabra",
    y="Frecuencia",
    color="Frecuencia",
    color_continuous_scale=[
        "#2e6da4",
        "#ff9505"
    ]
)

fig4.update_layout(
    **PLANTILLA,
    showlegend=False
)

st.plotly_chart(fig4, use_container_width=True)

# ----------------- Gráfico 7 -----------------

st.markdown("### ☁️ Nube de palabras (análisis semántico)")

st.caption(
    "Mapa visual donde el tamaño de cada palabra representa su frecuencia en los abstracts."
)

random.seed(42)

palabras_nube = frec[:28]

max_f = palabras_nube[0][1] if palabras_nube else 1

colores = [
    "#ff9505",
    "#2e6da4",
    "#7cc4f0",
    "#48a9a6",
    "#e0e1dd",
    "#f4a261"
]

xs, ys, textos, tamanos, cols = [], [], [], [], []

for i, (palabra, f) in enumerate(palabras_nube):
    xs.append(random.uniform(0, 100))
    ys.append(random.uniform(0, 100))

    textos.append(palabra)

    tamanos.append(
        14 + (f / max_f) * 46
    )

    cols.append(
        colores[i % len(colores)]
    )

fig_nube = go.Figure()

fig_nube.add_trace(
    go.Scatter(
        x=xs,
        y=ys,
        mode="text",
        text=textos,
        textfont=dict(size=tamanos, color=cols),
        hovertext=[
            f"{p}: {f} veces"
            for p, f in palabras_nube
        ],
        hoverinfo="text"
    )
)

fig_nube.update_layout(
    **PLANTILLA,
    height=420,
    xaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        range=[-10, 110]
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        range=[-10, 110]
    ),
)

st.plotly_chart(fig_nube, use_container_width=True)

# ----------------- Tabla interactiva -----------------

st.markdown("---")
st.markdown("### 🗂️ Explorar el dataset")

st.caption(
    "Tabla interactiva con los artículos analizados (respeta los filtros). "
    "Ordena por cualquier columna."
)

cols_t = [
    c for c in [
        "Authors",
        "Title",
        "Year",
        "Source title",
        "Cited by",
        "DOI",
        "Abstract"          # 👈 columna de abstracts a la derecha
    ]
    if c in df_f.columns
]

tabla_df = df_f[cols_t].copy()

if "Abstract" in tabla_df.columns:
    tabla_df["Abstract"] = tabla_df["Abstract"].astype(str).apply(
        lambda x: x[:220] + "..." if len(x) > 220 else x
    )

st.dataframe(
    tabla_df,
    use_container_width=True,
    height=380,
    column_config={
        "Abstract": st.column_config.TextColumn("Abstract", width="large"),
    }
)

# ----------------- Pie de página -----------------

st.markdown("---")

st.markdown("""
<p style='text-align:center; color:#94a3b8;'>
Este aplicativo interactivo y su código fuente están distribuidos bajo los términos de la licencia:
</p>

<div style='text-align:center; background: #f8fafc; border: 1px solid #cbd5e1;
border-radius: 12px; padding: 16px 24px; margin-bottom: 14px;'>
<span style="font-size: 16px; font-weight: bold; color: #0f172a;">
Apache License, Version 2.0 (Apache-2.0)
</span>
<br />
<small style="display: block; margin-top: 5px; color: #64748b;">
Esta licencia permite la utilización, modificación y distribución del software
con fines académicos y comerciales, respetando la atribución correspondiente.
</small>
</div>

<p style='text-align:center; color:#d6e6f2; font-size:13px;'>
Desarrollado con Streamlit · Fuente: Scopus (Elsevier)<br>
Proyecto académico · ISIL · Detección de Fake News con Inteligencia Artificial
</p>
""", unsafe_allow_html=True)

