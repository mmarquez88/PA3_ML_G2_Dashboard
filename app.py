import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# =====================================================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# =====================================================================
st.set_page_config(
    page_title="Dashboard Académico - BERT y Facebook",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS para asegurar una interfaz estética y profesional
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stTitle { color: #0f172a; font-family: 'Arial'; font-weight: bold; }
    .metric-box { background-color: #ffffff; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); text-align: center; border-top: 4px solid #0284c7; }
    .keyword-badge { background-color: #e0f2fe; color: #0369a1; padding: 6px 12px; border-radius: 20px; font-weight: bold; display: inline-block; margin: 5px; font-size: 13px; border: 1px solid #bae6fd; }
    .comentario-caja { background-color: #f1f5f9; padding: 15px; border-left: 5px solid #0284c7; border-radius: 4px; margin-bottom: 20px; color: #334155; font-size: 14px; }
    </style>
    """, unsafe_allow_html=True)

# =====================================================================
# 2. IDENTIFICACIÓN Y PREGUNTA DE INVESTIGACIÓN
# =====================================================================
st.title("¿De qué manera los modelos BERT permiten detectar noticias falsas publicadas en Facebook?")
st.markdown("🔍 **Análisis avanzado de minería de datos y mapeo científico**")
st.markdown("---")

# Despliegue de palabras clave
st.markdown("### 🏷️ Palabras clave del estudio")
st.markdown("""
    <span class="keyword-badge">BERT</span>
    <span class="keyword-badge">Noticias falsas</span>
    <span class="keyword-badge">Facebook</span>
    <span class="keyword-badge">Procesamiento de lenguaje natural</span>
""", unsafe_allow_html=True)

st.markdown("---")

# =====================================================================
# 3. CARGA DE DATOS OPTIMIZADA
# =====================================================================
@st.cache_data
def load_data():
    df = pd.read_csv("PA3_ML_scopus_limpio.csv")
    df = df.fillna("No registra")

    # Limpieza preventiva de espacios en blanco
    df.columns = df.columns.str.strip()

    # Detección automática de la columna País
    col_pais_detectada = None
    for col in df.columns:
        if col.lower() in ['country', 'país', 'pais']:
            col_pais_detectada = col
            break

    if col_pais_detectada:
        df = df.rename(columns={col_pais_detectada: 'Country'})
        df['Country'] = df['Country'].astype(str).str.strip()

    return df

df = load_data()

# =====================================================================
# 4. BARRA LATERAL
# =====================================================================
st.sidebar.header("🔍 Filtros dinámicos")

year_min, year_max = int(df['Year'].min()), int(df['Year'].max())

selected_years = st.sidebar.slider(
    "Filtrar rango de años de publicación:",
    year_min,
    year_max,
    (year_min, year_max)
)

df_filtered = df[
    (df['Year'] >= selected_years[0]) &
    (df['Year'] <= selected_years[1])
]

st.sidebar.markdown("---")
st.sidebar.markdown("### ⭳ Plantilla del dataset")

st.sidebar.markdown(
    "<small>Descarga el archivo CSV utilizado para este análisis si deseas replicar la estructura de columnas requerida.</small>",
    unsafe_allow_html=True
)

csv_data = df.to_csv(index=False).encode('utf-8')

st.sidebar.download_button(
    label="Descargar plantilla CSV",
    data=csv_data,
    file_name="plantilla_scopus_bert.csv",
    mime="text/csv"
)

# Datos del alumno
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎓 Datos del alumno")
st.sidebar.write("**Estudiante:** Grupo 2")

st.sidebar.markdown("""
* Chancafe Pisfil Liz
* Alzamora Graciela
* Marquez Marcelo
* Retamozo Sheila
""")

st.sidebar.write("**Código:** 6817")

# 5. KPIs
# =====================================================================
st.subheader("🚀 Indicadores clave del dataset")
kpi1, kpi2, kpi3 = st.columns(3)
with kpi1:
    total_articulos = len(df_filtered)
    st.markdown(
        f'<div class="metric-box"><h3 style="color: #0f172a !important;">{total_articulos} artículos</h3><p style="color: #64748b;">Muestra de literatura científica</p></div>',
        unsafe_allow_html=True
    )
with kpi2:
    pais_lider = "India"
    st.markdown(
        f'<div class="metric-box"><h3 style="color: #0f172a !important;">{pais_lider}</h3><p style="color: #64748b;">País con mayor producción</p></div>',
        unsafe_allow_html=True
    )
with kpi3:
    citas_totales = int(df_filtered['Cited by'].sum()) if 'Cited by' in df_filtered.columns else 0
    st.markdown(
        f'<div class="metric-box"><h3 style="color: #0f172a !important;">{citas_totales} citas</h3><p style="color: #64748b;">Impacto total en la comunidad científica</p></div>',
        unsafe_allow_html=True
    )
st.markdown("---")

# =====================================================================
# 6. ANÁLISIS DE MODELOS DE IA
# =====================================================================
st.subheader("🤖 Análisis de modelos de inteligencia artificial más utilizados")

col_abs1, col_abs2 = st.columns(2)

with col_abs1:
    st.write("**Eficacia comparativa de algoritmos en la detección de noticias falsas**")

    modelos_data = {
        'Algoritmo / Arquitectura': [
            'SVM + TF-IDF (Máximo)',
            'BERT (COVID-19 Dataset)',
            'BERT + Bi-GRU (FNID)',
            'Passive-Aggressive',
            'Linear SVM',
            'BERT + Bi-GRU (FNFD)',
            'CNN / LSTM Baseline'
        ],
        'Precisión Máxima (%)': [99.6, 98.41, 97.0, 94.0, 92.0, 91.0, 90.0]
    }

    df_models = pd.DataFrame(modelos_data)

    fig_models = px.bar(
        df_models,
        x='Precisión Máxima (%)',
        y='Algoritmo / Arquitectura',
        orientation='h',
        title="Porcentaje de accuracy / F1-score por arquitectura",
        color='Precisión Máxima (%)',
        color_continuous_scale="Blues"
    )

    fig_models.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        template="plotly_white"
    )

    st.plotly_chart(fig_models, use_container_width=True)

with col_abs2:
    st.write("**Presencia y menciones de modelos de IA en el dataset**")

    menciones_ia = {
        'Modelo / Variante': [
            'BERT',
            'LSTM',
            'CNN',
            'SVM',
            'RNN',
            'Random Forest',
            'RoBERTa',
            'DistilBERT',
            'XGBoost'
        ],
        'Menciones': [37, 12, 10, 7, 5, 5, 4, 0, 0]
    }

    df_menciones = pd.DataFrame(menciones_ia)

    fig_menciones = px.bar(
        df_menciones,
        x='Modelo / Variante',
        y='Menciones',
        title="Frecuencia de modelos identificados en títulos y resúmenes",
        color='Menciones',
        color_continuous_scale="Blues"
    )

    fig_menciones.update_layout(template="plotly_white")

    st.plotly_chart(fig_menciones, use_container_width=True)

st.markdown("""
<div class="comentario-caja">
BERT emerge como el modelo más prominente con 37 menciones, destacando su importancia en el procesamiento de lenguaje natural para la detección de noticias falsas. Le siguen arquitecturas profundas como LSTM y CNN, lo que evidencia el predominio de redes neuronales avanzadas frente a modelos tradicionales.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =====================================================================
# 7. DOCUMENTOS Y EDITORIALES
# =====================================================================
st.subheader("📄 Distribución de formatos académicos y ecosistema editorial")

col_doc1, col_doc2 = st.columns(2)

with col_doc1:
    st.write("**Distribución de publicaciones por tipo de documento**")

    tipo_doc_data = {
        'Tipo de Documento': ['Conference paper', 'Article', 'Book chapter'],
        'Cantidad': [20, 15, 25]
    }

    df_tipos = pd.DataFrame(tipo_doc_data)

    fig_tipos = px.pie(
        df_tipos,
        values='Cantidad',
        names='Tipo de Documento',
        title="Preferencia de difusión académica",
        color_discrete_sequence=px.colors.sequential.Blues_r
    )

    st.plotly_chart(fig_tipos, use_container_width=True)

with col_doc2:
    st.write("**Principales editoriales según número de publicaciones**")

    editores_data = {
        'Editor': [
            'IEEE',
            'Springer Deutschland',
            'Springer',
            'World Scientific',
            'ACM',
            'Inderscience',
            'CEUR-WS',
            'Incoma Ltd',
            'Elsevier Ltd',
            'IOS Press'
        ],
        'Publicaciones': [14, 6, 4, 2, 2, 1, 1, 1, 1, 1]
    }

    df_editores = pd.DataFrame(editores_data)

    fig_editores = px.bar(
        df_editores,
        x='Publicaciones',
        y='Editor',
        orientation='h',
        title="Editoriales con mayor presencia",
        color='Publicaciones',
        color_continuous_scale="Blues"
    )

    fig_editores.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        template="plotly_white"
    )

    st.plotly_chart(fig_editores, use_container_width=True)

st.markdown("---")

# =====================================================================
# 8. VARIABLES GEOGRÁFICAS Y TEMÁTICAS
# =====================================================================
st.subheader("🌐 Variables demográficas, idioma y entornos de desinformación")

col_geo1, col_geo2 = st.columns(2)

with col_geo1:
    st.write("**Contextos principales de desinformación**")

    crisis_data = {
        'Contexto Temático': [
            'Pandemia COVID-19 / Infodemia',
            'Campañas electorales y política',
            'Grupos médicos y de salud',
            'Crisis migratorias / Minorías',
            'Multimodalidad (memes + texto)'
        ],
        'Artículos Críticos': [14, 9, 6, 5, 3]
    }

    df_crisis = pd.DataFrame(crisis_data)

    fig_crisis = px.pie(
        df_crisis,
        values='Artículos Críticos',
        names='Contexto Temático',
        title="Distribución temática",
        color_discrete_sequence=px.colors.sequential.Blues_r
    )

    st.plotly_chart(fig_crisis, use_container_width=True)

with col_geo2:
    st.write("**Indicadores complementarios**")

    identificadores_data = {
        'Métrica Evaluada': [
            'Artículos en inglés',
            'Direcciones de correspondencia únicas',
            'PubMed IDs únicos'
        ],
        'Valor Registrado': [37, 35, 0]
    }

    df_ids = pd.DataFrame(identificadores_data)

    fig_ids = px.bar(
        df_ids,
        x='Métrica Evaluada',
        y='Valor Registrado',
        title="Distribución lingüística e identificadores",
        color='Valor Registrado',
        color_continuous_scale="Blues"
    )

    fig_ids.update_layout(template="plotly_white")

    st.plotly_chart(fig_ids, use_container_width=True)

st.markdown("---")

# =====================================================================
# 9. VISUALIZACIONES DEL REPOSITORIO
# =====================================================================
st.subheader("📈 Métricas estructuradas del repositorio")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribución de publicaciones por año")

    df_years = df_filtered.groupby('Year').size().reset_index(
        name='Cantidad de Artículos'
    )

    fig_line = px.line(
        df_years,
        x='Year',
        y='Cantidad de Artículos',
        title="Evolución temporal de la investigación",
        markers=True,
        template="plotly_white"
    )

    fig_line.update_traces(
        line_color="#0284c7",
        line_width=3
    )

    st.plotly_chart(fig_line, use_container_width=True)

with col2:
    st.subheader("Autores más citados")

    if 'Cited by' in df_filtered.columns and 'Authors' in df_filtered.columns:

        top_authors = df_filtered.sort_values(
            by='Cited by',
            ascending=False
        ).head(8)

        fig_author = px.bar(
            top_authors,
            x='Cited by',
            y='Authors',
            orientation='h',
            title="Top 8 autores con mayor impacto",
            labels={
                'Cited by': 'Número de citas',
                'Authors': 'Autor'
            },
            color='Cited by',
            color_continuous_scale="Blues"
        )

        fig_author.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            template="plotly_white"
        )

        st.plotly_chart(fig_author, use_container_width=True)

    else:
        st.warning("Las columnas 'Authors' o 'Cited by' no están disponibles.")

# =====================================================================
# 10. NUBE DE PALABRAS
# =====================================================================
st.markdown("---")

st.subheader("☁️ Análisis semántico de palabras en abstracts")

st.markdown("""
Este mapa visual identifica los términos más recurrentes en los resúmenes científicos.
""")

columna_texto = 'Abstract' if 'Abstract' in df.columns else 'Author Keywords'

text_abstracts = " ".join(df_filtered[columna_texto].astype(str))

stop_words = [
    "the", "and", "a", "of", "to", "in", "is",
    "that", "for", "on", "with", "as", "by",
    "an", "it", "this", "from", "No registra"
]

if text_abstracts.strip() and len(text_abstracts) > 100:

    wordcloud = WordCloud(
        width=1100,
        height=350,
        background_color='white',
        colormap='Blues',
        stopwords=set(stop_words)
    ).generate(text_abstracts)

    fig_wc, ax = plt.subplots(figsize=(11, 3.5))

    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')

    st.pyplot(fig_wc)

else:
    st.info(
        "Agrega la columna 'Abstract' en tu archivo CSV para generar la nube de palabras."
    )

# =====================================================================
# 11. TABLA DE REGISTROS
# =====================================================================
st.markdown("---")

with st.expander("📂 Desplegar registros de literatura científica"):

    columnas_tabla = [
        "Title",
        "Authors",
        "Year",
        "Source title",
        "Abstract",
        "Cited by"
    ]

    if 'Country' in df_filtered.columns:
        columnas_tabla.append("Country")

    st.dataframe(
        df_filtered[columnas_tabla],
        use_container_width=True
    )

# =====================================================================
# 12. LICENCIA
# =====================================================================

st.markdown("---")

st.subheader("Licencia")

st.write("**Apache License, Version 2.0 (Apache-2.0)**")

st.caption(
    "Esta licencia permite la utilización, modificación y distribución "
    "del software con fines académicos y comerciales, respetando la "
    "atribución correspondiente."
)
