import io
import pickle
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Visualización 2D
import altair as alt

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
st.set_page_config(page_title="Clusters de Películas", layout="wide")
st.title("🎬 Clasificación por Clusters de Pósters")

st.caption(
    "Sube un póster, se predice su cluster con tu modelo entrenado (.pkl), "
    "se muestran 5 películas representativas de ese cluster y se visualiza "
    "la distribución 2D. Filtro rápido por género incluido."
)

# ---------------------------------------------------------
# Embedding simple (cámbialo por el que usaste al entrenar)
# ---------------------------------------------------------
def image_embedding_rgb_hist(img: Image.Image, bins_per_channel: int = 8) -> np.ndarray:
    """Histograma RGB normalizado — rápido y sin dependencias pesadas.
    Sustituye por tu extractor REAL si entrenaste con otro."""
    arr = np.array(img.convert("RGB"))
    hist = []
    for ch in range(3):
        h, _ = np.histogram(arr[..., ch], bins=bins_per_channel, range=(0, 256), density=False)
        hist.append(h.astype(np.float32))
    vec = np.concatenate(hist, axis=0)
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    return vec

def embed_image_query(img: Image.Image, embedder_name: str = "rgb_hist") -> np.ndarray:
    if embedder_name == "rgb_hist":
        return image_embedding_rgb_hist(img, bins_per_channel=8)
#tod o añade aqui tu pipeline real (CLIP, ResNet, etc.)
    return image_embedding_rgb_hist(img, bins_per_channel=8)

# ---------------------------------------------------------
# Carga de artefactos
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts(pkl_bytes: bytes) -> Dict[str, Any]:
    artifacts = pickle.loads(pkl_bytes)
    required = ["df", "kmeans", "embeddings", "rep_indices"]
    missing = [k for k in required if k not in artifacts]
    if missing:
        raise ValueError(f"Faltan claves en el .pkl: {missing}")
    # Normalizaciones suaves
    if "proj_2d" not in artifacts:
        artifacts["proj_2d"] = None
    if "projector" not in artifacts:
        artifacts["projector"] = None
    if "posters_dir" not in artifacts:
        artifacts["posters_dir"] = "posters"
    if "embedder_name" not in artifacts:
        artifacts["embedder_name"] = "rgb_hist"
    # Tipos esperados
    artifacts["df"] = pd.DataFrame(artifacts["df"])
    artifacts["embeddings"] = np.asarray(artifacts["embeddings"], dtype=np.float32)
    return artifacts

# ---------------------------------------------------------
# Sidebar: .pkl y género
# ---------------------------------------------------------
st.sidebar.header("⚙️ Configuración")
pkl_file = st.sidebar.file_uploader("Sube tu modelo (.pkl)", type=["pkl", "pickle"])

genre_filter = st.sidebar.text_input("Filtrar por género (opcional, contiene):").strip().lower()

# ---------------------------------------------------------
# Entrada: imagen del usuario
# ---------------------------------------------------------
st.subheader("1) Subí el póster de la película")
up_img_file = st.file_uploader("Imagen (jpg/png/webp)", type=["jpg", "jpeg", "png", "webp"])

if (pkl_file is None) or (up_img_file is None):
    st.info("Sube el archivo .pkl y una imagen para continuar.")
    st.stop()

# Cargar artefactos
try:
    artifacts = load_artifacts(pkl_file.read())
except Exception as e:
    st.error(f"Error al leer el .pkl: {e}")
    st.stop()

df = artifacts["df"].copy()
embeddings = artifacts["embeddings"]
kmeans = artifacts["kmeans"]
rep_indices: Dict[int, List[int]] = artifacts["rep_indices"]
posters_dir = Path(artifacts.get("posters_dir", "posters"))
embedder_name = artifacts.get("embedder_name", "rgb_hist")
proj_2d = artifacts.get("proj_2d", None)
projector = artifacts.get("projector", None)

# Mostrar imagen de consulta
query_img = Image.open(up_img_file).convert("RGB")
st.image(query_img, caption="Imagen de consulta", use_column_width=False, width=320)

# ---------------------------------------------------------
# 2) Embedding y predicción de cluster
# ---------------------------------------------------------
query_vec = embed_image_query(query_img, embedder_name=embedder_name).reshape(1, -1)
try:
    cluster_id = int(kmeans.predict(query_vec)[0])
except Exception as e:
    st.error(f"No se pudo predecir el cluster con tu modelo: {e}")
    st.stop()

st.success(f"Cluster asignado: **{cluster_id}**")

# ---------------------------------------------------------
# 3) Top-5 representativas del cluster (con filtro por género opcional)
# ---------------------------------------------------------
st.subheader("2) Películas representativas del cluster")
indices_cluster = rep_indices.get(cluster_id, [])
if not indices_cluster:
    st.warning("No hay índices representativos almacenados para este cluster.")
else:
    # Filtro por género (si se escribió algo)
    def ok_genre(s: str) -> bool:
        return (genre_filter in s.lower()) if genre_filter else True

    shown = 0
    cols = st.columns(5)
    for i in indices_cluster:
        if i < 0 or i >= len(df):
            continue
        row = df.iloc[i]
        if not ok_genre(str(row.get("genres", ""))):
            continue
        img_path = posters_dir / str(row["poster_path"])
        with cols[shown % 5]:
            if img_path.exists():
                st.image(str(img_path), use_column_width=True)
            st.caption(f"{row['title']} ({int(row['year'])})\n{row.get('genres','')}")
        shown += 1
        if shown >= 5:
            break
    if shown == 0:
        st.info("No hay representantes que cumplan el filtro de género. Limpia el filtro o ajusta tus clusters.")

# ---------------------------------------------------------
# 4) Distribución 2D (resalta el cluster asignado)
# ---------------------------------------------------------
st.subheader("3) Distribución 2D (características visuales)")

# Calcula/recupera proyección 2D
try:
    if proj_2d is None:
        if projector is not None:
            proj = projector.transform(embeddings)
        else:
            # Fallback PCA 2D rápido (sin persistir)
            X = embeddings - embeddings.mean(axis=0, keepdims=True)
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            proj = (X @ Vt[:2].T)
        proj_2d = proj.astype(np.float32)
    else:
        proj_2d = np.asarray(proj_2d).astype(np.float32)
except Exception as e:
    st.error(f"No se pudo calcular/usar la proyección 2D: {e}")
    st.stop()

plot_df = pd.DataFrame({
    "x": proj_2d[:, 0],
    "y": proj_2d[:, 1],
    "title": df["title"],
    "year": df["year"],
    "genres": df["genres"],
})

# etiquetas de cluster para color (si existen)
try:
    labels = kmeans.labels_
    plot_df["cluster"] = labels.astype(int)
except Exception:
    plot_df["cluster"] = 0

# Marcar cuál es el cluster predicho
plot_df["is_query_cluster"] = (plot_df["cluster"] == cluster_id)

# Filtro de género opcional también en el scatter
if genre_filter:
    plot_df = plot_df[plot_df["genres"].str.lower().str.contains(genre_filter, na=False)]

chart = alt.Chart(plot_df).mark_circle(size=80).encode(
    x=alt.X("x", title="Dim 1"),
    y=alt.Y("y", title="Dim 2"),
    color=alt.Color("cluster:N", title="Cluster"),
    opacity=alt.condition("datum.is_query_cluster", alt.value(1.0), alt.value(0.35)),
    tooltip=["title","year","genres","cluster"]
).properties(height=520).interactive()

st.altair_chart(chart, use_container_width=True)

st.markdown("---")
st.caption("Tip: guarda en tu .pkl los representantes por cluster y, si puedes, también la proyección 2D entrenada para que todo cargue instantáneamente.")
