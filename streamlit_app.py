# app.py — minimal con géneros fijos (en inglés)
import pickle
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import altair as alt

st.set_page_config(page_title="Movie Poster Clusters", layout="wide")
st.title("🎬 Movie Poster Clusters")

st.caption(
    "Upload a poster → predict its cluster using your trained .pkl → show the 5 most representative "
    "movies of that cluster → 2D visualization. Genre filter uses a fixed English list."
)

# -------------------------
# Fixed genre vocabulary (edit if you wish)
# -------------------------
FIXED_GENRES = [
    "Action","Adventure","Animation","Comedy","Crime","Documentary","Drama","Family","Fantasy",
    "History","Horror","Music","Mystery","Romance","Science Fiction","Thriller","War","Western",
    "Biography","Sport","Musical","Noir"
]
# opcional: normalizador para comparar
def norm(s: str) -> str:
    return s.strip().lower()

FIXED_GENRES_NORM = [norm(g) for g in FIXED_GENRES]

# -------------------------
# Simple embedding (cámbialo por tu pipeline real si usaste otro)
# -------------------------
def image_embedding_rgb_hist(img: Image.Image, bins_per_channel: int = 8) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    hist = []
    for ch in range(3):
        h, _ = np.histogram(arr[..., ch], bins=bins_per_channel, range=(0, 256), density=False)
        hist.append(h.astype(np.float32))
    vec = np.concatenate(hist, axis=0)
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    return vec

def embed_image_query(img: Image.Image, embedder_name: str = "rgb_hist") -> np.ndarray:
    # Reemplaza por tu extractor real (CLIP, etc.) si entrenaste con otro
    return image_embedding_rgb_hist(img, bins_per_channel=8)

# -------------------------
# Carga de artefactos
# -------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts(pkl_bytes: bytes) -> Dict[str, Any]:
    artifacts = pickle.loads(pkl_bytes)
    req = ["df", "kmeans", "embeddings", "rep_indices"]
    miss = [k for k in req if k not in artifacts]
    if miss:
        raise ValueError(f"Missing keys in .pkl: {miss}")
    artifacts["df"] = pd.DataFrame(artifacts["df"])
    artifacts["embeddings"] = np.asarray(artifacts["embeddings"], dtype=np.float32)
    artifacts.setdefault("posters_dir", "posters")
    artifacts.setdefault("embedder_name", "rgb_hist")
    artifacts.setdefault("proj_2d", None)
    artifacts.setdefault("projector", None)
    return artifacts

# -------------------------
# Sidebar: modelo y filtro de género (multi-select fijo)
# -------------------------
st.sidebar.header("⚙️ Settings")
pkl_file = st.sidebar.file_uploader("Upload your model (.pkl)", type=["pkl", "pickle"])

selected_genres = st.sidebar.multiselect(
    "Genres (fixed list, English)",
    options=FIXED_GENRES,
    default=[]
)

# helper: coincide si ALGUNO de los géneros fijos seleccionados aparece en la cadena de géneros del df
def split_genres_semicolon(s: str) -> List[str]:
    return [g.strip() for g in str(s).split(";") if g.strip()]

def genre_match(genres_str: str, selected: List[str]) -> bool:
    if not selected:
        return True
    # normalizamos ambas partes para comparar de forma robusta
    movie_genres_norm = set(norm(g) for g in split_genres_semicolon(genres_str))
    selected_norm = set(norm(g) for g in selected)
    return len(movie_genres_norm.intersection(selected_norm)) > 0

# -------------------------
# Input: imagen y pkl
# -------------------------
st.subheader("1) Upload a movie poster")
up_img_file = st.file_uploader("Image (jpg/png/webp)", type=["jpg", "jpeg", "png", "webp"])

if (pkl_file is None) or (up_img_file is None):
    st.info("Upload both the .pkl model and an image to continue.")
    st.stop()

try:
    artifacts = load_artifacts(pkl_file.read())
except Exception as e:
    st.error(f"Couldn't read .pkl: {e}")
    st.stop()

df = artifacts["df"].copy()
embeddings = artifacts["embeddings"]
kmeans = artifacts["kmeans"]
rep_indices: Dict[int, List[int]] = artifacts["rep_indices"]
posters_dir = Path(artifacts["posters_dir"])
embedder_name = artifacts["embedder_name"]
proj_2d = artifacts["proj_2d"]
projector = artifacts["projector"]

# -------------------------
# 2) Predicción de cluster
# -------------------------
query_img = Image.open(up_img_file).convert("RGB")
st.image(query_img, caption="Query poster", width=320)

query_vec = embed_image_query(query_img, embedder_name=embedder_name).reshape(1, -1)
try:
    cluster_id = int(kmeans.predict(query_vec)[0])
except Exception as e:
    st.error(f"Cluster prediction failed: {e}")
    st.stop()

st.success(f"Predicted cluster: **{cluster_id}**")

# -------------------------
# 3) Top-5 representativas del cluster (filtradas por géneros fijos)
# -------------------------
st.subheader("2) Top-5 representative movies of the predicted cluster")

indices_cluster = rep_indices.get(cluster_id, [])
if not indices_cluster:
    st.warning("No stored representatives for this cluster.")
else:
    shown = 0
    cols = st.columns(5)
    for idx in indices_cluster:
        if idx < 0 or idx >= len(df):
            continue
        row = df.iloc[idx]
        if not genre_match(str(row.get("genres", "")), selected_genres):
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
        st.info("No representatives match the selected genres. Clear the genre filter or adjust your clusters.")

# -------------------------
# 4) Proyección 2D (también respeta el filtro de géneros fijos)
# -------------------------
st.subheader("3) 2D distribution of movies (visual features)")

try:
    if proj_2d is None:
        if projector is not None:
            proj = projector.transform(embeddings)
        else:
            X = embeddings - embeddings.mean(axis=0, keepdims=True)
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            proj = (X @ Vt[:2].T)
        proj_2d = proj.astype(np.float32)
    else:
        proj_2d = np.asarray(proj_2d).astype(np.float32)
except Exception as e:
    st.error(f"2D projection error: {e}")
    st.stop()

plot_df = pd.DataFrame({
    "x": proj_2d[:, 0],
    "y": proj_2d[:, 1],
    "title": df["title"],
    "year": df["year"],
    "genres": df["genres"],
})

try:
    labels = kmeans.labels_
    plot_df["cluster"] = labels.astype(int)
except Exception:
    plot_df["cluster"] = 0

plot_df["is_pred_cluster"] = (plot_df["cluster"] == cluster_id)

# aplicar Filtro de géneros fijos en el scatter
if selected_genres:
    mask = plot_df["genres"].apply(lambda s: genre_match(str(s), selected_genres))
    plot_df = plot_df[mask]

chart = alt.Chart(plot_df).mark_circle(size=80).encode(
    x=alt.X("x", title="Dim 1"),
    y=alt.Y("y", title="Dim 2"),
    color=alt.Color("cluster:N", title="Cluster"),
    opacity=alt.condition("datum.is_pred_cluster", alt.value(1.0), alt.value(0.35)),
    tooltip=["title","year","genres","cluster"]
).properties(height=520).interactive()

st.altair_chart(chart, use_container_width=True)

st.markdown("---")
st.caption("Fixed English genres are used for filtering (semicolon-separated in your df). Matching is case-insensitive; a movie passes if it has ANY of the selected genres.")
