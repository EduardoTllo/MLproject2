# streamlit_visual_movies_app.py (versión sin escrituras en el repo)
# ---------------------------------------------------------------
# Funciona en entornos de solo-lectura (p. ej., Streamlit Cloud).
# Lee CSV + posters del proyecto y solo usa /tmp para temporales.
#
# Requisitos:
#   pip install streamlit pillow scikit-learn numpy pandas altair

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# sklearn opcional (tenemos fallback en numpy)
try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    KMeans = None
    PCA = None
    cosine_similarity = None

# -------------------------
# Config de página
# -------------------------
st.set_page_config(page_title="Películas por Similitud Visual", layout="wide")
st.title("🎬 Buscador y Explorador Visual de Películas")

st.markdown(
    """
    1) **Búsqueda por similitud visual** (póster del dataset o imagen subida).  
    2) **Representantes por cluster**.  
    3) **Distribución 2D** (PCA) por rasgos visuales.  
    4) **Filtros** por género, año y otros metadatos.
    """
)

# -------------------------
# Utilidades
# -------------------------
@st.cache_data(show_spinner=False)
def load_metadata(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected = {"movie_id","title","genres","year","country","director","actors","poster_path"}
    miss = expected - set(df.columns)
    if miss:
        raise ValueError(f"Faltan columnas en el CSV: {miss}")
    return df

def open_image(img_path: Path, max_side=512) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1:
        img = img.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
    return img

def image_embedding_rgb_hist(img: Image.Image, bins_per_channel: int = 8) -> np.ndarray:
    arr = np.array(img)
    hist = []
    for ch in range(3):
        h, _ = np.histogram(arr[..., ch], bins=bins_per_channel, range=(0, 256), density=False)
        hist.append(h.astype(np.float32))
    vec = np.concatenate(hist, axis=0)
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    return vec

@st.cache_resource(show_spinner=False)
def compute_dataset_embeddings(df: pd.DataFrame, posters_dir: Path, bins: int) -> np.ndarray:
    embs = []
    for _, row in df.iterrows():
        p = posters_dir / str(row["poster_path"])
        if not p.exists():
            embs.append(np.zeros(3*bins, dtype=np.float32))
            continue
        img = open_image(p, max_side=224)
        embs.append(image_embedding_rgb_hist(img, bins_per_channel=bins))
    return np.vstack(embs) if embs else np.zeros((0, 3*bins), dtype=np.float32)

def nearest_neighbors(query_vec: np.ndarray, emb_matrix: np.ndarray, topk: int = 12) -> List[int]:
    if emb_matrix.size == 0:
        return []
    if cosine_similarity is None:
        q = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        M = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-8)
        sims = (M @ q)
    else:
        sims = cosine_similarity(emb_matrix, query_vec.reshape(1, -1)).ravel()
    return np.argsort(-sims)[:topk].tolist()

def fit_kmeans(emb_matrix: np.ndarray, n_clusters: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    if len(emb_matrix) < n_clusters:
        # No forzamos escritura ni nada: devolvemos etiqueta 0
        return np.zeros(len(emb_matrix), dtype=int), np.array([emb_matrix.mean(axis=0)]) if len(emb_matrix) else np.zeros((1, emb_matrix.shape[1]))
    if KMeans is None:
        # K-means simple en numpy (sin escribir disco)
        rng = np.random.default_rng(seed)
        n, d = emb_matrix.shape
        centers = emb_matrix[rng.choice(n, size=n_clusters, replace=False)].copy()
        for _ in range(25):
            dists = ((emb_matrix[:, None, :] - centers[None, :, :])**2).sum(axis=2)
            labels = np.argmin(dists, axis=1)
            for k in range(n_clusters):
                pts = emb_matrix[labels == k]
                if len(pts) > 0:
                    centers[k] = pts.mean(axis=0)
        return labels, centers
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    labels = km.fit_predict(emb_matrix)
    centers = km.cluster_centers_
    return labels, centers

def cluster_representatives(emb_matrix: np.ndarray, labels: np.ndarray, centers: np.ndarray, per_cluster: int = 3) -> List[List[int]]:
    reps = []
    K = centers.shape[0]
    for k in range(K):
        idxs = np.where(labels == k)[0]
        if len(idxs) == 0:
            reps.append([])
            continue
        c = centers[k]
        dists = np.linalg.norm(emb_matrix[idxs] - c[None, :], axis=1)
        order = idxs[np.argsort(dists)]
        reps.append(order[:per_cluster].tolist())
    return reps

def project_2d(emb_matrix: np.ndarray) -> np.ndarray:
    if len(emb_matrix) == 0:
        return np.zeros((0,2), dtype=np.float32)
    if PCA is None:
        X = emb_matrix - emb_matrix.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        comp = (X @ Vt[:2].T)
        return comp.astype(np.float32)
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(emb_matrix).astype(np.float32)

def parse_genres(s: str) -> List[str]:
    return [g.strip() for g in str(s).split(";") if g.strip()]

# -------------------------
# Sidebar (solo lectura)
# -------------------------
st.sidebar.header("⚙️ Configuración / Datos (solo lectura)")

# Por defecto, leemos del propio proyecto (sin crear nada)
default_csv = st.sidebar.text_input(
    "Ruta CSV (en el proyecto)",
    value=str(Path(__file__).with_name("movies_demo.csv"))
)
posters_dir_text = st.sidebar.text_input(
    "Carpeta con pósters (en el proyecto)",
    value=str(Path(__file__).with_name("posters"))
)

bins = st.sidebar.slider("Bins por canal (histograma RGB)", 4, 32, 8, 1)
n_clusters = st.sidebar.slider("Número de clusters", 2, 12, 6, 1)
per_cluster = st.sidebar.slider("Representantes por cluster", 1, 8, 3, 1)

uploaded_csv = st.sidebar.file_uploader("…o sube un CSV", type=["csv"])

# Carga CSV (sin escribir a disco)
df = pd.read_csv(uploaded_csv) if uploaded_csv is not None else load_metadata(default_csv)
df["genres_list"] = df["genres"].apply(parse_genres)

posters_path = Path(posters_dir_text)
if not posters_path.exists():
    st.error(f"No existe la carpeta de pósters: {posters_path}")
    st.stop()

# Embeddings
with st.spinner("Calculando embeddings del dataset…"):
    emb = compute_dataset_embeddings(df, posters_path, bins=bins)

# -------------------------
# Filtros
# -------------------------
st.subheader("🧭 Filtros por metadatos")
all_genres = sorted({g for lst in df["genres_list"] for g in lst})
sel_genres = st.multiselect("Género(s)", options=all_genres)
ymin, ymax = int(df["year"].min()), int(df["year"].max())
year_range = st.slider("Rango de año", ymin, ymax, (ymin, ymax))
countries = sorted(df["country"].dropna().unique().tolist())
sel_countries = st.multiselect("País(es)", options=countries)
director_q = st.text_input("Director (contiene)").strip().lower()
actor_q = st.text_input("Actor/Actriz (contiene)").strip().lower()
title_q = st.text_input("Título (contiene)").strip().lower()

def apply_filters(_df: pd.DataFrame) -> pd.Series:
    m = pd.Series(True, index=_df.index)
    if sel_genres:
        m &= _df["genres_list"].apply(lambda gs: any(g in gs for g in sel_genres))
    y = _df["year"].fillna(0).astype(int)
    m &= y.between(year_range[0], year_range[1])
    if sel_countries:
        m &= _df["country"].isin(sel_countries)
    if director_q:
        m &= _df["director"].fillna("").str.lower().str.contains(director_q)
    if actor_q:
        m &= _df["actors"].fillna("").str.lower().str.contains(actor_q)
    if title_q:
        m &= _df["title"].fillna("").str.lower().str.contains(title_q)
    return m

mask = apply_filters(df)
df_filt = df[mask].reset_index(drop=True)
emb_filt = emb[mask.values]
st.caption(f"Películas que pasan los filtros: **{len(df_filt)}** de {len(df)}")

# -------------------------
# A) Búsqueda por similitud visual
# -------------------------
st.subheader("A) 🔍 Búsqueda por similitud visual")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Opción 1:** Selecciona un póster del dataset")
    if len(df_filt) == 0:
        st.info("No hay resultados con los filtros actuales.")
        st.stop()
    idx_in_df = st.selectbox(
        "Película de referencia",
        options=list(range(len(df_filt))),
        format_func=lambda i: f"[{int(df_filt.loc[i,'movie_id'])}] {df_filt.loc[i,'title']} ({df_filt.loc[i,'year']})"
    )
    sel_img_path = posters_path / df_filt.loc[idx_in_df, "poster_path"]
    st.image(str(sel_img_path), caption=df_filt.loc[idx_in_df, "title"], use_column_width=True)

with c2:
    st.markdown("**Opción 2:** Sube una imagen propia (se procesa en memoria, no se guarda)")
    up_img = st.file_uploader("Subir imagen", type=["jpg","jpeg","png","webp"])
    query_vec = None
    if up_img is not None:
        img = Image.open(up_img).convert("RGB")
        st.image(img, caption="Consulta", use_column_width=True)
        query_vec = image_embedding_rgb_hist(img, bins_per_channel=bins)

# Vector de consulta
if query_vec is None:
    # usamos el seleccionado del dataset filtrado
    q_idx_global = df.index[df.index[mask]][idx_in_df]
    query_vec = emb[q_idx_global]

topk = st.slider("Top-K resultados", 4, 36, 12, 1)
nn_idxs_local = nearest_neighbors(query_vec, emb_filt, topk=topk)
st.markdown("**Resultados similares (dentro del subconjunto filtrado):**")

cols = st.columns(6)
for j, i_local in enumerate(nn_idxs_local):
    r = df_filt.loc[i_local]
    p = posters_path / r["poster_path"]
    with cols[j % 6]:
        st.image(str(p), use_column_width=True)
        st.caption(f"{r['title']} ({int(r['year'])})\n{r['genres']}")

# -------------------------
# B) Representantes por cluster
# -------------------------
st.subheader("B) 🧩 Películas representativas por cluster")
labels, centers = fit_kmeans(emb_filt, n_clusters=n_clusters)
reps = cluster_representatives(emb_filt, labels, centers, per_cluster=per_cluster)
for k, idxs in enumerate(reps):
    st.markdown(f"**Cluster {k}** — {int((labels==k).sum())} películas")
    cols = st.columns(max(1, min(per_cluster, 6)))
    for c, i in enumerate(idxs):
        if c >= len(cols): break
        r = df_filt.loc[i]
        with cols[c]:
            st.image(str(posters_path / r["poster_path"]), use_column_width=True)
            st.caption(f"{r['title']} ({int(r['year'])})\n{r['genres']}")

# -------------------------
# C) Proyección 2D
# -------------------------
st.subheader("C) 🗺️ Distribución 2D por características visuales")
proj = project_2d(emb_filt)
proj_df = pd.DataFrame({
    "x": proj[:,0] if len(proj) else [],
    "y": proj[:,1] if len(proj) else [],
    "title": df_filt["title"],
    "year": df_filt["year"],
    "genres": df_filt["genres"],
    "country": df_filt["country"],
    "cluster": labels.astype(int) if len(df_filt) else [],
})
import altair as alt
chart = alt.Chart(proj_df).mark_circle(size=100).encode(
    x=alt.X("x", title="Componente 1"),
    y=alt.Y("y", title="Componente 2"),
    color=alt.Color("cluster:N", title="Cluster"),
    tooltip=["title","year","genres","country"]
).interactive()
st.altair_chart(chart, use_container_width=True)

# -------------------------
# D) Tabla
# -------------------------
st.subheader("D) 📋 Resultados filtrados")
st.dataframe(
    df_filt[["movie_id","title","year","genres","country","director","actors","poster_path"]],
    use_container_width=True
)

st.markdown("---")
st.caption("Esta versión no escribe archivos en el directorio del proyecto. Para procesar imágenes subidas se trabaja en memoria; para temporales, se usaría /tmp si fuera necesario.")
