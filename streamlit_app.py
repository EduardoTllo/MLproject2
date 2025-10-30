# This cell creates a ready-to-run Streamlit app and a small demo dataset.
# Files saved under /mnt/data so you can download them.

import os, json, random, string, textwrap, zipfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

base = Path("/mnt/data/streamlit_visual_movies")
posters_dir = base / "posters"
base.mkdir(parents=True, exist_ok=True)
posters_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 1) Create a small demo dataset
# -----------------------------
random.seed(7)

genres_pool = ["Action", "Drama", "Comedy", "Sci-Fi", "Romance", "Thriller", "Horror", "Animation", "Adventure", "Crime"]
countries = ["USA", "UK", "France", "Japan", "India", "Spain", "Mexico", "Germany", "Brazil", "Korea"]
directors = ["A. Smith", "B. Johnson", "C. Lee", "D. García", "E. Müller", "F. Rossi", "G. Kim", "H. Suzuki"]
actors = ["Actor One; Actor Two", "Jane Doe; John Roe", "Foo Bar; Baz Qux", "María Pérez; Juan Díaz"]

def random_title():
    letters = ''.join(random.choice(string.ascii_uppercase) for _ in range(3))
    nouns = ["Legacy", "Echoes", "Horizons", "Pulse", "Odyssey", "Mirage", "Shadows", "Genesis", "Vertex", "Spectrum"]
    return f"{letters} {random.choice(nouns)}"

def make_poster(img_path, seed_color=None, text=""):
    w, h = 400, 600
    if seed_color is None:
        seed_color = tuple(random.randint(40, 215) for _ in range(3))
    img = Image.new("RGB", (w, h), seed_color)
    draw = ImageDraw.Draw(img)
    # Add a couple of shapes for variety
    draw.ellipse([w*0.1, h*0.15, w*0.9, h*0.55], outline=(255,255,255), width=6)
    draw.rectangle([w*0.2, h*0.6, w*0.8, h*0.9], outline=(255,255,255), width=6)
    # Add title text
    wrapped = "\n".join(textwrap.wrap(text, width=12))
    try:
        # Default PIL font; specific fonts may not exist in this environment
        draw.text((20, 20), wrapped, fill=(255,255,255))
    except Exception:
        pass
    img.save(img_path, quality=90)

rows = []
for i in range(60):
    title = random_title()
    year = random.randint(1980, 2025)
    gs = sorted(random.sample(genres_pool, k=random.randint(1,3)))
    country = random.choice(countries)
    director = random.choice(directors)
    actor = random.choice(actors)
    # Choose a color biased by genre to make clusters somewhat meaningful
    base_colors = {
        "Action": (200,60,60), "Drama": (80,120,180), "Comedy": (240,200,60), "Sci-Fi": (120,200,230),
        "Romance": (230,120,180), "Thriller": (130,130,130), "Horror": (80,0,0), "Animation": (80,220,80),
        "Adventure": (255,140,0), "Crime": (100,80,60)
    }
    seed_color = base_colors[random.choice(gs)]
    poster_path = posters_dir / f"{i:03d}.jpg"
    make_poster(poster_path, seed_color=seed_color, text=title)
    rows.append({
        "movie_id": i,
        "title": title,
        "genres": ";".join(gs),
        "year": year,
        "country": country,
        "director": director,
        "actors": actor,
        "poster_path": str(poster_path.name),
    })

df = pd.DataFrame(rows)
df.to_csv(base / "movies_demo.csv", index=False)

# ------------------------------------
# 2) Write the Streamlit app source
# ------------------------------------
app_code = r'''
# streamlit_visual_movies_app.py
# --------------------------------
# App de ejemplo: Búsqueda por similitud visual, clustering, proyección 2D y filtros por metadatos
#
# Cómo ejecutar:
#   1) Instala dependencias: pip install streamlit pillow scikit-learn numpy pandas altair
#   2) streamlit run streamlit_visual_movies_app.py
#
# Estructura esperada:
#   - Un CSV con columnas: [movie_id,title,genres,year,country,director,actors,poster_path]
#   - Carpeta "posters/" con las imágenes de los pósters (rutas relativas al CSV).
#
# Este script incluye:
#   - Cálculo sencillo de embeddings visuales (histogramas RGB normalizados) para evitar pesos pesados.
#   - Búsqueda de similitud (coseno)
#   - Agrupamiento KMeans y selección de representantes (más cercanos al centroide)
#   - Proyección 2D (PCA por defecto; si está UMAP, se puede activar)
#   - UI para filtrar por metadatos (género, año, país, etc.) y subir imagen propia

import os
import io
import math
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List

import streamlit as st
from PIL import Image

# Intentamos importar sklearn; si no está, mostramos un aviso
try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_similarity
except Exception as e:
    KMeans = None
    PCA = None
    cosine_similarity = None

# --------------------------------
# Configuración de la página
# --------------------------------
st.set_page_config(
    page_title="Películas por Similitud Visual",
    layout="wide",
)

st.title("🎬 Buscador y Explorador Visual de Películas")

st.markdown(
    """
    Esta app permite:
    1) **Buscar películas por similitud visual** a partir de un póster del dataset o una imagen subida.
    2) **Mostrar películas representativas de cada grupo (cluster)**.
    3) **Visualizar la distribución** en 2D según características visuales.
    4) **Filtrar** por género, año y otros metadatos.
    """
)

# --------------------------------
# Utilidades
# --------------------------------
@st.cache_data(show_spinner=False)
def load_metadata(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalizamos columnas esperadas
    expected_cols = {"movie_id","title","genres","year","country","director","actors","poster_path"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")
    return df

def open_image(img_path: Path, max_side=512) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    # Resize conservando aspecto
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
    return img

def image_embedding_rgb_hist(img: Image.Image, bins_per_channel: int = 8) -> np.ndarray:
    """Embedding simple: histograma RGB normalizado (bins_per_channel por canal)."""
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
        img_path = posters_dir / row["poster_path"]
        if not img_path.exists():
            # Imagen faltante -> vector cero (o aleatorio suave)
            vec = np.zeros(3*bins, dtype=np.float32)
        else:
            img = open_image(img_path, max_side=224)
            vec = image_embedding_rgb_hist(img, bins_per_channel=bins)
        embs.append(vec)
    return np.vstack(embs)

def nearest_neighbors(query_vec: np.ndarray, emb_matrix: np.ndarray, topk: int = 12) -> List[int]:
    if cosine_similarity is None:
        # fallback coseno manual
        q = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        M = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-8)
        sims = (M @ q)
    else:
        sims = cosine_similarity(emb_matrix, query_vec.reshape(1, -1)).ravel()
    idxs = np.argsort(-sims)[:topk]
    return idxs.tolist()

def fit_kmeans(emb_matrix: np.ndarray, n_clusters: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    if KMeans is None:
        # Implementación muy básica de k-means con numpy si sklearn no está
        # (para demos pequeñas funciona)
        rng = np.random.default_rng(seed)
        n, d = emb_matrix.shape
        centers = emb_matrix[rng.choice(n, size=n_clusters, replace=False)].copy()
        for _ in range(25):
            # asignación
            dists = ((emb_matrix[:, None, :] - centers[None, :, :])**2).sum(axis=2)
            labels = np.argmin(dists, axis=1)
            # actualización
            for k in range(n_clusters):
                pts = emb_matrix[labels == k]
                if len(pts) > 0:
                    centers[k] = pts.mean(axis=0)
        return labels, centers
    else:
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
        labels = km.fit_predict(emb_matrix)
        centers = km.cluster_centers_
        return labels, centers

def cluster_representatives(emb_matrix: np.ndarray, labels: np.ndarray, centers: np.ndarray, per_cluster: int = 3) -> List[List[int]]:
    reps = []
    for k in range(centers.shape[0]):
        idxs = np.where(labels == k)[0]
        if len(idxs) == 0:
            reps.append([])
            continue
        c = centers[k]
        # distancia euclídea al centroide
        dists = np.linalg.norm(emb_matrix[idxs] - c[None, :], axis=1)
        order = idxs[np.argsort(dists)]
        reps.append(order[:per_cluster].tolist())
    return reps

def project_2d(emb_matrix: np.ndarray, method: str = "PCA", seed: int = 42) -> np.ndarray:
    if method == "PCA" or PCA is None:
        # Fallback a PCA simple con numpy si sklearn no está
        X = emb_matrix - emb_matrix.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        comp = (X @ Vt[:2].T)
        return comp.astype(np.float32)
    else:
        pca = PCA(n_components=2, random_state=seed)
        comp = pca.fit_transform(emb_matrix)
        return comp.astype(np.float32)

def parse_genres(s: str) -> List[str]:
    if isinstance(s, str):
        return [g.strip() for g in s.split(";") if g.strip()]
    return []

# --------------------------------
# Sidebar: carga de datos y params
# --------------------------------
st.sidebar.header("⚙️ Configuración / Datos")

default_csv = st.sidebar.text_input(
    "Ruta CSV",
    value=str(Path(__file__).with_name("movies_demo.csv"))
)

posters_dir = st.sidebar.text_input(
    "Carpeta con pósters",
    value=str(Path(__file__).with_name("posters"))
)

bins = st.sidebar.slider("Bins por canal (histograma RGB)", 4, 32, 8, 1)
n_clusters = st.sidebar.slider("Número de clusters", 2, 12, 6, 1)
per_cluster = st.sidebar.slider("Representantes por cluster", 1, 8, 3, 1)

uploaded_csv = st.sidebar.file_uploader("…o sube un CSV alternativo", type=["csv"])

if uploaded_csv is not None:
    df = pd.read_csv(uploaded_csv)
else:
    df = load_metadata(default_csv)

# Aseguramos columnas y convertimos géneros a listas
df["genres_list"] = df["genres"].apply(parse_genres)

posters_path = Path(posters_dir)
if not posters_path.exists():
    st.sidebar.error(f"No existe la carpeta de posters: {posters_path}")
    st.stop()

# Embeddings del dataset
with st.spinner("Calculando embeddings del dataset…"):
    emb = compute_dataset_embeddings(df, posters_path, bins=bins)

# --------------------------------
# Filtros por metadatos
# --------------------------------
st.subheader("🧭 Filtros por metadatos")

all_genres = sorted({g for lst in df["genres_list"] for g in lst})
sel_genres = st.multiselect("Género(s)", options=all_genres, default=[])

min_year, max_year = int(df["year"].min()), int(df["year"].max())
year_range = st.slider("Rango de año", min_year, max_year, (min_year, max_year))

countries = sorted(df["country"].dropna().unique().tolist())
sel_countries = st.multiselect("País(es)", options=countries, default=[])

director_query = st.text_input("Director (contiene)", value="").strip().lower()
actor_query = st.text_input("Actor/Actriz (contiene)", value="").strip().lower()
title_query = st.text_input("Título (contiene)", value="").strip().lower()

def apply_filters(_df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=_df.index)
    if sel_genres:
        mask = mask & _df["genres_list"].apply(lambda gs: any(g in gs for g in sel_genres))
    y = _df["year"].fillna(0).astype(int)
    mask = mask & (y.between(year_range[0], year_range[1]))
    if sel_countries:
        mask = mask & _df["country"].isin(sel_countries)
    if director_query:
        mask = mask & _df["director"].fillna("").str.lower().str.contains(director_query)
    if actor_query:
        mask = mask & _df["actors"].fillna("").str.lower().str.contains(actor_query)
    if title_query:
        mask = mask & _df["title"].fillna("").str.lower().str.contains(title_query)
    return mask

mask = apply_filters(df)
df_filt = df[mask].reset_index(drop=True)
emb_filt = emb[mask.values]

st.caption(f"Películas que pasan los filtros: **{len(df_filt)}** de {len(df)}")

# --------------------------------
# A) Búsqueda por similitud visual
# --------------------------------
st.subheader("A) 🔍 Búsqueda por similitud visual")

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("**Opción 1:** Selecciona un póster del dataset")
    idx_in_df = st.selectbox(
        "Película de referencia",
        options=list(range(len(df_filt))),
        format_func=lambda i: f"[{int(df_filt.loc[i,'movie_id'])}] {df_filt.loc[i,'title']} ({df_filt.loc[i,'year']})"
    )
    sel_img_path = Path(posters_path) / df_filt.loc[idx_in_df, "poster_path"]
    st.image(str(sel_img_path), caption=df_filt.loc[idx_in_df, "title"], use_column_width=True)

with col2:
    st.markdown("**Opción 2:** Sube una imagen propia")
    up_img = st.file_uploader("Subir imagen", type=["jpg","jpeg","png","webp"])
    query_vec = None
    if up_img is not None:
        img = Image.open(up_img).convert("RGB")
        st.image(img, caption="Consulta", use_column_width=True)
        query_vec = image_embedding_rgb_hist(img, bins_per_channel=bins)

# Determinar vector de consulta
if query_vec is None:
    # usar el seleccionado del dataset filtrado
    q_idx_global = df.index[df.index[mask]][idx_in_df]
    query_vec = emb[q_idx_global]

topk = st.slider("Top-K resultados", 4, 36, 12, 1)
nn_idxs_local = nearest_neighbors(query_vec, emb_filt, topk=topk)
st.markdown("**Resultados similares (dentro del subconjunto filtrado):**")

grid_cols = st.columns(6)
for j, i_local in enumerate(nn_idxs_local):
    i = i_local
    with grid_cols[j % 6]:
        row = df_filt.loc[i]
        p = posters_path / row["poster_path"]
        st.image(str(p), use_column_width=True)
        st.caption(f"{row['title']} ({int(row['year'])})\n\n{row['genres']}")

# --------------------------------
# B) Representantes por cluster
# --------------------------------
st.subheader("B) 🧩 Películas representativas por cluster")
if len(df_filt) >= n_clusters:
    labels, centers = fit_kmeans(emb_filt, n_clusters=n_clusters)
    reps = cluster_representatives(emb_filt, labels, centers, per_cluster=per_cluster)
    for k, idxs in enumerate(reps):
        st.markdown(f"**Cluster {k}** — {sum(labels==k)} películas")
        cols = st.columns(per_cluster)
        for c, i in enumerate(idxs):
            if c >= len(cols): break
            with cols[c]:
                r = df_filt.loc[i]
                st.image(str(posters_path / r["poster_path"]), use_column_width=True)
                st.caption(f"{r['title']} ({int(r['year'])})\n{r['genres']}")
else:
    st.info("Ajusta el número de clusters o los filtros para tener suficientes películas.")

# --------------------------------
# C) Proyección 2D
# --------------------------------
st.subheader("C) 🗺️ Distribución 2D por características visuales")

proj = project_2d(emb_filt, method="PCA")
proj_df = pd.DataFrame({
    "x": proj[:,0],
    "y": proj[:,1],
    "title": df_filt["title"],
    "year": df_filt["year"],
    "genres": df_filt["genres"],
    "country": df_filt["country"],
})

# Color por cluster si está disponible
if 'labels' in locals():
    proj_df["cluster"] = labels.astype(int)
else:
    proj_df["cluster"] = 0

# Usamos Altair para un scatter interactivo
import altair as alt
chart = alt.Chart(proj_df).mark_circle(size=100).encode(
    x=alt.X("x", title="Componente 1"),
    y=alt.Y("y", title="Componente 2"),
    color=alt.Color("cluster:N", title="Cluster"),
    tooltip=["title","year","genres","country"]
).interactive()

st.altair_chart(chart, use_container_width=True)

# --------------------------------
# D) Tabla de resultados filtrados
# --------------------------------
st.subheader("D) 📋 Resultados filtrados (con metadatos)")
st.dataframe(
    df_filt[["movie_id","title","year","genres","country","director","actors","poster_path"]],
    use_container_width=True
)

st.markdown("---")
st.caption("Nota: los embeddings se basan en histogramas RGB normalizados de los pósters para una demo rápida y ligera. Puedes reemplazar `image_embedding_rgb_hist` por un extractor más potente (p. ej., CLIP, ResNet) si tu entorno lo permite.")
'''

app_path = base / "streamlit_visual_movies_app.py"
with open(app_path, "w", encoding="utf-8") as f:
    f.write(app_code)

# ------------------------------------
# 3) Package everything into a zip
# ------------------------------------
zip_path = "/mnt/data/streamlit_visual_movies_demo.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(app_path, arcname="streamlit_visual_movies_app.py")
    z.write(base / "movies_demo.csv", arcname="movies_demo.csv")
    # add posters
    for p in posters_dir.glob("*.jpg"):
        z.write(p, arcname=f"posters/{p.name}")

'''# Show a small preview dataframe
from caas_jupyter_tools import display_dataframe_to_user
display_dataframe_to_user("Demo movies (first 10)", pd.read_csv(base / "movies_demo.csv").head(10))

zip_path, str(app_path), str(base / "movies_demo.csv"), str(posters_dir)'''
