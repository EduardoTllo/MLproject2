# app.py
import os
import io
import json
import shutil
import tempfile
from pathlib import Path
from collections import deque  # para DBSCAN desde cero
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import streamlit as st

import cv2
from PIL import Image

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import umap

@st.cache_data(show_spinner=False)
def build_train_gallery(df_train_label: pd.DataFrame, train_dir: str):
    """
    Devuelve un DataFrame solo con items que tienen póster en disco,
    y una columna 'img_path' con la ruta del archivo.
    """
    df = df_train_label.copy()
    df["img_path"] = df["movieId"].apply(lambda mid: find_image_in_folder_by_id(mid, train_dir))
    df = df[~df["img_path"].isna()]  # mantener solo los que tienen imagen
    # normaliza columnas por si vienen vacías
    if "Genre" not in df.columns:
        df["Genre"] = ""
    df["title"] = df["title"].astype(str).fillna("")
    return df[["movieId", "title", "Genre", "img_path"]]



# ---------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------
path = 'LDA_UMAP_DBSCAN/'
TRAIN_IMAGE_DIR = os.path.join(path, "Train_image")

# ---------------------------------------------------------------------
# Config de Streamlit
# ---------------------------------------------------------------------
st.set_page_config(page_title="Recomendador por Pósters", layout="wide")
st.title("🎬 Recomendación de Películas por Similitud Visual")

st.markdown(
    """
**¿Cómo funciona?**  
1) **Entrada**: sube un póster o elige uno del set de entrenamiento y presiona 🔎 Recomendar.  
2) **Extracción**: obtenemos rasgos visuales de color (HSV), textura (LBP, GLCM), bordes (HOG) y forma (Hu).  
3) **Proyección**: mapeamos el espacio a 2D con **LDA → UMAP** para que la distancia ≈ similitud visual.  
4) **Clustering**: detectamos grupos naturales con **DBSCAN** (sin forzar forma ni número de clusters).  
5) **Recomendación**: buscamos los **10** más cercanos con **kNN** sobre la proyección.

"""
)


# ---------------------------------------------------------------------
# DBSCAN DESDE CERO
# ---------------------------------------------------------------------
def dbscan_desde_cero(X, eps=0.8, min_samples=15):
    n = X.shape[0]
    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    def obtener_vecinos(idx):
        dists = np.linalg.norm(X - X[idx], axis=1)
        return np.where(dists <= eps)[0]

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        vecinos = obtener_vecinos(i)

        if len(vecinos) < min_samples:
            labels[i] = -1
        else:
            cluster_id += 1
            labels[i] = cluster_id
            cola = deque(vecinos)
            while cola:
                j = cola.popleft()
                if not visited[j]:
                    visited[j] = True
                    vecinos_j = obtener_vecinos(j)
                    if len(vecinos_j) >= min_samples:
                        cola.extend(vecinos_j)
                if labels[j] == -1:
                    labels[j] = cluster_id
    return labels

# ---------------------------------------------------------------------
# Utilidades y extracción de features
# ---------------------------------------------------------------------
EPS = 1e-8
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif", ".JPG", ".PNG", ".JPEG"}

def list_images(root: Path):
    allow = {e.lower() for e in IMG_EXTS}
    return sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in allow])

def preprocess(img_bgr, size=(256,256)):
    img = cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

def hsv_24(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H,S,V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    def h1(ch, bins, rmax):
        h,_ = np.histogram(ch.ravel(), bins=bins, range=(0,rmax))
        h = h.astype(np.float32); h /= (h.sum()+EPS)
        return h
    return np.concatenate([h1(H,8,180), h1(S,8,256), h1(V,8,256)]).astype(np.float32)

def hsv_24_names():
    names = []
    for c in ["H","S","V"]:
        for i in range(8):
            names.append(f"hsv_{c}_bin{i}")
    return names

def hsv_stats(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H,S,V = hsv[:,:,0].astype(np.float32), hsv[:,:,1].astype(np.float32), hsv[:,:,2].astype(np.float32)
    def skew(x):
        m = x.mean(); s = x.std() + EPS
        return (((x-m)/s)**3).mean()
    p_dark = (V < 40).mean()
    p_sat  = (S > 180).mean()
    Vf = cv2.Sobel(V, cv2.CV_32F, 1, 0, ksize=3)**2 + cv2.Sobel(V, cv2.CV_32F, 0, 1, ksize=3)**2
    v_contrast = Vf.mean()
    return np.array([H.mean(), H.std(), skew(H),
                     S.mean(), S.std(),
                     V.mean(), V.std(),
                     p_dark, p_sat, v_contrast], dtype=np.float32)

def hsv_stats_names():
    return ["h_mean","h_std","h_skew","s_mean","s_std","v_mean","v_std","p_dark","p_sat","v_contrast"]

def lbp_uniform(gray, radius=1, P=8):
    g = gray.astype(np.int16)
    h,w = g.shape
    neighbors = [(-1,-1),(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0)]
    if radius != 1:
        neighbors = [(dx*radius, dy*radius) for (dx,dy) in neighbors]
    m = radius
    center = g[m:-m, m:-m]
    lbp = np.zeros_like(center, dtype=np.uint8)
    for bit,(dx,dy) in enumerate(neighbors):
        nbr = g[m+dy:h-m+dy, m+dx:w-m+dx]
        lbp |= ((nbr >= center).astype(np.uint8) << bit)
    maps = np.zeros(256, dtype=np.uint8)
    def transitions(x):
        b = ((x<<1)&0xFF) | (x>>7)
        return bin((x ^ b) & 0xFF).count("1")
    idx = 0
    for i in range(256):
        if transitions(i) <= 2:
            maps[i] = idx; idx += 1
        else:
            maps[i] = P+1
    lbp_u = maps[lbp]
    hist = np.bincount(lbp_u.ravel(), minlength=P+2).astype(np.float32)
    hist /= (hist.sum()+EPS)
    return hist

def lbp_names(radius=1, P=8):
    return [f"lbp_r{radius}_bin{i}" for i in range(P+2)]

def glcm_light(gray, levels=32):
    step = max(1, 256//levels)
    q = (gray // step).astype(np.uint8)
    h,w = q.shape
    feats = []
    for d in (1,2):
        for th in (0, np.pi/2):
            dx = int(round(np.cos(th)*d)); dy = int(round(np.sin(th)*d))
            x_from = max(0,-dx); x_to = min(w, w-dx)
            y_from = max(0,-dy); y_to = min(h, h-dy)
            I = q[y_from:y_to, x_from:x_to]
            J = q[y_from+dy:y_to+dy, x_from+dx:x_to+dx]
            P = np.zeros((levels,levels), dtype=np.float64)
            np.add.at(P, (I.ravel(), J.ravel()), 1)
            P = P + P.T
            s = P.sum()
            if s > 0: P /= s
            i = np.arange(levels)[:,None]
            j = np.arange(levels)[None,:]
            contrast = (((i-j)**2) * P).sum()
            hom      = (P / (1.0 + (i-j)**2)).sum()
            energy   = np.sqrt((P**2).sum())
            feats.extend([contrast, hom, energy])
    return np.array(feats, dtype=np.float32)

def glcm_light_names():
    names = []
    for d in (1,2):
        for th_name in ("0","90"):
            names.append(f"glcm_d{d}_a{th_name}_contrast")
            names.append(f"glcm_d{d}_a{th_name}_homogeneity")
            names.append(f"glcm_d{d}_a{th_name}_energy")
    return names

def hog_super_compacto(gray, cell=32, bins=6):
    g = gray.astype(np.float32)
    h, w = g.shape
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    ang = np.mod(ang, 180.0)
    ncy, ncx = h // cell, w // cell
    edges = np.linspace(0, 180, bins+1)
    cell_hists = []
    for cy in range(ncy):
        for cx in range(ncx):
            y0,y1 = cy*cell, (cy+1)*cell
            x0,x1 = cx*cell, (cx+1)*cell
            hist, _ = np.histogram(
                ang[y0:y1, x0:x1].ravel(),
                bins=edges,
                weights=mag[y0:y1, x0:x1].ravel()
            )
            hist = hist.astype(np.float32)
            hist /= (hist.sum()+EPS)
            cell_hists.append(hist)
    if not cell_hists:
        return np.zeros((bins*2,), dtype=np.float32)
    cell_hists = np.stack(cell_hists, axis=0)
    hog_mean = cell_hists.mean(axis=0)
    hog_std  = cell_hists.std(axis=0)
    return np.concatenate([hog_mean, hog_std]).astype(np.float32)

def hog_super_compacto_names(bins=6):
    names = [f"hog_mean_bin{i}" for i in range(bins)]
    names += [f"hog_std_bin{i}" for i in range(bins)]
    return names

def hu_feats(gray):
    m = cv2.moments(gray)
    hu = cv2.HuMoments(m).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu)+EPS)
    return hu.astype(np.float32)

def hu_feats_names():
    return [f"hu_{i+1}" for i in range(7)]

def extract_features_robust_capped(img_bgr, size=(256,256)):
    img, gray = preprocess(img_bgr, size=size)
    f_color = hsv_24(img)
    f_stats = hsv_stats(img)
    f_lbp1  = lbp_uniform(gray, radius=1)
    f_lbp2  = lbp_uniform(gray, radius=2)
    f_glcm  = glcm_light(gray)
    f_hog   = hog_super_compacto(gray)
    f_hu    = hu_feats(gray)
    feats = np.concatenate([f_color, f_stats, f_lbp1, f_lbp2, f_glcm, f_hog, f_hu]).astype(np.float32)
    return feats

def feature_headers():
    return (
        hsv_24_names() +
        hsv_stats_names() +
        lbp_names(radius=1) +
        lbp_names(radius=2) +
        glcm_light_names() +
        hog_super_compacto_names() +
        hu_feats_names()
    )

def build_features(in_dir, out_dir, size=(256,256)):
    in_dir = Path(in_dir); out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = list_images(in_dir)
    X_list, ids = [], []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] no se pudo leer {p}")
            continue
        v = extract_features_robust_capped(img, size=size)
        X_list.append(v)
        ids.append(p.stem)
    if len(X_list) == 0:
        raise RuntimeError("No se pudieron extraer características de ninguna imagen")
    X = np.vstack(X_list).astype(np.float32)
    ids = np.array(ids, dtype=object)
    headers = feature_headers()
    np.save(out_dir / "Y_hsv.npy", X)
    np.save(out_dir / "image_ids_test.npy", ids)
    with open(out_dir / "features_meta_test.json", "w") as f:
        json.dump({"num_images": int(len(ids)), "dim": int(X.shape[1]), "headers": headers}, f, indent=2)
    return X, ids, headers

# ---------------------------------------------------------------------
# Helpers de imágenes
# ---------------------------------------------------------------------
def _normalize_id(mid):
    return str(int(mid)).strip()

def find_image_in_folder_by_id(movie_id, folder):
    base = _normalize_id(movie_id)
    for ext in [".jpg", ".jpeg", ".png", ".webp", ".JPG", ".PNG", ".JPEG", ".WEBP"]:
        p = os.path.join(folder, base + ext)
        if os.path.exists(p):
            return p
    return None

# ---------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_labels_and_bins():
    try:
        generos = pd.read_csv(os.path.join(path, 'MovieGenre.csv'), encoding='latin-1')
        df_test = pd.read_csv(os.path.join(path, 'movies_test.csv'))
        df_train = pd.read_csv(os.path.join(path, 'movies_train.csv'))
        ids = pd.read_csv(os.path.join(path, 'links.csv'))

        df_test_label = (
            df_test.merge(ids, on='movieId', how='left').merge(generos, on='imdbId', how='left')
        ).drop_duplicates()
        df_train_label = (
            df_train.merge(ids, on='movieId', how='left').merge(generos, on='imdbId', how='left')
        ).drop_duplicates()

        imdb_col = None
        for c in ["IMDBScore", "IMDB Score", "IMDB_Score", "IMDB Rating", "IMDB_Rating"]:
            if c in df_test_label.columns and c in df_train_label.columns:
                imdb_col = c
                break
        if imdb_col is None:
            imdb_col = "IMDBScore"
            df_test_label[imdb_col] = np.nan
            df_train_label[imdb_col] = np.nan

        # Importante: columna correcta "title"
        df_test_label = df_test_label[["movieId", "title", "Genre", imdb_col]]
        df_train_label = df_train_label[["movieId", "title", "Genre", imdb_col]]

        df_test_label["genre_p"] = df_test_label["Genre"].str.split("|").str[0]
        df_train_label["genre_p"] = df_train_label["Genre"].str.split("|").str[0]

        genre_dummies_train = df_train_label["Genre"].str.get_dummies(sep="|")
        genre_dummies_test = df_test_label["Genre"].str.get_dummies(sep="|")

        df_train_bin = pd.concat(
            [df_train_label[["movieId", "title", imdb_col, "genre_p"]], genre_dummies_train], axis=1
        )
        df_test_bin = pd.concat(
            [df_test_label[["movieId", "title", imdb_col, "genre_p"]], genre_dummies_test], axis=1
        )
        return df_train_label, df_test_label, df_train_bin, df_test_bin
    except Exception as e:
        raise RuntimeError(f"Error cargando CSVs: {e}")

@st.cache_data(show_spinner=False)
def load_train_features():
    try:
        X_hsv = np.load(os.path.join(path, "Data Modelamiento", "X_hsv.npy"), mmap_mode="r")
        X_hsv_ids = np.load(os.path.join(path, "Data Modelamiento", "image_ids_train.npy"), allow_pickle=True)
        feature_names = [f"feature_{i+1}" for i in range(X_hsv.shape[1])]
        df_X_train = pd.DataFrame(X_hsv, columns=feature_names)
        df_X_train.insert(0, "movieId", X_hsv_ids)
        df_X_train["movieId"] = df_X_train["movieId"].astype("int64")
        return df_X_train
    except Exception as e:
        raise RuntimeError(f"Error cargando features de entrenamiento (.npy): {e}")

# ---------------------------------------------------------------------
# Entrenar proyección y clustering
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_projection_and_cluster(df_X_train, df_train_bin):
    try:
        df = df_X_train.merge(df_train_bin[["movieId", "genre_p"]], on="movieId", how="left")
        df = df.dropna(subset=["genre_p"])
        X = df.drop(columns=["movieId", "genre_p"]).values

        enc = LabelEncoder()
        y = enc.fit_transform(df["genre_p"].astype(str))

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        lda = LinearDiscriminantAnalysis(n_components=2)
        X_lda = lda.fit_transform(Xs, y)

        reducer = umap.UMAP(
            n_neighbors=30,
            min_dist=0.1,
            metric="cosine",
            random_state=42
        )
        X_umap = reducer.fit_transform(X_lda)

        # DBSCAN desde cero en UMAP
        eps = 0.8
        min_samples = 15
        labels = dbscan_desde_cero(X_umap, eps=eps, min_samples=min_samples)

        nn_global = NearestNeighbors(n_neighbors=5, metric="euclidean")
        nn_global.fit(X_umap)

        knn_genre = KNeighborsClassifier(n_neighbors=7, metric="euclidean")
        knn_genre.fit(X_umap, y)

        # ...
        train_ids = df["movieId"].values

        return {
        "scaler": scaler,
        "lda": lda,
        "umap": reducer,
        "labels": labels,
        "dbscan_params": {"eps": eps, "min_samples": min_samples},
        "X_umap": X_umap,
        "train_ids": train_ids,
        "label_encoder": enc,
        "nn_global": nn_global,
        "knn_genre": knn_genre,
        "train_cols": df_X_train.drop(columns=["movieId"], errors="ignore").columns,
        "id2idx": {int(mid): i for i, mid in enumerate(train_ids)}  # << NUEVO
        }

    except Exception as e:
        raise RuntimeError(f"Error entrenando proyección y clustering: {e}")

# ---------------------------------------------------------------------
# Sidebar: Input
# ---------------------------------------------------------------------
st.sidebar.header("Entrada")
input_mode = st.sidebar.radio(
    "¿Cómo quieres seleccionar la imagen base?",
    ["Subir imagen", "Elegir de Train"],
    index=0
)

uploaded_file = None
selected_train_movie = None

if input_mode == "Subir imagen":
    uploaded_file = st.sidebar.file_uploader("Sube un póster (JPG, PNG, WEBP)", type=["jpg","jpeg","png","webp"])
else:
    try:
        # Cargamos metadatos (train/test) una sola vez
        df_train_label, df_test_label, df_train_bin, df_test_bin = load_labels_and_bins()
        df_gallery = build_train_gallery(df_train_label, TRAIN_IMAGE_DIR)

        # ------- Filtros en sidebar -------
        st.sidebar.success("Metadatos cargados")
        q_title = st.sidebar.text_input("🔎 Buscar título (contiene):", value="")
        # lista de géneros desde el CSV
        all_genres = sorted({g for s in df_gallery["Genre"].fillna("").astype(str)
                               for g in s.split("|") if g})
        sel_genres = st.sidebar.multiselect("Filtrar por género", options=all_genres, default=[])

        # paginación
        page_size = st.sidebar.slider("Miniaturas por página", min_value=10, max_value=50, value=10, step=10)
        # aplicar filtros
        df_f = df_gallery
        if q_title:
            df_f = df_f[df_f["title"].str.contains(q_title, case=False, na=False)]
        if sel_genres:
            df_f = df_f[df_f["Genre"].fillna("").apply(lambda s: any(g in s.split("|") for g in sel_genres))]

        total = len(df_f)
        max_pages = max(1, int(np.ceil(total / page_size)))
        page = st.sidebar.number_input("Página", min_value=1, max_value=max_pages, value=1, step=1)

        # subset de la página
        start = (page - 1) * page_size
        end = start + page_size
        df_page = df_f.iloc[start:end]

        st.subheader("🖼️ Elige una película del Train")
        st.caption(f"Mostrando {len(df_page)} de {total} coincidencias")

        # ------- Galería con botones de selección -------
        ncols = 5
        rows = [df_page.iloc[i:i+ncols] for i in range(0, len(df_page), ncols)]
        for r in rows:
            cols = st.columns(ncols, gap="small")
            for col, (_, row) in zip(cols, r.iterrows()):
                with col:
                    st.image(row["img_path"], use_column_width=True,
                             caption=f'{int(row["movieId"])} — {row["title"][:40]}')
                    if st.button("Seleccionar", key=f"sel_{int(row['movieId'])}_{page}"):
                        st.session_state["selected_train_movie"] = int(row["movieId"])

        # Mostrar selección actual en el sidebar
        selected_train_movie = st.session_state.get("selected_train_movie")
        st.sidebar.write("🎯 Seleccionado:", selected_train_movie if selected_train_movie else "—")

    except Exception as e:
        st.sidebar.error(str(e))

btn_run = st.sidebar.button("🔎 Recomendar", use_container_width=True)


# ---------------------------------------------------------------------
# Estado de carga
# ---------------------------------------------------------------------
with st.expander("📦 Estado de carga de datos", expanded=True):
    try:
        if "df_train_bin" not in st.session_state:
            df_train_label, df_test_label, df_train_bin, df_test_bin = load_labels_and_bins()
            st.session_state["df_train_label"] = df_train_label
            st.session_state["df_test_label"] = df_test_label
            st.session_state["df_train_bin"] = df_train_bin
            st.session_state["df_test_bin"] = df_test_bin
        st.success("CSV y metadatos cargados")
    except Exception as e:
        st.error(str(e))

    try:
        if "df_X_train" not in st.session_state:
            st.session_state["df_X_train"] = load_train_features()
        st.success("Features de entrenamiento (.npy) cargados")
    except Exception as e:
        st.error(str(e))

    try:
        if "model" not in st.session_state:
            st.session_state["model"] = train_projection_and_cluster(
                st.session_state["df_X_train"],
                st.session_state["df_train_bin"]
            )
        st.success("Modelo LDA + UMAP + DBSCAN + kNN entrenado")
    except Exception as e:
        st.error(str(e))

# ---------------------------------------------------------------------
# Extraer features del input
# ---------------------------------------------------------------------
def extract_query_features_with_build_features(file_bytes: bytes, filename: str):
    tmpdir = tempfile.mkdtemp(prefix="query_")
    outdir = tempfile.mkdtemp(prefix="query_feats_")
    try:
        img_path = os.path.join(tmpdir, filename)
        with open(img_path, "wb") as f:
            f.write(file_bytes)

        X, ids, _ = build_features(tmpdir, outdir, size=(256,256))
        v = X[0].astype(np.float32)
        qid = str(ids[0])
        return v, qid
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
            shutil.rmtree(outdir, ignore_errors=True)
        except Exception:
            pass

# ---------------------------------------------------------------------
# Recomendación con exclusión de IDs
# ---------------------------------------------------------------------
def recommend_from_feature_vector(v_query: np.ndarray, exclude_ids=None, topk: int = 10):
    if exclude_ids is None:
        exclude_ids = []

    model = st.session_state["model"]
    scaler = model["scaler"]
    lda = model["lda"]
    reducer = model["umap"]
    labels = model["labels"]
    X_umap = model["X_umap"]
    train_ids = model["train_ids"]
    enc = model["label_encoder"]
    nn_global = model["nn_global"]
    knn_genre = model["knn_genre"]
    train_cols = model["train_cols"]

    # Verificación de dimensión
    if v_query.shape[0] != len(train_cols):
        raise RuntimeError(
            f"Dimensión de features del query ({v_query.shape[0]}) no coincide con train ({len(train_cols)}). "
            "Asegúrate de que los features del train y del query se generen con las mismas funciones y orden."
        )

    df_q = pd.DataFrame([v_query], columns=train_cols)
    vq_scaled = scaler.transform(df_q.values)
    vq_lda = lda.transform(vq_scaled)
    vq_umap = reducer.transform(vq_lda)

    # Vecinos globales para estimar cluster, excluyendo IDs si toca
    dists, idxs = nn_global.kneighbors(vq_umap, n_neighbors=7, return_distance=True)
    neigh_ids = train_ids[idxs[0]]
    neigh_labels = labels[idxs[0]]

    # Filtrar excluidos
    mask_keep = ~np.isin(neigh_ids, np.array(exclude_ids, dtype=train_ids.dtype))
    neigh_labels = neigh_labels[mask_keep]

    # Mayoría sobre vecinos no excluidos
    if len(neigh_labels) == 0:
        predicted_cluster = -1
    else:
        lab_candidates = [l for l in neigh_labels if l != -1]
        if len(lab_candidates) == 0:
            predicted_cluster = int(neigh_labels[0])
        else:
            vals, cnts = np.unique(lab_candidates, return_counts=True)
            predicted_cluster = int(vals[np.argmax(cnts)])

    # Predicción de género
    genre_idx = int(knn_genre.predict(vq_umap)[0])
    genre_proba = knn_genre.predict_proba(vq_umap)[0]
    top_genre = enc.inverse_transform([genre_idx])[0]
    top_genre_prob = float(np.max(genre_proba))

    # Espacio base por cluster
    mask_cluster = labels == predicted_cluster
    use_global = False
    if predicted_cluster == -1 or mask_cluster.sum() < topk:
        use_global = True
        base_space = X_umap
        base_ids = train_ids
    else:
        base_space = X_umap[mask_cluster]
        base_ids = train_ids[mask_cluster]

    # Excluir IDs antes de buscar vecinos finales
    if len(exclude_ids) > 0:
        valid_mask = ~np.isin(base_ids, np.array(exclude_ids, dtype=base_ids.dtype))
        base_space = base_space[valid_mask]
        base_ids = base_ids[valid_mask]

    if base_space.shape[0] == 0:
        return {
            "predicted_cluster": predicted_cluster,
            "use_global": True,
            "genre_pred": top_genre,
            "genre_conf": top_genre_prob,
            "ids_similares": [],
            "distancias": []
        }

    k = min(topk, base_space.shape[0])
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(base_space)
    d, ix = nn.kneighbors(vq_umap)
    ids_similares = [int(base_ids[j]) for j in ix[0]]


    return {
        "predicted_cluster": predicted_cluster,
        "use_global": use_global,
        "genre_pred": top_genre,
        "genre_conf": top_genre_prob,
        "ids_similares": ids_similares,
        "distancias": d[0].tolist(),
        "vq_umap": vq_umap.flatten()
    }

def plot_umap_distribution(result: dict, only_cluster: bool = False):
    """
    Proyección UMAP 2D coloreada por cluster (DBSCAN) SIN mostrar esos clusters en la leyenda.
    Solo se muestran en la leyenda: Query y Vecinos recomendados.
    """
    model = st.session_state["model"]
    X_umap = model["X_umap"]         # (N, 2)
    labels = model["labels"]         # (N,)
    train_ids = model["train_ids"]   # (N,)
    id2idx = model.get("id2idx", {int(mid): i for i, mid in enumerate(train_ids)})

    # Máscara opcional: solo cluster del query si corresponde
    q_cluster = int(result.get("predicted_cluster", -1))
    if only_cluster and q_cluster != -1:
        mask = labels == q_cluster
    else:
        mask = np.ones_like(labels, dtype=bool)

    Xp = X_umap[mask]
    Lp = labels[mask]
    visible_global_idx = np.where(mask)[0]
    global_to_local = {g: i for i, g in enumerate(visible_global_idx)}

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Distribución 2D (UMAP)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    # Colores por cluster, pero sin participar en la leyenda
    cmap = plt.get_cmap("tab20")
    unique_labs = np.unique(Lp)
    for lab in unique_labs:
        lab_mask = Lp == lab
        color = "lightgray" if lab == -1 else cmap(int(lab) % cmap.N)
        ax.scatter(
            Xp[lab_mask, 0], Xp[lab_mask, 1],
            s=10, c=[color], alpha=0.35, linewidths=0,
            label="_nolegend_"  # <- etiquetas que empiezan con "_" no salen en la leyenda
        )

    # Vecinos recomendados (anillos)
    neigh_ids = result.get("ids_similares", [])
    neigh_idx_global = [id2idx[mid] for mid in neigh_ids if int(mid) in id2idx]
    neigh_local = [global_to_local[g] for g in neigh_idx_global if g in global_to_local]
    if len(neigh_local) > 0:
        ax.scatter(
            Xp[neigh_local, 0], Xp[neigh_local, 1],
            s=90, facecolors="none", edgecolors="black",
            linewidths=1.1, label="Vecinos recomendados"
        )

    # Query (estrella)
    q_umap = np.array(result.get("vq_umap", [])).ravel()
    if q_umap.size == 2:
        ax.scatter(
            [q_umap[0]], [q_umap[1]],
            s=140, marker="*", c="red", edgecolors="k", linewidths=1.0,
            label="Query"
        )

    # Solo se mostrarán "Query" y "Vecinos recomendados"
    handles, labels_txt = ax.get_legend_handles_labels()
    if len(handles) > 0:
        ax.legend(frameon=True, fontsize=9, loc="best")

    ax.grid(True, ls="--", alpha=0.2)
    st.pyplot(fig, clear_figure=True)



# ---------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------
if btn_run:
    if "model" not in st.session_state or "df_X_train" not in st.session_state:
        st.error("El modelo o las features no están listos. Revisa la sección de estado de carga.")
    else:
        try:
            exclude_ids = []

            # 1) Obtener imagen base
            if input_mode == "Subir imagen":
                if uploaded_file is None:
                    st.error("Por favor sube una imagen")
                    st.stop()

                st.subheader("🖼️ Imagen base")
                st.image(uploaded_file, width=220)

                vq, qid = extract_query_features_with_build_features(
                    uploaded_file.getbuffer().tobytes(),
                    filename=uploaded_file.name if uploaded_file.name else "uploaded.jpg"
                )
                # Si el nombre del archivo es un movieId presente en train, excluir
                try:
                    qid_int = int(str(qid))
                    model = st.session_state["model"]
                    if qid_int in set(model["train_ids"].tolist()):
                        exclude_ids = [qid_int]
                except Exception:
                    pass

            else:
                # >>>> NUEVO: tomamos la selección hecha en la galería <<<<
                selected_train_movie = st.session_state.get("selected_train_movie")
                if selected_train_movie is None:
                    st.error("Selecciona una película del Train desde la galería antes de continuar.")
                    st.stop()

                img_path = find_image_in_folder_by_id(selected_train_movie, TRAIN_IMAGE_DIR)
                st.subheader("🖼️ Imagen base")
                if img_path is not None:
                    st.image(img_path, width=220, caption=f"movieId {selected_train_movie}")
                    with open(img_path, "rb") as f:
                        vq, qid = extract_query_features_with_build_features(
                            f.read(), filename=f"{_normalize_id(selected_train_movie)}.jpg"
                        )
                else:
                    st.error("No se encontró el póster en la carpeta de entrenamiento.")
                    st.stop()

                # Siempre excluimos el propio ID cuando se elige de train
                exclude_ids = [selected_train_movie]

            # 2) Recomendación
            st.info("Calculando proyección, asignando cluster y encontrando vecinos...")
            result = recommend_from_feature_vector(vq, exclude_ids=exclude_ids, topk=10)
            st.success("Búsqueda completada")

            # 3) Resumen
            colA, colB = st.columns(2)
            with colA:
                st.metric("Cluster asignado (DBSCAN)", str(result["predicted_cluster"]))
            with colB:
                st.metric("Género predicho", result["genre_pred"])

            if result["use_global"]:
                st.warning("Cluster muy pequeño o ruido. Se usó búsqueda global en todo el train.")
            if exclude_ids:
                st.info(f"Se excluyó el ID {exclude_ids[0]} de los resultados para evitar auto-match.")

            # 4) Mostrar recomendaciones
            st.subheader("🎯 Recomendaciones visualmente similares (KNN)")
            ids_sim = result["ids_similares"]
            dists = result["distancias"]
            if len(ids_sim) == 0:
                st.write("No se encontraron vecinos tras excluir el ID del query.")
            else:
                n_cols = 5
                rows = [ids_sim[i:i+n_cols] for i in range(0, len(ids_sim), n_cols)]
                rows_d = [dists[i:i+n_cols] for i in range(0, len(dists), n_cols)]
                for r_ids, r_ds in zip(rows, rows_d):
                    cols = st.columns(n_cols, gap="small")
                    for c, mid, dd in zip(cols, r_ids, r_ds):
                        with c:
                            p = find_image_in_folder_by_id(mid, TRAIN_IMAGE_DIR)
                            if p:
                                st.image(p, use_column_width=True)
                            else:
                                st.info("Imagen no encontrada")
                            st.caption(f"movieId {mid} • dist {dd:.3f}")
            # 5) Visualización 2D posterior a la predicción
            st.subheader("📈 Distribución 2D según características visuales")
            only_cluster = st.checkbox("Mostrar solo el cluster del query", value=False,
                                       help="Si el query fue clasificado como ruido (-1), se mostrará todo el set.")
            plot_umap_distribution(result, only_cluster=only_cluster)


        except Exception as e:
            st.error(f"Ocurrió un error: {e}")


# ---------------------------------------------------------------------
# Notas
# ---------------------------------------------------------------------
with st.expander("ℹ️ Ayuda y notas", expanded=False):
    st.markdown("""
- Pipeline: StandardScaler → LDA → UMAP → DBSCAN propio → kNN sobre UMAP.
""")
