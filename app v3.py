# app.py
import os
import io
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import cv2
from PIL import Image

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import umap

# ===================== CONFIG =====================
st.set_page_config(page_title="Recomendador por Pósters", layout="wide")
st.title("🎬 Recomendación de Películas por Similitud Visual")
st.caption("Sube un póster o elige uno del set de entrenamiento. Se extraen *features* visuales, "
           "se proyecta con LDA+UMAP, se asigna cluster (DBSCAN) y se buscan los 10 más cercanos con kNN.")

# ===================== UTILIDADES ORIGINALES (TUS FUNCIONES) =====================
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
    # Guardados tipo "test"
    np.save(out_dir / "Y_hsv.npy", X)
    np.save(out_dir / "image_ids_test.npy", ids)
    with open(out_dir / "features_meta_test.json", "w") as f:
        json.dump({"num_images": int(len(ids)), "dim": int(X.shape[1]), "headers": headers}, f, indent=2)
    return X, ids, headers

# ===================== CARGA DE DATA =====================
@st.cache_data(show_spinner=False)
def load_labels_and_bins():
    try:
        generos = pd.read_csv('MovieGenre.csv', encoding='latin-1')
        df_test = pd.read_csv('movies_test.csv')
        df_train = pd.read_csv('movies_train.csv')
        ids = pd.read_csv('links.csv')

        df_test_label = (
            df_test.merge(ids, on='movieId', how='left').merge(generos, on='imdbId', how='left')
        )[["movieId", "title", "Genre", "IMDB Score"]].drop_duplicates()
        df_train_label = (
            df_train.merge(ids, on='movieId', how='left').merge(generos, on='imdbId', how='left')
        )[["movieId", "title", "Genre", "IMDB Score"]].drop_duplicates()

        df_test_label["genre_p"] = df_test_label["Genre"].str.split("|").str[0]
        df_train_label["genre_p"] = df_train_label["Genre"].str.split("|").str[0]

        genre_dummies_train = df_train_label["Genre"].str.get_dummies(sep="|")
        genre_dummies_test = df_test_label["Genre"].str.get_dummies(sep="|")

        df_train_bin = pd.concat(
            [df_train_label[["movieId", "title", "IMDB Score", "genre_p"]], genre_dummies_train], axis=1
        )
        df_test_bin = pd.concat(
            [df_test_label[["movieId", "title", "IMDB Score", "genre_p"]], genre_dummies_test], axis=1
        )
        return df_train_label, df_test_label, df_train_bin, df_test_bin
    except Exception as e:
        raise RuntimeError(f"Error cargando CSVs: {e}")

@st.cache_data(show_spinner=False)
def load_train_features():
    try:
        X_hsv = np.load("X_hsv.npy", mmap_mode="r")
        X_hsv_ids = np.load("image_ids_train.npy", allow_pickle=True)
        feature_names = [f"feature_{i+1}" for i in range(X_hsv.shape[1])]
        df_X_train = pd.DataFrame(X_hsv, columns=feature_names)
        df_X_train.insert(0, "movieId", X_hsv_ids)
        df_X_train["movieId"] = df_X_train["movieId"].astype("int64")
        return df_X_train
    except Exception as e:
        raise RuntimeError(f"Error cargando features de entrenamiento (.npy): {e}")

def find_image(movie_id, folder):
    for ext in [".jpg", ".jpeg", ".png", ".webp", ".JPG", ".PNG"]:
        p = os.path.join(folder, f"{movie_id}{ext}")
        if os.path.exists(p):
            return p
    return None

# ===================== ENTRENAR PROYECCIÓN Y CLUSTER =====================
@st.cache_resource(show_spinner=False)
def train_projection_and_cluster(df_X_train, df_train_bin):
    try:
        # join para obtener género principal por movieId
        df = df_X_train.merge(df_train_bin[["movieId", "genre_p"]], on="movieId", how="left")
        df = df.dropna(subset=["genre_p"])
        X = df.drop(columns=["movieId", "genre_p"]).values
        # encode label
        enc = LabelEncoder()
        y = enc.fit_transform(df["genre_p"].astype(str))

        # scaler
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        # LDA
        lda = LinearDiscriminantAnalysis(n_components=2)
        X_lda = lda.fit_transform(Xs, y)

        # UMAP
        reducer = umap.UMAP(
            n_neighbors=30,
            min_dist=0.1,
            metric="cosine",
            random_state=42
        )
        X_umap = reducer.fit_transform(X_lda)

        # DBSCAN
        db = DBSCAN(eps=0.8, min_samples=15, metric="euclidean").fit(X_umap)
        labels = db.labels_

        # Vecino global para estimar cluster del query
        nn_global = NearestNeighbors(n_neighbors=5, metric="euclidean")
        nn_global.fit(X_umap)

        # kNN de género sobre el espacio UMAP
        knn_genre = KNeighborsClassifier(n_neighbors=7, metric="euclidean")
        knn_genre.fit(X_umap, y)

        # guardar ids en el mismo orden
        train_ids = df["movieId"].values

        return {
            "scaler": scaler,
            "lda": lda,
            "umap": reducer,
            "dbscan": db,
            "labels": labels,
            "X_umap": X_umap,
            "train_ids": train_ids,
            "label_encoder": enc,
            "nn_global": nn_global,
            "knn_genre": knn_genre,
            "train_cols": df_X_train.drop(columns=["movieId"], errors="ignore").columns
        }
    except Exception as e:
        raise RuntimeError(f"Error entrenando proyección y clustering: {e}")

# ===================== SIDEBAR: INPUT =====================
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
    # Cargar labels para listar títulos
    try:
        df_train_label, df_test_label, df_train_bin, df_test_bin = load_labels_and_bins()
        st.sidebar.success("Metadatos cargados correctamente")
        # opciones: "movieId - title"
        options = df_train_label.dropna(subset=["movieId","title"]).copy()
        options["opt"] = options["movieId"].astype(int).astype(str) + " - " + options["title"].astype(str)
        selected_opt = st.sidebar.selectbox("Selecciona una película del Train", options["opt"].tolist())
        selected_train_movie = int(selected_opt.split(" - ")[0]) if selected_opt else None
    except Exception as e:
        st.sidebar.error(str(e))

btn_run = st.sidebar.button("🔎 Recomendar", use_container_width=True)

# ===================== MAIN: CARGA BASES Y MODELO =====================
with st.expander("📦 Estado de carga de datos", expanded=True):
    # CSVs
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

    # Features de train
    try:
        if "df_X_train" not in st.session_state:
            st.session_state["df_X_train"] = load_train_features()
        st.success("Features de entrenamiento (.npy) cargados")
    except Exception as e:
        st.error(str(e))

    # Modelo de proyección y clustering
    try:
        if "model" not in st.session_state:
            st.session_state["model"] = train_projection_and_cluster(
                st.session_state["df_X_train"],
                st.session_state["df_train_bin"]
            )
        st.success("Modelo LDA + UMAP + DBSCAN + kNN entrenado")
    except Exception as e:
        st.error(str(e))

# ===================== FUNCIÓN: EXTRAER FEATURES DEL INPUT CON build_features =====================
def extract_query_features_with_build_features(file_bytes: bytes, filename: str):
    """
    Escribe la imagen a un directorio temporal y llama a build_features(in_dir, out_dir).
    Retorna (vec_features: np.ndarray shape (d,), query_id: str).
    """
    tmpdir = tempfile.mkdtemp(prefix="query_")
    outdir = tempfile.mkdtemp(prefix="query_feats_")
    try:
        # guardar imagen como único archivo en tmpdir
        img_path = os.path.join(tmpdir, filename)
        with open(img_path, "wb") as f:
            f.write(file_bytes)

        X, ids, _ = build_features(tmpdir, outdir, size=(256,256))
        # Tomamos el primero
        v = X[0].astype(np.float32)
        qid = str(ids[0])
        return v, qid
    finally:
        # limpia temporales
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
            shutil.rmtree(outdir, ignore_errors=True)
        except Exception:
            pass

# ===================== FUNCIÓN: PIPELINE DE RECOMENDACIÓN =====================
def recommend_from_feature_vector(v_query: np.ndarray, query_label_hint: str = None, topk: int = 10):
    """
    v_query: vector de features crudos (dimensionalidad igual al train)
    Retorna dict con cluster, género_predicho, ids_similares, distancias, etc.
    """
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

    # Alinear columnas con el train
    train_cols = model["train_cols"]
    # v_query ya es un vector en el mismo orden de features (porque usamos build_features igual que en train)
    # Llevamos a dataframe para poder estandarizar con las mismas columnas
    df_q = pd.DataFrame([v_query], columns=train_cols)

    # Escalado -> LDA -> UMAP
    vq_scaled = scaler.transform(df_q.values)
    vq_lda = lda.transform(vq_scaled)
    vq_umap = reducer.transform(vq_lda)

    # Estimar cluster: vecino global y mayoría de sus 5 vecinos
    dists, idxs = nn_global.kneighbors(vq_umap, n_neighbors=5, return_distance=True)
    neigh_labels = labels[idxs[0]]
    # mayoría ignorando -1 si es posible
    lab_candidates = [l for l in neigh_labels if l != -1]
    if len(lab_candidates) == 0:
        predicted_cluster = int(neigh_labels[0])
    else:
        # conteo simple
        vals, cnts = np.unique(lab_candidates, return_counts=True)
        predicted_cluster = int(vals[np.argmax(cnts)])

    # kNN de género en el espacio UMAP
    genre_idx = int(knn_genre.predict(vq_umap)[0])
    genre_proba = knn_genre.predict_proba(vq_umap)[0]
    top_genre = enc.inverse_transform([genre_idx])[0]
    top_genre_prob = float(np.max(genre_proba))

    # Buscar top-k similares dentro del cluster predicho
    mask_cluster = labels == predicted_cluster
    # si cluster es ruido o cluster muy pequeño, fallback global
    use_global = False
    if predicted_cluster == -1 or mask_cluster.sum() < topk:
        use_global = True
        base_space = X_umap
        base_ids = train_ids
    else:
        base_space = X_umap[mask_cluster]
        base_ids = train_ids[mask_cluster]

    nn = NearestNeighbors(n_neighbors=topk, metric="euclidean")
    nn.fit(base_space)
    d, ix = nn.kneighbors(vq_umap)
    ids_similares = [int(base_ids[j]) for j in ix[0]]

    return {
        "predicted_cluster": predicted_cluster,
        "use_global": use_global,
        "genre_pred": top_genre,
        "genre_conf": top_genre_prob,
        "ids_similares": ids_similares,
        "distancias": d[0].tolist()
    }

# ===================== EJECUCIÓN =====================
if btn_run:
    # Validar que el modelo y data existen
    if "model" not in st.session_state or "df_X_train" not in st.session_state:
        st.error("El modelo o las features no están listos. Revisa la sección de estado de carga.")
    else:
        try:
            # 1) Obtener imagen base y mostrarla
            if input_mode == "Subir imagen":
                if uploaded_file is None:
                    st.error("Por favor sube una imagen")
                    st.stop()
                # Vista previa
                st.subheader("🖼️ Imagen base")
                st.image(uploaded_file, width=220)
                st.info("Extrayendo características con `build_features`...")
                # Extraer features con build_features
                vq, qid = extract_query_features_with_build_features(
                    uploaded_file.getbuffer().tobytes(),
                    filename=uploaded_file.name if uploaded_file.name else "uploaded.jpg"
                )
                st.success("Características extraídas correctamente")

            else:
                if selected_train_movie is None:
                    st.error("Selecciona una película del Train")
                    st.stop()
                # Cargar imagen desde carpeta Train_image
                img_path = find_image(selected_train_movie, "Train_image")
                if img_path is None:
                    st.error("No se encontró el póster en la carpeta Train_image")
                    st.stop()
                st.subheader("🖼️ Imagen base")
                st.image(img_path, width=220, caption=f"movieId {selected_train_movie}")
                st.info("Extrayendo características con `build_features` sobre la imagen seleccionada...")
                with open(img_path, "rb") as f:
                    vq, qid = extract_query_features_with_build_features(f.read(), filename=f"{selected_train_movie}.jpg")
                st.success("Características extraídas correctamente")

            # 2) Recomendación
            st.info("Calculando proyección, asignando cluster y encontrando vecinos...")
            result = recommend_from_feature_vector(vq, topk=10)
            st.success("Búsqueda completada")

            # 3) Resumen de predicción
            colA, colB, colC = st.columns(3)
            with colA:
                st.metric("Cluster asignado (DBSCAN)", str(result["predicted_cluster"]))
            with colB:
                st.metric("Género predicho (kNN)", result["genre_pred"])
            with colC:
                st.metric("Confianza género", f"{result['genre_conf']*100:.1f}%")
            if result["use_global"]:
                st.warning("Cluster muy pequeño o ruido. Se usó búsqueda global en todo el train.")

            # 4) Mostrar 10 pósters más parecidos
            st.subheader("🎯 Recomendaciones visualmente similares")
            ids_sim = result["ids_similares"]
            dists = result["distancias"]

            # grid 5x2
            n_cols = 5
            rows = [ids_sim[i:i+n_cols] for i in range(0, len(ids_sim), n_cols)]
            rows_d = [dists[i:i+n_cols] for i in range(0, len(dists), n_cols)]

            for r_ids, r_ds in zip(rows, rows_d):
                cols = st.columns(n_cols, gap="small")
                for c, mid, dd in zip(cols, r_ids, r_ds):
                    with c:
                        p = find_image(mid, "Train_image")
                        if p:
                            st.image(p, use_column_width=True)
                        else:
                            st.info("Imagen no encontrada")
                        st.caption(f"movieId {mid} • dist {dd:.3f}")

        except Exception as e:
            st.error(f"Ocurrió un error: {e}")

# ===================== NOTAS DE ESTADO =====================
with st.expander("ℹ️ Ayuda y notas", expanded=False):
    st.markdown("""
- **Entrada**: puedes subir un archivo o seleccionar una película del conjunto de entrenamiento.
- **Extracción de *features***: se realiza **siempre** con tu función `build_features` escribiendo la imagen a un directorio temporal.
- **Proyección**: `StandardScaler` → `LDA` (supervisado con género) → `UMAP`.
- **Clustering**: `DBSCAN` sobre el espacio UMAP. El cluster del query se estima por mayoría de sus 5 vecinos globales.
- **kNN de género**: clasificador `KNeighborsClassifier` sobre UMAP para predecir el género principal.
- **Vecinos recomendados**: se buscan 10 más cercanos **dentro del cluster asignado**; si el cluster es ruido o hay pocos, se hace *fallback* a búsqueda global.
- **Mensajes**: se notifica cada paso de carga, extracción y recomendación, y se muestran errores cuando corresponde.
""")
