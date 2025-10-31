import pickle
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from pathlib import Path
from typing import Dict, Any

EPS = 1e-8
IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp",".jfif",".JPG",".PNG",".JPEG"}


def preprocess(img_bgr, size=(256,256)):
    img = cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

# ---------- COLOR ----------
def hsv_24(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H,S,V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    def h1(ch, bins, rmax):
        h,_ = np.histogram(ch.ravel(), bins=bins, range=(0,rmax))
        h = h.astype(np.float32); h /= (h.sum()+EPS)
        return h
    return np.concatenate([
        h1(H,8,180),
        h1(S,8,256),
        h1(V,8,256)
    ]).astype(np.float32)  # 24

def hsv_24_names():
    names = []
    for c in ["H","S","V"]:
        for i in range(8):
            names.append(f"hsv_{c}_bin{i}")
    return names  # 24 nombres

# ---------- STATS ----------
def hsv_stats(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H,S,V = hsv[:,:,0].astype(np.float32), hsv[:,:,1].astype(np.float32), hsv[:,:,2].astype(np.float32)
    def skew(x):
        m = x.mean(); s = x.std() + EPS
        return (((x-m)/s)**3).mean()
    p_dark = (V < 40).mean()
    p_sat  = (S > 180).mean()
    Vf = cv2.Sobel(V, cv2.CV_32F, 1, 0, ksize=3)*2 + cv2.Sobel(V, cv2.CV_32F, 0, 1, ksize=3)*2
    v_contrast = Vf.mean()
    return np.array([
        H.mean(), H.std(), skew(H),
        S.mean(), S.std(),
        V.mean(), V.std(),
        p_dark, p_sat,
        v_contrast
    ], dtype=np.float32)  # 10

def hsv_stats_names():
    return [
        "h_mean","h_std","h_skew",
        "s_mean","s_std",
        "v_mean","v_std",
        "p_dark","p_sat",
        "v_contrast"
    ]

# ---------- LBP ----------
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

    # mapping uniforme
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
    return hist  # 10

def lbp_names(radius=1, P=8):
    # P+2 bins
    return [f"lbp_r{radius}_bin{i}" for i in range(P+2)]

# ---------- GLCM LIGHT ----------
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
    return np.array(feats, dtype=np.float32)  # 12

def glcm_light_names():
    names = []
    for d in (1,2):
        for th_name in ("0","90"):
            names.append(f"glcm_d{d}_a{th_name}_contrast")
            names.append(f"glcm_d{d}_a{th_name}_homogeneity")
            names.append(f"glcm_d{d}_a{th_name}_energy")
    return names  # 12

# ---------- HOG ----------
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
    return names  # 12

# ---------- HU ----------
def hu_feats(gray):
    m = cv2.moments(gray)
    hu = cv2.HuMoments(m).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu)+EPS)
    return hu.astype(np.float32)  # 7

def extract_features(img_bgr: np.ndarray, size=(256, 256)):
    # Using the traditional feature extraction from the provided code
    img, gray = preprocess(img_bgr, size=size)

    f_color = hsv_24(img)  # 24
    f_stats = hsv_stats(img)  # 10
    f_lbp1 = lbp_uniform(gray, radius=1)  # 10
    f_lbp2 = lbp_uniform(gray, radius=2)  # 10
    f_glcm = glcm_light(gray)  # 12
    f_hog = hog_super_compacto(gray)  # 12
    f_hu = hu_feats(gray)  # 7

    feats = np.concatenate([
        f_color,
        f_stats,
        f_lbp1,
        f_lbp2,
        f_glcm,
        f_hog,
        f_hu
    ]).astype(np.float32)

    return feats.reshape(1, -1)


@st.cache_resource
def load_model(model_path: str) -> Dict[str, Any]:
    """
    Load the model from a file path or uploaded file object.
    
    Args:
        model_path: Either a string path to local file or StreamlitUploadedFile object
    
    Returns:
        Dict containing model components
    """
    try:
        # Handle both string paths and uploaded file objects
        if isinstance(model_path, str):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            # For StreamlitUploadedFile
            model = pickle.load(model_path)

        # Validate model components
        required_keys = ['scaler', 'pca', 'umap', 'cluster_labels', 'n_clusters']
        if not all(key in model for key in required_keys):
            raise ValueError("Missing required model components")

        return model

    except Exception as e:
        raise ValueError(f"Error loading model: {str(e)}")


def predict_cluster(model: Dict[str, Any], features: np.ndarray) -> int:
    # Process through pipeline
    scaled = model['scaler'].transform(features)
    # Apply PCA
    pca_transformed = model['pca'].transform(scaled)
    # Apply UMAP
    umap_transformed = model['umap'].transform(pca_transformed)
    # Get cluster prediction using the clustering model
    cluster = model['kmeans'].predict(umap_transformed)[0]
    return int(cluster)


st.title('Movie Poster Cluster Predictor')

uploaded_file = st.file_uploader("Choose a movie poster...", type=['jpg', 'png', 'jpeg'])
model_file = st.file_uploader("Upload model file...", type=['pkl'])
if model_file is None:
    # Load default model if no file is uploaded
    try:
        model_file = open('best_clustering_model.pkl', 'rb')
    except FileNotFoundError:
        st.error("Default model 'best_clustering_model.pkl' not found. Please upload a model file.")

if uploaded_file and model_file:
    # Load and process image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Movie Poster', width=300)

    # Load model
    model = load_model(model_file)

    # Process image and extract traditional features
    features = extract_features(image)

    # Predict cluster
    cluster = predict_cluster(model, features)

    st.write(f"Predicted Cluster: {cluster}")
