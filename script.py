# run_batch.py
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Optional
import hashlib

import numpy as np
import pandas as pd

from extract_vector import extract_vector  # Twoja funkcja

# ====== konfiguracja ======
IMAGES_DIR = Path("./images/zdjecia_kubus")
EXTS = {".jpg", ".jpeg", ".png"}  # dodaj inne jesli trzeba
OUT_DIR = Path("./out_cs")
CACHE_DIR = Path("./cache_embeddings")

# Modele wspierane przez DeepFace
# MODEL_NAMES = [  "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace",  "DeepID", "ArcFace", "Dlib", "SFace",  ]
MODEL_NAMES = ["ArcFace"]
# Detektory wspierane przez DeepFace (2025) – wybierz kilka praktycznych
DETECTOR_BACKENDS = [ "opencv", "ssd", "dlib", "mtcnn", "retinaface", "mediapipe", "yolov8", "yunet", "fastmtcnn", ]
ENFORCE_DETECTION = True  # ustaw False, gdy dużo trudnych zdjęć

# Dobór normalizacji zgodny z DeepFace (gdy None – użyje domyślnej)
NORMALIZATION_MAP = {
    "ArcFace":    "ArcFace",
    "Facenet":    "Facenet",
    "Facenet512": "Facenet",
    "VGG-Face":   "VGGFace",
    "OpenFace":   "base",
    "DeepFace":   "base",
    "DeepID":     "base",
    "Dlib":       "base",
    "SFace":      "base",   # bywa OK; jeśli są artefakty, ustaw None
}

# ====== przygotowanie listy obrazów i katalogów ======
IMAGES = sorted(
    [p for p in IMAGES_DIR.iterdir() if p.is_file() and p.suffix.lower() in EXTS],
    key=lambda p: p.name
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ====== helpery ======
def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v if n < eps else (v / n)

def _cache_key(img_path: Path, model: str, det: str, norm: Optional[str], enforce: bool) -> str:
    raw = f"{model}|{det}|{norm}|{enforce}|{img_path.name}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]

def embed_cached(
    img_path: Path,
    model_name: str,
    detector_backend: str,
    enforce_detection: bool,
    normalization: Optional[str]
) -> np.ndarray:
    key = _cache_key(img_path, model_name, detector_backend, normalization, enforce_detection)
    npy = CACHE_DIR / f"{key}.npy"
    if npy.exists():
        return np.load(npy)
    vec = extract_vector(
        img_path=str(img_path),
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        normalization=normalization
    )
    np.save(npy, vec)
    return vec

def compute_pairwise_rows(embeddings: Dict[str, np.ndarray]) -> List[Dict[str, object]]:
    """Zwraca listę wierszy: {'img1','img2','cos'} dla wszystkich par."""
    rows: List[Dict[str, object]] = []
    names = sorted(embeddings.keys())
    for a, b in combinations(names, 2):
        va = l2_normalize(embeddings[a])
        vb = l2_normalize(embeddings[b])
        cos = float(np.dot(va, vb))  # == cosine similarity po L2
        rows.append({"img1": a, "img2": b, "cos": cos})
    return rows

def summarize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Zwraca DataFrame ze statystykami min/max/median/mean oraz liczbą par."""
    if df.empty:
        return pd.DataFrame([{"detector": "ALL", "n_pairs": 0, "min": np.nan, "max": np.nan, "median": np.nan, "mean": np.nan}])

    # per-detector
    g = (
        df.groupby("detector")["cos"]
        .agg(n_pairs="count", min="min", max="max", median="median", mean="mean")
        .reset_index()
    )

    # global ALL
    all_stats = pd.DataFrame([{
        "detector": "ALL",
        "n_pairs": int(df["cos"].count()),
        "min": float(df["cos"].min()),
        "max": float(df["cos"].max()),
        "median": float(df["cos"].median()),
        "mean": float(df["cos"].mean()),
    }])

    return pd.concat([g, all_stats], ignore_index=True)

# ====== main ======
def main():
    if len(IMAGES) < 2:
        print("Potrzeba co najmniej 2 zdjec w katalogu IMAGES_DIR.")
        return

    for model in MODEL_NAMES:
        model_dir = OUT_DIR / model
        model_dir.mkdir(parents=True, exist_ok=True)
        normalization = NORMALIZATION_MAP.get(model, None)

        all_rows_for_model: List[pd.DataFrame] = []

        for det in DETECTOR_BACKENDS:
            print(f"[INFO] Model={model} | Detector={det} → embed...")
            # 1) Embed wszystkie obrazy dla (model, det)
            embeddings: Dict[str, np.ndarray] = {}
            for img in IMAGES:
                try:
                    embeddings[img.name] = embed_cached(
                        img_path=img,
                        model_name=model,
                        detector_backend=det,
                        enforce_detection=ENFORCE_DETECTION,
                        normalization=normalization
                    )
                except Exception as e:
                    print(f"[WARN] {model}/{det} {img.name}: {e}")

            # 2) Licz pary
            rows = compute_pairwise_rows(embeddings)
            df = pd.DataFrame(rows, columns=["img1", "img2", "cos"])

            # 3) Zapis detektor.csv
            det_csv = model_dir / f"{det}.csv"
            df.to_csv(det_csv, index=False)
            print(f"[OK] Zapisano {det_csv.name} (pairs={len(df)})")

            # 4) Dodaj kolumnę 'detector' i zbierz do wspólnego DF
            if not df.empty:
                df2 = df.copy()
                df2.insert(0, "detector", det)
                all_rows_for_model.append(df2)

        # 5) summary.csv (per model)
        if all_rows_for_model:
            big = pd.concat(all_rows_for_model, ignore_index=True)
        else:
            big = pd.DataFrame(columns=["detector", "img1", "img2", "cos"])

        summary = summarize_df(big)
        # formatowanie liczb do 6 miejsc po przecinku dla czytelności
        for col in ["min", "max", "median", "mean"]:
            if col in summary.columns:
                summary[col] = summary[col].astype(float).round(6)

        summary_csv = model_dir / "summary.csv"
        summary.to_csv(summary_csv, index=False)
        print(f"[OK] {model}: summary → {summary_csv}")

if __name__ == "__main__":
    main()
