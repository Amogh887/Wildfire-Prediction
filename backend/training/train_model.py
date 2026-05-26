"""Train an XGBoost wildfire-probability model.

Data path 1 (preferred): Kaggle dataset
  carlosparadis/fires-from-space-australia-and-new-zeland via kagglehub.
Data path 2 (fallback): synthetic but realistic fire/non-fire samples across Australia,
  weighted toward known fire-prone regions.

Features (must match ml_service.FEATURE_COLUMNS):
  temperature_max, relative_humidity_min, wind_speed_max, precipitation_30d,
  ndvi_proxy, fire_freq_historical, month, drought_index
"""
import logging
import os
import pickle

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("train")

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_BASE, "data", "wildfire_model.pkl")

FEATURE_COLUMNS = [
    "temperature_max", "relative_humidity_min", "wind_speed_max", "precipitation_30d",
    "ndvi_proxy", "fire_freq_historical", "month", "drought_index",
]

# Known fire-prone region centers (lat, lon) with relative weight.
FIRE_PRONE = [
    (-33.8, 150.9, 3.0),   # Greater Sydney / Blue Mountains NSW
    (-37.8, 145.5, 3.0),   # Victorian highlands / Gippsland
    (-35.3, 149.1, 2.0),   # ACT region
    (-31.9, 152.0, 2.0),   # Mid-North-Coast NSW
    (-34.9, 138.6, 2.0),   # Adelaide Hills SA
    (-27.5, 153.0, 1.5),   # SE Queensland
    (-31.9, 115.9, 2.0),   # Perth Hills WA
    (-42.0, 147.0, 1.5),   # Tasmania
    (-25.0, 133.0, 0.5),   # arid interior (low)
]


def _try_kaggle() -> pd.DataFrame | None:
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
        slug = "carlosparadis/fires-from-space-australia-and-new-zeland"
        path = kagglehub.dataset_download(slug)
        logger.info("Kaggle dataset downloaded to %s", path)
        csvs = []
        for root, _, files in os.walk(path):
            for f in files:
                if f.lower().endswith(".csv"):
                    csvs.append(os.path.join(root, f))
        logger.info("Found CSV files: %s", [os.path.basename(c) for c in csvs])
        # prefer the MODIS/VIIRS Australia file
        target = None
        for c in csvs:
            if "australia" in c.lower():
                target = c
                break
        target = target or (csvs[0] if csvs else None)
        if not target:
            return None
        df = pd.read_csv(target)
        logger.info("Loaded Kaggle file %s rows=%d cols=%s",
                    os.path.basename(target), len(df), list(df.columns)[:10])
        return df
    except Exception as e:
        logger.warning("Kaggle path failed (%s); will use synthetic data", e)
        return None


def _build_from_kaggle(df: pd.DataFrame) -> pd.DataFrame | None:
    """Aggregate fire points to H3 res-6 -> positives; sample negatives.
    Then synthesize plausible weather features correlated with label."""
    import h3
    latcol = next((c for c in df.columns if c.lower() in ("latitude", "lat")), None)
    loncol = next((c for c in df.columns if c.lower() in ("longitude", "lon", "long")), None)
    if not latcol or not loncol:
        logger.warning("Kaggle df missing lat/lon columns; falling back to synthetic")
        return None
    df = df[[latcol, loncol]].dropna()
    # keep Australia bbox
    df = df[(df[latcol].between(-44, -10)) & (df[loncol].between(112, 154))]
    if len(df) < 100:
        return None
    pos_cells = set()
    for lat, lon in zip(df[latcol], df[loncol]):
        try:
            pos_cells.add(h3.latlng_to_cell(float(lat), float(lon), 6))
        except Exception:
            continue
    pos_cells = list(pos_cells)
    logger.info("Kaggle positives: %d unique res-6 cells", len(pos_cells))
    # negatives: random res-6 cells over Australia not in positives
    rng = np.random.default_rng(42)
    neg = []
    while len(neg) < len(pos_cells):
        lat = rng.uniform(-44, -10)
        lon = rng.uniform(112, 154)
        c = h3.latlng_to_cell(lat, lon, 6)
        if c not in pos_cells:
            neg.append(c)
    rows = []
    for c in pos_cells:
        lat, lon = h3.cell_to_latlng(c)
        rows.append(_synth_features(lat, lon, label=1, rng=rng))
    for c in neg:
        lat, lon = h3.cell_to_latlng(c)
        rows.append(_synth_features(lat, lon, label=0, rng=rng))
    return pd.DataFrame(rows)


def _region_fire_weight(lat: float, lon: float) -> float:
    w = 0.2
    for flat, flon, fw in FIRE_PRONE:
        d2 = (lat - flat) ** 2 + (lon - flon) ** 2
        w += fw * np.exp(-d2 / 8.0)
    return w


def _synth_features(lat: float, lon: float, label: int, rng) -> dict:
    """Generate weather/static features. For positives skew toward hot/dry/windy."""
    base_hot = label == 1
    temp_max = rng.normal(34 if base_hot else 24, 5)
    hum_min = rng.normal(18 if base_hot else 45, 10)
    wind_max = rng.normal(35 if base_hot else 18, 12)
    precip_30d = abs(rng.normal(8 if base_hot else 60, 20))
    ndvi = np.clip(rng.normal(0.35 if base_hot else 0.5, 0.15), 0.05, 0.9)
    fire_freq = _region_fire_weight(lat, lon) * (1.4 if base_hot else 0.8)
    month = int(rng.integers(1, 13))
    # southern hemisphere fire season Nov-Mar -> boost
    if base_hot:
        month = int(rng.choice([12, 1, 2, 11, 3]))
    temp_max = float(np.clip(temp_max, -5, 50))
    hum_min = float(np.clip(hum_min, 2, 100))
    wind_max = float(np.clip(wind_max, 0, 120))
    drought = temp_max / (precip_30d + 1.0)
    return {
        "temperature_max": temp_max,
        "relative_humidity_min": hum_min,
        "wind_speed_max": wind_max,
        "precipitation_30d": float(precip_30d),
        "ndvi_proxy": float(ndvi),
        "fire_freq_historical": float(fire_freq),
        "month": month,
        "drought_index": float(drought),
        "label": label,
    }


def _build_synthetic(n_per_class: int = 4000) -> pd.DataFrame:
    import h3
    rng = np.random.default_rng(7)
    rows = []
    # positives: sample near fire-prone regions
    weights = np.array([w for _, _, w in FIRE_PRONE])
    weights = weights / weights.sum()
    n_pos = 0
    while n_pos < n_per_class:
        idx = rng.choice(len(FIRE_PRONE), p=weights)
        flat, flon, _ = FIRE_PRONE[idx]
        lat = flat + rng.normal(0, 2.0)
        lon = flon + rng.normal(0, 2.5)
        if not (-44 <= lat <= -10 and 112 <= lon <= 154):
            continue
        rows.append(_synth_features(lat, lon, 1, rng))
        n_pos += 1
    # negatives: uniform over Australia
    n_neg = 0
    while n_neg < n_per_class:
        lat = rng.uniform(-44, -10)
        lon = rng.uniform(112, 154)
        rows.append(_synth_features(lat, lon, 0, rng))
        n_neg += 1
    logger.info("Synthetic dataset: %d positives + %d negatives", n_per_class, n_per_class)
    return pd.DataFrame(rows)


def main():
    source = "synthetic"
    df = None
    kdf = _try_kaggle()
    if kdf is not None:
        built = _build_from_kaggle(kdf)
        if built is not None and len(built) > 200:
            df = built
            source = "kaggle"
    if df is None:
        df = _build_synthetic()

    logger.info("DATA SOURCE USED: %s  (rows=%d)", source.upper(), len(df))

    X = df[FEATURE_COLUMNS]
    y = df["label"].astype(int)

    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    pos = int((y_tr == 1).sum())
    neg = int((y_tr == 0).sum())
    spw = neg / max(pos, 1)

    model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=spw, eval_metric="logloss",
        n_jobs=4, random_state=1,
    )
    model.fit(X_tr, y_tr)

    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
        logger.info("Test AUC: %.3f", auc)
    except Exception as e:
        logger.warning("AUC calc failed: %s", e)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "features": FEATURE_COLUMNS, "source": source}, f)
    logger.info("Saved model to %s (source=%s)", MODEL_PATH, source)


if __name__ == "__main__":
    main()
