"""Train an XGBoost wildfire-probability model.

Data path 1 (preferred): Kaggle dataset + ERA5 real weather
  carlosparadis/fires-from-space-australia-and-new-zeland via kagglehub,
  with actual meteorological data from the Open-Meteo ERA5 archive.
Data path 2 (fallback): Kaggle dataset with synthetic weather features.
Data path 3 (fallback): synthetic but realistic fire/non-fire samples across Australia,
  weighted toward known fire-prone regions.

Features (must match ml_service.FEATURE_COLUMNS):
  temperature_max, relative_humidity_min, wind_speed_max, precipitation_30d,
  ndvi_proxy, fire_freq_historical, month, drought_index
"""
import logging
import os
import pickle
import time
from collections import defaultdict
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests

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


def _build_from_kaggle_real_weather(df: pd.DataFrame) -> pd.DataFrame | None:
    """Build training data using Kaggle fire locations + real ERA5 weather.

    Returns a DataFrame with FEATURE_COLUMNS + 'label', or None on failure.
    """
    import h3

    # --- 1. Extract lat / lon / date columns ---
    latcol = next((c for c in df.columns if c.lower() in ("latitude", "lat")), None)
    loncol = next((c for c in df.columns if c.lower() in ("longitude", "lon", "long")), None)
    datecol = next((c for c in df.columns if c.lower() in ("acq_date", "date")), None)
    if not latcol or not loncol:
        logger.warning("ERA5 path: missing lat/lon columns")
        return None

    sub = df[[latcol, loncol] + ([datecol] if datecol else [])].copy()
    sub = sub.dropna(subset=[latcol, loncol])
    # Keep Australia bounding box
    sub = sub[(sub[latcol].between(-44, -10)) & (sub[loncol].between(112, 154))]
    if len(sub) < 100:
        logger.warning("ERA5 path: too few rows after bbox filter (%d)", len(sub))
        return None

    # --- 2. Map each fire point to H3 res-4 cell; track most-common fire month ---
    cell_months: dict[str, list[int]] = defaultdict(list)
    cell_coords: dict[str, tuple[float, float]] = {}

    for row in sub.itertuples(index=False):
        lat = float(getattr(row, latcol))
        lon = float(getattr(row, loncol))
        try:
            cell = h3.latlng_to_cell(lat, lon, 4)
        except Exception:
            continue
        cell_coords[cell] = (lat, lon)
        if datecol:
            raw_date = getattr(row, datecol)
            try:
                m = pd.to_datetime(raw_date).month
                cell_months[cell].append(m)
            except Exception:
                pass

    pos_cells = list(cell_coords.keys())
    logger.info("ERA5 path: %d unique res-4 positive cells", len(pos_cells))

    # --- 3. Cap at 500 positive cells ---
    if len(pos_cells) > 500:
        rng_seed = np.random.default_rng(42)
        pos_cells = list(rng_seed.choice(pos_cells, size=500, replace=False))

    # Determine most-common fire month per cell
    def _most_common_month(cell: str) -> int:
        months = cell_months.get(cell, [])
        if not months:
            return 1  # default January if no date info
        counts: dict[int, int] = defaultdict(int)
        for m in months:
            counts[m] += 1
        return max(counts, key=lambda k: counts[k])

    cell_fire_month = {c: _most_common_month(c) for c in pos_cells}

    # --- 4. Group by fire month; determine ERA5 window ---
    today = date.today()
    era5_cutoff = today - timedelta(days=7)

    def _era5_window(month: int, rep_year: int) -> tuple[str, str]:
        end_dt = date(rep_year, month, 28)
        start_dt = end_dt - timedelta(days=29)
        end_dt = min(end_dt, era5_cutoff)
        start_dt = min(start_dt, era5_cutoff)
        return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")

    # Group positive cells by fire month
    month_to_cells: dict[int, list[str]] = defaultdict(list)
    for cell in pos_cells:
        month_to_cells[cell_fire_month[cell]].append(cell)

    # --- 5 & 6. Fetch ERA5 in batches of 50; extract features ---
    ERA5_URL = "https://archive-api.open-meteo.com/v1/era5"
    BATCH_SIZE = 50

    def _fetch_era5_batch(
        lats: list[float], lons: list[float], start: str, end: str
    ) -> list[dict | None]:
        params = {
            "latitude": ",".join(f"{v:.4f}" for v in lats),
            "longitude": ",".join(f"{v:.4f}" for v in lons),
            "daily": "temperature_2m_max,relative_humidity_2m_min,wind_speed_10m_max,precipitation_sum",
            "start_date": start,
            "end_date": end,
            "timezone": "UTC",
        }
        try:
            resp = requests.get(ERA5_URL, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            # Single coord → dict; multiple → list
            if isinstance(data, dict):
                data = [data]
            return data
        except Exception as exc:
            logger.warning("ERA5 fetch failed (%s); skipping batch", exc)
            return [None] * len(lats)

    def _extract_weather(entry: dict) -> dict | None:
        if entry is None:
            return None
        daily = entry.get("daily", {})
        t_vals = [v for v in (daily.get("temperature_2m_max") or []) if v is not None]
        h_vals = [v for v in (daily.get("relative_humidity_2m_min") or []) if v is not None]
        w_vals = [v for v in (daily.get("wind_speed_10m_max") or []) if v is not None]
        p_vals = [v for v in (daily.get("precipitation_sum") or []) if v is not None]
        # Fall back to reasonable defaults if all values are missing
        temp_max = max(t_vals) if t_vals else 30.0
        hum_min = min(h_vals) if h_vals else 20.0
        wind_max = max(w_vals) if w_vals else 25.0
        precip_30d = sum(p_vals) if p_vals else 5.0
        return {
            "temperature_max": float(temp_max),
            "relative_humidity_min": float(hum_min),
            "wind_speed_max": float(wind_max),
            "precipitation_30d": float(precip_30d),
        }

    # Fetch weather for all positive cells
    cell_weather: dict[str, dict] = {}

    for month, cells_in_month in month_to_cells.items():
        rep_year = 2020 if month <= 9 else 2019
        start_str, end_str = _era5_window(month, rep_year)
        logger.info(
            "ERA5 positive: month=%d rep_year=%d window=%s..%s cells=%d",
            month, rep_year, start_str, end_str, len(cells_in_month),
        )
        for i in range(0, len(cells_in_month), BATCH_SIZE):
            batch_cells = cells_in_month[i: i + BATCH_SIZE]
            lats = [h3.cell_to_latlng(c)[0] for c in batch_cells]
            lons = [h3.cell_to_latlng(c)[1] for c in batch_cells]
            results = _fetch_era5_batch(lats, lons, start_str, end_str)
            for cell, entry in zip(batch_cells, results):
                w = _extract_weather(entry)
                if w is not None:
                    cell_weather[cell] = w
            time.sleep(0.5)

    # --- 7. Build positive rows ---
    rows = []
    for cell in pos_cells:
        w = cell_weather.get(cell)
        if w is None:
            continue  # batch failed; skip
        lat, lon = h3.cell_to_latlng(cell)
        precip = w["precipitation_30d"]
        ndvi_proxy = float(np.clip(0.3 + min(precip, 100) / 250.0, 0.05, 0.9))
        fire_freq = _region_fire_weight(lat, lon)
        drought_index = w["temperature_max"] / (precip + 1.0)
        month = cell_fire_month[cell]
        rows.append({
            "temperature_max": w["temperature_max"],
            "relative_humidity_min": w["relative_humidity_min"],
            "wind_speed_max": w["wind_speed_max"],
            "precipitation_30d": precip,
            "ndvi_proxy": ndvi_proxy,
            "fire_freq_historical": float(fire_freq),
            "month": month,
            "drought_index": float(drought_index),
            "label": 1,
        })

    n_pos = len(rows)
    logger.info("ERA5 path: %d positive rows built", n_pos)
    if n_pos < 50:
        logger.warning("ERA5 path: too few positives (%d < 50); giving up", n_pos)
        return None

    # --- 8. Build negative rows ---
    pos_cell_set = set(pos_cells)
    rng = np.random.default_rng(99)
    neg_cells: list[str] = []
    while len(neg_cells) < n_pos:
        lat = rng.uniform(-44, -10)
        lon = rng.uniform(112, 154)
        try:
            c = h3.latlng_to_cell(lat, lon, 4)
        except Exception:
            continue
        if c not in pos_cell_set and c not in {nc for nc in neg_cells}:
            neg_cells.append(c)

    winter_months = [5, 6, 7, 8]  # Australian winter — low fire risk
    neg_month_groups: dict[int, list[str]] = defaultdict(list)
    for idx, c in enumerate(neg_cells):
        neg_month_groups[winter_months[idx % len(winter_months)]].append(c)

    neg_cell_weather: dict[str, dict] = {}
    for month, cells_in_month in neg_month_groups.items():
        start_str, end_str = _era5_window(month, 2020)
        logger.info(
            "ERA5 negative: month=%d window=%s..%s cells=%d",
            month, start_str, end_str, len(cells_in_month),
        )
        for i in range(0, len(cells_in_month), BATCH_SIZE):
            batch_cells = cells_in_month[i: i + BATCH_SIZE]
            lats = [h3.cell_to_latlng(c)[0] for c in batch_cells]
            lons = [h3.cell_to_latlng(c)[1] for c in batch_cells]
            results = _fetch_era5_batch(lats, lons, start_str, end_str)
            for cell, entry in zip(batch_cells, results):
                w = _extract_weather(entry)
                if w is not None:
                    neg_cell_weather[cell] = w
            time.sleep(0.5)

    for cell in neg_cells:
        w = neg_cell_weather.get(cell)
        if w is None:
            continue
        lat, lon = h3.cell_to_latlng(cell)
        precip = w["precipitation_30d"]
        ndvi_proxy = float(np.clip(0.3 + min(precip, 100) / 250.0, 0.05, 0.9))
        fire_freq = _region_fire_weight(lat, lon)
        drought_index = w["temperature_max"] / (precip + 1.0)
        # Assign the month used for ERA5 fetch
        assigned_month = next(
            (m for m, cs in neg_month_groups.items() if cell in cs), 6
        )
        rows.append({
            "temperature_max": w["temperature_max"],
            "relative_humidity_min": w["relative_humidity_min"],
            "wind_speed_max": w["wind_speed_max"],
            "precipitation_30d": precip,
            "ndvi_proxy": ndvi_proxy,
            "fire_freq_historical": float(fire_freq),
            "month": assigned_month,
            "drought_index": float(drought_index),
            "label": 0,
        })

    result = pd.DataFrame(rows)
    logger.info(
        "ERA5 path complete: %d total rows (%d pos, %d neg)",
        len(result),
        int((result["label"] == 1).sum()),
        int((result["label"] == 0).sum()),
    )
    # --- 9. Return None if fewer than 50 positive rows ---
    if int((result["label"] == 1).sum()) < 50:
        return None
    return result


def main():
    source = "synthetic"
    df = None
    kdf = _try_kaggle()
    if kdf is not None:
        # Try ERA5 real-weather path first
        era5_df = _build_from_kaggle_real_weather(kdf)
        if era5_df is not None and len(era5_df) >= 100:
            df = era5_df
            source = "kaggle+era5"
            logger.info("Using Kaggle + ERA5 real weather data (%d rows)", len(df))
        else:
            logger.info("ERA5 path unavailable; falling back to synthetic Kaggle features")
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
