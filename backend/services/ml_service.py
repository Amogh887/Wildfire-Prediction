"""ML inference service. Loads the XGBoost model and computes per-hex wildfire
probability from live weather + static historical fire frequency + month, then applies
live modifiers (extreme weather, active fires). Falls back to a physics-based score if
the model file is missing."""
import logging
import os
import pickle
from datetime import datetime, timezone

import h3
import numpy as np

from services.hex_service import grid, RES_COARSE
from services import weather_service, firms_service

logger = logging.getLogger("ml_service")

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_BASE, "data", "wildfire_model.pkl")

FEATURE_COLUMNS = [
    "temperature_max", "relative_humidity_min", "wind_speed_max", "precipitation_30d",
    "ndvi_proxy", "fire_freq_historical", "month", "drought_index",
]

# Fire-prone region centers reused for static historical fire frequency.
FIRE_PRONE = [
    (-33.8, 150.9, 3.0), (-37.8, 145.5, 3.0), (-35.3, 149.1, 2.0),
    (-31.9, 152.0, 2.0), (-34.9, 138.6, 2.0), (-27.5, 153.0, 1.5),
    (-31.9, 115.9, 2.0), (-42.0, 147.0, 1.5), (-25.0, 133.0, 0.5),
]


def fire_freq_historical(lat: float, lon: float) -> float:
    w = 0.2
    for flat, flon, fw in FIRE_PRONE:
        d2 = (lat - flat) ** 2 + (lon - flon) ** 2
        w += fw * np.exp(-d2 / 8.0)
    return float(w)


def ndvi_proxy(lat: float, lon: float, precip_30d: float) -> float:
    """Crude greenness proxy: wetter + coastal -> greener."""
    base = 0.3 + min(precip_30d, 100) / 250.0
    return float(np.clip(base, 0.05, 0.9))


def risk_band(p: float) -> str:
    if p < 0.25:
        return "LOW"
    if p < 0.5:
        return "MODERATE"
    if p < 0.75:
        return "HIGH"
    return "EXTREME"


class MLService:
    def __init__(self):
        self.model = None
        self.features = FEATURE_COLUMNS
        self.loaded = False
        # results keyed by res-4 cell
        self.results: dict[str, dict] = {}
        self.updated_at: datetime | None = None

    def load(self):
        try:
            with open(MODEL_PATH, "rb") as f:
                bundle = pickle.load(f)
            self.model = bundle["model"]
            self.features = bundle.get("features", FEATURE_COLUMNS)
            self.loaded = True
            logger.info("Loaded ML model (source=%s)", bundle.get("source"))
        except Exception as e:
            logger.warning("Model load failed (%s); using physics fallback", e)
            self.model = None
            self.loaded = False

    def _feature_row(self, cell: str, month: int) -> tuple[dict, dict]:
        lat, lon = grid.centroid(cell)
        w = weather_service.cache.for_cell(cell)
        precip_30d = w["precipitation_sum"]  # daily sum proxy for 30d
        feats = {
            "temperature_max": w["temperature_max"],
            "relative_humidity_min": w["humidity_min"],
            "wind_speed_max": w["wind_speed_max"],
            "precipitation_30d": precip_30d,
            "ndvi_proxy": ndvi_proxy(lat, lon, precip_30d),
            "fire_freq_historical": fire_freq_historical(lat, lon),
            "month": month,
            "drought_index": w["temperature_max"] / (precip_30d + 1.0),
        }
        return feats, w

    @staticmethod
    def _physics_score(w: dict) -> float:
        t = np.clip((w["temperature_max"] - 10) / 35, 0, 1)
        dry = np.clip((80 - w["humidity_min"]) / 80, 0, 1)
        wind = np.clip(w["wind_speed_max"] / 60, 0, 1)
        nodry = np.clip(1 - w["precipitation_sum"] / 100, 0, 1)
        return float(np.clip(0.35 * t + 0.3 * dry + 0.2 * wind + 0.15 * nodry, 0, 1))

    def run_inference(self):
        month = datetime.now(timezone.utc).month
        cells = grid.cells_coarse
        rows = []
        meta = []
        for c in cells:
            feats, w = self._feature_row(c, month)
            rows.append([feats[k] for k in self.features])
            meta.append((c, feats, w))

        physics_probs = [self._physics_score(w) for _, _, w in meta]
        if self.model is not None and rows:
            try:
                import pandas as pd
                X = pd.DataFrame(rows, columns=self.features)
                ml_probs = self.model.predict_proba(X)[:, 1]
                # take the higher of ML and physics so current weather always drives
                # a visible geographic gradient even when the seasonal ML signal is weak
                base_probs = [max(float(ml), phys) for ml, phys in zip(ml_probs, physics_probs)]
            except Exception as e:
                logger.warning("predict failed (%s); physics fallback", e)
                base_probs = physics_probs
        else:
            base_probs = physics_probs

        results = {}
        for (c, feats, w), base in zip(meta, base_probs):
            prob = float(base)
            fires = firms_service.cache.fires_for_cell(c)
            # live modifiers
            if w["temperature_max"] > 38 and w["humidity_min"] < 15 and w["wind_speed_max"] > 50:
                prob *= 1.8
            if fires >= 2:
                prob = max(prob, 0.9)
            elif fires == 1:
                prob = max(prob, 0.7)
            prob = float(np.clip(prob, 0.0, 1.0))
            lat, lon = grid.centroid(c)
            results[c] = {
                "probability": round(prob, 4),
                "risk": risk_band(prob),
                "temperature": round(w["temperature"], 1),
                "humidity": round(w["humidity"], 0),
                "wind_speed": round(w["wind_speed"], 1),
                "precipitation": round(w["precipitation"], 2),
                "active_fires": int(fires),
                "lat": round(lat, 4),
                "lon": round(lon, 4),
            }
        self.results = results
        self.updated_at = datetime.now(timezone.utc)
        logger.info("Inference complete for %d res-%d cells", len(results), RES_COARSE)

    def hex_payload(self, cell: str) -> dict:
        """Build the API payload for any-resolution cell (fine cells inherit res-4 parent)."""
        res = h3.get_resolution(cell)
        parent = cell if res == RES_COARSE else h3.cell_to_parent(cell, RES_COARSE)
        data = self.results.get(parent)
        if data is None:
            # edge cell whose res-4 parent fell outside the coarse fill:
            # inherit from the nearest available res-4 neighbor.
            for k in (1, 2):
                for nb in h3.grid_disk(parent, k):
                    if nb in self.results:
                        data = self.results[nb]
                        break
                if data is not None:
                    break
        lat, lon = grid.centroid(cell)
        if data is None:
            return {
                "h3": cell, "lat": round(lat, 4), "lon": round(lon, 4),
                "probability": 0.0, "risk": "LOW", "temperature": None,
                "humidity": None, "wind_speed": None, "precipitation": None,
                "active_fires": 0,
            }
        return {
            "h3": cell, "lat": round(lat, 4), "lon": round(lon, 4),
            "probability": data["probability"], "risk": data["risk"],
            "temperature": data["temperature"], "humidity": data["humidity"],
            "wind_speed": data["wind_speed"], "precipitation": data["precipitation"],
            "active_fires": data["active_fires"],
        }


service = MLService()
