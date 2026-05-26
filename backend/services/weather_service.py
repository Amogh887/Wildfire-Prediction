"""Weather service using WeatherAPI.com. One call per res-2 weather cell (~89 cells).
Finer hexes inherit their res-2 parent's weather."""
import logging
import os
import time
from datetime import datetime, timezone

import h3
import requests
from requests.adapters import HTTPAdapter

from services.hex_service import grid, RES_WEATHER

logger = logging.getLogger("weather_service")

WEATHERAPI_URL = "https://api.weatherapi.com/v1/forecast.json"
WEATHERAPI_KEY = os.environ.get("WEATHERAPI_KEY", "")

_session = requests.Session()
_session.headers.update({"User-Agent": "wildfire-prediction-app/1.0 contact:aarao22@wisc.edu"})
_adapter = HTTPAdapter(max_retries=0, pool_connections=8, pool_maxsize=16)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)

_DEFAULT = {
    "temperature": 20.0, "humidity": 60.0, "wind_speed": 10.0, "precipitation": 0.0,
    "temperature_max": 22.0, "humidity_min": 55.0, "wind_speed_max": 12.0,
    "precipitation_sum": 30.0,
}


def _precip_30d_estimate(lat: float, month: int) -> float:
    if lat > -20:
        return 80.0 if (month >= 11 or month <= 4) else 8.0
    if lat > -30:
        return 25.0
    if lat > -35:
        return 45.0 if 5 <= month <= 9 else 20.0
    return 50.0


class WeatherCache:
    def __init__(self):
        self.by_cell: dict[str, dict] = {}
        self.updated_at: datetime | None = None
        self.ok = False

    def _fetch_one(self, cell: str, month: int) -> dict | None:
        if not WEATHERAPI_KEY:
            return None
        lat, lon = grid.centroid(cell)
        try:
            r = _session.get(WEATHERAPI_URL, params={
                "key": WEATHERAPI_KEY,
                "q": f"{round(lat, 4)},{round(lon, 4)}",
                "days": 1,
                "aqi": "no",
                "alerts": "no",
            }, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.warning("weather fetch failed for %s: %s", cell, e)
            return None
        cur = data.get("current") or {}
        day = ((data.get("forecast") or {}).get("forecastday") or [{}])[0].get("day") or {}
        humidity = _num(cur.get("humidity"), _DEFAULT["humidity"])
        avg_hum = _num(day.get("avghumidity"), humidity)
        return {
            "temperature": _num(cur.get("temp_c"), _DEFAULT["temperature"]),
            "humidity": humidity,
            "wind_speed": _num(cur.get("wind_kph"), _DEFAULT["wind_speed"]),
            "precipitation": _num(cur.get("precip_mm"), _DEFAULT["precipitation"]),
            "temperature_max": _num(day.get("maxtemp_c"), _DEFAULT["temperature_max"]),
            "humidity_min": round(avg_hum * 0.7, 1),
            "wind_speed_max": _num(day.get("maxwind_kph"), _DEFAULT["wind_speed_max"]),
            "precipitation_sum": _precip_30d_estimate(lat, month),
        }

    def refresh(self):
        month = datetime.now(timezone.utc).month
        cells = grid.weather_cells()
        result: dict[str, dict] = {}
        n_ok = n_fail = 0
        for cell in cells:
            w = self._fetch_one(cell, month)
            if w is not None:
                result[cell] = w
                n_ok += 1
            else:
                result[cell] = self.by_cell.get(cell, dict(_DEFAULT))
                n_fail += 1
            time.sleep(0.2)
        if n_ok > 0:
            self.by_cell = result
            self.updated_at = datetime.now(timezone.utc)
            self.ok = True
            logger.info("Weather refreshed: %d/%d cells ok (res-%d)",
                        n_ok, n_ok + n_fail, RES_WEATHER)
        else:
            logger.warning("Weather refresh: all %d cells failed; keeping previous cache "
                           "(%d cells, ok=%s)", n_fail, len(self.by_cell), self.ok)

    def for_cell(self, cell: str) -> dict:
        res = h3.get_resolution(cell)
        wcell = cell if res == RES_WEATHER else h3.cell_to_parent(cell, RES_WEATHER)
        return self.by_cell.get(wcell, dict(_DEFAULT))


def _num(v, default):
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


cache = WeatherCache()
