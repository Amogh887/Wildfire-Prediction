"""Weather service. Uses WeatherAPI.com when key is available; falls back to Australian
climate normals (lat/month lookup) so geographic variation is always present."""
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

# Australian climate normals by latitude band, indexed [band][month-1].
# Bands: 0=tropical (lat>-20), 1=subtropical (-20 to -30),
#        2=temperate (-30 to -37), 3=southern (lat<-37)
_TEMP_MAX = [
    [33,33,33,32,29,27,27,28,31,33,33,33],
    [29,29,28,25,22,19,18,20,23,26,28,29],
    [27,27,25,21,17,14,13,15,18,21,24,26],
    [22,22,20,17,13,11,10,11,13,16,18,20],
]
_HUM_MIN = [
    [35,38,45,38,22,15,13,15,20,28,32,35],
    [55,57,55,48,42,38,35,35,38,43,48,52],
    [38,40,43,50,56,62,63,58,48,42,38,36],
    [48,50,53,60,65,70,71,66,58,52,49,47],
]
_WIND_MAX = [
    [20,20,19,18,21,24,25,23,22,21,20,20],
    [28,26,26,26,27,29,29,29,29,29,29,28],
    [33,31,30,29,28,28,29,31,32,33,34,34],
    [36,35,34,33,33,34,35,37,37,38,37,37],
]
_PRECIP_30D = [
    [310,280,220,70,12,2,1,2,12,45,115,210],
    [155,145,125,75,65,55,45,45,45,85,95,125],
    [70,75,90,95,100,110,80,65,60,65,70,70],
    [48,43,57,67,73,78,78,68,63,63,53,48],
]


def _lat_band(lat: float) -> int:
    if lat > -20: return 0
    if lat > -30: return 1
    if lat > -37: return 2
    return 3


def _climate_normal(lat: float, lon: float, month: int) -> dict:
    m = month - 1
    b = _lat_band(lat)
    temp_max = float(_TEMP_MAX[b][m])
    hum_min = float(_HUM_MIN[b][m])
    wind_max = float(_WIND_MAX[b][m])
    p30 = float(_PRECIP_30D[b][m])
    # Interior of Australia (Outback) is hotter and drier than coast at same latitude
    if -33 < lat < -18 and 125 < lon < 142:
        interior = min(1.0, min(lon - 115, 145 - lon) / 15.0)
        temp_max += interior * 7
        hum_min = max(5.0, hum_min - interior * 15)
        p30 = max(1.0, p30 * (1 - interior * 0.6))
    return {
        "temperature": round(temp_max - 4.0, 1),
        "humidity": round(min(100.0, hum_min / 0.65), 1),
        "wind_speed": round(wind_max * 0.7, 1),
        "precipitation": round(p30 / 30.0, 2),
        "temperature_max": round(temp_max, 1),
        "humidity_min": round(hum_min, 1),
        "wind_speed_max": round(wind_max, 1),
        "precipitation_sum": round(p30, 1),
    }


class WeatherCache:
    def __init__(self):
        self.by_cell: dict[str, dict] = {}
        self.updated_at: datetime | None = None
        self.ok = False
        self.live = False  # True when data comes from real API calls

    def _fetch_live(self, cell: str, month: int) -> dict | None:
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
            logger.warning("WeatherAPI fetch failed for %s: %s", cell, e)
            return None
        cur = data.get("current") or {}
        day = ((data.get("forecast") or {}).get("forecastday") or [{}])[0].get("day") or {}
        humidity = _num(cur.get("humidity"), 60.0)
        avg_hum = _num(day.get("avghumidity"), humidity)
        lat, lon = grid.centroid(cell)
        return {
            "temperature": _num(cur.get("temp_c"), 20.0),
            "humidity": humidity,
            "wind_speed": _num(cur.get("wind_kph"), 15.0),
            "precipitation": _num(cur.get("precip_mm"), 0.0),
            "temperature_max": _num(day.get("maxtemp_c"), 22.0),
            "humidity_min": round(avg_hum * 0.7, 1),
            "wind_speed_max": _num(day.get("maxwind_kph"), 20.0),
            "precipitation_sum": _climate_normal(lat, lon, month)["precipitation_sum"],
        }

    def refresh(self):
        month = datetime.now(timezone.utc).month
        cells = grid.weather_cells()
        result: dict[str, dict] = {}
        n_live = n_climate = 0
        for cell in cells:
            lat, lon = grid.centroid(cell)
            w = None
            if WEATHERAPI_KEY:
                w = self._fetch_live(cell, month)
                time.sleep(0.2)
            if w is not None:
                result[cell] = w
                n_live += 1
            else:
                result[cell] = _climate_normal(lat, lon, month)
                n_climate += 1
        self.by_cell = result
        self.updated_at = datetime.now(timezone.utc)
        self.ok = True
        self.live = n_live > 0
        if n_live > 0:
            logger.info("Weather: %d live + %d climate normals (%d res-%d cells)",
                        n_live, n_climate, len(result), RES_WEATHER)
        else:
            logger.info("Weather: %d cells from climate normals (no live API)", n_climate)

    def for_cell(self, cell: str) -> dict:
        res = h3.get_resolution(cell)
        wcell = cell if res == RES_WEATHER else h3.cell_to_parent(cell, RES_WEATHER)
        if wcell not in self.by_cell:
            lat, lon = grid.centroid(wcell)
            month = datetime.now(timezone.utc).month
            return _climate_normal(lat, lon, month)
        return self.by_cell[wcell]


def _num(v, default):
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


cache = WeatherCache()
