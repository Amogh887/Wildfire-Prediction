"""Open-Meteo weather service (no API key). Fetches current + daily forecast for a
coarse res-3 grid; finer hexes inherit their res-3 parent's weather."""
import logging
import time
from datetime import datetime, timezone

import h3
import requests

from services.hex_service import grid, RES_WEATHER

logger = logging.getLogger("weather_service")

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
BATCH_SIZE = 100         # Open-Meteo accepts comma-separated coords per request
MAX_WEATHER_CELLS = 800  # safety cap on startup

# default values if a fetch fails — deliberately neutral (cool, humid, wet) so that
# cells with missing weather don't generate false fire alerts.
_DEFAULT = {
    "temperature": 20.0, "humidity": 60.0, "wind_speed": 10.0, "precipitation": 0.0,
    "temperature_max": 22.0, "humidity_min": 55.0, "wind_speed_max": 12.0,
    "precipitation_sum": 30.0,
}


class WeatherCache:
    def __init__(self):
        # res-3 cell -> weather dict
        self.by_cell: dict[str, dict] = {}
        self.updated_at: datetime | None = None

    def _fetch_batch(self, cells: list[str]) -> dict[str, dict]:
        lats, lons = [], []
        for c in cells:
            lat, lon = grid.centroid(c)
            lats.append(round(lat, 4))
            lons.append(round(lon, 4))
        params = {
            "latitude": ",".join(str(x) for x in lats),
            "longitude": ",".join(str(x) for x in lons),
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
            "daily": "temperature_2m_max,relative_humidity_2m_min,wind_speed_10m_max,precipitation_sum",
            "timezone": "auto",
            "past_days": 30,
            "forecast_days": 1,
        }
        r = requests.get(FORECAST_URL, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        # single point returns dict, multiple returns list
        if isinstance(data, dict):
            data = [data]
        out = {}
        for cell, entry in zip(cells, data):
            cur = entry.get("current", {}) or {}
            daily = entry.get("daily", {}) or {}

            out[cell] = {
                "temperature": _num(cur.get("temperature_2m"), _DEFAULT["temperature"]),
                "humidity": _num(cur.get("relative_humidity_2m"), _DEFAULT["humidity"]),
                "wind_speed": _num(cur.get("wind_speed_10m"), _DEFAULT["wind_speed"]),
                "precipitation": _num(cur.get("precipitation"), _DEFAULT["precipitation"]),
                "temperature_max": _num(_last_val(daily.get("temperature_2m_max"), None), _DEFAULT["temperature_max"]),
                "humidity_min": _num(_last_val(daily.get("relative_humidity_2m_min"), None), _DEFAULT["humidity_min"]),
                "wind_speed_max": _num(_last_val(daily.get("wind_speed_10m_max"), None), _DEFAULT["wind_speed_max"]),
                "precipitation_sum": _num(_sum_vals(daily.get("precipitation_sum", [])[:30], None), _DEFAULT["precipitation_sum"]),
            }
        return out

    def refresh(self):
        cells = grid.weather_cells()
        if len(cells) > MAX_WEATHER_CELLS:
            logger.warning("weather cells %d > cap %d; sampling", len(cells), MAX_WEATHER_CELLS)
            step = len(cells) // MAX_WEATHER_CELLS + 1
            cells = cells[::step]
        result: dict[str, dict] = {}
        for i in range(0, len(cells), BATCH_SIZE):
            batch = cells[i:i + BATCH_SIZE]
            try:
                result.update(self._fetch_batch(batch))
            except Exception as e:
                logger.warning("weather batch %d failed: %s", i, e)
                for c in batch:
                    result[c] = dict(_DEFAULT)
            time.sleep(0.3)  # be polite to the free API
        if result:
            self.by_cell = result
            self.updated_at = datetime.now(timezone.utc)
            logger.info("Weather refreshed for %d res-%d cells", len(result), RES_WEATHER)
        else:
            logger.warning("Weather refresh produced no data")

    def for_cell(self, cell: str) -> dict:
        """Look up weather for any-resolution cell via its res-3 parent."""
        res = h3.get_resolution(cell)
        wcell = cell if res == RES_WEATHER else h3.cell_to_parent(cell, RES_WEATHER)
        return self.by_cell.get(wcell, dict(_DEFAULT))


def _last_val(lst, default):
    """Return the last non-None element of a list, or default if none exists."""
    if isinstance(lst, list):
        for v in reversed(lst):
            if v is not None:
                return v
    return default


def _sum_vals(lst, default):
    """Sum all non-None elements of a list; return default if the list is empty or all None."""
    if isinstance(lst, list):
        values = [v for v in lst if v is not None]
        if values:
            return sum(values)
    return default


def _num(v, default):
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


cache = WeatherCache()
