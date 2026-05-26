"""Australian state fire agency incident feed service. No API key needed —
these are public government feeds. Fetches NSW RFS and VIC Emergency incident
GeoJSON, maps incidents to H3 res-4 cells, and exposes a fires_for_cell() lookup
compatible with the FirmsCache interface in firms_service.py."""
import logging
from datetime import datetime, timezone

import h3
import requests

from services.hex_service import RES_COARSE

logger = logging.getLogger("aus_fires_service")

NSW_RFS_URL = "https://www.rfs.nsw.gov.au/feeds/majorIncidents.json"
VIC_EMERGENCY_URL = "https://emergency.vic.gov.au/public/osom-geojson.json"


class AusFiresCache:
    def __init__(self):
        # res-4 cell -> incident count
        self.fires_by_cell: dict[str, int] = {}
        self.updated_at: datetime | None = None

    def _fetch_nsw(self) -> list[tuple[float, float]]:
        """Fetch NSW RFS major incidents and return (lat, lon) pairs for fire incidents."""
        r = requests.get(NSW_RFS_URL, timeout=30)
        r.raise_for_status()
        data = r.json()
        points = []
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            category = props.get("category", "") or ""
            if "fire" not in category.lower():
                continue
            geom = feature.get("geometry") or {}
            if geom.get("type") != "Point":
                continue
            coords = geom.get("coordinates")
            if not coords or len(coords) < 2:
                continue
            lon, lat = float(coords[0]), float(coords[1])
            points.append((lat, lon))
        return points

    def _fetch_vic(self) -> list[tuple[float, float]]:
        """Fetch VIC Emergency incidents and return (lat, lon) pairs for fire incidents."""
        r = requests.get(VIC_EMERGENCY_URL, timeout=30)
        r.raise_for_status()
        data = r.json()
        points = []
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            category1 = props.get("category1", "") or ""
            if category1.lower() != "fire":
                continue
            geom = feature.get("geometry") or {}
            if geom.get("type") != "Point":
                continue
            coords = geom.get("coordinates")
            if not coords or len(coords) < 2:
                continue
            lon, lat = float(coords[0]), float(coords[1])
            points.append((lat, lon))
        return points

    def refresh(self):
        counts: dict[str, int] = {}
        total_points = 0

        for feed_name, fetch_fn in [("NSW RFS", self._fetch_nsw), ("VIC Emergency", self._fetch_vic)]:
            try:
                points = fetch_fn()
                for lat, lon in points:
                    try:
                        cell = h3.latlng_to_cell(lat, lon, RES_COARSE)
                    except Exception:
                        continue
                    counts[cell] = counts.get(cell, 0) + 1
                total_points += len(points)
                logger.info("%s: %d fire incidents fetched", feed_name, len(points))
            except Exception as e:
                logger.warning("%s feed failed: %s", feed_name, e)

        self.fires_by_cell = counts
        self.updated_at = datetime.now(timezone.utc)
        logger.info("AUS fires: %d total incidents -> %d affected cells", total_points, len(counts))

    def fires_for_cell(self, cell: str) -> int:
        """Active fires for any-resolution cell, mapped to its res-4 parent."""
        res = h3.get_resolution(cell)
        ccell = cell if res == RES_COARSE else h3.cell_to_parent(cell, RES_COARSE)
        return self.fires_by_cell.get(ccell, 0)


cache = AusFiresCache()
