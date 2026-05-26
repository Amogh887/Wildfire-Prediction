"""NASA FIRMS real-time fire hotspot service. Optional MAP key via NASA_FIRMS_MAP_KEY env.
Without a key it returns empty hotspots (app still works)."""
import io
import logging
import os
from datetime import datetime, timezone

import h3
import requests

from services.hex_service import grid, RES_COARSE

logger = logging.getLogger("firms_service")

# Australia bbox: west,south,east,north
BBOX = "112,-44,154,-10"
SENSOR = "VIIRS_SNPP_NRT"
DAYS = 1


class FirmsCache:
    def __init__(self):
        # res-4 cell -> active fire count (incl. neighbor spillover)
        self.fires_by_cell: dict[str, int] = {}
        self.updated_at: datetime | None = None
        self.enabled = bool(os.environ.get("NASA_FIRMS_MAP_KEY"))

    def _seed_demo(self) -> dict[str, int]:
        """Optional demo active-fires (FIRMS_DEMO=1) so the flame layer is visible
        without a NASA key. A handful of realistic fire-prone coordinates."""
        demo_points = [
            (-33.7, 150.3), (-37.4, 148.2), (-31.5, 152.1),
            (-34.6, 138.9), (-27.9, 152.6), (-31.8, 116.3),
            (-42.3, 146.7), (-23.5, 133.9), (-19.5, 146.2),
        ]
        counts: dict[str, int] = {}
        for lat, lon in demo_points:
            try:
                cell = h3.latlng_to_cell(lat, lon, RES_COARSE)
            except Exception:
                continue
            counts[cell] = counts.get(cell, 0) + 1
        return counts

    def refresh(self):
        key = os.environ.get("NASA_FIRMS_MAP_KEY")
        if not key:
            if os.environ.get("FIRMS_DEMO") == "1":
                self.fires_by_cell = self._seed_demo()
                self.updated_at = datetime.now(timezone.utc)
                self.enabled = True
                logger.warning("FIRMS_DEMO=1: seeded %d demo active-fire cells (not real data)",
                               len(self.fires_by_cell))
                return
            logger.warning("NASA_FIRMS_MAP_KEY not set; FIRMS disabled (0 hotspots)")
            self.fires_by_cell = {}
            self.updated_at = datetime.now(timezone.utc)
            self.enabled = False
            return
        self.enabled = True
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{SENSOR}/{BBOX}/{DAYS}"
        try:
            import pandas as pd
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            counts: dict[str, int] = {}
            if "latitude" in df.columns and "longitude" in df.columns:
                for lat, lon in zip(df["latitude"], df["longitude"]):
                    try:
                        cell = h3.latlng_to_cell(float(lat), float(lon), RES_COARSE)
                    except Exception:
                        continue
                    counts[cell] = counts.get(cell, 0) + 1
            self.fires_by_cell = counts
            self.updated_at = datetime.now(timezone.utc)
            logger.info("FIRMS: %d hotspot rows -> %d affected cells", len(df), len(counts))
        except Exception as e:
            logger.warning("FIRMS refresh failed: %s", e)
            self.fires_by_cell = {}
            self.updated_at = datetime.now(timezone.utc)

    def fires_for_cell(self, cell: str) -> int:
        """Active fires for any-resolution cell, mapped to its res-4 parent."""
        res = h3.get_resolution(cell)
        ccell = cell if res == RES_COARSE else h3.cell_to_parent(cell, RES_COARSE)
        return self.fires_by_cell.get(ccell, 0)


cache = FirmsCache()
