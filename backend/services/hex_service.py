"""H3 grid service: fill Australia boundary with hex cells and manage resolution mapping."""
import json
import logging
import os

import h3

logger = logging.getLogger("hex_service")

# Resolution constants (tunable)
RES_COARSE = 3   # ML inference + live-weather granularity (~620 cells, ~12k km²/hex)
RES_FINE = 4     # zoomed display resolution (~1770 km²/hex, ~4500 cells)
RES_WEATHER = 2  # weather fetched at this resolution; finer hexes map to res-2 parent

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOUNDARY_PATH = os.path.join(_BASE, "data", "australia_boundary.geojson")

# Coarse fallback polygon: mainland Australia + Tasmania (lon, lat)
_FALLBACK_POLY = {
    "type": "Polygon",
    "coordinates": [[
        [113.0, -22.0], [114.0, -26.0], [115.0, -34.0], [118.0, -35.0],
        [123.0, -34.0], [129.0, -32.0], [134.0, -33.0], [138.0, -35.0],
        [140.0, -38.0], [144.0, -38.5], [147.0, -38.0], [150.0, -37.5],
        [151.5, -33.0], [153.5, -28.0], [153.0, -25.0], [146.0, -19.0],
        [142.0, -11.0], [136.0, -12.0], [130.0, -11.0], [126.0, -14.0],
        [122.0, -18.0], [113.0, -22.0]
    ]]
}
# Tasmania
_FALLBACK_TAS = {
    "type": "Polygon",
    "coordinates": [[
        [144.7, -40.7], [148.3, -40.8], [148.3, -43.6], [145.5, -43.6], [144.7, -40.7]
    ]]
}


class HexGrid:
    def __init__(self):
        self.cells_coarse: list[str] = []
        self.cells_fine: list[str] = []
        self.centroids: dict[str, tuple[float, float]] = {}  # cell -> (lat, lon)
        self.boundary_geojson: dict | None = None

    def _load_boundary(self) -> list[dict]:
        """Return a list of GeoJSON Polygon geometries to fill."""
        try:
            with open(BOUNDARY_PATH) as f:
                gj = json.load(f)
            self.boundary_geojson = gj
            polys = []
            feats = gj.get("features", [gj]) if gj.get("type") != "Feature" else [gj]
            for feat in feats:
                geom = feat.get("geometry", feat) if "geometry" in feat else feat
                if geom["type"] == "Polygon":
                    polys.append(geom)
                elif geom["type"] == "MultiPolygon":
                    for coords in geom["coordinates"]:
                        polys.append({"type": "Polygon", "coordinates": coords})
            if polys:
                logger.info("Loaded boundary with %d polygons", len(polys))
                return polys
        except Exception as e:
            logger.warning("Failed to load boundary file (%s); using fallback polygon", e)

        # fallback
        fc = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {}, "geometry": _FALLBACK_POLY},
            {"type": "Feature", "properties": {}, "geometry": _FALLBACK_TAS},
        ]}
        self.boundary_geojson = fc
        return [_FALLBACK_POLY, _FALLBACK_TAS]

    def _fill(self, polys: list[dict], res: int) -> list[str]:
        cells: set[str] = set()
        for geom in polys:
            try:
                cells.update(h3.geo_to_cells(geom, res))
            except Exception as e:
                logger.warning("geo_to_cells failed for a polygon at res %d: %s", res, e)
        return sorted(cells)

    def build(self):
        polys = self._load_boundary()
        self.cells_coarse = self._fill(polys, RES_COARSE)
        self.cells_fine = self._fill(polys, RES_FINE)
        for c in self.cells_coarse:
            self.centroids[c] = h3.cell_to_latlng(c)
        for c in self.cells_fine:
            self.centroids[c] = h3.cell_to_latlng(c)
        logger.info("Built H3 grid: res%d=%d cells, res%d=%d cells",
                    RES_COARSE, len(self.cells_coarse), RES_FINE, len(self.cells_fine))

    def centroid(self, cell: str) -> tuple[float, float]:
        if cell not in self.centroids:
            self.centroids[cell] = h3.cell_to_latlng(cell)
        return self.centroids[cell]

    @staticmethod
    def parent(cell: str, res: int) -> str:
        return h3.cell_to_parent(cell, res)

    @staticmethod
    def neighbors(cell: str, k: int = 1) -> list[str]:
        return list(h3.grid_disk(cell, k))

    def weather_cells(self) -> list[str]:
        """Unique res-3 parents of coarse cells: live-weather sampling grid."""
        return sorted({h3.cell_to_parent(c, RES_WEATHER) for c in self.cells_coarse})

    def cells_for_resolution(self, res: int) -> list[str]:
        return self.cells_fine if res == RES_FINE else self.cells_coarse


# Singleton
grid = HexGrid()
