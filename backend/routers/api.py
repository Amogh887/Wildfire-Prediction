"""API endpoints matching API_CONTRACT.md."""
import json

from fastapi import APIRouter, HTTPException, Query

from services.hex_service import grid, RES_COARSE, RES_FINE
from services.ml_service import service as ml
from services import weather_service, firms_service
from services.region_service import region_name

router = APIRouter(prefix="/api")


def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ") if dt else None


@router.get("/hexagons")
def hexagons(resolution: int = Query(RES_COARSE)):
    if resolution not in (RES_COARSE, RES_FINE):
        raise HTTPException(400, f"resolution must be {RES_COARSE} or {RES_FINE}")
    cells = grid.cells_for_resolution(resolution)
    hexes = [ml.hex_payload(c) for c in cells]
    return {
        "resolution": resolution,
        "updated_at": _iso(ml.updated_at),
        "hexagons": hexes,
    }


@router.get("/hex/{h3id}")
def hex_detail(h3id: str):
    import h3 as _h3
    if not _h3.is_valid_cell(h3id):
        raise HTTPException(404, "invalid h3 cell")
    payload = ml.hex_payload(h3id)
    payload["region"] = region_name(payload["lat"], payload["lon"])
    return payload


@router.get("/status")
def status():
    return {
        "firms_updated": _iso(firms_service.cache.updated_at),
        "weather_updated": _iso(weather_service.cache.updated_at),
        "model_loaded": ml.loaded,
        "alert_count": sum(1 for r in ml.results.values() if r["probability"] >= 0.7),
        "hex_count": len(ml.results),
    }


@router.get("/alerts")
def alerts():
    out = []
    for cell, r in ml.results.items():
        if r["probability"] >= 0.7:
            out.append({
                "h3": cell,
                "lat": r["lat"],
                "lon": r["lon"],
                "probability": r["probability"],
                "risk": r["risk"],
                "region": region_name(r["lat"], r["lon"]),
            })
    out.sort(key=lambda x: x["probability"], reverse=True)
    return {"alerts": out}


@router.get("/boundary")
def boundary():
    gj = grid.boundary_geojson
    if gj is None:
        raise HTTPException(503, "boundary not loaded")
    if gj.get("type") == "FeatureCollection":
        return gj
    if gj.get("type") == "Feature":
        return {"type": "FeatureCollection", "features": [gj]}
    # bare geometry
    return {"type": "FeatureCollection",
            "features": [{"type": "Feature", "properties": {}, "geometry": gj}]}
