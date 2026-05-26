# API Contract (source of truth for frontend + backend)

Backend: `http://localhost:8000`  · Frontend: `http://localhost:5173` · CORS: allow all origins in dev.

## GET /api/hexagons?resolution={4|6}
Returns all hexes covering Australia with risk + live data.
```json
{
  "resolution": 4,
  "updated_at": "2026-05-26T12:00:00Z",
  "hexagons": [
    {
      "h3": "84be8d3ffffffff",
      "lat": -33.87,
      "lon": 151.21,
      "probability": 0.67,
      "risk": "HIGH",
      "temperature": 38.2,
      "humidity": 12,
      "wind_speed": 45.0,
      "precipitation": 0.0,
      "active_fires": 3
    }
  ]
}
```
- `probability`: 0..1 wildfire probability (ML base score modified by live weather + active fires)
- `risk`: `LOW` (<0.25) | `MODERATE` (<0.5) | `HIGH` (<0.75) | `EXTREME` (>=0.75)
- resolution 4 ≈ globe view (~hundreds–few thousand hexes), resolution 6 ≈ zoomed flat view

## GET /api/hex/{h3}
Single-hex detail, same fields as above plus `"region": "New South Wales"`.

## GET /api/status
```json
{ "firms_updated": "2026-05-26T11:40:00Z", "weather_updated": "2026-05-26T11:55:00Z",
  "model_loaded": true, "alert_count": 12, "hex_count": 3500 }
```

## GET /api/alerts
```json
{ "alerts": [ { "h3": "...", "lat": -33.8, "lon": 151.2, "probability": 0.82, "risk": "EXTREME", "region": "..." } ] }
```
Only hexes with probability >= 0.7, sorted descending.

## GET /api/boundary
Australia outline as GeoJSON `FeatureCollection` (for the white map outline).
