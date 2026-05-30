# Pyroscope — Australia Wildfire Risk Intelligence Platform

A real-time wildfire risk prediction system covering all of Australia. The platform fuses satellite fire detections, live weather telemetry, and an XGBoost machine learning model to render per-hex risk scores on an interactive hexagonal map — updated every 30 minutes, served globally.

**Live:** [pyroscope.vercel.app](https://pyroscope.vercel.app) · Backend: [Railway](https://railway.app)

---

## What it does

- Divides Australia into ~620 H3 hexagonal cells (Uber H3 resolution 3, ~12,000 km² each)
- Fetches live weather per cell every 30 minutes via WeatherAPI.com
- Polls NASA FIRMS satellite fire hotspots every 60 minutes and Australian DEA Hotspots every 30 minutes
- Runs an XGBoost classifier (trained on 184k historical fire events) blended with a physics-based fire danger score
- Applies min-max normalization for geographic contrast, then live modifiers (extreme heat/wind/dryness, active satellite detections)
- Streams results to the frontend via a REST API
- Renders an interactive deck.gl map with per-hex colour gradients, hover cards, and a sliding detail panel

---

## Architecture

```
Frontend (React + TypeScript + Vite + deck.gl)
  └── H3HexagonLayer  →  colour-mapped risk scores
  └── GeoJsonLayer    →  Australia coastline mask
  └── HoverCard       →  on-hover weather snapshot
  └── SidePanel       →  on-click deep detail + fire authority links

Backend (Python + FastAPI)
  ├── /api/hexagons   →  all cells + probabilities
  ├── /api/hex/:id    →  single cell detail
  ├── /api/status     →  data freshness + alert count
  └── /api/alerts     →  high-risk cells (raw_score > 0.55)

  Services
  ├── ml_service      →  XGBoost inference + physics blend + live modifiers
  ├── weather_service →  WeatherAPI.com polling (res-2 H3 grid, ~90 cells)
  ├── firms_service   →  NASA FIRMS VIIRS NRT satellite hotspots
  ├── aus_fires_service → Australian DEA Hotspots (WFS feed)
  └── hex_service     →  H3 grid generation + centroid cache
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend framework | React 18 + TypeScript + Vite |
| Map rendering | deck.gl 9 (H3HexagonLayer, GeoJsonLayer, ScatterplotLayer, TextLayer) |
| Hex grid | Uber H3 (Python `h3` + JS `h3-js`) |
| Backend | FastAPI + Uvicorn |
| ML model | XGBoost (trained offline, loaded at startup) |
| Scheduling | APScheduler (background jobs for weather + fire feeds) |
| Weather API | WeatherAPI.com (1M free calls/month) |
| Fire data | NASA FIRMS VIIRS NRT + Australian DEA Hotspots WFS |
| Frontend deploy | Vercel |
| Backend deploy | Railway |
| E2E tests | Playwright |

---

## Data sources

| Source | Data | Cadence |
|---|---|---|
| NASA FIRMS | VIIRS SNPP Near Real-Time fire hotspots | Every 60 min |
| Australian DEA Hotspots | Government satellite fire detections | Every 30 min |
| WeatherAPI.com | Temperature, humidity, wind speed, precipitation | Every 30 min |
| Kaggle historical datasets | 184k fire events (training only) | Static |

---

## ML model

The XGBoost classifier is trained on historical Australian fire events aggregated to H3 cells. Features per cell:

- `temperature_max` — max temperature in past 24h
- `relative_humidity_min` — lowest humidity in past 24h
- `wind_speed_max` — peak wind speed
- `precipitation_30d` — 30-day rolling rainfall
- `ndvi_proxy` — vegetation density proxy (moisture-based)
- `fire_freq_historical` — Gaussian decay from known fire-prone regions
- `month` — seasonal signal
- `drought_index` — derived from temperature and precipitation deficit

**Live modifiers applied post-inference:**
- Extreme weather (temp > 38°C, humidity < 15%, wind > 50 km/h) → multiply by 1.8
- Active satellite fire in cell → floor probability at 0.7–0.9
- No confirmed fire → hard cap at 0.94 to prevent false extremes
- Min-max normalization across all cells for geographic contrast

---

## Risk colour scale

| Probability | Risk band | Colour |
|---|---|---|
| 0 – 25% | LOW | Near black |
| 25 – 50% | MODERATE | Yellow |
| 50 – 75% | HIGH | Orange |
| 75 – 100% | EXTREME | Deep red |

---

## Local development

### Prerequisites

- Python 3.11+
- Node.js 20+
- NASA FIRMS API key (free at [firms.modaps.eosdis.nasa.gov](https://firms.modaps.eosdis.nasa.gov/api/area/))
- WeatherAPI.com key (free at [weatherapi.com](https://www.weatherapi.com))

### Backend

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# create .env
echo "NASA_FIRMS_MAP_KEY=your_key" >> .env
echo "WEATHERAPI_KEY=your_key" >> .env

uvicorn main:app --reload --port 8000
# → http://localhost:8000/docs
```

### Frontend

```bash
cd frontend
npm install
npm run dev      # → http://localhost:5173
```

### E2E tests

```bash
cd frontend
npm run test:e2e        # headless Playwright
npm run test:e2e:ui     # interactive Playwright UI
```

The test suite auto-starts the Vite dev server on port 5174 and runs against it. Tests gracefully skip scenarios that require a live backend connection.

---

## Environment variables

| Variable | Where | Purpose |
|---|---|---|
| `NASA_FIRMS_MAP_KEY` | Backend `.env` / Railway | NASA FIRMS API access |
| `WEATHERAPI_KEY` | Backend `.env` / Railway | WeatherAPI.com access |
| `VITE_API_BASE` | Frontend `.env` / Vercel | Backend URL override |

Never commit `.env` files. The backend falls back to Australian BOM-derived climate normals if `WEATHERAPI_KEY` is absent.

---

## API reference

Full contract in [`API_CONTRACT.md`](API_CONTRACT.md).

```
GET /api/hexagons?resolution=3   →  all ~620 cells + risk scores
GET /api/hex/{h3id}              →  single cell detail
GET /api/status                  →  data freshness, alert count
GET /api/alerts                  →  cells with raw_score > 0.55 or active fires
```
