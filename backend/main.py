"""Wildfire-prediction backend (FastAPI). Builds H3 grid, loads ML model, fetches live
weather + FIRMS, runs inference, and schedules periodic refreshes."""
import logging
import threading

from dotenv import load_dotenv
load_dotenv()

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import api
from services.hex_service import grid
from services.ml_service import service as ml
from services import weather_service, firms_service, aus_fires_service

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("main")

app = FastAPI(title="Australia Wildfire Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api.router)

scheduler = BackgroundScheduler()


def refresh_weather():
    try:
        weather_service.cache.refresh()
        ml.run_inference()
    except Exception:
        logger.exception("weather refresh failed")


def refresh_firms():
    try:
        firms_service.cache.refresh()
        ml.run_inference()
    except Exception:
        logger.exception("FIRMS refresh failed")


def refresh_aus_fires():
    try:
        aus_fires_service.cache.refresh()
        ml.run_inference()
    except Exception:
        logger.exception("AUS fires refresh failed")


def initial_load():
    """Run the first data fetch + inference (in background so startup is fast)."""
    try:
        firms_service.cache.refresh()
        aus_fires_service.cache.refresh()
        weather_service.cache.refresh()
        ml.run_inference()
        logger.info("Initial data load complete")
    except Exception:
        logger.exception("initial load failed")


@app.on_event("startup")
def startup():
    grid.build()
    ml.load()
    ml.load_snapshot()
    # serve immediately with a quick physics pass over empty weather so endpoints work,
    # then kick the real fetch off in the background.
    try:
        ml.run_inference()  # uses default weather until real fetch lands
    except Exception:
        logger.exception("initial inference failed")
    threading.Thread(target=initial_load, daemon=True).start()

    scheduler.add_job(refresh_weather, "interval", minutes=15, id="weather")
    scheduler.add_job(refresh_firms, "interval", minutes=30, id="firms")
    scheduler.add_job(refresh_aus_fires, "interval", minutes=15, id="aus_fires")
    scheduler.start()
    logger.info("Startup complete; scheduler running")


@app.on_event("shutdown")
def shutdown():
    try:
        scheduler.shutdown(wait=False)
    except Exception:
        pass


@app.get("/")
def root():
    return {"service": "australia-wildfire-api", "docs": "/docs"}
