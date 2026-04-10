"""
AutoSR FastAPI application entry point.

Start with:
    uvicorn app.main:app --reload --port 8000

Pages:
    GET  /          → Web UI (app/static/index.html)
    GET  /docs      → Swagger UI

API:
    POST /api/search  — PICO → candidate papers
    POST /api/screen  — papers + PICO → inclusion decisions
    GET  /api/reviews — list benchmark reviews
    GET  /api/health  — health check
"""

import json
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.routers import search, screening

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("autosr")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AutoSR",
    description="Automated Systematic Review — literature search and screening API.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Static files & root page
# ---------------------------------------------------------------------------
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", include_in_schema=False)
def root():
    return FileResponse(
        str(STATIC_DIR / "index.html"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(search.router)
app.include_router(screening.router)

# ---------------------------------------------------------------------------
# Utility endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health", tags=["utility"])
def health():
    return {"status": "ok"}


BENCH_PATH = Path(__file__).parent.parent / "data" / "benchmarks" / "bench_review.json"

@app.get("/api/reviews", tags=["utility"], summary="List benchmark systematic reviews")
def list_reviews():
    """Return the 10 benchmark systematic reviews (PMID + title + PICO)."""
    if not BENCH_PATH.exists():
        return {"reviews": []}
    with open(BENCH_PATH, encoding="utf-8") as f:
        reviews = json.load(f)
    return {
        "count": len(reviews),
        "reviews": [
            {"pmid": r["PMID"], "title": r["title"], "pico": r["PICO"]}
            for r in reviews
        ],
    }
