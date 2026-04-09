"""
AutoSR FastAPI application entry point.

Start with:
    uvicorn app.main:app --reload --port 8000

Endpoints:
    POST /api/search  — search PubMed from PICO
    POST /api/screen  — screen papers against PICO-derived criteria
    GET  /api/health  — health check
    GET  /docs        — Swagger UI (auto-generated)
"""

import logging
import json
from pathlib import Path
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    description=(
        "Automated Systematic Review — literature search and screening API.\n\n"
        "Based on the Agent-MAS framework. "
        "Powered by OpenRouter / qwen3.6-plus and the PubMed eUtils API."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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
async def health():
    return {"status": "ok"}


BENCH_PATH = Path(__file__).parent.parent / "data" / "benchmarks" / "bench_review.json"

@app.get("/api/reviews", tags=["utility"], summary="List benchmark systematic reviews")
async def list_reviews():
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
