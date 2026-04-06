"""
AutoSR FastAPI application.

Start with:
    uvicorn web.api.main:app --reload --port 8000

Or from project root:
    python -m uvicorn web.api.main:app --reload
"""
from __future__ import annotations

# Load .env FIRST — before any src.* imports that read os.environ
from dotenv import load_dotenv
load_dotenv()

import asyncio
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from .database import create_tables
from .routes.reviews import router as reviews_router
from .routes.websocket import router as ws_router, set_main_loop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="AutoSR API",
    description="Automated Systematic Review — REST + WebSocket API",
    version="1.0.0",
)

# ─────────────────────────────────────────────────────────────────────────────
# CORS — allow the Vite dev server (port 5173) to call the API
# ─────────────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Routers
# ─────────────────────────────────────────────────────────────────────────────
app.include_router(reviews_router)
app.include_router(ws_router)


# ─────────────────────────────────────────────────────────────────────────────
# Lifecycle
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup() -> None:
    # Initialize SQLite tables
    create_tables()
    # Capture the running event loop for thread-safe WS broadcasting
    set_main_loop(asyncio.get_event_loop())


@app.on_event("shutdown")
async def shutdown() -> None:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Serve built React frontend (production mode)
# ─────────────────────────────────────────────────────────────────────────────
_FRONTEND_DIST = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "frontend", "dist"
)
if os.path.isdir(_FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
