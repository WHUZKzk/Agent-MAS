"""
WebSocket endpoint and connection / progress-queue management.

Architecture (thread-safe progress streaming):
  - The synchronous pipeline runs in a ThreadPoolExecutor.
  - It calls sync_push_progress(review_id, msg) from the worker thread.
  - sync_push_progress uses loop.call_soon_threadsafe to safely enqueue
    the message onto the asyncio event loop.
  - The WebSocket coroutine dequeues and sends to the browser.

Usage from background task:
    from web.api.routes.websocket import sync_push_progress, ensure_queue

    ensure_queue(review_id)
    callback = lambda msg: sync_push_progress(review_id, msg)
    orchestrator = SystematicReviewOrchestrator(progress_callback=callback)
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger("autosr.websocket")

router = APIRouter()

# ─────────────────────────────────────────────────────────────────────────────
# In-memory state (process-scoped — good enough for single-server deployment)
# ─────────────────────────────────────────────────────────────────────────────

# review_id → asyncio.Queue of progress dicts
_queues: Dict[str, asyncio.Queue] = {}

# Reference to the main asyncio event loop (set at app startup)
_main_loop: asyncio.AbstractEventLoop | None = None


def set_main_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Called once at app startup to capture the running event loop."""
    global _main_loop
    _main_loop = loop


def ensure_queue(review_id: str) -> asyncio.Queue:
    """Create a queue for this review if it doesn't exist."""
    if review_id not in _queues:
        _queues[review_id] = asyncio.Queue()
    return _queues[review_id]


def sync_push_progress(review_id: str, message: dict) -> None:
    """
    Thread-safe: called from the synchronous pipeline thread.
    Schedules the message onto the asyncio event loop so the WebSocket
    coroutine can pick it up.
    """
    if not isinstance(message.get("timestamp"), str) or not message.get("timestamp"):
        message = {**message, "timestamp": datetime.now(timezone.utc).isoformat()}

    queue = _queues.get(review_id)
    if queue is None or _main_loop is None:
        logger.debug("[WS] No queue or loop for review %s — dropping message.", review_id)
        return
    try:
        _main_loop.call_soon_threadsafe(queue.put_nowait, message)
    except Exception as exc:
        logger.warning("[WS] Failed to push progress for %s: %s", review_id, exc)


def drop_queue(review_id: str) -> None:
    """Remove the queue when the review pipeline finishes (frees memory)."""
    _queues.pop(review_id, None)


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.websocket("/ws/reviews/{review_id}/progress")
async def progress_ws(websocket: WebSocket, review_id: str) -> None:
    """
    Stream pipeline progress events to the browser.

    - Accepts the connection immediately.
    - Creates a queue for this review (idempotent).
    - Polls the queue and forwards each message as JSON.
    - Sends a heartbeat every 15 s to keep the connection alive.
    - Cleans up on disconnect.
    """
    await websocket.accept()
    logger.info("[WS] Client connected for review %s", review_id)

    queue = ensure_queue(review_id)

    # Send an initial "connected" log so the UI knows the stream is live.
    await websocket.send_json({
        "type": "log",
        "stage": None,
        "message": f"WebSocket connected for review {review_id}.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    try:
        while True:
            try:
                # Wait up to 15 s for the next message; send heartbeat if none.
                msg = await asyncio.wait_for(queue.get(), timeout=15.0)
                await websocket.send_json(msg)
            except asyncio.TimeoutError:
                # Heartbeat ping
                await websocket.send_json({
                    "type": "log",
                    "stage": None,
                    "message": "heartbeat",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
    except WebSocketDisconnect:
        logger.info("[WS] Client disconnected for review %s", review_id)
    except Exception as exc:
        logger.error("[WS] Unexpected error for review %s: %s", review_id, exc)
