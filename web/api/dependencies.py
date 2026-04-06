"""
FastAPI dependency injectors — database session, etc.
"""
from __future__ import annotations

from typing import Generator

from .database import SessionLocal


def get_db() -> Generator:
    """Yield a SQLAlchemy session, closing it when the request finishes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
