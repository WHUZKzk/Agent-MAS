"""
SQLAlchemy ORM model and engine setup.

Uses SQLite at data/autosr.db (created on first run).
"""
from __future__ import annotations

import os
from datetime import datetime

from sqlalchemy import Column, DateTime, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

_DB_PATH = os.environ.get("AUTOSR_DB", "data/autosr.db")
os.makedirs(os.path.dirname(_DB_PATH) if os.path.dirname(_DB_PATH) else ".", exist_ok=True)

engine = create_engine(
    f"sqlite:///{_DB_PATH}",
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class ReviewDB(Base):
    """
    Persists review identity and status.
    Actual pipeline state lives in per-review checkpoint files.

    status values:
        created → running → search_complete → screening_complete
                          → extraction_complete | failed
    """
    __tablename__ = "reviews"

    id = Column(String(36), primary_key=True)          # UUID4
    title = Column(String(500), nullable=False)
    status = Column(String(50), nullable=False, default="created")
    current_stage = Column(String(50), nullable=False, default="init")
    review_config_json = Column(Text, nullable=False)  # JSON of ReviewConfig
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow,
                        onupdate=datetime.utcnow)


def create_tables() -> None:
    Base.metadata.create_all(bind=engine)
