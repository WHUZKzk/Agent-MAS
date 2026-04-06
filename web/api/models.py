"""
API-layer Pydantic models (request / response).

These are distinct from src/schemas/ which owns the core pipeline schemas.
We re-use src/schemas where it makes sense (ReviewConfig, PICODefinition).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel

from src.schemas.common import PICODefinition


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ReviewCreateRequest(BaseModel):
    """User-submitted form data to create a new review."""
    title: str
    abstract: str
    pico: PICODefinition
    target_characteristics: List[str] = []
    target_outcomes: List[str] = []


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class ReviewResponse(BaseModel):
    id: str
    title: str
    status: Literal[
        "created", "running",
        "search_complete", "screening_complete", "extraction_complete",
        "failed",
    ]
    current_stage: Literal["init", "search", "screening", "extraction"]
    error_message: Optional[str] = None
    created_at: str
    updated_at: str


class ReviewListResponse(BaseModel):
    reviews: List[ReviewResponse]
    total: int


# ---------------------------------------------------------------------------
# WebSocket / progress message
# ---------------------------------------------------------------------------

class ProgressMessage(BaseModel):
    """
    Pushed over WS /ws/reviews/{id}/progress.
    Matches the spec in docs/08_WEB_INTERFACE.md §3.2.
    """
    type: Literal[
        "node_start", "node_complete",
        "stage_start", "stage_complete",
        "error", "log",
    ]
    stage: Optional[str] = None                # "search" | "screening" | "extraction"
    node_id: Optional[str] = None              # DAG node id
    item_id: Optional[str] = None              # e.g. "PMID:12345"
    progress: Optional[Dict[str, int]] = None  # {"current": 15, "total": 200}
    message: str = ""
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Search / Screening results summaries (lightweight, for UI display)
# ---------------------------------------------------------------------------

class PRISMASearchSummary(BaseModel):
    initial_query_results: int
    augmented_query_results: int
    after_deduplication: int
    final_candidate_count: int


class SearchResultsSummary(BaseModel):
    total_candidates: int
    prisma: PRISMASearchSummary
    query_history: List[Dict[str, Any]] = []
    pico_terms: Optional[Dict[str, Any]] = None


class PRISMAScreeningSummary(BaseModel):
    total_screened: int
    excluded_by_metadata: int
    excluded_by_dual_review: int
    sent_to_adjudication: int
    included_after_screening: int


class ScreeningResultsSummary(BaseModel):
    cohens_kappa: float
    prisma: PRISMAScreeningSummary
    included_pmids: List[str]
    decisions: List[Dict[str, Any]] = []


class ExtractionResultsSummary(BaseModel):
    papers_processed: int
    outcomes_extracted: int
    effect_sizes_computed: int


# ---------------------------------------------------------------------------
# Upload response
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    pmid: str
    filename: str
    size_bytes: int
    message: str
