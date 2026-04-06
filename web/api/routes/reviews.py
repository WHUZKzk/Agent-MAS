"""
REST endpoints for review CRUD and pipeline control.

Routes:
  POST   /api/reviews                    Create review
  GET    /api/reviews                    List reviews
  GET    /api/reviews/{id}               Get review status
  POST   /api/reviews/{id}/start         Start / resume pipeline
  GET    /api/reviews/{id}/search        Get search results
  GET    /api/reviews/{id}/screening     Get screening results
  POST   /api/reviews/{id}/uploads       Upload full-text file
  GET    /api/reviews/{id}/extraction    Get extraction results
  GET    /api/reviews/{id}/export        Download outputs as ZIP
  DELETE /api/reviews/{id}               Delete review
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from typing import List, Optional

from fastapi import (
    APIRouter, BackgroundTasks, Depends, File, HTTPException,
    Response, UploadFile,
)
from sqlalchemy.orm import Session

from src.main import load_review_config
from src.orchestrator import SystematicReviewOrchestrator
from src.schemas.common import PICODefinition, ReviewConfig

from ..database import ReviewDB
from ..dependencies import get_db
from ..models import (
    ExtractionResultsSummary,
    ReviewCreateRequest,
    ReviewListResponse,
    ReviewResponse,
    ScreeningResultsSummary,
    SearchResultsSummary,
    UploadResponse,
)
from .websocket import drop_queue, ensure_queue, sync_push_progress

logger = logging.getLogger("autosr.api.reviews")
router = APIRouter(prefix="/api/reviews", tags=["reviews"])

_CHECKPOINT_BASE = "data/checkpoints"
_UPLOADS_BASE = "data/uploads"
_OUTPUTS_BASE = "data/outputs"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _review_to_response(r: ReviewDB) -> ReviewResponse:
    return ReviewResponse(
        id=r.id,
        title=r.title,
        status=r.status,
        current_stage=r.current_stage,
        error_message=r.error_message,
        created_at=r.created_at.isoformat(),
        updated_at=r.updated_at.isoformat(),
    )


def _get_review_or_404(review_id: str, db: Session) -> ReviewDB:
    r = db.get(ReviewDB, review_id)
    if r is None:
        raise HTTPException(status_code=404, detail=f"Review {review_id} not found.")
    return r


def _checkpoint_dir(review_id: str) -> str:
    return os.path.join(_CHECKPOINT_BASE, review_id)


def _uploads_dir(review_id: str) -> str:
    return os.path.join(_UPLOADS_BASE, review_id)


def _outputs_dir(review_id: str) -> str:
    return os.path.join(_OUTPUTS_BASE, review_id)


def _load_stage_data(review_id: str, stage: str) -> Optional[dict]:
    """Load checkpoint JSON for the given stage, returning the app_state dict."""
    cp_path = os.path.join(_checkpoint_dir(review_id), f"{review_id}_{stage}.json")
    # Legacy path for reviews that use pmid as prefix
    if not os.path.exists(cp_path):
        # Try to glob for any file matching *_{stage}.json
        cp_dir = _checkpoint_dir(review_id)
        if os.path.isdir(cp_dir):
            for fname in os.listdir(cp_dir):
                if fname.endswith(f"_{stage}.json"):
                    cp_path = os.path.join(cp_dir, fname)
                    break
    if not os.path.exists(cp_path):
        return None
    with open(cp_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("app_state")


# ─────────────────────────────────────────────────────────────────────────────
# Background pipeline task
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline_sync(
    review_id: str,
    review_config: ReviewConfig,
    db_session_factory,
) -> None:
    """
    Blocking function executed in a ThreadPoolExecutor.
    Runs the full orchestrator pipeline, updating DB status on completion.
    """
    def _update_db(status: str, stage: str, error: str = None):
        db = db_session_factory()
        try:
            r = db.get(ReviewDB, review_id)
            if r:
                r.status = status
                r.current_stage = stage
                r.error_message = error
                r.updated_at = datetime.utcnow()
                db.commit()
        finally:
            db.close()

    def progress_cb(msg: dict) -> None:
        sync_push_progress(review_id, msg)

    try:
        orchestrator = SystematicReviewOrchestrator(
            checkpoint_dir=_checkpoint_dir(review_id),
            uploads_dir=_uploads_dir(review_id),
            outputs_dir=_outputs_dir(review_id),
            progress_callback=progress_cb,
        )
        final_state = orchestrator.run(review_config)
        _update_db("extraction_complete", final_state.current_stage)

        sync_push_progress(review_id, {
            "type": "stage_complete",
            "stage": "extraction",
            "message": "Pipeline complete.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as exc:
        logger.exception("[Pipeline] Error for review %s: %s", review_id, exc)
        _update_db("failed", "extraction", error_message=str(exc))
        sync_push_progress(review_id, {
            "type": "error",
            "stage": None,
            "message": f"Pipeline failed: {exc}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    finally:
        drop_queue(review_id)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.post("", response_model=ReviewResponse, status_code=201)
def create_review(
    req: ReviewCreateRequest,
    db: Session = Depends(get_db),
) -> ReviewResponse:
    """Create a new review from a user-submitted PICO form."""
    review_id = str(uuid.uuid4())
    config = ReviewConfig(
        pmid=review_id,          # use review UUID as the PMID key for checkpoints
        title=req.title,
        abstract=req.abstract,
        pico=req.pico,
        target_characteristics=req.target_characteristics or [
            "Mean Age", "% Female", "Sample Size", "Study Duration"
        ],
        target_outcomes=req.target_outcomes or [
            "Physical Activity", "MVPA", "Step Count", "BMI"
        ],
    )
    now = datetime.utcnow()
    db_review = ReviewDB(
        id=review_id,
        title=req.title,
        status="created",
        current_stage="init",
        review_config_json=config.model_dump_json(),
        created_at=now,
        updated_at=now,
    )
    db.add(db_review)
    db.commit()
    db.refresh(db_review)
    logger.info("[API] Created review %s: %s", review_id, req.title[:60])
    return _review_to_response(db_review)


@router.post("/upload-bench", response_model=ReviewResponse, status_code=201)
async def create_review_from_bench(
    file: UploadFile = File(...),
    index: int = 0,
    db: Session = Depends(get_db),
) -> ReviewResponse:
    """
    Create a review by uploading a bench_review.json file.
    The `index` query parameter selects which review to use (default 0).
    """
    content = await file.read()
    tmp_path = f"/tmp/bench_{uuid.uuid4().hex}.json"
    with open(tmp_path, "wb") as f:
        f.write(content)

    try:
        config = load_review_config(tmp_path, index)
    except (IndexError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    finally:
        os.remove(tmp_path)

    review_id = str(uuid.uuid4())
    # Override the pmid from bench file with our UUID so checkpoints use review_id
    config = config.model_copy(update={"pmid": review_id})

    now = datetime.utcnow()
    db_review = ReviewDB(
        id=review_id,
        title=config.title,
        status="created",
        current_stage="init",
        review_config_json=config.model_dump_json(),
        created_at=now,
        updated_at=now,
    )
    db.add(db_review)
    db.commit()
    db.refresh(db_review)
    return _review_to_response(db_review)


@router.get("", response_model=ReviewListResponse)
def list_reviews(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
) -> ReviewListResponse:
    reviews = db.query(ReviewDB).order_by(ReviewDB.created_at.desc()).offset(skip).limit(limit).all()
    total = db.query(ReviewDB).count()
    return ReviewListResponse(
        reviews=[_review_to_response(r) for r in reviews],
        total=total,
    )


@router.get("/{review_id}", response_model=ReviewResponse)
def get_review(
    review_id: str,
    db: Session = Depends(get_db),
) -> ReviewResponse:
    r = _get_review_or_404(review_id, db)
    return _review_to_response(r)


@router.post("/{review_id}/start", response_model=ReviewResponse)
async def start_pipeline(
    review_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> ReviewResponse:
    """
    Start or resume the pipeline for this review.
    Returns immediately; progress is streamed via WebSocket.
    """
    r = _get_review_or_404(review_id, db)

    if r.status == "running":
        raise HTTPException(status_code=409, detail="Pipeline is already running.")
    if r.status == "extraction_complete":
        raise HTTPException(status_code=409, detail="Pipeline already complete.")

    review_config = ReviewConfig.model_validate_json(r.review_config_json)
    ensure_queue(review_id)
    os.makedirs(_checkpoint_dir(review_id), exist_ok=True)
    os.makedirs(_uploads_dir(review_id), exist_ok=True)
    os.makedirs(_outputs_dir(review_id), exist_ok=True)

    # Update status to running
    r.status = "running"
    r.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(r)

    from ..database import SessionLocal

    loop = asyncio.get_event_loop()
    background_tasks.add_task(
        loop.run_in_executor,
        None,
        _run_pipeline_sync,
        review_id,
        review_config,
        SessionLocal,
    )

    logger.info("[API] Started pipeline for review %s", review_id)
    return _review_to_response(r)


@router.get("/{review_id}/search")
def get_search_results(
    review_id: str,
    db: Session = Depends(get_db),
) -> SearchResultsSummary:
    _get_review_or_404(review_id, db)
    data = _load_stage_data(review_id, "search")
    if data is None:
        raise HTTPException(status_code=404, detail="Search results not yet available.")

    search_out = data.get("search_output") or {}
    prisma = search_out.get("prisma_numbers") or {}
    return SearchResultsSummary(
        total_candidates=len(search_out.get("pmids") or []),
        prisma={
            "initial_query_results": prisma.get("initial_query_results", 0),
            "augmented_query_results": prisma.get("augmented_query_results", 0),
            "after_deduplication": prisma.get("after_deduplication", 0),
            "final_candidate_count": prisma.get("final_candidate_count", 0),
        },
        query_history=search_out.get("query_history") or [],
        pico_terms=search_out.get("pico_terms"),
    )


@router.get("/{review_id}/screening")
def get_screening_results(
    review_id: str,
    db: Session = Depends(get_db),
) -> ScreeningResultsSummary:
    _get_review_or_404(review_id, db)
    data = _load_stage_data(review_id, "screening")
    if data is None:
        raise HTTPException(status_code=404, detail="Screening results not yet available.")

    sc_out = data.get("screening_output") or {}
    prisma = sc_out.get("prisma_numbers") or {}
    decisions = sc_out.get("decisions") or {}
    decision_list = [
        {"pmid": pmid, **d} for pmid, d in decisions.items()
    ]
    return ScreeningResultsSummary(
        cohens_kappa=sc_out.get("cohens_kappa", 0.0),
        prisma={
            "total_screened": prisma.get("total_screened", 0),
            "excluded_by_metadata": prisma.get("excluded_by_metadata", 0),
            "excluded_by_dual_review": prisma.get("excluded_by_dual_review", 0),
            "sent_to_adjudication": prisma.get("sent_to_adjudication", 0),
            "included_after_screening": prisma.get("included_after_screening", 0),
        },
        included_pmids=sc_out.get("included_pmids") or [],
        decisions=decision_list,
    )


@router.post("/{review_id}/uploads", response_model=UploadResponse)
async def upload_fulltext(
    review_id: str,
    pmid: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> UploadResponse:
    """Store a full-text file (XML or PDF) for the given PMID."""
    _get_review_or_404(review_id, db)

    ext = os.path.splitext(file.filename or "")[-1].lower()
    if ext not in (".xml", ".pdf"):
        raise HTTPException(status_code=422, detail="Only .xml and .pdf files are accepted.")

    dest_dir = _uploads_dir(review_id)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, f"{pmid}{ext}")

    content = await file.read()
    with open(dest_path, "wb") as f:
        f.write(content)

    logger.info("[API] Uploaded %s for review %s PMID %s", file.filename, review_id, pmid)
    return UploadResponse(
        pmid=pmid,
        filename=file.filename or f"{pmid}{ext}",
        size_bytes=len(content),
        message=f"File saved as {pmid}{ext}.",
    )


@router.get("/{review_id}/extraction")
def get_extraction_results(
    review_id: str,
    db: Session = Depends(get_db),
) -> ExtractionResultsSummary:
    _get_review_or_404(review_id, db)
    data = _load_stage_data(review_id, "extraction")
    if data is None:
        raise HTTPException(status_code=404, detail="Extraction results not yet available.")

    ex_outputs = data.get("extraction_outputs") or {}
    effect_count = sum(
        len(v.get("effect_sizes", []))
        for v in ex_outputs.values()
        if isinstance(v, dict)
    )
    outcome_count = sum(
        len(v.get("raw_outcomes", []))
        for v in ex_outputs.values()
        if isinstance(v, dict)
    )
    return ExtractionResultsSummary(
        papers_processed=len(ex_outputs),
        outcomes_extracted=outcome_count,
        effect_sizes_computed=effect_count,
    )


@router.get("/{review_id}/export")
def export_results(
    review_id: str,
    db: Session = Depends(get_db),
) -> Response:
    """Download all output CSV files for this review as a ZIP archive."""
    _get_review_or_404(review_id, db)
    out_dir = _outputs_dir(review_id)
    if not os.path.isdir(out_dir):
        raise HTTPException(status_code=404, detail="No output files available yet.")

    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(out_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                arcname = os.path.relpath(fpath, out_dir)
                zf.write(fpath, arcname)

    buf.seek(0)
    return Response(
        content=buf.read(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="autosr_{review_id}.zip"'},
    )


@router.delete("/{review_id}", status_code=204)
def delete_review(
    review_id: str,
    db: Session = Depends(get_db),
) -> None:
    """Delete a review, its checkpoints, uploads, and outputs."""
    r = _get_review_or_404(review_id, db)
    if r.status == "running":
        raise HTTPException(status_code=409, detail="Cannot delete a running review.")

    db.delete(r)
    db.commit()

    for base in (_checkpoint_dir(review_id), _uploads_dir(review_id), _outputs_dir(review_id)):
        if os.path.isdir(base):
            shutil.rmtree(base)

    logger.info("[API] Deleted review %s", review_id)
