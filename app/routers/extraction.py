"""
POST /api/extract        — Extract data from uploaded PDFs
POST /api/extract/stream — Same with SSE streaming progress
"""

import json
import logging
import os
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from autosr.agents.extraction_agent import ExtractionAgent
from autosr.schemas.models import PICODefinition
from autosr.schemas.extraction_models import (
    ExtractionFieldDefinition,
    ExtractionOutput,
    ExtractionSummary,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/extract", tags=["extraction"])

UPLOAD_DIR = Path(__file__).parent.parent.parent / "data" / "uploads"


# ---------------------------------------------------------------------------
# Request metadata (sent as JSON string in Form field alongside file uploads)
# ---------------------------------------------------------------------------

class ExtractionMetadata(BaseModel):
    pico: PICODefinition
    study_characteristics_fields: List[ExtractionFieldDefinition] = Field(default_factory=list)
    study_results_fields: List[ExtractionFieldDefinition] = Field(default_factory=list)
    top_k: int = Field(default=15, ge=1, le=30)
    max_concurrency: int = Field(default=10, ge=1, le=50)


# ---------------------------------------------------------------------------
# Helper: save uploaded files, return paths
# ---------------------------------------------------------------------------

async def _save_uploads(files: List[UploadFile]) -> List[str]:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in files:
        if not f.filename or not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=422, detail=f"Not a PDF file: {f.filename}")
        dest = UPLOAD_DIR / f.filename
        content = await f.read()
        with open(dest, "wb") as out:
            out.write(content)
        paths.append(str(dest))
    return paths


def _cleanup(paths: List[str]):
    for fp in paths:
        try:
            os.unlink(fp)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# POST /api/extract — full response (non-streaming)
# ---------------------------------------------------------------------------

@router.post("", summary="Extract data from uploaded PDFs")
async def extract_data(
    files: List[UploadFile] = File(..., description="PDF files to extract from"),
    metadata: str = Form(..., description="JSON string with pico, fields, and options"),
):
    # Parse metadata JSON
    try:
        meta = ExtractionMetadata.model_validate_json(metadata)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid metadata JSON: {e}")

    if not meta.study_characteristics_fields and not meta.study_results_fields:
        raise HTTPException(status_code=422, detail="At least one extraction field must be defined")

    logger.info(
        "POST /api/extract  files=%d  char_fields=%d  result_fields=%d",
        len(files), len(meta.study_characteristics_fields), len(meta.study_results_fields),
    )

    file_paths = await _save_uploads(files)
    try:
        agent = ExtractionAgent()
        result = agent.run(
            file_paths=file_paths,
            pico=meta.pico,
            char_fields=meta.study_characteristics_fields,
            result_fields=meta.study_results_fields,
            top_k=meta.top_k,
            max_concurrency=meta.max_concurrency,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("ExtractionAgent failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        _cleanup(file_paths)

    return result.model_dump()


# ---------------------------------------------------------------------------
# POST /api/extract/stream — SSE streaming progress
# ---------------------------------------------------------------------------

@router.post("/stream", summary="Extract data with real-time SSE progress")
async def extract_data_stream(
    files: List[UploadFile] = File(..., description="PDF files to extract from"),
    metadata: str = Form(..., description="JSON string with pico, fields, and options"),
):
    # Parse metadata JSON
    try:
        meta = ExtractionMetadata.model_validate_json(metadata)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid metadata JSON: {e}")

    if not meta.study_characteristics_fields and not meta.study_results_fields:
        raise HTTPException(status_code=422, detail="At least one extraction field must be defined")

    logger.info(
        "POST /api/extract/stream  files=%d  char_fields=%d  result_fields=%d",
        len(files), len(meta.study_characteristics_fields), len(meta.study_results_fields),
    )

    file_paths = await _save_uploads(files)

    def event_generator():
        agent = ExtractionAgent()
        try:
            for event in agent.run_stream(
                file_paths=file_paths,
                pico=meta.pico,
                char_fields=meta.study_characteristics_fields,
                result_fields=meta.study_results_fields,
                top_k=meta.top_k,
                max_concurrency=meta.max_concurrency,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.exception("ExtractionAgent stream failed")
            yield f"data: {json.dumps({'type': 'error', 'data': str(exc)})}\n\n"
        finally:
            _cleanup(file_paths)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
