"""
POST /api/screen      — full Stage 0-2 screening (v2 PICOS pipeline)
POST /api/screen/stream — SSE streaming variant
POST /api/screen/review — Stage 3 review of uncertain papers (supports PDF upload)
POST /api/screen/review/stream — SSE streaming variant of review
"""

import json
import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from autosr.agents.screening_agent_v2 import ScreeningAgentV2
from autosr.schemas.models import (
    PICODefinition,
    Paper,
    StudyDesignFilter,
    MatchingCriteria,
    PaperDecisionV2,
    ScreeningResultV2,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/screen", tags=["screening"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class PaperInput(BaseModel):
    pmid: str
    title: str
    abstract: str
    authors: Optional[str] = None
    year: Optional[str] = None
    journal: Optional[str] = None
    publication_type: Optional[str] = None


class ScreenRequest(BaseModel):
    pico: PICODefinition
    papers: List[PaperInput] = Field(description="Papers to screen")
    study_design_filter: StudyDesignFilter = Field(
        default=StudyDesignFilter.BOTH,
        description="rct_only | obs_only | both",
    )
    max_concurrency: int = Field(
        default=50, ge=1, le=200,
        description="Max parallel LLM requests",
    )


class ReviewRequestBody(BaseModel):
    """JSON body for the review endpoint (papers + context)."""
    pico: PICODefinition
    uncertain_decisions: List[PaperDecisionV2]
    papers: List[PaperInput]
    criteria: MatchingCriteria
    max_concurrency: int = Field(default=10, ge=1, le=50)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_papers(inputs: List[PaperInput]) -> List[Paper]:
    return [
        Paper(
            pmid=p.pmid, title=p.title, abstract=p.abstract,
            authors=p.authors, year=p.year,
            journal=p.journal, publication_type=p.publication_type,
        )
        for p in inputs
    ]


def _to_papers_map(inputs: List[PaperInput]) -> Dict[str, Paper]:
    return {
        p.pmid: Paper(
            pmid=p.pmid, title=p.title, abstract=p.abstract,
            authors=p.authors, year=p.year,
            journal=p.journal, publication_type=p.publication_type,
        )
        for p in inputs
    }


# ---------------------------------------------------------------------------
# POST /api/screen — full Stage 0-2 screening
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=ScreeningResultV2,
    summary="Screen papers using four-stage PICOS pipeline (Stages 0-2)",
)
def screen_papers(request: ScreenRequest):
    """
    Four-stage PICOS-based screening pipeline:

    - **Stage 0**: Rule-based pre-filtering (publication type + study design)
    - **Stage 1**: PICOS structured extraction from title + abstract
    - **Stage 2**: Criteria-based dimension matching with CoT reasoning

    Papers marked UNCERTAIN can be sent to ``/api/screen/review`` for Stage 3.
    """
    logger.info(
        "POST /api/screen  papers=%d  design_filter=%s  concurrency=%d",
        len(request.papers), request.study_design_filter.value,
        request.max_concurrency,
    )

    papers = _to_papers(request.papers)

    try:
        agent = ScreeningAgentV2()
        result = agent.run(
            papers=papers,
            pico=request.pico,
            study_design_filter=request.study_design_filter,
            max_concurrency=request.max_concurrency,
        )
    except Exception as exc:
        logger.exception("ScreeningAgentV2 failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return result


# ---------------------------------------------------------------------------
# POST /api/screen/stream — SSE streaming
# ---------------------------------------------------------------------------

@router.post("/stream", summary="Screen papers with real-time SSE progress")
def screen_papers_stream(request: ScreenRequest):
    """
    Same pipeline as ``POST /api/screen`` but streams progress events via SSE.

    Event types::

        stage0_done, stage1_done, criteria_generated,
        paper_decided, summary, done, error
    """
    logger.info(
        "POST /api/screen/stream  papers=%d  design_filter=%s",
        len(request.papers), request.study_design_filter.value,
    )

    papers = _to_papers(request.papers)

    def event_generator():
        agent = ScreeningAgentV2()
        try:
            for event in agent.run_stream(
                papers=papers,
                pico=request.pico,
                study_design_filter=request.study_design_filter,
                max_concurrency=request.max_concurrency,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.exception("ScreeningAgentV2 stream failed")
            yield f"data: {json.dumps({'type': 'error', 'data': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# POST /api/screen/review — Stage 3 review of uncertain papers
# ---------------------------------------------------------------------------

@router.post(
    "/review",
    summary="Stage 3: review uncertain papers with stronger model",
)
def review_uncertain_papers(request: ReviewRequestBody):
    """
    Re-evaluate UNCERTAIN papers using Claude Sonnet.

    Accepts the uncertain decisions from Stage 2 along with
    the original papers and matching criteria. Returns updated
    decisions with final INCLUDE / EXCLUDE determinations.

    For PDF upload, use the ``/review/stream`` endpoint with multipart form data.
    """
    logger.info(
        "POST /api/screen/review  papers=%d  concurrency=%d",
        len(request.uncertain_decisions), request.max_concurrency,
    )

    papers_map = _to_papers_map(request.papers)

    try:
        agent = ScreeningAgentV2()
        updated = agent.review(
            uncertain_decisions=request.uncertain_decisions,
            papers_map=papers_map,
            pico=request.pico,
            criteria=request.criteria,
            max_concurrency=request.max_concurrency,
        )
    except Exception as exc:
        logger.exception("Stage 3 review failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return updated


# ---------------------------------------------------------------------------
# POST /api/screen/review/stream — Stage 3 streaming with optional PDF
# ---------------------------------------------------------------------------

@router.post(
    "/review/stream",
    summary="Stage 3 review with SSE streaming (supports PDF upload)",
)
async def review_uncertain_papers_stream(
    request_json: str = Form(..., description="JSON string of ReviewRequestBody"),
    pdfs: Optional[List[UploadFile]] = File(None, description="Optional PDF files keyed by PMID filename"),
):
    """
    SSE-streaming Stage 3 review. Supports optional PDF upload via multipart form.

    - ``request_json``: JSON string containing pico, uncertain_decisions, papers, criteria
    - ``pdfs``: Optional PDF files; filename should be ``{pmid}.pdf``

    Event types: ``review_decided``, ``review_done``, ``error``
    """
    try:
        body = ReviewRequestBody.model_validate_json(request_json)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid request_json: {exc}")

    logger.info(
        "POST /api/screen/review/stream  papers=%d  pdfs=%d",
        len(body.uncertain_decisions), len(pdfs) if pdfs else 0,
    )

    papers_map = _to_papers_map(body.papers)

    # Process uploaded PDFs: parse with Docling + BM25 chunking
    pdf_map: Dict[str, str] = {}
    if pdfs:
        from autosr.tools.pdf_parser import parse_pdfs
        from autosr.tools.chunker import (
            chunk_document, build_context_chunks, format_chunks_with_citations,
        )
        import tempfile, os

        for upload in pdfs:
            pmid = os.path.splitext(upload.filename)[0]
            # Save to temp file for Docling
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                content = await upload.read()
                tmp.write(content)
                tmp_path = tmp.name

            try:
                parsed_list = parse_pdfs([tmp_path])
                if parsed_list:
                    parsed = parsed_list[0]
                    md_text = parsed[1] if isinstance(parsed, tuple) else parsed.text
                    tables = parsed[2] if isinstance(parsed, tuple) else parsed.tables
                    body_chunks, table_chunks = chunk_document(md_text, tables)
                    pico_queries = [
                        ("Population", body.pico.P),
                        ("Intervention", body.pico.I),
                        ("Comparison", body.pico.C),
                        ("Outcome", body.pico.O),
                    ]
                    relevant = build_context_chunks(
                        body_chunks, table_chunks,
                        field_names_and_descs=pico_queries,
                        top_k=10,
                    )
                    pdf_map[pmid] = format_chunks_with_citations(relevant)
            finally:
                os.unlink(tmp_path)

    def event_generator():
        agent = ScreeningAgentV2()
        try:
            for event in agent.review_stream(
                uncertain_decisions=body.uncertain_decisions,
                papers_map=papers_map,
                pico=body.pico,
                criteria=body.criteria,
                pdf_map=pdf_map,
                max_concurrency=body.max_concurrency,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.exception("Stage 3 review stream failed")
            yield f"data: {json.dumps({'type': 'error', 'data': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
