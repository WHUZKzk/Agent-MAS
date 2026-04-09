"""
POST /api/screen

Accepts a list of papers + PICO → runs ScreeningAgent → returns per-paper decisions.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import logging

from autosr.agents.screening_agent import ScreeningAgent
from autosr.schemas.models import PICODefinition, Paper, ScreeningResult

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
    papers: List[PaperInput] = Field(description="Papers to screen (title + abstract required)")
    num_title_criteria: int = Field(default=3, ge=1, le=5)
    num_content_criteria: int = Field(default=3, ge=1, le=5)
    batch_size: int = Field(default=10, ge=1, le=50, description="Papers processed in parallel per LLM batch")


class CriteriaOut(BaseModel):
    title_criteria: List[str]
    content_criteria: List[str]
    eligibility_analysis: List[str]


class DecisionOut(BaseModel):
    pmid: str
    title: str
    evaluations: List[str]
    decision: str  # INCLUDE | EXCLUDE | UNCERTAIN


class SummaryOut(BaseModel):
    total: int
    included: int
    excluded: int
    uncertain: int


class ScreenResponse(BaseModel):
    criteria: CriteriaOut
    decisions: List[DecisionOut]
    summary: SummaryOut


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("", response_model=ScreenResponse, summary="Screen papers for inclusion in meta-analysis")
async def screen_papers(request: ScreenRequest):
    """
    Generate PICO-based eligibility criteria, evaluate every paper against
    all criteria, and return per-paper INCLUDE / EXCLUDE / UNCERTAIN decisions.
    """
    logger.info(
        "POST /api/screen  papers=%d  criteria=%d+%d  batch=%d",
        len(request.papers),
        request.num_title_criteria,
        request.num_content_criteria,
        request.batch_size,
    )

    # Convert request papers to domain Paper objects
    papers = [
        Paper(
            pmid=p.pmid,
            title=p.title,
            abstract=p.abstract,
            authors=p.authors,
            year=p.year,
            journal=p.journal,
            publication_type=p.publication_type,
        )
        for p in request.papers
    ]

    try:
        agent = ScreeningAgent()
        result: ScreeningResult = agent.run(
            papers=papers,
            pico=request.pico,
            num_title_criteria=request.num_title_criteria,
            num_content_criteria=request.num_content_criteria,
            batch_size=request.batch_size,
        )
    except Exception as exc:
        logger.exception("ScreeningAgent failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return ScreenResponse(
        criteria=CriteriaOut(
            title_criteria=result.criteria.title_criteria,
            content_criteria=result.criteria.content_criteria,
            eligibility_analysis=result.criteria.eligibility_analysis,
        ),
        decisions=[
            DecisionOut(
                pmid=d.pmid,
                title=d.title,
                evaluations=d.evaluations,
                decision=d.decision,
            )
            for d in result.decisions
        ],
        summary=SummaryOut(
            total=result.summary.total,
            included=result.summary.included,
            excluded=result.summary.excluded,
            uncertain=result.summary.uncertain,
        ),
    )
