"""
POST /api/search

Accepts PICO → runs SearchAgent → returns candidate paper list.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import logging

from autosr.agents.search_agent import SearchAgent
from autosr.schemas.models import PICODefinition, SearchResult

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/search", tags=["search"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    pico: PICODefinition
    retmax: int = Field(default=1000, ge=1, le=1000, description="Max papers to retrieve (1-1000)")


class SearchResponse(BaseModel):
    query_url: str
    total_count: int
    retrieved_count: int
    search_terms: dict
    papers: list


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("", response_model=SearchResponse, summary="Search PubMed for candidate papers")
async def search_papers(request: SearchRequest):
    """
    Generate domain-agnostic search terms from PICO, query PubMed,
    and return up to `retmax` papers with title + abstract metadata.
    """
    logger.info("POST /api/search  retmax=%d", request.retmax)
    try:
        agent = SearchAgent()
        result: SearchResult = agent.run(pico=request.pico, retmax=request.retmax)
    except Exception as exc:
        logger.exception("SearchAgent failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return SearchResponse(
        query_url=result.query_url,
        total_count=result.total_count,
        retrieved_count=result.retrieved_count,
        search_terms={
            "populations":   result.search_terms.populations,
            "interventions": result.search_terms.interventions,
            "outcomes":      result.search_terms.outcomes,
        },
        papers=[p.model_dump() for p in result.papers],
    )
