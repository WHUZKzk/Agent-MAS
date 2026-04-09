"""
Shared Pydantic data models for AutoSR.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------

class PICODefinition(BaseModel):
    P: str = Field(description="Population / Problem")
    I: str = Field(description="Intervention")
    C: str = Field(description="Comparison / Control")
    O: str = Field(description="Outcome")


# ---------------------------------------------------------------------------
# Paper models
# ---------------------------------------------------------------------------

class Paper(BaseModel):
    pmid: str
    title: str
    abstract: str
    authors: Optional[str] = None
    year: Optional[str] = None
    journal: Optional[str] = None
    publication_type: Optional[str] = None


# ---------------------------------------------------------------------------
# Search result models
# ---------------------------------------------------------------------------

class SearchTerms(BaseModel):
    populations: List[str] = Field(default_factory=list)
    interventions: List[str] = Field(default_factory=list)
    outcomes: List[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    query_url: str
    total_count: int
    retrieved_count: int
    papers: List[Paper]
    search_terms: SearchTerms


# ---------------------------------------------------------------------------
# Screening result models
# ---------------------------------------------------------------------------

class CriteriaSet(BaseModel):
    title_criteria: List[str] = Field(default_factory=list)
    content_criteria: List[str] = Field(default_factory=list)
    eligibility_analysis: List[str] = Field(default_factory=list)


class PaperDecision(BaseModel):
    pmid: str
    title: str
    evaluations: List[str] = Field(description="YES/NO/UNCERTAIN per criterion")
    decision: str = Field(description="INCLUDE | EXCLUDE | UNCERTAIN")


class ScreeningSummary(BaseModel):
    total: int
    included: int
    excluded: int
    uncertain: int


class ScreeningResult(BaseModel):
    criteria: CriteriaSet
    decisions: List[PaperDecision]
    summary: ScreeningSummary
