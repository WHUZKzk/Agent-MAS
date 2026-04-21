"""
Shared Pydantic data models for AutoSR.
"""

from enum import Enum
from typing import Dict, List, Optional
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


# ===========================================================================
# Screening v2 models (four-stage PICOS-based pipeline)
# ===========================================================================

class StudyDesignFilter(str, Enum):
    """User-specified study design inclusion scope."""
    RCT_ONLY = "rct_only"
    OBSERVATIONAL_ONLY = "obs_only"
    BOTH = "both"


# ---- Stage 1: PICOS extraction ----

class PICOSProfile(BaseModel):
    """Structured PICOS summary extracted from title + abstract."""
    P_population: str = Field(description="Study population / participants")
    I_intervention: str = Field(description="Intervention or exposure")
    C_comparison: str = Field(description="Comparator or control")
    O_outcome: str = Field(description="Outcome measures")
    S_study_design: str = Field(description="Study design type")
    sample_size: str = Field(default="Not reported")
    duration: str = Field(default="Not reported")


# ---- Stage 2: criteria & matching ----

class DimensionCriteria(BaseModel):
    """Matching criteria for a single PICO dimension."""
    core: str = Field(description="Essential requirement")
    acceptable_variations: str = Field(description="Broader scope that still qualifies")
    exclusion_boundary: str = Field(description="What clearly does NOT match")


class StudyDesignCriteria(BaseModel):
    """Matching criteria for study design dimension."""
    acceptable_designs: List[str] = Field(default_factory=list)
    excluded_designs: List[str] = Field(default_factory=list)


class MatchingCriteria(BaseModel):
    """Complete set of PICOS matching criteria (generated once per review)."""
    P_criteria: DimensionCriteria
    I_criteria: DimensionCriteria
    C_criteria: DimensionCriteria
    O_criteria: DimensionCriteria
    S_criteria: StudyDesignCriteria


class DimensionResult(BaseModel):
    """Per-paper dimension matching result from Stage 2."""
    reasoning: Dict[str, str] = Field(
        description="CoT reasoning per dimension: {P: '...', I: '...', ...}"
    )
    dimensions: Dict[str, str] = Field(
        description="Decision per dimension: {P: 'MATCH', I: 'UNCERTAIN', ...}"
    )
    overall_decision: str = Field(description="INCLUDE | EXCLUDE | UNCERTAIN")


# ---- Stage 3: uncertain review ----

class ReviewResult(BaseModel):
    """Stage 3 review result for borderline papers."""
    review_reasoning: str = Field(description="Detailed reasoning for final decision")
    resolved_dimensions: Dict[str, str] = Field(
        description="{P: 'MATCH', I: 'STILL_UNCERTAIN', ...}"
    )
    final_decision: str = Field(description="INCLUDE | EXCLUDE (no UNCERTAIN)")
    confidence: str = Field(description="HIGH | MEDIUM | LOW")


# ---- Final decision record ----

class PaperDecisionV2(BaseModel):
    """Full auditable decision record for a single paper."""
    pmid: str
    title: str

    # Stage 0
    stage0_result: str = Field(description="KEEP | EXCLUDED_pub_type | EXCLUDED_study_design")

    # Stage 1 (only for stage0=KEEP papers)
    picos_profile: Optional[PICOSProfile] = None

    # Stage 2 (only for papers passing Stage 1)
    dimension_result: Optional[DimensionResult] = None

    # Stage 3 (only for UNCERTAIN papers)
    review_result: Optional[ReviewResult] = None

    # Final
    final_decision: str = Field(description="INCLUDE | EXCLUDE")
    decision_stage: str = Field(
        description="Stage where final decision was made: stage0 | stage1 | stage2 | stage3"
    )


class ScreeningSummaryV2(BaseModel):
    """Aggregated screening statistics."""
    total: int
    stage0_excluded: int = Field(description="Excluded by publication type / study design rules")
    stage1_excluded: int = Field(description="Excluded by study design cross-validation")
    stage2_included: int = Field(description="Directly included at Stage 2")
    stage2_excluded: int = Field(description="Directly excluded at Stage 2")
    stage3_reviewed: int = Field(description="Papers sent to Stage 3 review")
    stage3_included: int = Field(description="Included after Stage 3 review")
    stage3_excluded: int = Field(description="Excluded after Stage 3 review")
    final_included: int
    final_excluded: int


class ScreeningResultV2(BaseModel):
    """Complete screening output with full audit trail."""
    criteria: MatchingCriteria
    decisions: List[PaperDecisionV2]
    summary: ScreeningSummaryV2
