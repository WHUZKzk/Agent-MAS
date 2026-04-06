"""
Screening-stage Pydantic schemas.
Spec: docs/02_SCHEMA_CONTRACT.md §4
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, model_validator


class BinaryQuestion(BaseModel):
    question_id: str                    # e.g. "Q_P1", "Q_I2"
    dimension: Literal["P", "I", "C", "O"]
    question_text: str
    answerable_by: Literal["YES", "NO", "UNCERTAIN"]


class ScreeningCriteria(BaseModel):
    """Output of Node 2.1 (Criteria Binarization)."""
    questions: List[BinaryQuestion]
    reflexion_rounds: int

    @model_validator(mode="after")
    def all_dimensions_covered(self) -> "ScreeningCriteria":
        dims = {q.dimension for q in self.questions}
        missing = {"P", "I", "C", "O"} - dims
        if missing:
            raise ValueError(
                f"ScreeningCriteria MUST have >= 1 question per dimension; "
                f"missing: {missing}"
            )
        return self


class QuestionAnswer(BaseModel):
    question_id: str
    answer: Literal["YES", "NO", "UNCERTAIN"]
    reasoning: str      # Exact sentence(s) from abstract supporting the answer


class ReviewerOutput(BaseModel):
    """Output of a single reviewer for a single paper."""
    reviewer_model: str                         # Logical model name from ModelRegistry
    answers: Dict[str, QuestionAnswer]          # question_id → answer


class ConflictRecord(BaseModel):
    question_id: str
    reasoning_1: str            # Blinded — no model identity exposed
    reasoning_2: str
    adjudicated_answer: Optional[Literal["YES", "NO", "UNCERTAIN"]] = None
    adjudication_reasoning: Optional[str] = None


class ScreeningDecision(BaseModel):
    pmid: str
    reviewer_a: ReviewerOutput
    reviewer_b: ReviewerOutput
    conflicts: List[ConflictRecord]
    individual_status_a: Literal["INCLUDE", "EXCLUDE", "UNCERTAIN_FOR_FULL_TEXT"]
    individual_status_b: Literal["INCLUDE", "EXCLUDE", "UNCERTAIN_FOR_FULL_TEXT"]
    final_status: Literal["INCLUDED", "EXCLUDED", "EXCLUDED_BY_METADATA", "FAILED"]
    exclusion_reasons: List[str]        # e.g. ["Population mismatch (Q_P1)"]


class PRISMAScreeningData(BaseModel):
    total_screened: int
    excluded_by_metadata: int
    excluded_by_dual_review: int
    sent_to_adjudication: int
    included_after_screening: int
    exclusion_reason_counts: Dict[str, int]     # reason → count


class ScreeningOutput(BaseModel):
    criteria: ScreeningCriteria
    decisions: Dict[str, ScreeningDecision]     # pmid → decision
    included_pmids: List[str]
    cohens_kappa: float
    prisma_numbers: PRISMAScreeningData
