"""
Search-stage Pydantic schemas.
Spec: docs/02_SCHEMA_CONTRACT.md §3
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, field_validator

from .common import PaperMetadata


class PICOTerm(BaseModel):
    original: str                   # LLM-generated term
    normalized: Optional[str] = None
    status: Literal["valid_mesh", "mapped", "fuzzy_mapped", "not_found"]
    search_field: Literal["[MeSH Terms]", "[tiab]"]
    similarity_score: Optional[float] = None    # For fuzzy matches


class PICOTermSet(BaseModel):
    P: List[PICOTerm]
    I: List[PICOTerm]
    C: List[PICOTerm]
    O: List[PICOTerm]

    @field_validator("P", "I", "C", "O")
    @classmethod
    def at_least_one_term(cls, v: List[PICOTerm]) -> List[PICOTerm]:
        if len(v) < 1:
            raise ValueError("Each PICO dimension MUST have >= 1 term")
        return v


class QueryRecord(BaseModel):
    query_string: str
    stage: Literal["initial", "augmented"]
    result_count: int
    webenv: Optional[str] = None
    query_key: Optional[str] = None


class PRISMASearchData(BaseModel):
    initial_query_results: int
    augmented_query_results: int
    after_deduplication: int
    final_candidate_count: int


class AugmentedTerm(BaseModel):
    """A single term produced by the Pearl Growing LLM call."""
    term: str
    dimension: Literal["P", "I", "C", "O"]
    rationale: Optional[str] = None     # Why the LLM added this term


class PearlGrowingOutput(BaseModel):
    """Output schema for the pearl_growing Skill YAML (LLM response)."""
    new_terms: List[AugmentedTerm]


class SearchOutput(BaseModel):
    pmids: List[str]
    papers: Dict[str, PaperMetadata]    # pmid → metadata
    query_history: List[QueryRecord]
    pico_terms: PICOTermSet
    prisma_numbers: PRISMASearchData
