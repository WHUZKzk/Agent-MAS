"""
Pydantic data models for the ExtractionAgent pipeline.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from autosr.schemas.models import PICODefinition


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------

class ExtractionFieldDefinition(BaseModel):
    """A single user-defined extraction field."""
    name: str = Field(description="Field name, e.g. 'Sample Size' or 'Effect Size'")
    description: str = Field(
        default="",
        description="Optional description guiding the LLM on what to extract",
    )


# ---------------------------------------------------------------------------
# Intermediate models (internal pipeline state)
# ---------------------------------------------------------------------------

class ParsedPDF(BaseModel):
    """Result of Docling PDF parsing for a single paper."""
    filename: str
    markdown_text: str = Field(default="", description="Full text converted to Markdown")
    tables: List[str] = Field(
        default_factory=list,
        description="Tables extracted as Markdown-formatted strings",
    )
    num_pages: int = 0


class TextChunk(BaseModel):
    """A chunk of text from a parsed PDF."""
    text: str
    source: str = Field(description="'body' or 'table'")
    start_char: int = 0
    end_char: int = 0


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

class FieldExtraction(BaseModel):
    """Extracted value for a single field from a single paper."""
    field_name: str
    value: str = Field(default="NOT FOUND", description="Extracted raw value")
    citation: str = Field(
        default="",
        description="Verbatim text from the paper supporting this extraction",
    )
    confidence: str = Field(
        default="LOW",
        description="HIGH | MEDIUM | LOW",
    )


class CharacteristicsRow(BaseModel):
    """One row in the study_characteristics table (one per paper)."""
    filename: str
    extractions: List[FieldExtraction] = Field(default_factory=list)


class ResultsRow(BaseModel):
    """One row in the study_results table (potentially multiple per paper)."""
    filename: str
    outcome_label: str = Field(
        default="",
        description="Label for this outcome group, e.g. 'Primary: BMI at 6 months'",
    )
    extractions: List[FieldExtraction] = Field(default_factory=list)


class ExtractionOutput(BaseModel):
    """Complete extraction output for all papers."""
    characteristics: List[CharacteristicsRow] = Field(default_factory=list)
    results: List[ResultsRow] = Field(default_factory=list)


class ExtractionSummary(BaseModel):
    """Summary statistics for the extraction run."""
    total_papers: int = 0
    papers_parsed: int = 0
    papers_extracted: int = 0
    total_characteristics_fields: int = 0
    total_results_fields: int = 0
