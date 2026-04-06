"""
Extraction-stage Pydantic schemas.
Spec: docs/02_SCHEMA_CONTRACT.md §5
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Document parsing
# ---------------------------------------------------------------------------

class TextChunk(BaseModel):
    chunk_id: str           # e.g. "Table_1", "Results_Para_3"
    section: str            # e.g. "Methods", "Results", "Abstract"
    content: str
    chunk_type: Literal["paragraph", "table", "figure_caption", "list"]


class ArmInfo(BaseModel):
    arm_id: str             # e.g. "arm_intervention", "arm_control"
    arm_name: str           # e.g. "Exergame group", "Usual care"
    total_n: Optional[int] = None


class TimepointInfo(BaseModel):
    timepoint_id: str
    label: str              # e.g. "Baseline", "12 Weeks", "6 Months"


class DocumentMap(BaseModel):
    pmid: str
    source_type: Literal["pmc_xml", "user_xml", "user_pdf"]
    chunks: List[TextChunk]
    arms: List[ArmInfo]
    timepoints: List[TimepointInfo]


# ---------------------------------------------------------------------------
# Raw extraction (LLM reads verbatim — NO computation)
# ---------------------------------------------------------------------------

class RawDataPoint(BaseModel):
    """Exactly what the LLM reads from the paper. No computation allowed."""
    data_type: Literal[
        "mean_sd", "mean_se", "mean_95ci",
        "median_iqr", "median_range",
        "events_total", "percentage_total",
        "or_ci",
        "not_reported",
    ]
    val1: Optional[float] = None    # mean / median / events / OR / percentage
    val2: Optional[float] = None    # sd / se / lower_ci / Q1
    val3: Optional[float] = None    # upper_ci / Q3
    n: Optional[int] = None
    raw_text: Optional[str] = None  # Exact text snippet from paper


class RawOutcomeExtraction(BaseModel):
    outcome: str
    timepoint: str
    arms: Dict[str, RawDataPoint]       # arm_id → raw data
    relevant_chunk_ids: List[str]
    extractor_confidence: Optional[Literal["high", "medium", "low"]] = None


class CharacteristicsExtraction(BaseModel):
    pmid: str
    values: Dict[str, Any]              # characteristic_name → extracted value
    source_chunks: List[str]            # chunk_ids used


# ---------------------------------------------------------------------------
# Standardized data (after MathSandbox HardNode)
# ---------------------------------------------------------------------------

class StandardizedDataPoint(BaseModel):
    """All continuous data converted to Mean ± SD by HardNode math."""
    original_type: str
    mean: Optional[float] = None
    sd: Optional[float] = None
    n: Optional[int] = None
    events: Optional[int] = None        # Dichotomous outcomes
    total: Optional[int] = None
    standardization_method: Optional[str] = None    # e.g. "se_to_sd", "wan_iqr"
    is_valid: bool = True
    validation_notes: Optional[str] = None          # e.g. "SD <= 0, marked invalid"


class StandardizedOutcome(BaseModel):
    outcome: str
    timepoint: str
    arms: Dict[str, StandardizedDataPoint]  # arm_id → standardized


# ---------------------------------------------------------------------------
# Effect sizes (computed by HardNode — never by LLM)
# ---------------------------------------------------------------------------

class EffectSizeData(BaseModel):
    outcome: str
    timepoint: str
    effect_measure: Literal["SMD", "OR", "RR", "MD"]
    effect_value: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    se: Optional[float] = None
    computation_method: str             # e.g. "hedges_g", "log_or"
    is_valid: bool = True


# ---------------------------------------------------------------------------
# Aggregate output per paper
# ---------------------------------------------------------------------------

class ExtractionOutput(BaseModel):
    pmid: str
    document_map: DocumentMap
    characteristics: CharacteristicsExtraction
    raw_outcomes: List[RawOutcomeExtraction]
    standardized_outcomes: List[StandardizedOutcome]
    effect_sizes: List[EffectSizeData]
