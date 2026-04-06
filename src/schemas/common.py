"""
Common / shared Pydantic schemas used across all pipeline stages.
Spec: docs/02_SCHEMA_CONTRACT.md §1-2, §7
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, field_validator


# ---------------------------------------------------------------------------
# Core data primitives
# ---------------------------------------------------------------------------

class PaperMetadata(BaseModel):
    pmid: str
    title: str
    abstract: str
    publication_types: List[str]   # e.g. ["Journal Article", "Randomized Controlled Trial"]
    mesh_terms: List[str]          # e.g. ["Diabetes Mellitus, Type 2", "Exercise"]
    fetch_date: datetime


class PICODefinition(BaseModel):
    P: str   # Population
    I: str   # Intervention
    C: str   # Comparison
    O: str   # Outcome


class ReviewConfig(BaseModel):
    """Top-level configuration parsed from bench_review.json."""
    pmid: str
    title: str
    abstract: str
    pico: PICODefinition
    target_characteristics: List[str]   # e.g. ["Mean Age", "Sample Size", ...]
    target_outcomes: List[str]          # e.g. ["IL-6", "CRP", ...]


# ---------------------------------------------------------------------------
# DAG schema types
# ---------------------------------------------------------------------------

class NodeDefinition(BaseModel):
    node_id: str                        # e.g. "search.1_2_mesh_validation"
    node_type: Literal["soft", "hard"]
    skill_id: Optional[str] = None      # Required when node_type == "soft"
    implementation: str                 # Python dotted path
    description: str

    @field_validator("skill_id")
    @classmethod
    def skill_required_for_soft(cls, v: Optional[str], info: Any) -> Optional[str]:
        # Access via info.data for cross-field validation
        if info.data.get("node_type") == "soft" and not v:
            raise ValueError("skill_id is required for SoftNodes")
        return v


class EdgeDefinition(BaseModel):
    from_node: str
    to_node: str
    guard: Optional[str] = None         # Python expression evaluated against state dict


class DAGDefinition(BaseModel):
    dag_id: str
    nodes: List[NodeDefinition]
    edges: List[EdgeDefinition]
    entry_node: str
    terminal_nodes: List[str]


# ---------------------------------------------------------------------------
# Master application state
# ---------------------------------------------------------------------------

class AppState(BaseModel):
    """Passed between stages and serialized to checkpoint files."""
    review_config: ReviewConfig
    # Populated after each stage completes:
    search_output: Optional[Any] = None          # SearchOutput
    screening_output: Optional[Any] = None       # ScreeningOutput
    extraction_outputs: Optional[Dict[str, Any]] = None  # pmid → ExtractionOutput
    current_stage: str = "init"
    skill_generation_complete: bool = False


# ---------------------------------------------------------------------------
# Checkpoint (serialized to data/checkpoints/)
# ---------------------------------------------------------------------------

class Checkpoint(BaseModel):
    timestamp: datetime
    stage_completed: Literal["search", "screening", "extraction"]
    app_state: AppState
    metadata: Dict[str, Any] = {}   # runtime info: total_time, error_count, etc.
