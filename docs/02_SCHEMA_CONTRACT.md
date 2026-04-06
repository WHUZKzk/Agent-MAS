# 02 — SCHEMA CONTRACT

> **Dependencies:** Read `01_MASTER_BLUEPRINT.md` first.
>
> **Purpose:** This document defines every cross-module data structure as a
> Pydantic schema specification. Implement these in `src/schemas/` BEFORE
> writing any pipeline code.

---

## 1. Common Types (`src/schemas/common.py`)

```
class PaperMetadata(BaseModel):
    pmid: str
    title: str
    abstract: str
    publication_types: List[str]       # e.g., ["Journal Article", "Randomized Controlled Trial"]
    mesh_terms: List[str]              # e.g., ["Diabetes Mellitus, Type 2", "Exercise"]
    fetch_date: datetime

class PICODefinition(BaseModel):
    P: str      # Population
    I: str      # Intervention
    C: str      # Comparison
    O: str      # Outcome

class ReviewConfig(BaseModel):
    """Top-level configuration parsed from bench_review.json."""
    pmid: str                          # PMID of the review article itself
    title: str
    abstract: str
    pico: PICODefinition
    target_characteristics: List[str]  # e.g., ["Mean Age", "Sample Size", ...]
    target_outcomes: List[str]         # e.g., ["IL-6", "CRP", ...]

class AppState(BaseModel):
    """Master state object passed between stages and serialized to checkpoints."""
    review_config: ReviewConfig
    search_output: Optional[SearchOutput] = None
    screening_output: Optional[ScreeningOutput] = None
    extraction_outputs: Optional[Dict[str, ExtractionOutput]] = None  # pmid → output
    current_stage: str = "init"
    skill_generation_complete: bool = False
```

---

## 2. DAG Types (`src/schemas/common.py`)

```
class NodeDefinition(BaseModel):
    node_id: str                       # e.g., "search.1_2_mesh_validation"
    node_type: Literal["soft", "hard"]
    skill_id: Optional[str] = None     # Required if node_type == "soft"
    implementation: str                # Python dotted path, e.g., "stages.search.mesh_validation"
    description: str                   # Human-readable purpose

class EdgeDefinition(BaseModel):
    from_node: str                     # node_id
    to_node: str                       # node_id
    guard: Optional[str] = None        # Python expression evaluated against pipeline state
                                       # None = unconditional edge

class DAGDefinition(BaseModel):
    dag_id: str                        # e.g., "search_pipeline"
    nodes: List[NodeDefinition]
    edges: List[EdgeDefinition]
    entry_node: str                    # node_id of the first node
    terminal_nodes: List[str]          # node_ids that mark completion
```

---

## 3. Search Stage Schemas (`src/schemas/search.py`)

```
class PICOTerm(BaseModel):
    original: str                      # LLM-generated term
    normalized: Optional[str] = None   # After MeSH alignment
    status: Literal["valid_mesh", "mapped", "fuzzy_mapped", "not_found"]
    search_field: Literal["[MeSH Terms]", "[tiab]"]
    similarity_score: Optional[float] = None  # For fuzzy matches

class PICOTermSet(BaseModel):
    P: List[PICOTerm]                  # MUST have >= 1
    I: List[PICOTerm]                  # MUST have >= 1
    C: List[PICOTerm]                  # MUST have >= 1
    O: List[PICOTerm]                  # MUST have >= 1

class QueryRecord(BaseModel):
    query_string: str                  # The full Boolean query
    stage: Literal["initial", "augmented"]
    result_count: int
    webenv: Optional[str] = None
    query_key: Optional[str] = None

class PRISMASearchData(BaseModel):
    initial_query_results: int
    augmented_query_results: int
    after_deduplication: int
    final_candidate_count: int

class SearchOutput(BaseModel):
    pmids: List[str]
    papers: Dict[str, PaperMetadata]   # pmid → metadata
    query_history: List[QueryRecord]
    pico_terms: PICOTermSet            # The validated terms used
    prisma_numbers: PRISMASearchData
```

---

## 4. Screening Stage Schemas (`src/schemas/screening.py`)

```
class BinaryQuestion(BaseModel):
    question_id: str                   # e.g., "Q_P1", "Q_I2"
    dimension: Literal["P", "I", "C", "O"]
    question_text: str                 # The binary question itself
    answerable_by: Literal["YES", "NO", "UNCERTAIN"]  # Expected answer domain

class ScreeningCriteria(BaseModel):
    """Output of Step 2.1: Criteria Binarization."""
    questions: List[BinaryQuestion]
    # Validation: MUST have >= 1 question per dimension (P, I, C, O)
    reflexion_rounds: int              # How many self-correction rounds were used

class QuestionAnswer(BaseModel):
    question_id: str
    answer: Literal["YES", "NO", "UNCERTAIN"]
    reasoning: str                     # Exact sentence(s) from abstract supporting the answer

class ReviewerOutput(BaseModel):
    """Output of a single reviewer for a single paper."""
    reviewer_model: str                # Logical model name from ModelRegistry
    answers: Dict[str, QuestionAnswer] # question_id → answer

class ConflictRecord(BaseModel):
    question_id: str
    reasoning_1: str                   # Blinded — no model identity
    reasoning_2: str                   # Blinded — no model identity
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
    exclusion_reasons: List[str]       # e.g., ["Population mismatch (Q_P1)"]

class PRISMAScreeningData(BaseModel):
    total_screened: int
    excluded_by_metadata: int
    excluded_by_dual_review: int
    sent_to_adjudication: int
    included_after_screening: int
    exclusion_reason_counts: Dict[str, int]  # reason → count

class ScreeningOutput(BaseModel):
    criteria: ScreeningCriteria
    decisions: Dict[str, ScreeningDecision]   # pmid → decision
    included_pmids: List[str]
    cohens_kappa: float
    prisma_numbers: PRISMAScreeningData
```

---

## 5. Extraction Stage Schemas (`src/schemas/extraction.py`)

```
# --- Document Parsing ---

class TextChunk(BaseModel):
    chunk_id: str                      # e.g., "Table_1", "Results_Para_3"
    section: str                       # e.g., "Methods", "Results", "Abstract"
    content: str                       # The actual text
    chunk_type: Literal["paragraph", "table", "figure_caption", "list"]

class ArmInfo(BaseModel):
    arm_id: str                        # e.g., "arm_intervention", "arm_control"
    arm_name: str                      # e.g., "Exergame group", "Usual care"
    total_n: Optional[int] = None

class TimepointInfo(BaseModel):
    timepoint_id: str
    label: str                         # e.g., "Baseline", "12 Weeks", "6 Months"

class DocumentMap(BaseModel):
    pmid: str
    source_type: Literal["pmc_xml", "user_xml", "user_pdf"]
    chunks: List[TextChunk]
    arms: List[ArmInfo]
    timepoints: List[TimepointInfo]

# --- Raw Extraction ---

class RawDataPoint(BaseModel):
    """Exactly what the LLM reads from the paper — NO computation."""
    data_type: Literal[
        "mean_sd", "mean_se", "mean_95ci",
        "median_iqr", "median_range",
        "events_total", "percentage_total",
        "or_ci",                       # Paper directly reports OR + 95%CI
        "not_reported"
    ]
    val1: Optional[float] = None       # mean, median, events, OR, percentage
    val2: Optional[float] = None       # sd, se, lower_ci, Q1/IQR_lower, lower_CI
    val3: Optional[float] = None       # upper_ci, Q3/IQR_upper, upper_CI
    n: Optional[int] = None            # Sample size for this arm at this timepoint
    raw_text: Optional[str] = None     # The exact text snippet the value was read from

class RawOutcomeExtraction(BaseModel):
    outcome: str
    timepoint: str
    arms: Dict[str, RawDataPoint]      # arm_id → raw data
    relevant_chunk_ids: List[str]      # Which chunks were used for extraction
    extractor_confidence: Optional[Literal["high", "medium", "low"]] = None

class CharacteristicsExtraction(BaseModel):
    pmid: str
    values: Dict[str, Any]             # characteristic_name → extracted value
    source_chunks: List[str]           # chunk_ids used

# --- Standardized Data ---

class StandardizedDataPoint(BaseModel):
    """After Math Sandbox processing — all continuous data as Mean ± SD."""
    original_type: str                 # What data_type was before standardization
    mean: Optional[float] = None
    sd: Optional[float] = None
    n: Optional[int] = None
    events: Optional[int] = None       # For dichotomous outcomes
    total: Optional[int] = None        # For dichotomous outcomes
    standardization_method: Optional[str] = None  # e.g., "se_to_sd", "wan_iqr"
    is_valid: bool = True
    validation_notes: Optional[str] = None  # e.g., "SD <= 0, marked invalid"

class StandardizedOutcome(BaseModel):
    outcome: str
    timepoint: str
    arms: Dict[str, StandardizedDataPoint]  # arm_id → standardized

# --- Effect Size ---

class EffectSizeData(BaseModel):
    outcome: str
    timepoint: str
    effect_measure: Literal["SMD", "OR", "RR", "MD"]
    effect_value: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    se: Optional[float] = None
    computation_method: str            # e.g., "hedges_g", "log_or"
    is_valid: bool = True

# --- Aggregate Output ---

class ExtractionOutput(BaseModel):
    pmid: str
    document_map: DocumentMap
    characteristics: CharacteristicsExtraction
    raw_outcomes: List[RawOutcomeExtraction]
    standardized_outcomes: List[StandardizedOutcome]
    effect_sizes: List[EffectSizeData]
```

---

## 6. Skill YAML Schema

The Skill YAML schema is defined in `04_SKILL_FRAMEWORK.md`. It is referenced
here for completeness. Each Skill YAML file is validated against this structure
at load time by `ContextManager`.

---

## 7. Checkpoint Schema

```
class Checkpoint(BaseModel):
    """Serialized to data/checkpoints/checkpoint_{stage}.json"""
    timestamp: datetime
    stage_completed: str               # "search" | "screening" | "extraction"
    app_state: AppState
    metadata: Dict[str, Any]           # Runtime info: total_time, error_count, etc.
```

---

## 8. Model Registry Schema (`configs/models.yaml`)

```yaml
models:
  model_a:
    provider: "anthropic"              # "anthropic" | "openai" | "google"
    model_id: "claude-sonnet-4-20250514"
    api_base: "https://api.anthropic.com"
    max_context_tokens: 200000
    supports_vision: true

  model_b:
    provider: "openai"
    model_id: "gpt-4o-mini"
    api_base: "https://api.openai.com/v1"
    max_context_tokens: 128000
    supports_vision: true

defaults:
  executor: "model_a"                  # Used by ExecutorAgent
  reviewer_a: "model_a"               # Screening Reviewer A
  reviewer_b: "model_b"               # Screening Reviewer B (heterogeneous)
  adjudicator: "model_a"              # Screening Adjudicator
  skill_generator: "model_a"          # SkillGenerator LLM calls
```

> **MUST:** Reviewer A and Reviewer B MUST use different base models to ensure
> true architectural heterogeneity. This is enforced by the Model Registry —
> `defaults.reviewer_a` and `defaults.reviewer_b` MUST point to different
> model entries.
