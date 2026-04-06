# 07 — EXTRACTION STAGE SPEC

> **Dependencies:** You MUST have implemented the Core Engine, Skill Framework,
> Search Stage, and Screening Stage before this module.
>
> **Purpose:** Implement the 6-node Extraction DAG. Processes full-text
> documents to extract, standardize, and compute effect sizes for quantitative
> outcomes.

---

## 1. Stage Overview

**Input:** `ScreeningOutput.included_pmids` + user-uploaded full-text files
(XML or PDF) from `data/uploads/`

**Output:** `ExtractionOutput` per study (see `02_SCHEMA_CONTRACT.md`)

This stage demonstrates Innovation #1 (DCR) at its most extreme: the full-text
document is never fed to the LLM in its entirety. Instead, a two-level
filtering pipeline ensures the LLM sees only the chunks relevant to each
specific extraction task.

**Critical design principle:** LLMs are transcriptionists — they copy numbers.
All mathematical operations happen in the Math Sandbox (Hard Node).

---

## 2. DAG Definition

```
Nodes: [3.1, 3.2, 3.3a, 3.3b, 3.4, 3.5, 3.6]
Entry: 3.1
Terminal: 3.6

Edges (all unconditional, linear):
  3.1 → 3.2 → 3.3a → 3.3b → 3.4 → 3.5 → 3.6
```

This is a linear DAG. Nodes 3.3a–3.3b internally contain nested loops
over outcomes and timepoints, but the DAG itself has no branches.

---

## 3. Processing Model

The Extraction DAG runs in a PER-PAPER sequential loop:

```
for pmid in included_pmids:
    file_path = resolve_upload(pmid)  # data/uploads/{pmid}.xml or .pdf
    run_extraction_dag(pmid, file_path)
```

Within each paper, Nodes 3.3a–3.3b loop over outcomes and timepoints.

---

## 4. Full-Text Parsing Strategy

Before Node 3.1 executes, the system must parse the uploaded file into a
structured representation. This is handled by a utility, not a DAG node.

### 4.1 XML Parsing (Primary Path)

```
For PubMed Central XML or user-uploaded XML:
1. Parse with lxml.etree.
2. Extract sections: Abstract, Methods, Results, Discussion, Tables,
   Figure Captions.
3. For each section, create TextChunk objects:
   - paragraph → one TextChunk per <p> element
   - table → one TextChunk per <table-wrap>, preserving structure
4. Assign chunk_ids: "{section}_{element_type}_{index}"
   e.g., "Results_Para_3", "Results_Table_1"
```

### 4.2 PDF Parsing (Fallback Path)

```
For user-uploaded PDFs:
1. Render each page to image (PNG) using pymupdf.
2. Send each page image to a vision-capable LLM (via ExecutorAgent):
   - System prompt: "Extract all text from this page. Preserve table
     structure using Markdown table format. Identify the section
     (Methods, Results, etc.) based on headings."
   - Output: structured text per page.
3. Merge page outputs into TextChunk objects using the same schema
   as XML parsing.

MUST: The DocumentMap schema is identical regardless of source format.
      Downstream nodes never know whether the input was XML or PDF.
```

---

## 5. Node Specifications

### Node 3.1: Document Cartography

| Field | Value |
|-------|-------|
| **Type** | Hard Node + Soft Node (composite) |
| **Input** | Parsed TextChunks from full-text file |
| **Output** | `DocumentMap` |

**Sub-step A (Hard Node):** Organize chunks.

```
1. Parse the full-text file (XML or PDF, using §4 strategy).
2. Create TextChunk objects with chunk_ids and section labels.
3. Store in a preliminary DocumentMap (chunks filled, arms/timepoints empty).
```

**Sub-step B (Soft Node):** Map study structure.

```
Skill: extraction.document_cartography
Input: ONLY the Methods section chunks + Abstract
LLM task:
  - Identify study arms (intervention/control names + total N)
  - Identify timepoints (Baseline, follow-up durations)
Output: arms: List[ArmInfo], timepoints: List[TimepointInfo]
Merge into DocumentMap.
```

### Node 3.2: Track A — Study Characteristics Extraction

| Field | Value |
|-------|-------|
| **Type** | Soft Node |
| **Skill** | `extraction.characteristics_extraction` |
| **Input** | Abstract + Methods + Table 1 chunks (DCR filtered) |
| **Output** | `CharacteristicsExtraction` |

**DCR Context:** Load ONLY:
- Abstract chunks
- Methods section chunks
- The first table (typically "Table 1: Baseline Characteristics")

```
LLM task:
  - Extract values for each item in review_config.target_characteristics
  - Example: {"Mean Age": 55.2, "Sample Size": 120, "Study Design": "RCT",
              "Country": "Japan", "% Male": 62.5}
  - If a characteristic is not found, return null for that key.
```

| Constraint | Level |
|------------|-------|
| LLM extracts ONLY what is written — no inference | MUST |
| Missing values → null (not guessed) | MUST |

### Node 3.3a: Chunk Relevance Scoring (Two-Level DCR, Level 1)

| Field | Value |
|-------|-------|
| **Type** | Soft Node |
| **Skill** | `extraction.chunk_relevance_scoring` |
| **Input** | Full chunk list (IDs + section + first sentence only), current outcome name |
| **Output** | List of relevant chunk_ids |

**Purpose:** This is the first level of two-level DCR for outcome extraction.
Instead of using keyword regex (which misses synonyms), an LLM evaluates
which chunks are likely to contain data for the target outcome.

```
For each outcome in target_outcomes:
    # Send the LLM a lightweight index of all chunks:
    chunk_index = [
        {"chunk_id": c.chunk_id, "section": c.section,
         "preview": first_sentence(c.content),
         "chunk_type": c.chunk_type}
        for c in document_map.chunks
    ]

    # LLM marks which chunks are potentially relevant.
    # Also always include ALL table-type chunks as candidates
    # (tables are the primary data source).

    relevant_ids = llm_score(chunk_index, outcome)

    # Hard override: ensure all table chunks are included
    for c in document_map.chunks:
        if c.chunk_type == "table" and c.chunk_id not in relevant_ids:
            relevant_ids.append(c.chunk_id)
```

| Constraint | Level |
|------------|-------|
| ALL table-type chunks MUST be included in candidates regardless of LLM scoring | MUST |
| LLM sees only chunk metadata (ID, section, first sentence), NOT full content | MUST |
| This is a lightweight classification — keep the prompt short | SHOULD |

### Node 3.3b: Focused Outcome Extraction (Two-Level DCR, Level 2)

| Field | Value |
|-------|-------|
| **Type** | Soft Node (inside nested loop) |
| **Skill** | `extraction.outcome.{outcome_name}` (dynamically generated by SkillGenerator) |
| **Input** | Relevant chunks (full content) + arms + timepoints |
| **Output** | `RawOutcomeExtraction` |

**Loop structure:**

```
for outcome in target_outcomes:
    relevant_ids = node_3_3a_output[outcome]
    relevant_chunks = [c for c in document_map.chunks if c.chunk_id in relevant_ids]

    for timepoint in document_map.timepoints:
        # Mount the dynamically generated Skill for this outcome.
        # DCR injects ONLY the relevant chunks.
        raw_extraction = run_soft_node(
            skill_id=f"extraction.outcome.{outcome}",
            data={
                "relevant_chunks": relevant_chunks,
                "arms": document_map.arms,
                "timepoint": timepoint,
            }
        )
        store_raw_extraction(pmid, outcome, timepoint, raw_extraction)
```

**The LLM outputs a `RawOutcomeExtraction`:**

```json
{
  "outcome": "IL-6",
  "timepoint": "12 Weeks",
  "arms": {
    "arm_intervention": {
      "data_type": "mean_sd",
      "val1": 3.42,
      "val2": 1.15,
      "val3": null,
      "n": 60,
      "raw_text": "IL-6 levels in the intervention group were 3.42 ± 1.15 pg/mL (n=60)"
    },
    "arm_control": {
      "data_type": "mean_se",
      "val1": 4.87,
      "val2": 0.38,
      "val3": null,
      "n": 58,
      "raw_text": "Control group: 4.87 (SE 0.38) pg/mL"
    }
  },
  "relevant_chunk_ids": ["Results_Table_2", "Results_Para_5"]
}
```

| Constraint | Level |
|------------|-------|
| LLM MUST copy numbers exactly as written | MUST |
| LLM MUST NOT calculate, derive, or infer any numbers | MUST |
| LLM MUST include raw_text — the exact snippet the number was read from | MUST |
| If data not found → data_type = "not_reported", all values null | MUST |
| LLM MUST correctly identify the data_type (mean_sd vs mean_se vs mean_95ci etc.) | MUST |

### Node 3.4: Neuro-Symbolic Math Sandbox

| Field | Value |
|-------|-------|
| **Type** | Hard Node |
| **Input** | All `RawOutcomeExtraction` objects for this paper |
| **Output** | `StandardizedOutcome` list |

**Implementation:** `src/math_sandbox.py`

**Standardization rules — route by `data_type`:**

```python
def standardize(raw: RawDataPoint) -> StandardizedDataPoint:
    if raw.data_type == "not_reported":
        return StandardizedDataPoint(is_valid=False,
                                     validation_notes="Data not reported")

    if raw.data_type == "mean_sd":
        # Direct pass-through
        mean, sd, n = raw.val1, raw.val2, raw.n
        method = "direct"

    elif raw.data_type == "mean_se":
        # SD = SE × √n
        mean = raw.val1
        se = raw.val2
        n = raw.n
        sd = se * math.sqrt(n)
        method = "se_to_sd"

    elif raw.data_type == "mean_95ci":
        # SD = √n × (CI_upper - CI_lower) / 3.92
        mean = raw.val1
        ci_lower, ci_upper = raw.val2, raw.val3
        n = raw.n
        sd = math.sqrt(n) * (ci_upper - ci_lower) / 3.92
        method = "ci_to_sd"

    elif raw.data_type == "median_iqr":
        # Wan et al. (2014) — Estimating Mean and SD from Median + IQR + n
        median, q1, q3 = raw.val1, raw.val2, raw.val3
        n = raw.n
        # Mean estimate:
        mean = (q1 + median + q3) / 3
        # SD estimate:
        # SD ≈ (Q3 - Q1) / (2 × Φ⁻¹((0.75n - 0.125) / (n + 0.25)))
        from scipy.stats import norm
        denominator = 2 * norm.ppf((0.75 * n - 0.125) / (n + 0.25))
        sd = (q3 - q1) / denominator
        method = "wan_iqr"

    elif raw.data_type == "median_range":
        # Wan et al. (2014) — Estimating Mean and SD from Median + Range + n
        median, min_val, max_val = raw.val1, raw.val2, raw.val3
        n = raw.n
        # Mean estimate:
        mean = (min_val + 2 * median + max_val) / 4
        # SD estimate:
        # SD ≈ (max - min) / (2 × Φ⁻¹((n - 0.375) / (n + 0.25)))
        from scipy.stats import norm
        denominator = 2 * norm.ppf((n - 0.375) / (n + 0.25))
        sd = (max_val - min_val) / denominator
        method = "wan_range"

    elif raw.data_type == "events_total":
        # Dichotomous — pass through
        return StandardizedDataPoint(
            original_type="events_total",
            events=int(raw.val1), total=raw.n,
            standardization_method="direct_dichotomous",
            is_valid=True
        )

    elif raw.data_type == "percentage_total":
        # Convert percentage to events
        events = round(raw.val1 / 100.0 * raw.n)
        return StandardizedDataPoint(
            original_type="percentage_total",
            events=events, total=raw.n,
            standardization_method="percentage_to_events",
            is_valid=True
        )

    elif raw.data_type == "or_ci":
        # Paper directly reports OR + 95%CI. Cannot extract raw counts.
        # Store as-is; effect size computation will use directly.
        return StandardizedDataPoint(
            original_type="or_ci",
            mean=raw.val1,        # Store OR in mean field
            sd=None,
            n=raw.n,
            standardization_method="direct_or",
            is_valid=True,
            validation_notes="OR+CI reported directly; raw counts unavailable"
        )

    # --- Sanity Checks ---
    assert sd is not None and sd > 0, f"SD must be positive, got {sd}"
    assert n is not None and n > 0, f"N must be positive, got {n}"

    return StandardizedDataPoint(
        original_type=raw.data_type,
        mean=mean, sd=sd, n=n,
        standardization_method=method,
        is_valid=True
    )
```

**Error handling for assertion failures:**

```
try:
    standardized = standardize(raw_point)
except AssertionError as e:
    standardized = StandardizedDataPoint(
        original_type=raw_point.data_type,
        is_valid=False,
        validation_notes=str(e)
    )
    logger.error(f"[MathSandbox] PMID {pmid}, {outcome}, {arm}: {e}")
```

| Constraint | Level |
|------------|-------|
| ALL math in this node — LLMs MUST NOT compute | MUST |
| Wan et al. formulas MUST be implemented natively (numpy/scipy) | MUST |
| Failed assertions → mark as invalid, do NOT propagate hallucinated data | MUST |
| Log every standardization with method used | SHOULD |

### Node 3.5: Effect Size Computation

| Field | Value |
|-------|-------|
| **Type** | Hard Node |
| **Input** | `StandardizedOutcome` list |
| **Output** | `EffectSizeData` list |

**For continuous outcomes (Mean ± SD + N for two arms):**

```
# Standardized Mean Difference (Hedges' g):
pooled_sd = sqrt(
    ((n_ig - 1) * sd_ig**2 + (n_cg - 1) * sd_cg**2) /
    (n_ig + n_cg - 2)
)
cohens_d = (mean_ig - mean_cg) / pooled_sd

# Hedges' correction factor:
J = 1 - (3 / (4 * (n_ig + n_cg - 2) - 1))
hedges_g = cohens_d * J

# SE of Hedges' g:
se_g = sqrt(
    (n_ig + n_cg) / (n_ig * n_cg) +
    hedges_g**2 / (2 * (n_ig + n_cg))
)

# 95% CI:
ci_lower = hedges_g - 1.96 * se_g
ci_upper = hedges_g + 1.96 * se_g
```

**For dichotomous outcomes (Events + Total for two arms):**

```
# Odds Ratio:
a = events_ig
b = n_ig - events_ig
c = events_cg
d = n_cg - events_cg

# Handle zero cells: add 0.5 continuity correction
if a == 0 or b == 0 or c == 0 or d == 0:
    a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5

OR = (a * d) / (b * c)
ln_OR = math.log(OR)
se_ln_OR = sqrt(1/a + 1/b + 1/c + 1/d)

ci_lower = math.exp(ln_OR - 1.96 * se_ln_OR)
ci_upper = math.exp(ln_OR + 1.96 * se_ln_OR)
```

**For direct OR + CI from paper:**

```
# If raw.data_type == "or_ci", the paper already reported the effect size.
# Use directly without computation.
OR = raw.val1
ci_lower = raw.val2
ci_upper = raw.val3
se_ln_OR = (math.log(ci_upper) - math.log(ci_lower)) / 3.92
```

| Constraint | Level |
|------------|-------|
| Use Hedges' g (not Cohen's d) for continuous outcomes | MUST |
| Apply continuity correction for zero cells in OR calculation | MUST |
| If either arm has invalid standardized data → skip effect size, mark invalid | MUST |

### Node 3.6: Benchmark Output Alignment

| Field | Value |
|-------|-------|
| **Type** | Hard Node |
| **Input** | All extraction data for this paper |
| **Output** | CSV files in `data/outputs/` |

**Output files per paper:**

```
1. data/outputs/study_characteristics/{pmid}.csv
   Columns match the benchmark format:
   First Author, Year, Title, Country, [+ target_characteristics columns]

2. data/outputs/study_results/{pmid}.csv
   For dichotomous outcomes (matching benchmark):
   First Author, Year, Title, Follow-up, OR, 95% CI Lower, 95% CI Upper

   For continuous outcomes:
   First Author, Year, Title, Follow-up, SMD, 95% CI Lower, 95% CI Upper

   Also export raw data CSV:
   data/outputs/study_raw/{pmid}_{outcome}.csv
   Arm, Timepoint, Mean, SD, N (or Events, Total for dichotomous)
```

---

## 6. Stage-Level Constraints

| Rule | Level |
|------|-------|
| Full-text documents come from user uploads only — no automatic fetching | MUST |
| XML and PDF inputs produce the same DocumentMap schema | MUST |
| Two-level DCR: Chunk Relevance Scoring (3.3a) → Focused Extraction (3.3b) | MUST |
| All table-type chunks are always included in relevance candidates | MUST |
| Math Sandbox formulas are implemented natively in Python (no R, no external tools) | MUST |
| Effect sizes are computed for all valid outcome-timepoint-arm combinations | SHOULD |

---

## 7. Skill Files to Create

| Skill ID | File | Notes |
|----------|------|-------|
| `extraction.document_cartography` | `src/skills/extraction/document_cartography.yaml` | Static |
| `extraction.characteristics_extraction` | `src/skills/extraction/characteristics_extraction.yaml` | Static |
| `extraction.chunk_relevance_scoring` | `src/skills/extraction/chunk_relevance_scoring.yaml` | Static |
| `extraction.outcome.{name}` | `src/skills/extraction/outcome_{name}.yaml` | Dynamic (SkillGenerator) |

---

## 8. Implementation Checklist

1. [ ] Full-text parsing utility: XML parser + PDF-to-multimodal pipeline
2. [ ] `src/skills/extraction/document_cartography.yaml`
3. [ ] `src/skills/extraction/characteristics_extraction.yaml`
4. [ ] `src/skills/extraction/chunk_relevance_scoring.yaml`
5. [ ] `src/guidelines/data_extraction_rules.md` (stub)
6. [ ] `src/math_sandbox.py`:
   - [ ] All 7 data_type standardization routes
   - [ ] Wan et al. (2014) formulas for median_iqr and median_range
   - [ ] Assertion-based sanity checks
7. [ ] Effect size computation functions (Hedges' g, OR, continuity correction)
8. [ ] `src/stages/extraction_pipeline.py`:
   - [ ] Node 3.1: Document Cartography
   - [ ] Node 3.2: Characteristics Extraction
   - [ ] Node 3.3a: Chunk Relevance Scoring
   - [ ] Node 3.3b: Focused Outcome Extraction (nested loop)
   - [ ] Node 3.4: Math Sandbox
   - [ ] Node 3.5: Effect Size Computation
   - [ ] Node 3.6: CSV Output Alignment
9. [ ] DAG declaration for Extraction
10. [ ] Unit tests for Math Sandbox (all 7 data_type routes)
11. [ ] Unit tests for Effect Size computation
