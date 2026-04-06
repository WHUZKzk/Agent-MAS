# 04 — SKILL FRAMEWORK

> **Dependencies:** You MUST have read `01_MASTER_BLUEPRINT.md`,
> `02_SCHEMA_CONTRACT.md`, and `03_CORE_ENGINE_SPEC.md`.
>
> **Purpose:** This document defines the Skill YAML standard format, the
> SkillGenerator (benchmark-driven compiler), and the guidelines library
> structure. Skills are the atomic units of DCR — each one describes what
> context a specific DAG Soft Node needs.

---

## 1. Design Principles

1. **Skills are declarations, not prompts.** A Skill YAML describes *what
   information a node needs*, not the literal prompt text. The ContextManager
   assembles the actual prompt at runtime.
2. **Skills are never hardcoded in Python.** All prompt logic lives in YAML
   files under `src/skills/`. Python code references skills by `skill_id`.
3. **Some Skills are static, some are generated.** Search and Screening skills
   are defined by the developer. Extraction skills for specific outcomes are
   generated dynamically by the SkillGenerator based on `bench_review.json`.
4. **Guidelines are separate from Skills.** A Skill references a guideline
   file by ID. The guideline text is stored in `src/guidelines/` and loaded
   at mount time.

---

## 2. Skill YAML Schema

Every Skill YAML file MUST conform to this structure:

```yaml
# --- Identity ---
skill_id: string                     # Unique ID: "{stage}.{skill_name}"
                                     # MUST match file path: src/skills/{stage}/{skill_name}.yaml
node_type: "soft"                    # Always "soft" — only SoftNodes have Skills
description: string                  # Human-readable one-liner

# --- Model Requirements ---
model_requirement: string            # "any" | "executor" | "reviewer" | "adjudicator"
                                     # Maps to a role in ModelRegistry.defaults

# --- Context Template ---
context_template:
  role: string                       # System message template.
                                     # May contain {variable} placeholders resolved at mount time.
                                     # Example: "You are a {persona} screening papers for a
                                     #           systematic review on {review_topic}."

  guidelines_source: string | null   # ID of a guideline file in src/guidelines/.
                                     # Example: "cochrane_ch7_screening"
                                     # → loads src/guidelines/cochrane_ch7_screening.md
                                     # null if no methodological guidelines needed.

  input_slots:                       # List of data inputs injected into user message.
    - name: string                   # Slot name (used in user_message_template)
      source: string                 # Dot-path into pipeline state.
                                     # Example: "current_paper.abstract"
      required: bool                 # Default: true. If true and missing → mount fails.

  user_message_template: string      # Template for the user message.
                                     # References input slots by {slot_name}.
                                     # Example: |
                                     #   Paper Title: {paper_title}
                                     #   Abstract: {paper_abstract}
                                     #   Answer the following questions: {criteria_json}

# --- Output Requirements ---
output_schema: string                # Name of Pydantic model in src/schemas/.
                                     # Example: "ReviewerOutput"
response_format: "json"              # Always "json" for SoftNodes.

# --- Execution Constraints ---
constraints:
  max_retries: int                   # Default: 2. Max parse/validation retries.
  temperature_override: float | null # If set, overrides the Agent's default.
                                     # null = use Agent default.

# --- Hard Constraints (injected into prompt) ---
hard_constraints:                    # List of invariant rules appended to every prompt.
  - string                           # These are methodological rules that NEVER change
  - string                           # regardless of the specific paper or outcome.
```

### 2.1 Schema Validation

At load time, `SkillRegistry.load()` MUST validate every YAML file against
the schema above. Invalid Skills MUST raise an error with the file path
and specific validation failure.

---

## 3. Example Skills

### 3.1 Search — PICO Term Generation

```yaml
# src/skills/search/pico_generation.yaml
skill_id: "search.pico_generation"
node_type: "soft"
description: "Generate PICO search terms from natural language review question"

model_requirement: "executor"

context_template:
  role: |
    You are a medical librarian expert in systematic review search strategy.
    Your task is to decompose a clinical review question into structured
    PICO terms suitable for PubMed Boolean query construction.

  guidelines_source: "cochrane_ch4_pico"

  input_slots:
    - name: "review_question"
      source: "review_config.title"
      required: true
    - name: "pico_definition"
      source: "review_config.pico"
      required: true

  user_message_template: |
    ## Review Question
    {review_question}

    ## PICO Definition
    {pico_definition}

    ## Task
    For each PICO dimension (P, I, C, O), generate:
    - Core terms (the most specific, standard terms)
    - Synonyms (alternative names, abbreviations)
    - Candidate MeSH terms (your best guesses — these will be validated separately)

    Return a JSON object with keys P, I, C, O. Each key maps to a list of
    term objects with fields: term, type ("core"|"synonym"|"mesh_candidate").

output_schema: "PICOTermSet"
response_format: "json"

constraints:
  max_retries: 2
  temperature_override: null

hard_constraints:
  - "Every PICO dimension MUST have at least 1 term."
  - "MeSH candidates are GUESSES. Do NOT fabricate certainty. They will be validated by a separate system."
  - "Include common abbreviations (e.g., T2DM for Type 2 Diabetes Mellitus)."
  - "Include both US and UK spellings where applicable."
```

### 3.2 Screening — Reviewer Screening

```yaml
# src/skills/screening/reviewer_screening.yaml
skill_id: "screening.reviewer_screening"
node_type: "soft"
description: "Answer binary screening questions for a single paper's title and abstract"

model_requirement: "reviewer"

context_template:
  role: |
    You are a systematic review screener evaluating whether a paper meets
    the inclusion criteria for a review on: {review_topic}.

    You will be given a paper's title and abstract, along with a set of
    binary questions. For each question, you MUST provide:
    1. An answer: YES, NO, or UNCERTAIN
    2. The exact sentence or phrase from the abstract that supports your answer.

    If the abstract does not contain enough information to answer, respond UNCERTAIN.

  guidelines_source: "cochrane_ch7_screening"

  input_slots:
    - name: "review_topic"
      source: "review_config.title"
      required: true
    - name: "paper_title"
      source: "current_paper.title"
      required: true
    - name: "paper_abstract"
      source: "current_paper.abstract"
      required: true
    - name: "criteria_json"
      source: "pipeline_state.screening_criteria"
      required: true

  user_message_template: |
    ## Paper
    **Title:** {paper_title}
    **Abstract:** {paper_abstract}

    ## Screening Questions
    {criteria_json}

    ## Instructions
    For each question, respond with a JSON object.
    Key = question_id, Value = {{"answer": "YES|NO|UNCERTAIN", "reasoning": "..."}}

    CRITICAL: You MUST NOT output a final INCLUDE or EXCLUDE decision.
    You are ONLY answering the individual questions.

output_schema: "ReviewerOutput"
response_format: "json"

constraints:
  max_retries: 2
  temperature_override: null

hard_constraints:
  - "You MUST answer every question. Do not skip any."
  - "Your reasoning MUST quote or closely paraphrase the abstract text."
  - "If the abstract is silent on a topic, answer UNCERTAIN — never guess YES or NO."
  - "You MUST NOT output a final INCLUDE/EXCLUDE verdict."
```

### 3.3 Screening — Adjudicator Resolution

```yaml
# src/skills/screening/adjudicator_resolution.yaml
skill_id: "screening.adjudicator_resolution"
node_type: "soft"
description: "Resolve conflicting reviewer answers via blinded chain-of-thought"

model_requirement: "adjudicator"

context_template:
  role: |
    You are an impartial methodological adjudicator for a systematic review.

    Two independent reviewers have given conflicting answers to a screening
    question. You will see:
    - The question
    - The paper's abstract (for reference)
    - Reasoning 1 (from an anonymous reviewer)
    - Reasoning 2 (from a different anonymous reviewer)

    Your job is to evaluate which reasoning is more methodologically sound
    based on the evidence in the abstract, and provide your final answer.

  guidelines_source: "cochrane_ch7_screening"

  input_slots:
    - name: "paper_abstract"
      source: "current_paper.abstract"
      required: true
    - name: "conflict_data"
      source: "current_conflict"
      required: true

  user_message_template: |
    ## Abstract
    {paper_abstract}

    ## Conflicting Question
    **Question:** {conflict_data.question_text}

    **Reasoning 1:** {conflict_data.reasoning_1}
    **Reasoning 2:** {conflict_data.reasoning_2}

    ## Task
    1. Evaluate which reasoning is more faithful to the abstract text.
    2. Identify any logical errors or unsupported assumptions in either reasoning.
    3. Provide your final answer: YES, NO, or UNCERTAIN.

    Return JSON: {{"adjudicated_answer": "...", "adjudication_reasoning": "..."}}

output_schema: "ConflictRecord"
response_format: "json"

constraints:
  max_retries: 2
  temperature_override: 0.0

hard_constraints:
  - "You MUST NOT know which model or persona produced each reasoning."
  - "Judge ONLY the logical quality of the reasoning against the abstract."
  - "If both reasonings are equally valid, answer UNCERTAIN."
  - "Do NOT introduce information not present in the abstract."
```

---

## 4. SkillGenerator (`src/skill_generator.py`)

### 4.1 Purpose

The SkillGenerator runs once during system initialization. It reads the
benchmark configuration and dynamically generates extraction Skills tailored
to the specific outcomes and characteristics defined in `bench_review.json`.

### 4.2 Interface

```
class SkillGenerator:
    def __init__(self, review_config: ReviewConfig,
                 context_manager: ContextManager,
                 skills_dir: str = "src/skills/extraction/"):
        ...

    def generate(self) -> List[str]:
        """
        Generate customized extraction Skill YAMLs.

        Returns: List of generated skill_ids.

        Process:
        1. Schema Analysis (Hard Node):
           - For each outcome in review_config.target_outcomes:
             a. Infer data type from PICO.O description:
                - Keywords ["rate", "incidence", "events", "odds", "risk"]
                  → "dichotomous"
                - Keywords ["level", "score", "count", "mean", "change"]
                  → "continuous"
                - Ambiguous → "continuous" (safer default)
             b. Record in extraction_plan: {outcome, inferred_type}

           - For target_characteristics:
             a. These are always qualitative/descriptive.
             b. No dynamic generation needed — use the static
                characteristics_extraction.yaml skill.

        2. Skill Compilation (Soft Node):
           - For each outcome in extraction_plan:
             a. Mount the "skill_generator" skill with:
                - outcome name
                - inferred data type
                - the Skill template skeleton (see §4.3)
             b. LLM generates the dynamic fields:
                - primary_search_terms (synonyms, abbreviations)
                - expected_data_format (how this outcome is typically reported)
                - common_reporting_patterns (table formats, in-text patterns)
             c. Merge LLM output with the fixed template skeleton.
             d. Validate against Skill YAML schema.
             e. Write to: src/skills/extraction/outcome_{outcome_name}.yaml

        3. Reload SkillRegistry to pick up new files.
        """
```

### 4.3 Skill Template Skeleton for Outcomes

This is the fixed template. The SkillGenerator fills in the `# DYNAMIC`
sections using an LLM call, while the `# FIXED` sections remain unchanged.

```yaml
# src/skills/extraction/outcome_{outcome_name}.yaml
# AUTO-GENERATED by SkillGenerator — do not edit manually

skill_id: "extraction.outcome.{outcome_name}"    # DYNAMIC
node_type: "soft"
description: "Extract {outcome_name} data"        # DYNAMIC

model_requirement: "executor"

context_template:
  role: |                                          # PARTIALLY DYNAMIC
    You are extracting quantitative outcome data from a clinical trial paper.
    You are a precise transcriptionist — you copy numbers exactly as written.

    Target Outcome: {outcome_name}
    Data Type: {data_type_hint}

    Search Terms (look for these words/phrases in the text):
    {primary_search_terms}                         # DYNAMIC - LLM generated

    Expected Data Format:
    {expected_data_format}                         # DYNAMIC - LLM generated

    Common Reporting Patterns:
    {common_reporting_patterns}                    # DYNAMIC - LLM generated

  guidelines_source: "data_extraction_rules"

  input_slots:
    - name: "relevant_chunks"
      source: "dcr_filtered_chunks"
      required: true
    - name: "arms"
      source: "document_map.arms"
      required: true
    - name: "timepoints"
      source: "document_map.timepoints"
      required: true

  user_message_template: |
    ## Study Arms
    {arms}

    ## Timepoints
    {timepoints}

    ## Relevant Text & Tables
    {relevant_chunks}

    ## Task
    For each arm and timepoint, extract the raw numerical data for
    this outcome. Return JSON matching the RawOutcomeExtraction schema.

output_schema: "RawOutcomeExtraction"
response_format: "json"

constraints:
  max_retries: 2
  temperature_override: 0.0

hard_constraints:                                  # FIXED — never change
  - "You are a TRANSCRIPTIONIST. Copy numbers exactly as they appear in the text."
  - "Do NOT calculate, derive, or infer any numbers."
  - "If text provides a range (Min-Max), do NOT record it as SD."
  - "Record the data_type exactly as reported: mean_sd, mean_se, mean_95ci, median_iqr, median_range, events_total, percentage_total, or not_reported."
  - "If you cannot find the data for a specific arm/timepoint, set data_type to 'not_reported' and all values to null."
  - "Always include the raw_text field — paste the exact snippet you read the number from."
  - "If a table has the data, specify the table ID in relevant_chunk_ids."
```

---

## 5. Guidelines Library (`src/guidelines/`)

### 5.1 Structure

Each file is a Markdown document containing a focused excerpt of
methodological guidance. Files are referenced by Skill YAMLs via
`guidelines_source`.

```
src/guidelines/
├── cochrane_ch4_pico.md           # PICO framework rules
├── cochrane_ch4_search.md         # Search strategy methodology
├── cochrane_ch7_screening.md      # Title/abstract screening rules
├── data_extraction_rules.md       # General data extraction rules
├── pearl_growing_method.md        # Pearl growing & term augmentation
└── effect_size_reference.md       # Effect size computation reference
```

### 5.2 File Format

```markdown
# {Guideline Title}

## Source
{Citation or chapter reference}

## Rules
{The actual methodological guidance, written as clear instructions
that can be injected directly into an LLM system prompt.}
```

### 5.3 Authoring Guidance

- **SHOULD** be concise: aim for 200–500 words per file.
- **MUST** be written as direct instructions, not academic prose.
  Good: "If the study design is not explicitly stated, look for keywords
  such as 'randomized', 'blinded', 'placebo-controlled'."
  Bad: "According to the Cochrane Handbook (Higgins et al., 2019),
  study design identification is a critical step..."
- **MUST NOT** contain implementation details (no Python, no JSON schemas).
- **MAY** be initially created as stub files with TODO markers.
  The system will raise a clear error if a Skill references a missing
  guideline, reminding the developer to fill it in.

---

## 6. Skill Lifecycle Summary

```
Initialization:
  1. Developer creates static Skills (search/, screening/)
  2. Developer creates guideline stubs (src/guidelines/)
  3. SkillGenerator reads bench_review.json
  4. SkillGenerator creates dynamic extraction Skills
  5. SkillRegistry loads all Skills from disk

Runtime (per SoftNode execution):
  1. DAGRunner reaches a SoftNode
  2. ContextManager.mount(skill_id, state):
     a. SkillRegistry.load(skill_id) → SkillDefinition
     b. Load guidelines from src/guidelines/{guidelines_source}.md
     c. Resolve input_slots from state
     d. Render system_message + user_message
     e. Return MountedContext
  3. Agent.call(mounted_context) → raw JSON string
  4. Validate against output_schema (Pydantic)
  5. On failure: retry up to max_retries
  6. ContextManager.unmount()
```
