"""
Search query prompts — domain-agnostic version of TrialMind's search_query.py.

Key changes from TrialMind:
  - Removed "clinical specialist" persona → "systematic review specialist"
  - Removed "medical conditions / treatments" language → generic research constructs
  - Renamed output keys: CONDITIONS→POPULATION_TERMS, TREATMENTS→INTERVENTION_TERMS
  - Otherwise keeps TrialMind's proven 2-step structure (primary terms → refine+expand)
"""

# ---------------------------------------------------------------------------
# Step 1  –  quick bootstrap: extract 1-3 primary terms to seed PubMed reference search
# ---------------------------------------------------------------------------

PRIMARY_TERM_EXTRACTION = """\
You are a systematic review specialist. You are conducting a systematic review and meta-analysis.
The research is defined by the following PICO elements:
P (Population): {P}
I (Intervention): {I}
C (Comparison): {C}
O (Outcome): {O}

## Task
Identify 1 to 3 primary search terms that best represent the core topic of this research.
- Terms must be specific and searchable (e.g., named interventions, specific constructs, target behaviors, health conditions).
- Do NOT include generic words such as "patients", "participants", "intervention", "outcome", "study", or "effect".

## Reply Format
Output ONLY valid JSON, no explanation:

{{
    "terms": ["term1", "term2"]
}}
"""

# ---------------------------------------------------------------------------
# Step 2  –  full term extraction + refinement + expansion
#            (informed by reference papers fetched in Step 1)
# ---------------------------------------------------------------------------

SEARCH_TERM_EXTRACTION = """\
## Background

You are a systematic review specialist conducting a systematic review and meta-analysis.
The research is defined by the following PICO elements:
P (Population): {P}
I (Intervention): {I}
C (Comparison): {C}
O (Outcome): {O}

## Reference Papers

You have already retrieved these related papers from PubMed:
{pubmed_reference_text}

## Task

Expand the literature search by completing the following 3 steps.

### Step 1 — Extract terms from reference papers
Provide three lists of query terms found in or implied by the reference papers above:

POPULATION_TERMS : terms describing the study population, target group, or condition being studied
INTERVENTION_TERMS : terms describing the intervention, program, technology, or independent variable
OUTCOME_TERMS : terms describing the measured outcomes, endpoints, or dependent variables

### Step 2 — Refine (keep only directly PICO-relevant terms)
Remove any term that is not directly relevant to the PICO elements of this research.
Provide three refined lists:

CORE_POPULATION   : ~5 refined population/condition terms
CORE_INTERVENTION : ~5 refined intervention/program terms
CORE_OUTCOME      : ~5 refined outcome/measurement terms

### Step 3 — Expand (synonyms, abbreviations, alternate forms)
For each core term, add:
1. Synonyms and alternative names / phrasing
2. Common abbreviations or their full forms
3. Elements obtained by splitting compound phrases

Provide three expanded lists:

EXPAND_POPULATION   : ~10 expanded population terms
EXPAND_INTERVENTION : ~10 expanded intervention terms
EXPAND_OUTCOME      : ~10 expanded outcome terms

## Reply Format
There must be no overlap between any two lists in the same position (e.g., CORE_* and EXPAND_*).
Output ONLY valid JSON:

{{
    "step 1": {{
        "POPULATION_TERMS":   ["term1", "term2", ...],
        "INTERVENTION_TERMS": ["term1", "term2", ...],
        "OUTCOME_TERMS":      ["term1", "term2", ...]
    }},
    "step 2": {{
        "CORE_POPULATION":   ["term1", "term2", ...],
        "CORE_INTERVENTION": ["term1", "term2", ...],
        "CORE_OUTCOME":      ["term1", "term2", ...]
    }},
    "step 3": {{
        "EXPAND_POPULATION":   ["term1", "term2", ...],
        "EXPAND_INTERVENTION": ["term1", "term2", ...],
        "EXPAND_OUTCOME":      ["term1", "term2", ...]
    }}
}}
"""
