"""
Stage 2a — PICOS matching criteria generation prompt.

Given a review's PICO definition and study-design requirement, the LLM
produces structured matching criteria (core / acceptable_variations /
exclusion_boundary) for each of the five PICOS dimensions.

This prompt is called ONCE per screening task (not per paper).
"""

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

CRITERIA_GENERATION_V2_PROMPT = """\
# ROLE
You are a systematic review methodologist with expertise in designing eligibility criteria.

# TASK
Based on the following PICO definition for a systematic review, generate specific matching
criteria for each PICOS dimension. These criteria will be used to systematically evaluate
whether candidate studies should be included.

# REVIEW PICO DEFINITION
- P (Population):   {P}
- I (Intervention): {I}
- C (Comparison):   {C}
- O (Outcome):      {O}
- Acceptable Study Designs: {study_design_description}

# INSTRUCTIONS
For each PICO dimension (P, I, C, O), provide three components:
1. **core**: The essential requirement that a study must address (the most important aspect).
2. **acceptable_variations**: Broader scope or related concepts that still qualify for inclusion.
3. **exclusion_boundary**: What clearly does NOT match and should lead to exclusion.

For the Study Design dimension (S), list acceptable and excluded designs.

Be specific and actionable. Avoid vague criteria. The criteria should enable consistent,
reproducible screening decisions.

# OUTPUT FORMAT
Call the function `submit_criteria` with the generated criteria.
"""

# ---------------------------------------------------------------------------
# Function-calling tool schema
# ---------------------------------------------------------------------------

_DIM_CRITERIA_SCHEMA = {
    "type": "object",
    "properties": {
        "core": {
            "type": "string",
            "description": "The essential requirement for this dimension.",
        },
        "acceptable_variations": {
            "type": "string",
            "description": "Broader scope or related concepts that still qualify.",
        },
        "exclusion_boundary": {
            "type": "string",
            "description": "What clearly does NOT match this dimension.",
        },
    },
    "required": ["core", "acceptable_variations", "exclusion_boundary"],
}

CRITERIA_GENERATION_V2_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_criteria",
        "description": "Submit the generated PICOS matching criteria for all five dimensions.",
        "parameters": {
            "type": "object",
            "properties": {
                "P_criteria": _DIM_CRITERIA_SCHEMA,
                "I_criteria": _DIM_CRITERIA_SCHEMA,
                "C_criteria": _DIM_CRITERIA_SCHEMA,
                "O_criteria": _DIM_CRITERIA_SCHEMA,
                "S_criteria": {
                    "type": "object",
                    "properties": {
                        "acceptable_designs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Study designs acceptable for inclusion.",
                        },
                        "excluded_designs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Study designs that should be excluded.",
                        },
                    },
                    "required": ["acceptable_designs", "excluded_designs"],
                },
            },
            "required": [
                "P_criteria",
                "I_criteria",
                "C_criteria",
                "O_criteria",
                "S_criteria",
            ],
        },
    },
}
