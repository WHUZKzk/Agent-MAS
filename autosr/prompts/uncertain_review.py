"""
Stage 3 — Uncertain-paper review prompt.

A stronger model (Claude Sonnet) re-evaluates borderline papers that
Stage 2 could not confidently classify.  The review has access to the
full Stage 1 PICOS profile, Stage 2 reasoning chain, the original
title + abstract, and optionally user-uploaded full-text PDF content.
"""

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

UNCERTAIN_REVIEW_PROMPT = """\
# ROLE
You are a senior systematic review expert performing a careful secondary review
of a borderline study that could not be clearly classified in the initial screening.

# CONTEXT
This study was flagged as UNCERTAIN in the initial automated screening.
You have access to the original article information, the extracted PICOS profile,
the initial screening reasoning, and the review's eligibility criteria.
Your task is to make a final INCLUDE or EXCLUDE determination.

# REVIEW PICO DEFINITION
- P (Population):   {P}
- I (Intervention): {I}
- C (Comparison):   {C}
- O (Outcome):      {O}

# ELIGIBILITY CRITERIA
{criteria_json}

# CANDIDATE STUDY — ORIGINAL ARTICLE
Title: {title}
Abstract: {abstract}

# CANDIDATE STUDY — EXTRACTED PICOS PROFILE (from Stage 1)
- Population:    {study_P}
- Intervention:  {study_I}
- Comparison:    {study_C}
- Outcome:       {study_O}
- Study Design:  {study_S}
- Sample Size:   {study_sample_size}
- Duration:      {study_duration}

# INITIAL SCREENING REASONING (from Stage 2)
{stage2_reasoning_json}

# DIMENSIONS THAT CAUSED UNCERTAINTY
{uncertain_dimensions_detail}
{fulltext_section}\

# INSTRUCTIONS
1. Carefully review ALL available evidence, including the original abstract.
2. Focus especially on the dimensions that were marked UNCERTAIN — try to resolve them.
3. Consider whether the study COULD plausibly be relevant to this systematic review,
   even if not all information is explicitly stated in the abstract.
4. Apply the Cochrane principle: "When in doubt, INCLUDE."
   It is far better to include a borderline study for full-text review
   than to miss a potentially relevant one.
5. Only assign EXCLUDE if you find clear, specific evidence that the study
   does NOT match the review's Population or Intervention criteria.

# OUTPUT FORMAT
Call the function `submit_review` with your final determination.
"""

# ---------------------------------------------------------------------------
# Function-calling tool schema
# ---------------------------------------------------------------------------

_RESOLVED_ENUM = {"type": "string", "enum": ["MATCH", "MISMATCH", "STILL_UNCERTAIN"]}

UNCERTAIN_REVIEW_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_review",
        "description": "Submit the final review determination for an uncertain study.",
        "parameters": {
            "type": "object",
            "properties": {
                "review_reasoning": {
                    "type": "string",
                    "description": (
                        "Detailed reasoning for the final decision, "
                        "addressing each uncertain dimension."
                    ),
                },
                "resolved_dimensions": {
                    "type": "object",
                    "description": "Updated dimension assessments after review.",
                    "properties": {
                        "P": _RESOLVED_ENUM,
                        "I": _RESOLVED_ENUM,
                        "C": _RESOLVED_ENUM,
                        "O": _RESOLVED_ENUM,
                        "S": _RESOLVED_ENUM,
                    },
                    "required": ["P", "I", "C", "O", "S"],
                },
                "final_decision": {
                    "type": "string",
                    "enum": ["INCLUDE", "EXCLUDE"],
                    "description": "Final binary decision. No UNCERTAIN allowed.",
                },
                "confidence": {
                    "type": "string",
                    "enum": ["HIGH", "MEDIUM", "LOW"],
                    "description": "Confidence level in the final decision.",
                },
            },
            "required": [
                "review_reasoning",
                "resolved_dimensions",
                "final_decision",
                "confidence",
            ],
        },
    },
}
