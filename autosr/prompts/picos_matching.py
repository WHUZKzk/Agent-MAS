"""
Stage 2b — PICOS dimension matching prompt.

For each candidate paper, the LLM compares its extracted PICOS profile
against the review's matching criteria with chain-of-thought reasoning
on every dimension, then outputs MATCH / MISMATCH / UNCERTAIN per dimension.
"""

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

PICOS_MATCHING_PROMPT = """\
# ROLE
You are a systematic review screener evaluating a candidate study for inclusion.

# TASK
Compare the candidate study's PICOS profile against the review's eligibility criteria.
Evaluate each dimension with EXPLICIT reasoning, then make an overall decision.

# REVIEW ELIGIBILITY CRITERIA
{criteria_json}

# CANDIDATE STUDY PICOS PROFILE
- Population:    {study_P}
- Intervention:  {study_I}
- Comparison:    {study_C}
- Outcome:       {study_O}
- Study Design:  {study_S}
- Sample Size:   {study_sample_size}
- Duration:      {study_duration}

# ORIGINAL ARTICLE INFO
Title: {title}

# INSTRUCTIONS
For EACH PICOS dimension, you must:
1. State what the study reports (from the PICOS profile above).
2. State what the review requires (from the eligibility criteria above).
3. Reason about whether they match, citing specific evidence.
4. Assign: MATCH / MISMATCH / UNCERTAIN.

CRITICAL GUIDELINES:
- "Not reported" in the study does NOT mean MISMATCH — it means UNCERTAIN.
  The absence of information is NOT evidence of non-eligibility.
- Only assign MISMATCH when there is a CLEAR, EXPLICIT contradiction between
  the study and the criteria.
- When in doubt, assign UNCERTAIN rather than MISMATCH.
- For Population and Intervention: be particularly careful — these are the most
  important dimensions. A wrong MISMATCH here means losing a relevant study.

# OUTPUT FORMAT
Call the function `submit_screening` with your evaluation.
"""

# ---------------------------------------------------------------------------
# Function-calling tool schema
# ---------------------------------------------------------------------------

_DIMENSION_ENUM = {"type": "string", "enum": ["MATCH", "MISMATCH", "UNCERTAIN"]}

PICOS_MATCHING_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_screening",
        "description": "Submit the dimension-by-dimension screening evaluation.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "object",
                    "description": "Chain-of-thought reasoning for each PICOS dimension.",
                    "properties": {
                        "P": {"type": "string"},
                        "I": {"type": "string"},
                        "C": {"type": "string"},
                        "O": {"type": "string"},
                        "S": {"type": "string"},
                    },
                    "required": ["P", "I", "C", "O", "S"],
                },
                "dimensions": {
                    "type": "object",
                    "description": "Decision for each dimension.",
                    "properties": {
                        "P": _DIMENSION_ENUM,
                        "I": _DIMENSION_ENUM,
                        "C": _DIMENSION_ENUM,
                        "O": _DIMENSION_ENUM,
                        "S": _DIMENSION_ENUM,
                    },
                    "required": ["P", "I", "C", "O", "S"],
                },
                "overall_decision": {
                    "type": "string",
                    "enum": ["INCLUDE", "EXCLUDE", "UNCERTAIN"],
                    "description": "Overall screening decision.",
                },
            },
            "required": ["reasoning", "dimensions", "overall_decision"],
        },
    },
}
