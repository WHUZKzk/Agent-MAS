"""
Stage 1 — PICOS structured extraction prompt.

Pure information extraction: the LLM reads title + abstract and extracts
Population, Intervention, Comparison, Outcome, Study design into a
structured JSON via function calling.  NO eligibility judgment is made here.
"""

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

PICOS_EXTRACTION_PROMPT = """\
# ROLE
You are a biomedical information extraction specialist.

# TASK
Extract the PICOS elements from the following research article's title and abstract.
This is a PURE EXTRACTION task — do NOT make any eligibility judgments or comparisons.
Simply extract what the article reports.

# ARTICLE
Title: {title}
Abstract: {abstract}

# INSTRUCTIONS
For each PICOS element, extract the relevant information DIRECTLY from the text.
- Be concise but complete.
- Use the article's own terminology.
- If information is not explicitly stated or cannot be determined, respond with "Not reported".
- For Study Design, classify into one of the predefined categories.

# OUTPUT FORMAT
Call the function `submit_picos` with the extracted information.
"""

# ---------------------------------------------------------------------------
# Function-calling tool schema
# ---------------------------------------------------------------------------

PICOS_EXTRACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_picos",
        "description": "Submit the extracted PICOS elements from the article.",
        "parameters": {
            "type": "object",
            "properties": {
                "P_population": {
                    "type": "string",
                    "description": (
                        "The study population / participants. "
                        "Include key demographics, conditions, and sample size if stated."
                    ),
                },
                "I_intervention": {
                    "type": "string",
                    "description": "The intervention or exposure being studied.",
                },
                "C_comparison": {
                    "type": "string",
                    "description": "The comparator or control group.",
                },
                "O_outcome": {
                    "type": "string",
                    "description": "The outcome measures reported.",
                },
                "S_study_design": {
                    "type": "string",
                    "enum": [
                        "RCT",
                        "Quasi-experimental",
                        "Cohort",
                        "Case-control",
                        "Cross-sectional",
                        "Before-after",
                        "Case series",
                        "Qualitative",
                        "Mixed methods",
                        "Other",
                        "Not reported",
                    ],
                    "description": "The study design type.",
                },
                "sample_size": {
                    "type": "string",
                    "description": "Total sample size or 'Not reported'.",
                },
                "duration": {
                    "type": "string",
                    "description": "Study duration or follow-up period, or 'Not reported'.",
                },
            },
            "required": [
                "P_population",
                "I_intervention",
                "C_comparison",
                "O_outcome",
                "S_study_design",
            ],
        },
    },
}
