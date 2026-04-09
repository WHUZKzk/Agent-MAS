"""
Screening criteria generation prompt.
Adapted from TrialMind's screen_criteria.py — kept nearly identical,
only minor wording polish for clarity.
"""

SCREENING_CRITERIA_GENERATION = """\
You are a systematic review specialist. You are conducting a systematic review and meta-analysis.
The research is defined by the following PICO elements:
P (Population): {P}
I (Intervention): {I}
C (Comparison): {C}
O (Outcome): {O}

## Task
Design eligibility criteria for selecting studies into this meta-analysis by completing the 3 steps below.

### Step 1
Based on the PRISMA guidelines and the PICO elements above, identify five eligibility criteria
for studies to be included in the meta-analysis. Provide a brief rationale for each.

ELIGIBILITY_ANALYSIS: your items and rationale here …

### Step 2
Create {num_title_criteria} binary yes/no questions that allow you to select or reject studies
based on their **title alone**.
- A "YES" answer means the study meets the criterion.
- A "NO" answer means it does not.
- The information needed to answer must be readily identifiable from the title.

TITLE_CRITERIA n: …

### Step 3
Create {num_abstract_criteria} binary yes/no questions to further filter studies based on their
**abstract content**.
- Same YES/NO convention as above.
- The information needed will be more detailed and found within the abstract.

CONTENT_CRITERIA n: …

## Reply Format
Output ONLY valid JSON:

{{
    "ELIGIBILITY_ANALYSIS": ["rationale 1", "rationale 2", ...],
    "TITLE_CRITERIA":   ["criterion 1", "criterion 2", ...],
    "CONTENT_CRITERIA": ["criterion 1", "criterion 2", ...]
}}
"""
