# 06 — SCREENING STAGE SPEC

> **Dependencies:** You MUST have implemented the Core Engine, Skill Framework,
> and Search Stage before this module.
>
> **Purpose:** Implement the 6-node Screening DAG. Takes SearchOutput and
> produces a list of included PMIDs via heterogeneous dual-blind review with
> adjudication.

---

## 1. Stage Overview

**Input:** `SearchOutput` (from Search Stage)
**Output:** `ScreeningOutput` (see `02_SCHEMA_CONTRACT.md`)

This stage is the primary showcase for Innovation #3: Epistemic-Aware
Heterogeneous Dual-Adjudication. Two different base LLMs independently
screen each paper, and conflicts are resolved by a blinded adjudicator.

**Critical design principle:** LLMs are "reading comprehension bots" — they
answer binary questions with YES/NO/UNCERTAIN. The final INCLUDE/EXCLUDE
decision is made entirely by Python code (Hard Node).

---

## 2. DAG Definition

```
Nodes: [2.1, 2.2, 2.3, 2.4, 2.5, 2.6]
Entry: 2.1
Terminal: 2.6

Edges:
  2.1 → 2.2  (unconditional)
  2.2 → 2.3  (unconditional)
  2.3 → 2.4  (unconditional)
  2.4 → 2.5  (guard: state.has_conflicts == True)
  2.4 → 2.6  (guard: state.has_conflicts == False)
  2.5 → 2.4  (guard: state.adjudication_complete == True)
```

**Loop behavior:** Node 2.4 and 2.5 form a conditional loop.
After adjudication, the logic gate re-evaluates with the adjudicated answers.
Maximum loop iterations: 2 (one adjudication round should resolve all
conflicts for a single paper; a second pass is a safety net).

---

## 3. Processing Model

The Screening DAG is instantiated ONCE, but its core nodes (2.3, 2.4, 2.5)
execute in a PER-PAPER loop controlled by the ScreeningPipeline:

```
# In ScreeningPipeline:
criteria = run_node_2_1(review_config)     # once
filtered_papers = run_node_2_2(all_papers) # once

for paper in filtered_papers:              # sequential loop
    reviewer_outputs = run_node_2_3(paper, criteria)   # may parallelize A/B
    decision = run_node_2_4(reviewer_outputs)
    if decision.has_conflicts:
        adjudicated = run_node_2_5(decision.conflicts, paper)
        decision = run_node_2_4_again(adjudicated)     # re-evaluate
    store_decision(paper.pmid, decision)

final_output = run_node_2_6(all_decisions) # once
```

---

## 4. Node Specifications

### Node 2.1: Criteria Binarization & Zero-Shot Reflexion

| Field | Value |
|-------|-------|
| **Type** | Soft Node |
| **Skill** | `screening.criteria_binarization` |
| **Input** | `review_config.pico`, `review_config.title` |
| **Output** | `ScreeningCriteria` |
| **Constraints** | See below |

**Process:**

```
Round 1 (Generation):
  - LLM converts the PICO into binary questions.
  - Each question assigned to a dimension (P, I, C, O).
  - Each question answerable by YES / NO / UNCERTAIN.

Hard Validation:
  - Parse JSON output against ScreeningCriteria schema.
  - Assert: len(questions where dimension == "P") >= 1
  - Assert: len(questions where dimension == "I") >= 1
  - Assert: len(questions where dimension == "C") >= 1
  - Assert: len(questions where dimension == "O") >= 1
  - If any assertion fails → go to Round 2.
  - If all pass → proceed.

Round 2 (Reflexion, only if Round 1 failed validation):
  - Feed the Round 1 output back to the LLM with critique prompt:
    "Your criteria are missing coverage for dimension(s): {missing}.
     Rewrite the complete criteria set with better coverage."
  - Re-validate.
  - If Round 2 still fails → HALT pipeline with error.
    (Criteria are foundational; we cannot proceed without valid criteria.)

Maximum rounds: 2.
```

| Constraint | Level |
|------------|-------|
| ≥1 question per PICO dimension | MUST |
| SHOULD have 2–3 questions per dimension for granularity | SHOULD |
| Questions MUST be binary (YES/NO/UNCERTAIN answerable) | MUST |
| Questions MUST be mutually exclusive within a dimension | SHOULD |
| Reflexion loop MUST exit after at most 2 rounds | MUST |

### Node 2.2: Deterministic Pre-Filtering

| Field | Value |
|-------|-------|
| **Type** | Hard Node |
| **Input** | `SearchOutput.papers` |
| **Output** | Filtered paper list + exclusion records |

**Logic:**

```
EXCLUDE_PUBLICATION_TYPES = {
    "Review", "Systematic Review", "Meta-Analysis",
    "Editorial", "Letter", "Comment", "Case Reports",
    "Published Erratum", "Retracted Publication"
}

EXCLUDE_MESH_PATTERNS = ["Animals"] without ["Humans"]
# i.e., exclude if MeSH contains "Animals" but NOT "Humans"

for paper in all_papers:
    pub_types = set(paper.publication_types)
    if pub_types & EXCLUDE_PUBLICATION_TYPES:
        decision = "EXCLUDED_BY_METADATA"
        reason = f"Publication type: {pub_types & EXCLUDE_PUBLICATION_TYPES}"
    elif "Animals" in paper.mesh_terms and "Humans" not in paper.mesh_terms:
        decision = "EXCLUDED_BY_METADATA"
        reason = "Animal study without human subjects"
    else:
        decision = "PASS_TO_SCREENING"

    record_decision(paper.pmid, decision, reason)
```

### Node 2.3: Heterogeneous Dual-Blind Screening

| Field | Value |
|-------|-------|
| **Type** | Soft Node (×2, parallelizable) |
| **Skill** | `screening.reviewer_screening` |
| **Input** | Paper title + abstract + `ScreeningCriteria` |
| **Output** | Two `ReviewerOutput` objects (one per reviewer) |

**Execution model:**

```
For each paper (in the sequential outer loop):

    # Both reviewers answer the EXACT SAME questions.
    # Heterogeneity comes from different base models.

    reviewer_a = ReviewerAdjudicatorAgent(
        role="reviewer",
        model_id=model_registry.get_default("reviewer_a")
    )
    reviewer_b = ReviewerAdjudicatorAgent(
        role="reviewer",
        model_id=model_registry.get_default("reviewer_b")
    )

    # MAY run in parallel via asyncio.gather:
    output_a, output_b = await asyncio.gather(
        run_soft_node(reviewer_a, paper, criteria),
        run_soft_node(reviewer_b, paper, criteria)
    )

    # Each reviewer sees:
    # - System: screening skill role + guidelines
    # - User: paper title, abstract, criteria questions
    # Each reviewer returns:
    # - Dict[question_id, QuestionAnswer]
    # Reviewers MUST NOT output INCLUDE/EXCLUDE.
```

| Constraint | Level |
|------------|-------|
| Reviewer A and B MUST use different base models | MUST |
| Both reviewers answer ALL questions (not split by dimension) | MUST |
| Temperature = 0.0 for both reviewers | MUST |
| Reviewers MUST NOT produce INCLUDE/EXCLUDE verdicts | MUST |

### Node 2.4: Symbolic Logic Gate (The Decider)

| Field | Value |
|-------|-------|
| **Type** | Hard Node |
| **Input** | `ReviewerOutput` from A and B |
| **Output** | `ScreeningDecision` (per paper) |

**Decision rules (applied independently to each reviewer's answers):**

```
def compute_reviewer_status(answers: Dict[str, QuestionAnswer]) -> str:
    answer_values = [a.answer for a in answers.values()]

    if "NO" in answer_values:
        return "EXCLUDE"
    elif all(a == "YES" for a in answer_values):
        return "INCLUDE"
    else:
        # No NOs, but at least one UNCERTAIN
        return "UNCERTAIN_FOR_FULL_TEXT"
        # Treat as INCLUDE for this stage — paper proceeds.

status_a = compute_reviewer_status(output_a.answers)
status_b = compute_reviewer_status(output_b.answers)
```

**Agreement logic:**

```
# Concordance groups:
#   INCLUDE and UNCERTAIN_FOR_FULL_TEXT are in the same group (both pass).
#   EXCLUDE is its own group.

PASS_STATUSES = {"INCLUDE", "UNCERTAIN_FOR_FULL_TEXT"}

if status_a in PASS_STATUSES and status_b in PASS_STATUSES:
    final_status = "INCLUDED"
    has_conflicts = False

elif status_a == "EXCLUDE" and status_b == "EXCLUDE":
    final_status = "EXCLUDED"
    has_conflicts = False

else:
    # One says EXCLUDE, the other says INCLUDE/UNCERTAIN
    # → Identify the specific conflicting questions
    has_conflicts = True
    conflicts = []
    for q_id in all_question_ids:
        ans_a = output_a.answers[q_id].answer
        ans_b = output_b.answers[q_id].answer
        if ans_a != ans_b:
            conflicts.append(ConflictRecord(
                question_id=q_id,
                reasoning_1=output_a.answers[q_id].reasoning,  # Blinded
                reasoning_2=output_b.answers[q_id].reasoning,  # Blinded
            ))
    # Note: reasoning_1 and reasoning_2 are anonymous.
    # We do NOT label which came from Reviewer A vs B.
```

### Node 2.5: Epistemic Adjudication Sandbox

| Field | Value |
|-------|-------|
| **Type** | Soft Node |
| **Skill** | `screening.adjudicator_resolution` |
| **Input** | Conflict records + paper abstract |
| **Output** | Updated `ConflictRecord` objects with adjudicated answers |

**For EACH conflict in the conflict queue:**

```
adjudicator = ReviewerAdjudicatorAgent(
    role="adjudicator",
    model_id=model_registry.get_default("adjudicator")
)

# DCR Context (via ContextManager.mount):
# System message: Adjudicator role + screening guidelines
# User message: ONLY the abstract + the specific conflicting question
#               + Reasoning 1 + Reasoning 2 (blinded, no model identity)

adjudicated_answer = run_soft_node(adjudicator, conflict_context)
conflict.adjudicated_answer = adjudicated_answer.answer
conflict.adjudication_reasoning = adjudicated_answer.reasoning
```

**Blinding protocol:**

| Constraint | Level |
|------------|-------|
| Adjudicator MUST NOT see "Reviewer A" or "Reviewer B" labels | MUST |
| Use generic labels: "Reasoning 1" and "Reasoning 2" | MUST |
| Adjudicator MUST NOT see the reviewers' overall decisions | MUST |
| Adjudicator sees ONLY the abstract + one conflicting question at a time | MUST |

**After adjudication:** The adjudicated answer replaces the conflicting
answers, and the paper is sent back to Node 2.4 for re-evaluation.
In the re-evaluation, for each adjudicated question, use the adjudicator's
answer for BOTH reviewers' entries, then recompute status.

### Node 2.6: PRISMA Reporting & Consistency Evaluation

| Field | Value |
|-------|-------|
| **Type** | Hard Node |
| **Input** | All `ScreeningDecision` objects |
| **Output** | Final `ScreeningOutput` |

**Logic:**

```
1. Cohen's Kappa Calculation:
   - Compute on the INITIAL (pre-adjudication) decisions of Reviewer A and B.
   - Binary classification: PASS (INCLUDE or UNCERTAIN) vs EXCLUDE.
   - Use standard Kappa formula:
     κ = (p_o - p_e) / (1 - p_e)
     where p_o = observed agreement, p_e = expected agreement by chance.

2. Exclusion Reason Tallying:
   - For each excluded paper, identify which question(s) triggered
     the NO answer.
   - Map question_id to dimension name for human-readable reasons.
   - Example: "Excluded: Population mismatch (Q_P1)"

3. PRISMA Numbers:
   - total_screened
   - excluded_by_metadata (from Node 2.2)
   - excluded_by_dual_review
   - sent_to_adjudication
   - included_after_screening

4. Produce final ScreeningOutput:
   - included_pmids: list of PMIDs with final_status == "INCLUDED"
   - decisions: full dict of all decisions
   - cohens_kappa: float
   - prisma_numbers
```

---

## 5. Stage-Level Constraints

| Rule | Level |
|------|-------|
| Node 2.1 (Criteria Binarization) runs ONCE per review. | MUST |
| Node 2.2 (Pre-Filtering) runs ONCE on the full paper set. | MUST |
| Nodes 2.3–2.5 run in a PER-PAPER sequential loop. | MUST |
| Within each paper, Reviewer A and B MAY run in parallel. | MAY |
| Node 2.6 runs ONCE after all papers are processed. | MUST |
| Cohen's Kappa is computed on initial (pre-adjudication) decisions. | MUST |
| Papers with final status UNCERTAIN_FOR_FULL_TEXT are treated as INCLUDED. | MUST |

---

## 6. Skill Files to Create

| Skill ID | File |
|----------|------|
| `screening.criteria_binarization` | `src/skills/screening/criteria_binarization.yaml` |
| `screening.reviewer_screening` | `src/skills/screening/reviewer_screening.yaml` |
| `screening.adjudicator_resolution` | `src/skills/screening/adjudicator_resolution.yaml` |

Example YAML content for these skills is provided in `04_SKILL_FRAMEWORK.md` §3.

---

## 7. Implementation Checklist

1. [ ] `src/skills/screening/criteria_binarization.yaml`
2. [ ] `src/skills/screening/reviewer_screening.yaml`
3. [ ] `src/skills/screening/adjudicator_resolution.yaml`
4. [ ] `src/guidelines/cochrane_ch7_screening.md` (stub)
5. [ ] `src/stages/screening_pipeline.py`:
   - [ ] Node 2.1: Criteria Binarization + Reflexion loop
   - [ ] Node 2.2: Metadata Pre-Filter
   - [ ] Node 2.3: Dual-Blind Screening (with async parallel option)
   - [ ] Node 2.4: Symbolic Logic Gate
   - [ ] Node 2.5: Adjudication Sandbox
   - [ ] Node 2.6: PRISMA Reporting + Kappa
6. [ ] DAG declaration for Screening
7. [ ] Integration test: run full screening on a small sample (5 papers)
