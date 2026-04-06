"""
Screening Pipeline — 6-node DAG implementation.

Spec: docs/06_SCREENING_STAGE.md

Nodes:
  2.1 (Soft)       Criteria Binarization + Zero-Shot Reflexion
  2.2 (Hard)       Deterministic Pre-Filtering
  2.3 (Soft×2)     Heterogeneous Dual-Blind Screening (A=deepseek, B=gemini)
  2.4 (Hard)       Symbolic Logic Gate
  2.5 (Soft)       Epistemic Adjudication Sandbox (adjudicator=qwen, blinded)
  2.6 (Hard)       PRISMA Reporting + Cohen's Kappa

Public module-level functions (tested independently):
  compute_reviewer_status(answers) → str
  apply_logic_gate(pmid, output_a, output_b, criteria) → ScreeningDecision
  compute_cohens_kappa(decisions) → float
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from src.clients.pubmed_client import PubMedClient
from src.engine.agents import ExecutorAgent, ReviewerAdjudicatorAgent
from src.engine.context_manager import ContextManager, MountedContext
from src.engine.dag import DAGRunner
from src.engine.model_registry import ModelRegistry
from src.schemas.common import (
    DAGDefinition, EdgeDefinition, NodeDefinition, PaperMetadata, ReviewConfig,
)
from src.schemas.screening import (
    BinaryQuestion,
    ConflictRecord,
    PRISMAScreeningData,
    QuestionAnswer,
    ReviewerOutput,
    ScreeningCriteria,
    ScreeningDecision,
    ScreeningOutput,
)
from src.schemas.search import SearchOutput

logger = logging.getLogger("autosr.screening_pipeline")

_EXCLUDE_PUBLICATION_TYPES = {
    "Review", "Systematic Review", "Meta-Analysis",
    "Editorial", "Letter", "Comment", "Case Reports",
    "Published Erratum", "Retracted Publication",
}
_PASS_STATUSES = {"INCLUDE", "UNCERTAIN_FOR_FULL_TEXT"}


# ─────────────────────────────────────────────────────────────────────────────
# Module-level pure functions (testable independently)
# ─────────────────────────────────────────────────────────────────────────────

def compute_reviewer_status(answers: Dict[str, QuestionAnswer]) -> str:
    """
    Derive a reviewer's overall status from their question answers.

    Rules (spec §4 Node 2.4):
      - Any NO  → EXCLUDE
      - All YES → INCLUDE
      - No NOs, at least one UNCERTAIN → UNCERTAIN_FOR_FULL_TEXT
    """
    answer_values = [a.answer for a in answers.values()]
    if "NO" in answer_values:
        return "EXCLUDE"
    if all(v == "YES" for v in answer_values):
        return "INCLUDE"
    return "UNCERTAIN_FOR_FULL_TEXT"


def apply_logic_gate(
    pmid: str,
    output_a: ReviewerOutput,
    output_b: ReviewerOutput,
    criteria: ScreeningCriteria,
) -> ScreeningDecision:
    """
    Symbolic Logic Gate (Node 2.4) — pure Python, no LLM.

    Derives final_status and collects ConflictRecords.
    Blinding: conflict reasoning is anonymous (reasoning_1/2 only).

    Returns a ScreeningDecision with:
    - individual_status_a / b
    - final_status:  INCLUDED | EXCLUDED  (conflicts → caller re-evaluates)
    - conflicts:     [] if no conflicts, otherwise list of ConflictRecord
    """
    status_a = compute_reviewer_status(output_a.answers)
    status_b = compute_reviewer_status(output_b.answers)

    if status_a in _PASS_STATUSES and status_b in _PASS_STATUSES:
        final_status = "INCLUDED"
        conflicts: List[ConflictRecord] = []

    elif status_a == "EXCLUDE" and status_b == "EXCLUDE":
        final_status = "EXCLUDED"
        conflicts = []

    else:
        # Conflict: one says PASS, the other EXCLUDE.
        # Identify the specific disagreeing questions (blinded).
        conflicts = []
        all_qids = {q.question_id for q in criteria.questions}
        for q_id in sorted(all_qids):
            ans_a = output_a.answers.get(q_id)
            ans_b = output_b.answers.get(q_id)
            if ans_a is None or ans_b is None:
                continue
            if ans_a.answer != ans_b.answer:
                conflicts.append(ConflictRecord(
                    question_id=q_id,
                    # Blinded: we do NOT label which reasoning came from A or B.
                    reasoning_1=ans_a.reasoning,
                    reasoning_2=ans_b.reasoning,
                ))
        # Provisional status while conflicts await adjudication.
        # Caller (Node 2.5) will re-call apply_logic_gate after patching answers.
        final_status = "EXCLUDED"   # conservative default until adjudicated

    return ScreeningDecision(
        pmid=pmid,
        reviewer_a=output_a,
        reviewer_b=output_b,
        conflicts=conflicts,
        individual_status_a=status_a,
        individual_status_b=status_b,
        final_status=final_status,
        exclusion_reasons=_build_exclusion_reasons(output_a, output_b, criteria),
    )


def _build_exclusion_reasons(
    output_a: ReviewerOutput,
    output_b: ReviewerOutput,
    criteria: ScreeningCriteria,
) -> List[str]:
    """Collect human-readable exclusion reasons from NO answers."""
    qid_to_dim = {q.question_id: q.dimension for q in criteria.questions}
    reasons: List[str] = []
    seen: set = set()
    for output in (output_a, output_b):
        for q_id, qa in output.answers.items():
            if qa.answer == "NO" and q_id not in seen:
                dim = qid_to_dim.get(q_id, "?")
                reasons.append(f"Criterion not met: {dim} dimension ({q_id})")
                seen.add(q_id)
    return reasons


def compute_cohens_kappa(decisions: Dict[str, ScreeningDecision]) -> float:
    """
    Compute Cohen's Kappa on the INITIAL (pre-adjudication) reviewer decisions.

    Binary classification: PASS (INCLUDE or UNCERTAIN_FOR_FULL_TEXT) vs EXCLUDE.
    Only dual-reviewed papers are included (EXCLUDED_BY_METADATA papers are skipped).

    Formula:
        κ = (p_o - p_e) / (1 - p_e)
        p_o = observed agreement rate
        p_e = expected agreement by chance
        p_e = p_a_pass * p_b_pass + (1 - p_a_pass) * (1 - p_b_pass)

    Edge cases:
        - No dual-reviewed decisions → return 0.0
        - p_e == 1.0 and p_o == 1.0 → return 1.0 (perfect agreement, degenerate)
        - p_e == 1.0 and p_o != 1.0 → return 0.0 (undefined)
    """
    # Only include papers that went through dual review (not metadata-excluded)
    dual_reviewed = [
        d for d in decisions.values()
        if d.final_status != "EXCLUDED_BY_METADATA"
    ]
    n = len(dual_reviewed)
    if n == 0:
        return 0.0

    def _is_pass(status: str) -> bool:
        return status in _PASS_STATUSES

    agree_count = sum(
        1 for d in dual_reviewed
        if _is_pass(d.individual_status_a) == _is_pass(d.individual_status_b)
    )
    p_o = agree_count / n

    p_a_pass = sum(1 for d in dual_reviewed if _is_pass(d.individual_status_a)) / n
    p_b_pass = sum(1 for d in dual_reviewed if _is_pass(d.individual_status_b)) / n

    p_e = p_a_pass * p_b_pass + (1 - p_a_pass) * (1 - p_b_pass)

    denom = 1 - p_e
    if abs(denom) < 1e-12:
        # Degenerate: all raters always agree (or always disagree) by chance
        return 1.0 if abs(p_o - 1.0) < 1e-12 else 0.0

    return (p_o - p_e) / denom


# ─────────────────────────────────────────────────────────────────────────────
# Helper: criteria → text for LLM context
# ─────────────────────────────────────────────────────────────────────────────

def _criteria_to_text(criteria: ScreeningCriteria) -> str:
    """Convert ScreeningCriteria to a human-readable string for LLM prompts."""
    lines = []
    for q in criteria.questions:
        lines.append(f"[{q.question_id}] ({q.dimension}) {q.question_text}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Node 2.1: Criteria Binarization
# ─────────────────────────────────────────────────────────────────────────────

class _Node21_CriteriaBinarization:
    """
    Soft Node 2.1 — Criteria Binarization + Zero-Shot Reflexion.

    The DAGRunner mounts the skill before calling us; we access the mounted
    context via context_manager._current_mount.

    Reflexion loop (≤2 rounds):
      Round 1: generate criteria.
      Validate: ≥1 question per PICO dimension.
      If fails: Round 2 with critique.
      If Round 2 fails: raise RuntimeError (pipeline halts).
    """

    def __init__(self, context_manager: ContextManager, agent: ExecutorAgent) -> None:
        self._cm = context_manager
        self._agent = agent

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        review_config: ReviewConfig = state["review_config"]

        for round_num in range(1, 3):
            mounted: MountedContext = self._cm._current_mount
            raw = self._agent.call(mounted)

            try:
                criteria = ScreeningCriteria.model_validate_json(raw)
                logger.info(
                    "[Node 2.1] Round %d: %d questions across dims %s",
                    round_num, len(criteria.questions),
                    {q.dimension for q in criteria.questions},
                )
                # Validation passed
                return {**state, "screening_criteria": criteria}
            except Exception as exc:
                logger.warning("[Node 2.1] Round %d parse/validation error: %s", round_num, exc)
                if round_num == 2:
                    raise RuntimeError(
                        f"[Node 2.1] Criteria binarization failed after 2 rounds: {exc}"
                    ) from exc
                # Reflexion: re-mount with critique context
                # (The DAGRunner has already mounted; we force a re-mount with extra state)
                missing = {"P", "I", "C", "O"}
                try:
                    # Try to get partial parse to find which dims are present
                    partial = json.loads(raw)
                    present = {q.get("dimension", "?") for q in partial.get("questions", [])}
                    missing = {"P", "I", "C", "O"} - present
                except Exception:
                    pass

                critique = (
                    f"Your previous output failed validation. "
                    f"Missing dimension coverage: {missing}. "
                    f"Rewrite the complete criteria set ensuring at least 1 question "
                    f"per dimension (P, I, C, O)."
                )
                reflexion_state = {**state, "reflexion_critique": critique}
                self._cm.mount("screening.criteria_binarization", reflexion_state)

        # Should never reach here
        raise RuntimeError("[Node 2.1] Criteria binarization exhausted retries.")


# ─────────────────────────────────────────────────────────────────────────────
# Node 2.2: Deterministic Pre-Filtering
# ─────────────────────────────────────────────────────────────────────────────

class _Node22_PreFilter:
    """
    Hard Node 2.2 — Deterministic metadata pre-filtering.

    Excludes: reviews, editorials, animal-only studies, etc.
    Papers that pass go to Node 2.3 for dual-blind screening.
    """

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        search_output: SearchOutput = state["search_output"]
        all_papers: Dict[str, PaperMetadata] = search_output.papers

        papers_to_screen: List[PaperMetadata] = []
        metadata_decisions: Dict[str, ScreeningDecision] = {}
        excluded_by_metadata = 0

        for pmid, paper in all_papers.items():
            pub_types = set(paper.publication_types)
            excluded_types = pub_types & _EXCLUDE_PUBLICATION_TYPES
            is_animal_only = (
                "Animals" in paper.mesh_terms and "Humans" not in paper.mesh_terms
            )

            if excluded_types:
                reason = f"Publication type: {', '.join(sorted(excluded_types))}"
                metadata_decisions[pmid] = _make_metadata_excluded_decision(
                    pmid, reason)
                excluded_by_metadata += 1
            elif is_animal_only:
                metadata_decisions[pmid] = _make_metadata_excluded_decision(
                    pmid, "Animal study without human subjects")
                excluded_by_metadata += 1
            else:
                papers_to_screen.append(paper)

        logger.info(
            "[Node 2.2] %d papers: %d excluded by metadata, %d proceed to screening.",
            len(all_papers), excluded_by_metadata, len(papers_to_screen),
        )

        return {
            **state,
            "papers_to_screen": papers_to_screen,
            "metadata_decisions": metadata_decisions,
            "excluded_by_metadata_count": excluded_by_metadata,
        }


def _make_metadata_excluded_decision(pmid: str, reason: str) -> ScreeningDecision:
    """Build a placeholder ScreeningDecision for metadata-excluded papers."""
    # Create minimal reviewer outputs (no real answers — just placeholders)
    empty_out = ReviewerOutput(reviewer_model="none", answers={})
    return ScreeningDecision(
        pmid=pmid,
        reviewer_a=empty_out,
        reviewer_b=empty_out,
        conflicts=[],
        individual_status_a="EXCLUDE",
        individual_status_b="EXCLUDE",
        final_status="EXCLUDED_BY_METADATA",
        exclusion_reasons=[reason],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Node 2.3: Heterogeneous Dual-Blind Screening
# ─────────────────────────────────────────────────────────────────────────────

class _Node23_DualBlindScreening:
    """
    Soft Node 2.3 — Runs Reviewer A (deepseek) and Reviewer B (gemini) in parallel.

    This node is called once per paper (in ScreeningPipeline._screen_paper).
    The context_manager must be mounted for 'screening.reviewer_screening' before
    this node is called (DAGRunner handles this).

    Model constraints (spec §4 Node 2.3):
    - Reviewer A: reviewer_a model (deepseek)
    - Reviewer B: reviewer_b model (gemini)
    - Both at temperature = 0.0
    - Neither outputs INCLUDE/EXCLUDE
    """

    def __init__(
        self,
        context_manager: ContextManager,
        reviewer_a: ReviewerAdjudicatorAgent,
        reviewer_b: ReviewerAdjudicatorAgent,
    ) -> None:
        self._cm = context_manager
        self._reviewer_a = reviewer_a
        self._reviewer_b = reviewer_b

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        paper: PaperMetadata = state["current_paper"]
        criteria: ScreeningCriteria = state["screening_criteria"]

        # Build text representation of criteria for LLM context
        criteria_text = _criteria_to_text(criteria)
        state_with_criteria = {**state, "screening_criteria_text": criteria_text}

        # Mount context for reviewer A, call, then mount again for reviewer B.
        # (Parallelizing two synchronous LLM calls via threading would require
        # a thread pool; here we call sequentially to keep the event loop clean.
        # Both use the SAME skill / mounted context structure.)

        # --- Reviewer A ---
        self._cm.mount("screening.reviewer_screening", state_with_criteria)
        mounted_a: MountedContext = self._cm._current_mount
        raw_a = self._reviewer_a.call(mounted_a)
        self._cm.unmount()

        # --- Reviewer B ---
        self._cm.mount("screening.reviewer_screening", state_with_criteria)
        mounted_b: MountedContext = self._cm._current_mount
        raw_b = self._reviewer_b.call(mounted_b)
        self._cm.unmount()

        try:
            output_a = ReviewerOutput.model_validate_json(raw_a)
        except Exception as exc:
            logger.error("[Node 2.3] Reviewer A parse error: %s", exc)
            output_a = _fallback_reviewer_output(
                self._reviewer_a.model_id, criteria, "UNCERTAIN")

        try:
            output_b = ReviewerOutput.model_validate_json(raw_b)
        except Exception as exc:
            logger.error("[Node 2.3] Reviewer B parse error: %s", exc)
            output_b = _fallback_reviewer_output(
                self._reviewer_b.model_id, criteria, "UNCERTAIN")

        return {**state, "reviewer_output_a": output_a, "reviewer_output_b": output_b}


def _fallback_reviewer_output(
    model_id: str,
    criteria: ScreeningCriteria,
    default_answer: str,
) -> ReviewerOutput:
    """Build a conservative fallback output when LLM response cannot be parsed."""
    answers = {
        q.question_id: QuestionAnswer(
            question_id=q.question_id,
            answer=default_answer,
            reasoning="Parse error — conservative fallback.",
        )
        for q in criteria.questions
    }
    return ReviewerOutput(reviewer_model=model_id, answers=answers)


# ─────────────────────────────────────────────────────────────────────────────
# Node 2.4: Symbolic Logic Gate
# ─────────────────────────────────────────────────────────────────────────────

class _Node24_LogicGate:
    """
    Hard Node 2.4 — Symbolic Logic Gate.

    Sets state["has_conflicts"] to control the conditional DAG edge.
    Sets state["adjudication_complete"] = False (reset for new paper).
    """

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pmid: str = state["current_paper"].pmid
        output_a: ReviewerOutput = state["reviewer_output_a"]
        output_b: ReviewerOutput = state["reviewer_output_b"]
        criteria: ScreeningCriteria = state["screening_criteria"]

        decision = apply_logic_gate(pmid, output_a, output_b, criteria)
        has_conflicts = len(decision.conflicts) > 0

        logger.info(
            "[Node 2.4] pmid=%s  A=%s  B=%s  final=%s  conflicts=%d",
            pmid, decision.individual_status_a, decision.individual_status_b,
            decision.final_status, len(decision.conflicts),
        )

        return {
            **state,
            "current_decision": decision,
            "has_conflicts": has_conflicts,
            "adjudication_complete": False,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node 2.5: Epistemic Adjudication Sandbox
# ─────────────────────────────────────────────────────────────────────────────

class _Node25_AdjudicationSandbox:
    """
    Soft Node 2.5 — Blinded adjudication of each conflict question.

    Blinding protocol (spec §4 Node 2.5):
    - Uses generic labels 'Reasoning 1' and 'Reasoning 2'.
    - Adjudicator sees ONLY abstract + one question + two anonymous reasonings.
    - Adjudicator MUST NOT see model identities or overall reviewer verdicts.

    After adjudication, patches reviewer outputs with the adjudicator's answer
    and sets state["adjudication_complete"] = True so Node 2.4 re-evaluates.
    """

    def __init__(
        self,
        context_manager: ContextManager,
        adjudicator: ReviewerAdjudicatorAgent,
        criteria: ScreeningCriteria,
    ) -> None:
        self._cm = context_manager
        self._adjudicator = adjudicator
        self._qid_to_question: Dict[str, BinaryQuestion] = {
            q.question_id: q for q in criteria.questions
        }

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        decision: ScreeningDecision = state["current_decision"]
        paper: PaperMetadata = state["current_paper"]
        output_a: ReviewerOutput = state["reviewer_output_a"]
        output_b: ReviewerOutput = state["reviewer_output_b"]

        # Adjudicate each conflict question independently
        adjudicated_answers: Dict[str, QuestionAnswer] = {}

        for conflict in decision.conflicts:
            q_id = conflict.question_id
            question_obj = self._qid_to_question.get(q_id)
            question_text = question_obj.question_text if question_obj else q_id

            # Build blinded context state for this conflict
            conflict_state = {
                **state,
                "conflict_paper_abstract": paper.abstract,
                "conflict_question_text": f"[{q_id}] {question_text}",
                "conflict_reasoning_1": conflict.reasoning_1,
                "conflict_reasoning_2": conflict.reasoning_2,
            }

            self._cm.mount("screening.adjudicator_resolution", conflict_state)
            mounted: MountedContext = self._cm._current_mount
            raw = self._adjudicator.call(mounted)
            self._cm.unmount()

            try:
                adj_answer = QuestionAnswer.model_validate_json(raw)
            except Exception as exc:
                logger.error("[Node 2.5] Adjudicator parse error for %s: %s", q_id, exc)
                adj_answer = QuestionAnswer(
                    question_id=q_id,
                    answer="UNCERTAIN",
                    reasoning="Adjudication parse error — defaulting to UNCERTAIN.",
                )

            adjudicated_answers[q_id] = adj_answer
            logger.info("[Node 2.5] Adjudicated %s → %s", q_id, adj_answer.answer)

        # Patch reviewer outputs: for adjudicated questions, replace both A and B
        # with the adjudicator's answer (spec §4 Node 2.5 re-evaluation logic).
        new_answers_a = dict(output_a.answers)
        new_answers_b = dict(output_b.answers)
        for q_id, adj_qa in adjudicated_answers.items():
            new_answers_a[q_id] = adj_qa
            new_answers_b[q_id] = adj_qa

        patched_a = ReviewerOutput(
            reviewer_model=output_a.reviewer_model,
            answers=new_answers_a,
        )
        patched_b = ReviewerOutput(
            reviewer_model=output_b.reviewer_model,
            answers=new_answers_b,
        )

        return {
            **state,
            "reviewer_output_a": patched_a,
            "reviewer_output_b": patched_b,
            "adjudication_complete": True,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node 2.6: PRISMA Reporting + Cohen's Kappa
# ─────────────────────────────────────────────────────────────────────────────

class _Node26_PRISMAReport:
    """
    Hard Node 2.6 — Aggregate all decisions, compute kappa, produce ScreeningOutput.
    """

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        all_decisions: Dict[str, ScreeningDecision] = state["all_decisions"]
        criteria: ScreeningCriteria = state["screening_criteria"]

        included_pmids = [
            pmid for pmid, d in all_decisions.items()
            if d.final_status == "INCLUDED"
        ]

        # Cohen's Kappa (pre-adjudication, dual-reviewed only)
        kappa = compute_cohens_kappa(all_decisions)

        # PRISMA numbers
        excluded_by_metadata = sum(
            1 for d in all_decisions.values()
            if d.final_status == "EXCLUDED_BY_METADATA"
        )
        excluded_by_review = sum(
            1 for d in all_decisions.values()
            if d.final_status == "EXCLUDED"
        )
        sent_to_adj = sum(
            1 for d in all_decisions.values()
            if len(d.conflicts) > 0
        )

        # Exclusion reason tallying
        reason_counts: Dict[str, int] = {}
        for d in all_decisions.values():
            for reason in d.exclusion_reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

        prisma = PRISMAScreeningData(
            total_screened=len(all_decisions),
            excluded_by_metadata=excluded_by_metadata,
            excluded_by_dual_review=excluded_by_review,
            sent_to_adjudication=sent_to_adj,
            included_after_screening=len(included_pmids),
            exclusion_reason_counts=reason_counts,
        )

        screening_output = ScreeningOutput(
            criteria=criteria,
            decisions=all_decisions,
            included_pmids=included_pmids,
            cohens_kappa=kappa,
            prisma_numbers=prisma,
        )

        logger.info(
            "[Node 2.6] Screening complete: %d included, %d excluded (metadata=%d, review=%d), "
            "adj=%d, κ=%.3f",
            len(included_pmids), excluded_by_review + excluded_by_metadata,
            excluded_by_metadata, excluded_by_review, sent_to_adj, kappa,
        )

        return {**state, "screening_output": screening_output}


# ─────────────────────────────────────────────────────────────────────────────
# DAG Definition
# ─────────────────────────────────────────────────────────────────────────────
#
# The screening DAG is used inside ScreeningPipeline._screen_paper() for the
# per-paper loop (nodes 2.3–2.5). Nodes 2.1, 2.2, and 2.6 run once outside
# the loop and are called directly.
#
# Per-paper DAG:
#   s2_3 → s2_4 → s2_5 (guard: has_conflicts==True)  → back to s2_4
#                ↓ (guard: has_conflicts==False)
#               s2_end  [terminal placeholder — pipeline reads current_decision]

PAPER_SCREENING_DAG = DAGDefinition(
    dag_id="paper_screening",
    entry_node="s2_3",
    terminal_nodes=["s2_4_final"],
    nodes=[
        NodeDefinition(
            node_id="s2_3", node_type="soft",
            skill_id="screening.reviewer_screening",
            implementation="stages.screening_pipeline._Node23_DualBlindScreening",
            description="Dual-blind screening by Reviewer A (deepseek) and Reviewer B (gemini)",
        ),
        NodeDefinition(
            node_id="s2_4", node_type="hard",
            implementation="stages.screening_pipeline._Node24_LogicGate",
            description="Symbolic Logic Gate — derive final status or flag conflicts",
        ),
        NodeDefinition(
            node_id="s2_5", node_type="soft",
            skill_id="screening.adjudicator_resolution",
            implementation="stages.screening_pipeline._Node25_AdjudicationSandbox",
            description="Blinded adjudication of conflicting questions (qwen adjudicator)",
        ),
        NodeDefinition(
            node_id="s2_4_final", node_type="hard",
            implementation="stages.screening_pipeline._Node24_LogicGate",
            description="Logic Gate re-evaluation after adjudication (terminal)",
        ),
    ],
    edges=[
        EdgeDefinition(from_node="s2_3",  to_node="s2_4"),
        # 2.4 → 2.5 when conflicts exist; 2.4 → terminal when no conflicts
        EdgeDefinition(from_node="s2_4",  to_node="s2_5",
                       guard='state.get("has_conflicts") == True'),
        EdgeDefinition(from_node="s2_4",  to_node="s2_4_final",
                       guard='state.get("has_conflicts") == False'),
        # 2.5 → back to 2.4 (adjudication_complete=True → re-evaluate)
        EdgeDefinition(from_node="s2_5",  to_node="s2_4_final",
                       guard='state.get("adjudication_complete") == True'),
    ],
)


# ─────────────────────────────────────────────────────────────────────────────
# ScreeningPipeline
# ─────────────────────────────────────────────────────────────────────────────

class ScreeningPipeline:
    """
    Implements the 6-node Screening DAG.

    Processing model (spec §3):
      - Node 2.1 (criteria binarization): once per review.
      - Node 2.2 (pre-filter): once on full paper set.
      - Nodes 2.3–2.5 (dual-blind screening + adjudication): per paper.
      - Node 2.6 (PRISMA + kappa): once after all papers processed.

    Usage:
        pipeline = ScreeningPipeline(review_config, context_manager,
                                     model_registry, search_output)
        screening_output = pipeline.run()
    """

    def __init__(
        self,
        review_config: ReviewConfig,
        context_manager: ContextManager,
        model_registry: ModelRegistry,
        search_output: SearchOutput,
        progress_callback: Optional[Any] = None,
    ) -> None:
        self._review_config = review_config
        self._cm = context_manager
        self._search_output = search_output
        self._progress_callback = progress_callback

        # Executor for Node 2.1 (criteria binarization)
        exec_cfg = model_registry.get_default("executor")
        exec_name = model_registry.default_name("executor")
        self._executor = ExecutorAgent(model_id=exec_name, model_config=exec_cfg)

        # Reviewer A: deepseek (reviewer_a)
        ra_cfg = model_registry.get_default("reviewer_a")
        ra_name = model_registry.default_name("reviewer_a")
        self._reviewer_a = ReviewerAdjudicatorAgent(
            model_id=ra_name, model_config=ra_cfg, role="reviewer")

        # Reviewer B: gemini (reviewer_b)
        rb_cfg = model_registry.get_default("reviewer_b")
        rb_name = model_registry.default_name("reviewer_b")
        self._reviewer_b = ReviewerAdjudicatorAgent(
            model_id=rb_name, model_config=rb_cfg, role="reviewer")

        # Adjudicator: qwen
        adj_cfg = model_registry.get_default("adjudicator")
        adj_name = model_registry.default_name("adjudicator")
        self._adjudicator = ReviewerAdjudicatorAgent(
            model_id=adj_name, model_config=adj_cfg, role="adjudicator")

    def run(self) -> ScreeningOutput:
        """Execute the full Screening pipeline and return ScreeningOutput."""
        base_state: Dict[str, Any] = {
            "review_config": self._review_config,
            "search_output": self._search_output,
        }

        # ── Node 2.1: Criteria Binarization (once) ───────────────────────────
        node21 = _Node21_CriteriaBinarization(self._cm, self._executor)
        self._cm.mount("screening.criteria_binarization", base_state)
        state_after_21 = node21(base_state)
        self._cm.unmount()
        criteria: ScreeningCriteria = state_after_21["screening_criteria"]

        # ── Node 2.2: Pre-Filtering (once) ───────────────────────────────────
        node22 = _Node22_PreFilter()
        state_after_22 = node22(state_after_21)
        papers_to_screen: List[PaperMetadata] = state_after_22["papers_to_screen"]
        all_decisions: Dict[str, ScreeningDecision] = dict(
            state_after_22["metadata_decisions"])

        # ── Nodes 2.3–2.5: Per-paper screening loop ──────────────────────────
        node23 = _Node23_DualBlindScreening(
            self._cm, self._reviewer_a, self._reviewer_b)
        node24 = _Node24_LogicGate()
        node25 = _Node25_AdjudicationSandbox(self._cm, self._adjudicator, criteria)
        # Terminal node 2.4 — same logic, just re-used as final gate
        node24_final = _Node24_LogicGate()

        node_registry = {
            "s2_3": node23,
            "s2_4": node24,
            "s2_5": node25,
            "s2_4_final": node24_final,
        }

        total_papers = len(papers_to_screen)
        for idx, paper in enumerate(papers_to_screen):
            if self._progress_callback:
                try:
                    self._progress_callback({
                        "type": "log",
                        "stage": "paper_screening",
                        "node_id": "s2_3",
                        "item_id": f"PMID:{paper.pmid}",
                        "progress": {"current": idx + 1, "total": total_papers},
                        "message": f"Screening paper {idx + 1}/{total_papers}: PMID {paper.pmid}",
                        "timestamp": "",
                    })
                except Exception:
                    pass
            paper_state = {
                **state_after_22,
                "current_paper": paper,
            }
            runner = DAGRunner(
                dag=PAPER_SCREENING_DAG,
                context_manager=self._cm,
                node_registry=node_registry,
                max_iterations=2,
                progress_callback=self._progress_callback,
            )
            final_paper_state = runner.run(paper_state)
            decision: ScreeningDecision = final_paper_state["current_decision"]
            all_decisions[paper.pmid] = decision
            logger.info(
                "[Screening] pmid=%s → %s", paper.pmid, decision.final_status)

        # ── Node 2.6: PRISMA Reporting (once) ────────────────────────────────
        node26 = _Node26_PRISMAReport()
        final_state = node26({
            **state_after_22,
            "all_decisions": all_decisions,
            "screening_criteria": criteria,
        })

        return final_state["screening_output"]
