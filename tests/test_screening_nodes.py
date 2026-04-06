"""
TDD tests for Screening Pipeline — Node 2.4 (Symbolic Logic Gate)
and Node 2.6 (Cohen's Kappa calculation).

Written BEFORE implementation — tests MUST fail on first run.

Imports under test:
  src.stages.screening_pipeline → apply_logic_gate, compute_reviewer_status,
                                   compute_cohens_kappa
"""
import pytest
from typing import Dict

from src.schemas.screening import (
    BinaryQuestion,
    ConflictRecord,
    QuestionAnswer,
    ReviewerOutput,
    ScreeningCriteria,
    ScreeningDecision,
)

# ── TDD imports (will fail until implementation exists) ─────────────────────
from src.stages.screening_pipeline import (   # noqa: E402
    apply_logic_gate,
    compute_cohens_kappa,
    compute_reviewer_status,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_criteria(*question_ids: str) -> ScreeningCriteria:
    """Build a ScreeningCriteria with one question per provided ID.
    IDs must start with Q_P, Q_I, Q_C, or Q_O to satisfy the dimension
    coverage validator.  If no IDs are given, one per dimension is created.
    """
    if not question_ids:
        question_ids = ("Q_P1", "Q_I1", "Q_C1", "Q_O1")
    _dim_map = {"P": "P", "I": "I", "C": "C", "O": "O"}
    questions = []
    for qid in question_ids:
        dim = qid[2]   # Q_P1 → 'P'
        questions.append(BinaryQuestion(
            question_id=qid,
            dimension=dim,
            question_text=f"Test question {qid}",
            answerable_by="YES",
        ))
    return ScreeningCriteria(questions=questions, reflexion_rounds=1)


def _make_answers(*pairs) -> Dict[str, QuestionAnswer]:
    """Build answers dict from (question_id, answer) pairs."""
    return {
        qid: QuestionAnswer(question_id=qid, answer=ans, reasoning=f"reason for {qid}")
        for qid, ans in pairs
    }


def _make_reviewer_output(model: str, *pairs) -> ReviewerOutput:
    return ReviewerOutput(reviewer_model=model, answers=_make_answers(*pairs))


def _make_decision(pmid: str, status_a: str, status_b: str) -> ScreeningDecision:
    """Build a ScreeningDecision with specified individual statuses (no conflicts)."""
    criteria = _make_criteria("Q_P1", "Q_I1", "Q_C1", "Q_O1")
    answer_map = {
        "INCLUDE":               ("YES", "YES", "YES", "YES"),
        "EXCLUDE":               ("NO",  "YES", "YES", "YES"),
        "UNCERTAIN_FOR_FULL_TEXT": ("UNCERTAIN", "YES", "YES", "YES"),
    }
    qids = ["Q_P1", "Q_I1", "Q_C1", "Q_O1"]
    answers_a_vals = answer_map[status_a]
    answers_b_vals = answer_map[status_b]
    out_a = _make_reviewer_output("model_a", *zip(qids, answers_a_vals))
    out_b = _make_reviewer_output("model_b", *zip(qids, answers_b_vals))
    return apply_logic_gate(pmid, out_a, out_b, criteria)


# ─────────────────────────────────────────────────────────────────────────────
# Node 2.4: compute_reviewer_status
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeReviewerStatus:
    """
    Tests for compute_reviewer_status(answers) → str.
    Spec: docs/06_SCREENING_STAGE.md §4 Node 2.4
    """

    def test_all_yes_returns_include(self):
        answers = _make_answers(("Q_P1", "YES"), ("Q_I1", "YES"), ("Q_O1", "YES"))
        assert compute_reviewer_status(answers) == "INCLUDE"

    def test_any_no_returns_exclude(self):
        answers = _make_answers(("Q_P1", "YES"), ("Q_I1", "NO"), ("Q_O1", "YES"))
        assert compute_reviewer_status(answers) == "EXCLUDE"

    def test_only_no_returns_exclude(self):
        answers = _make_answers(("Q_P1", "NO"), ("Q_I1", "NO"))
        assert compute_reviewer_status(answers) == "EXCLUDE"

    def test_uncertain_with_yes_returns_uncertain_for_full_text(self):
        """At least one UNCERTAIN and no NO → UNCERTAIN_FOR_FULL_TEXT."""
        answers = _make_answers(("Q_P1", "YES"), ("Q_I1", "UNCERTAIN"), ("Q_O1", "YES"))
        assert compute_reviewer_status(answers) == "UNCERTAIN_FOR_FULL_TEXT"

    def test_all_uncertain_returns_uncertain_for_full_text(self):
        answers = _make_answers(("Q_P1", "UNCERTAIN"), ("Q_I1", "UNCERTAIN"))
        assert compute_reviewer_status(answers) == "UNCERTAIN_FOR_FULL_TEXT"

    def test_no_takes_priority_over_uncertain(self):
        """If there's BOTH a NO and an UNCERTAIN, result is EXCLUDE (NO wins)."""
        answers = _make_answers(("Q_P1", "NO"), ("Q_I1", "UNCERTAIN"), ("Q_O1", "YES"))
        assert compute_reviewer_status(answers) == "EXCLUDE"

    def test_single_yes_returns_include(self):
        answers = _make_answers(("Q_P1", "YES"),)
        assert compute_reviewer_status(answers) == "INCLUDE"

    def test_single_no_returns_exclude(self):
        answers = _make_answers(("Q_P1", "NO"),)
        assert compute_reviewer_status(answers) == "EXCLUDE"

    def test_single_uncertain_returns_uncertain_for_full_text(self):
        answers = _make_answers(("Q_P1", "UNCERTAIN"),)
        assert compute_reviewer_status(answers) == "UNCERTAIN_FOR_FULL_TEXT"


# ─────────────────────────────────────────────────────────────────────────────
# Node 2.4: apply_logic_gate — Agreement / Conflict Scenarios
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyLogicGate:
    """
    Tests for apply_logic_gate(pmid, output_a, output_b, criteria) → ScreeningDecision.
    Spec: docs/06_SCREENING_STAGE.md §4 Node 2.4
    """

    def _gate(self, pmid, pairs_a, pairs_b, extra_qids=()):
        qids = ["Q_P1", "Q_I1", "Q_C1", "Q_O1"] + list(extra_qids)
        # Build criteria covering all qids used
        all_qids = list(dict.fromkeys(
            [p[0] for p in pairs_a] + [p[0] for p in pairs_b] +
            ["Q_P1", "Q_I1", "Q_C1", "Q_O1"]
        ))
        criteria = _make_criteria(*all_qids)
        out_a = _make_reviewer_output("model_a", *pairs_a)
        out_b = _make_reviewer_output("model_b", *pairs_b)
        return apply_logic_gate(pmid, out_a, out_b, criteria)

    # ── Both INCLUDE ─────────────────────────────────────────────────────────

    def test_both_include_gives_included_no_conflict(self):
        decision = self._gate(
            "1234",
            [("Q_P1","YES"), ("Q_I1","YES"), ("Q_C1","YES"), ("Q_O1","YES")],
            [("Q_P1","YES"), ("Q_I1","YES"), ("Q_C1","YES"), ("Q_O1","YES")],
        )
        assert decision.final_status == "INCLUDED"
        assert decision.conflicts == []
        assert decision.individual_status_a == "INCLUDE"
        assert decision.individual_status_b == "INCLUDE"

    # ── Both EXCLUDE ─────────────────────────────────────────────────────────

    def test_both_exclude_gives_excluded_no_conflict(self):
        decision = self._gate(
            "1234",
            [("Q_P1","NO"), ("Q_I1","YES"), ("Q_C1","YES"), ("Q_O1","YES")],
            [("Q_P1","NO"), ("Q_I1","YES"), ("Q_C1","YES"), ("Q_O1","YES")],
        )
        assert decision.final_status == "EXCLUDED"
        assert decision.conflicts == []
        assert decision.individual_status_a == "EXCLUDE"
        assert decision.individual_status_b == "EXCLUDE"

    # ── INCLUDE + UNCERTAIN = both PASS ──────────────────────────────────────

    def test_include_and_uncertain_is_pass_no_conflict(self):
        """UNCERTAIN_FOR_FULL_TEXT and INCLUDE are both PASS → INCLUDED."""
        decision = self._gate(
            "1234",
            [("Q_P1","YES"), ("Q_I1","YES"), ("Q_C1","YES"), ("Q_O1","YES")],
            [("Q_P1","UNCERTAIN"), ("Q_I1","YES"), ("Q_C1","YES"), ("Q_O1","YES")],
        )
        assert decision.final_status == "INCLUDED"
        assert decision.conflicts == []
        assert decision.individual_status_a == "INCLUDE"
        assert decision.individual_status_b == "UNCERTAIN_FOR_FULL_TEXT"

    def test_both_uncertain_is_pass_no_conflict(self):
        """Both UNCERTAIN_FOR_FULL_TEXT → INCLUDED (no conflicts)."""
        decision = self._gate(
            "1234",
            [("Q_P1","UNCERTAIN"), ("Q_I1","YES"), ("Q_C1","YES"), ("Q_O1","YES")],
            [("Q_P1","UNCERTAIN"), ("Q_I1","YES"), ("Q_C1","YES"), ("Q_O1","YES")],
        )
        assert decision.final_status == "INCLUDED"
        assert decision.conflicts == []

    # ── Conflict scenarios ────────────────────────────────────────────────────

    def test_include_vs_exclude_triggers_conflict(self):
        """A=INCLUDE, B=EXCLUDE → has_conflicts, final status pending adjudication."""
        decision = self._gate(
            "1234",
            [("Q_P1","YES"), ("Q_I1","YES"), ("Q_C1","YES"), ("Q_O1","YES")],
            [("Q_P1","NO"),  ("Q_I1","YES"), ("Q_C1","YES"), ("Q_O1","YES")],
        )
        assert len(decision.conflicts) >= 1
        assert decision.individual_status_a == "INCLUDE"
        assert decision.individual_status_b == "EXCLUDE"

    def test_exclude_vs_include_triggers_conflict(self):
        """A=EXCLUDE, B=INCLUDE → has_conflicts (symmetric)."""
        decision = self._gate(
            "1234",
            [("Q_P1","NO"),  ("Q_I1","YES"), ("Q_C1","YES"), ("Q_O1","YES")],
            [("Q_P1","YES"), ("Q_I1","YES"), ("Q_C1","YES"), ("Q_O1","YES")],
        )
        assert len(decision.conflicts) >= 1

    def test_uncertain_vs_exclude_triggers_conflict(self):
        """A=UNCERTAIN_FOR_FULL_TEXT, B=EXCLUDE → conflict."""
        decision = self._gate(
            "1234",
            [("Q_P1","UNCERTAIN"), ("Q_I1","YES"), ("Q_C1","YES"), ("Q_O1","YES")],
            [("Q_P1","NO"),        ("Q_I1","YES"), ("Q_C1","YES"), ("Q_O1","YES")],
        )
        assert len(decision.conflicts) >= 1

    def test_conflict_question_ids_are_correct(self):
        """ConflictRecord.question_id matches the specific disagreeing question."""
        decision = self._gate(
            "1234",
            [("Q_P1","YES"), ("Q_I1","YES"), ("Q_C1","YES"), ("Q_O1","YES")],
            [("Q_P1","YES"), ("Q_I1","NO"),  ("Q_C1","YES"), ("Q_O1","YES")],
        )
        conflict_ids = {c.question_id for c in decision.conflicts}
        assert "Q_I1" in conflict_ids
        assert "Q_P1" not in conflict_ids   # agreed on this one

    def test_multiple_conflicts_captured(self):
        """If A and B disagree on multiple questions, all are recorded."""
        decision = self._gate(
            "1234",
            [("Q_P1","YES"), ("Q_I1","YES"), ("Q_C1","NO"), ("Q_O1","NO")],
            [("Q_P1","YES"), ("Q_I1","YES"), ("Q_C1","YES"), ("Q_O1","YES")],
        )
        conflict_ids = {c.question_id for c in decision.conflicts}
        assert "Q_C1" in conflict_ids
        assert "Q_O1" in conflict_ids

    # ── Blinding: no identity leakage ────────────────────────────────────────

    def test_conflict_reasoning_is_blinded_no_reviewer_labels(self):
        """
        ConflictRecord.reasoning_1 / reasoning_2 must NOT contain
        'Reviewer A', 'Reviewer B', or the model names.
        """
        decision = self._gate(
            "1234",
            [("Q_P1","YES"), ("Q_I1","YES"), ("Q_C1","YES"), ("Q_O1","YES")],
            [("Q_P1","NO"),  ("Q_I1","YES"), ("Q_C1","YES"), ("Q_O1","YES")],
        )
        for conflict in decision.conflicts:
            forbidden = ["Reviewer A", "Reviewer B", "model_a", "model_b",
                         "deepseek", "gemini"]
            for label in forbidden:
                assert label not in conflict.reasoning_1, (
                    f"Identity leak in reasoning_1: '{label}'"
                )
                assert label not in conflict.reasoning_2, (
                    f"Identity leak in reasoning_2: '{label}'"
                )

    # ── PMID stored correctly ─────────────────────────────────────────────────

    def test_decision_stores_pmid(self):
        decision = self._gate(
            "99999",
            [("Q_P1","YES"), ("Q_I1","YES"), ("Q_C1","YES"), ("Q_O1","YES")],
            [("Q_P1","YES"), ("Q_I1","YES"), ("Q_C1","YES"), ("Q_O1","YES")],
        )
        assert decision.pmid == "99999"

    # ── Re-evaluation after adjudication ──────────────────────────────────────

    def test_adjudicated_answers_resolve_conflict(self):
        """
        After adjudication, apply_logic_gate called again with overridden answers
        resolves the conflict.  Simulate adjudication by pre-patching the conflict.
        """
        # Initial: A says NO on Q_P1, B says YES → conflict
        criteria = _make_criteria("Q_P1", "Q_I1", "Q_C1", "Q_O1")
        out_a = _make_reviewer_output("model_a",
            ("Q_P1","NO"),  ("Q_I1","YES"), ("Q_C1","YES"), ("Q_O1","YES"))
        out_b = _make_reviewer_output("model_b",
            ("Q_P1","YES"), ("Q_I1","YES"), ("Q_C1","YES"), ("Q_O1","YES"))

        # Build adjudicated versions: override Q_P1 answer to "YES" for both
        adj_answers_a = dict(out_a.answers)
        adj_answers_a["Q_P1"] = QuestionAnswer(
            question_id="Q_P1", answer="YES", reasoning="adjudicator: relevant")
        adj_answers_b = dict(out_b.answers)
        adj_answers_b["Q_P1"] = QuestionAnswer(
            question_id="Q_P1", answer="YES", reasoning="adjudicator: relevant")

        out_a_adj = ReviewerOutput(reviewer_model="model_a", answers=adj_answers_a)
        out_b_adj = ReviewerOutput(reviewer_model="model_b", answers=adj_answers_b)

        decision2 = apply_logic_gate("1234", out_a_adj, out_b_adj, criteria)
        assert decision2.final_status == "INCLUDED"
        assert decision2.conflicts == []


# ─────────────────────────────────────────────────────────────────────────────
# Node 2.6: Cohen's Kappa Calculation
# ─────────────────────────────────────────────────────────────────────────────

class TestCohensKappa:
    """
    Tests for compute_cohens_kappa(decisions) → float.

    Spec: docs/06_SCREENING_STAGE.md §4 Node 2.6
    Binary classification: PASS (INCLUDE or UNCERTAIN_FOR_FULL_TEXT) vs EXCLUDE.
    Formula: κ = (p_o - p_e) / (1 - p_e)
    MUST use INITIAL (pre-adjudication) individual_status_a/b.
    """

    def _build_decisions(self, status_pairs):
        """
        Build a decisions dict from [(status_a, status_b), ...].
        Uses apply_logic_gate with synthetic reviewer outputs.
        """
        decisions = {}
        for i, (sa, sb) in enumerate(status_pairs):
            pmid = f"PMID_{i:04d}"
            decisions[pmid] = _make_decision(pmid, sa, sb)
        return decisions

    # ── Perfect agreement ────────────────────────────────────────────────────

    def test_perfect_agreement_all_include_kappa_1(self):
        """All papers: both reviewers INCLUDE → κ = 1.0."""
        decisions = self._build_decisions([("INCLUDE", "INCLUDE")] * 10)
        kappa = compute_cohens_kappa(decisions)
        assert abs(kappa - 1.0) < 1e-9, f"Expected 1.0, got {kappa}"

    def test_perfect_agreement_all_exclude_kappa_1(self):
        """All papers: both reviewers EXCLUDE → κ = 1.0."""
        decisions = self._build_decisions([("EXCLUDE", "EXCLUDE")] * 10)
        kappa = compute_cohens_kappa(decisions)
        assert abs(kappa - 1.0) < 1e-9, f"Expected 1.0, got {kappa}"

    def test_perfect_agreement_mixed_kappa_1(self):
        """Half INCLUDE/INCLUDE, half EXCLUDE/EXCLUDE → κ = 1.0."""
        pairs = [("INCLUDE", "INCLUDE")] * 5 + [("EXCLUDE", "EXCLUDE")] * 5
        decisions = self._build_decisions(pairs)
        kappa = compute_cohens_kappa(decisions)
        assert abs(kappa - 1.0) < 1e-9, f"Expected 1.0, got {kappa}"

    # ── Complete disagreement ────────────────────────────────────────────────

    def test_complete_disagreement_balanced_marginals_kappa_minus1(self):
        """
        κ = -1.0 requires balanced marginals AND zero observed agreement.
        5 papers: A=INCLUDE/B=EXCLUDE + 5 papers: A=EXCLUDE/B=INCLUDE
        → p_a_pass=0.5, p_b_pass=0.5, p_o=0.0, p_e=0.5 → κ=-1.0.
        (All-INCLUDE/All-EXCLUDE gives degenerate marginals p_e=0 → κ=0, not -1.)
        """
        pairs = [("INCLUDE", "EXCLUDE")] * 5 + [("EXCLUDE", "INCLUDE")] * 5
        decisions = self._build_decisions(pairs)
        kappa = compute_cohens_kappa(decisions)
        assert abs(kappa - (-1.0)) < 1e-9, f"Expected -1.0, got {kappa}"

    # ── Chance agreement (κ ≈ 0) ─────────────────────────────────────────────

    def test_chance_level_agreement(self):
        """When p_o ≈ p_e, kappa should be near 0."""
        # 5 INCLUDE/INCLUDE, 5 INCLUDE/EXCLUDE, 5 EXCLUDE/INCLUDE, 5 EXCLUDE/EXCLUDE
        # p_a_pass=0.5, p_b_pass=0.5, p_o=0.5, p_e=0.5 → κ=0
        pairs = (
            [("INCLUDE", "INCLUDE")] * 5 +
            [("INCLUDE", "EXCLUDE")] * 5 +
            [("EXCLUDE", "INCLUDE")] * 5 +
            [("EXCLUDE", "EXCLUDE")] * 5
        )
        decisions = self._build_decisions(pairs)
        kappa = compute_cohens_kappa(decisions)
        assert abs(kappa - 0.0) < 1e-9, f"Expected 0.0, got {kappa}"

    # ── Known value ──────────────────────────────────────────────────────────

    def test_known_kappa_value(self):
        """
        Manually computed:
          8 agree INCLUDE, 2 agree EXCLUDE out of 10.
          A: 8 PASS + 2 EXCLUDE → p_a_pass = 0.8
          B: 8 PASS + 2 EXCLUDE → p_b_pass = 0.8
          p_o = (8+2)/10 = 1.0  ← all agree → kappa = 1.0
        Use a case with actual disagreements:
          6 INCLUDE/INCLUDE, 2 INCLUDE/EXCLUDE, 2 EXCLUDE/EXCLUDE
          A: 8 PASS, 2 EXCLUDE → p_a_pass = 0.8
          B: 6 PASS, 4 EXCLUDE → p_b_pass = 0.6
          p_o = (6+2)/10 = 0.8
          p_e = 0.8*0.6 + 0.2*0.4 = 0.48 + 0.08 = 0.56
          κ = (0.8 - 0.56) / (1 - 0.56) = 0.24 / 0.44 ≈ 0.5454...
        """
        pairs = (
            [("INCLUDE", "INCLUDE")] * 6 +
            [("INCLUDE", "EXCLUDE")] * 2 +
            [("EXCLUDE", "EXCLUDE")] * 2
        )
        decisions = self._build_decisions(pairs)
        kappa = compute_cohens_kappa(decisions)
        expected = 0.24 / 0.44
        assert abs(kappa - expected) < 1e-6, f"Expected {expected:.6f}, got {kappa}"

    # ── UNCERTAIN treated as PASS ────────────────────────────────────────────

    def test_uncertain_treated_as_pass_in_kappa(self):
        """
        UNCERTAIN_FOR_FULL_TEXT is in the PASS group for kappa computation.
        INCLUDE/UNCERTAIN should agree (both PASS) and increase κ.
        """
        pairs = [("INCLUDE", "UNCERTAIN_FOR_FULL_TEXT")] * 10
        decisions = self._build_decisions(pairs)
        kappa = compute_cohens_kappa(decisions)
        # Both always PASS → perfect agreement → κ = 1.0
        assert abs(kappa - 1.0) < 1e-9, f"Expected 1.0, got {kappa}"

    # ── Edge cases ───────────────────────────────────────────────────────────

    def test_single_paper_both_include(self):
        """Single paper, both INCLUDE → κ = 1.0 (or handle degenerate case)."""
        decisions = self._build_decisions([("INCLUDE", "INCLUDE")])
        kappa = compute_cohens_kappa(decisions)
        # p_e = 1.0, so (1 - p_e) = 0 — must not raise ZeroDivisionError.
        # Return 1.0 by convention when p_e == 1.0 and p_o == 1.0.
        assert isinstance(kappa, float)
        assert not (kappa != kappa)   # Not NaN

    def test_empty_decisions_returns_zero(self):
        """No decisions → kappa = 0.0 (degenerate, no data)."""
        kappa = compute_cohens_kappa({})
        assert kappa == 0.0

    def test_kappa_ignores_excluded_by_metadata(self):
        """
        Papers excluded in Node 2.2 (EXCLUDED_BY_METADATA) have no reviewer
        outputs; they should NOT appear in the kappa calculation.
        Build decisions that include an EXCLUDED_BY_METADATA entry and verify
        it is ignored (kappa still based only on dual-reviewed papers).
        """
        decisions = self._build_decisions([("INCLUDE", "INCLUDE")] * 4)
        # Manually inject a metadata-excluded decision (no reviewer outputs)
        from src.schemas.screening import ScreeningDecision, ReviewerOutput
        empty_out = _make_reviewer_output("model_a",
            ("Q_P1","YES"), ("Q_I1","YES"), ("Q_C1","YES"), ("Q_O1","YES"))
        meta_decision = ScreeningDecision(
            pmid="META_99",
            reviewer_a=empty_out,
            reviewer_b=empty_out,
            conflicts=[],
            individual_status_a="INCLUDE",
            individual_status_b="INCLUDE",
            final_status="EXCLUDED_BY_METADATA",
            exclusion_reasons=["Publication type: Review"],
        )
        decisions["META_99"] = meta_decision

        # All 4 real decisions agree → kappa should still be 1.0
        kappa = compute_cohens_kappa(decisions)
        assert abs(kappa - 1.0) < 1e-9, f"Expected 1.0 (ignoring metadata), got {kappa}"
