"""
Integration test — end-to-end pipeline state passing and checkpointing.

TDD: Written BEFORE src/orchestrator.py exists.
Must fail (ImportError) on first run, then pass after implementation.

Tests:
  1. Search → Screening → Extraction state passing (correct objects forwarded).
  2. Checkpoint files created after each stage.
  3. Orchestrator resumes from checkpoint (skips completed stages).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest

from src.schemas.common import (
    AppState, Checkpoint, PICODefinition, ReviewConfig,
)
from src.schemas.screening import (
    BinaryQuestion, PRISMAScreeningData, QuestionAnswer, ReviewerOutput,
    ScreeningCriteria, ScreeningDecision, ScreeningOutput,
)
from src.schemas.search import (
    PICOTerm, PICOTermSet, PRISMASearchData, QueryRecord, SearchOutput,
)
from src.schemas.common import PaperMetadata

# TDD import — will fail until src/orchestrator.py is implemented
from src.orchestrator import SystematicReviewOrchestrator  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: minimal valid schema instances
# ─────────────────────────────────────────────────────────────────────────────

def make_review_config() -> ReviewConfig:
    return ReviewConfig(
        pmid="TEST001",
        title="Integration Test Review",
        abstract="Abstract for integration testing purposes.",
        pico=PICODefinition(
            P="Healthy adults",
            I="Exercise intervention",
            C="No exercise control",
            O="Physical fitness measures",
        ),
        target_characteristics=["Age", "N"],
        target_outcomes=["BMI"],
    )


def make_pico_term(original: str = "exercise") -> PICOTerm:
    return PICOTerm(original=original, status="not_found", search_field="[tiab]")


def make_search_output() -> SearchOutput:
    paper = PaperMetadata(
        pmid="P001",
        title="Dummy Paper Title",
        abstract="Dummy abstract text.",
        publication_types=["Journal Article"],
        mesh_terms=["Exercise"],
        fetch_date=datetime.now(timezone.utc),
    )
    term = make_pico_term()
    return SearchOutput(
        pmids=["P001"],
        papers={"P001": paper},
        query_history=[
            QueryRecord(query_string='"exercise"[tiab]', stage="initial", result_count=1)
        ],
        pico_terms=PICOTermSet(P=[term], I=[term], C=[term], O=[term]),
        prisma_numbers=PRISMASearchData(
            initial_query_results=1,
            augmented_query_results=0,
            after_deduplication=1,
            final_candidate_count=1,
        ),
    )


def make_screening_output(included_pmids=("P001",)) -> ScreeningOutput:
    dims = ["P", "I", "C", "O"]
    questions = [
        BinaryQuestion(
            question_id=f"Q_{d}1",
            dimension=d,
            question_text=f"Does the study satisfy the {d} criterion?",
            answerable_by="YES",
        )
        for d in dims
    ]
    criteria = ScreeningCriteria(questions=questions, reflexion_rounds=1)

    reviewer_out = ReviewerOutput(
        reviewer_model="mock-model",
        answers={
            q.question_id: QuestionAnswer(
                question_id=q.question_id, answer="YES", reasoning="Mock reasoning."
            )
            for q in questions
        },
    )
    decisions: Dict = {}
    for pmid in included_pmids:
        decisions[pmid] = ScreeningDecision(
            pmid=pmid,
            reviewer_a=reviewer_out,
            reviewer_b=reviewer_out,
            conflicts=[],
            individual_status_a="INCLUDE",
            individual_status_b="INCLUDE",
            final_status="INCLUDED",
            exclusion_reasons=[],
        )

    return ScreeningOutput(
        criteria=criteria,
        decisions=decisions,
        included_pmids=list(included_pmids),
        cohens_kappa=1.0,
        prisma_numbers=PRISMAScreeningData(
            total_screened=len(decisions),
            excluded_by_metadata=0,
            excluded_by_dual_review=0,
            sent_to_adjudication=0,
            included_after_screening=len(included_pmids),
            exclusion_reason_counts={},
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build orchestrator with all external deps mocked
# ─────────────────────────────────────────────────────────────────────────────

def _make_orchestrator(tmp_path, search_out, screening_out, extraction_out):
    """
    Patch ModelRegistry, ContextManager, SkillGenerator, and all three pipeline
    classes. Returns (orchestrator, mock_search_cls, mock_screening_cls, mock_extraction_cls).
    """
    checkpoint_dir = str(tmp_path / "checkpoints")

    patches = {
        "registry": patch("src.orchestrator.ModelRegistry"),
        "cm":       patch("src.orchestrator.ContextManager"),
        "sg":       patch("src.orchestrator.SkillGenerator"),
        "search":   patch("src.orchestrator.SearchPipeline"),
        "screen":   patch("src.orchestrator.ScreeningPipeline"),
        "extract":  patch("src.orchestrator.ExtractionPipeline"),
        "pubmed":   patch("src.orchestrator.PubMedClient"),
    }

    mocks = {name: p.start() for name, p in patches.items()}

    # Wire mock return values
    mocks["search"].return_value.run.return_value = search_out
    mocks["screen"].return_value.run.return_value = screening_out
    mocks["extract"].return_value.run.return_value = extraction_out
    mocks["sg"].return_value.generate.return_value = []

    orchestrator = SystematicReviewOrchestrator(
        config_path="configs/models.yaml",
        checkpoint_dir=checkpoint_dir,
        uploads_dir="data/uploads",
        outputs_dir="data/outputs",
    )

    return orchestrator, mocks, patches, checkpoint_dir


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: State passing — search_output → screening, screening_output → extraction
# ─────────────────────────────────────────────────────────────────────────────

def test_state_passing_search_to_screening(tmp_path):
    """ScreeningPipeline must receive the SearchOutput returned by SearchPipeline."""
    search_out = make_search_output()
    screening_out = make_screening_output()

    orchestrator, mocks, patches, _ = _make_orchestrator(
        tmp_path, search_out, screening_out, {}
    )
    try:
        review_config = make_review_config()
        orchestrator.run(review_config)

        # ScreeningPipeline was constructed with the correct search_output
        screen_call_kwargs = mocks["screen"].call_args.kwargs
        assert screen_call_kwargs["search_output"] is search_out, (
            "ScreeningPipeline must receive the SearchOutput from SearchPipeline"
        )
    finally:
        for p in patches.values():
            p.stop()


def test_state_passing_screening_to_extraction(tmp_path):
    """ExtractionPipeline must receive the ScreeningOutput returned by ScreeningPipeline."""
    search_out = make_search_output()
    screening_out = make_screening_output()

    orchestrator, mocks, patches, _ = _make_orchestrator(
        tmp_path, search_out, screening_out, {}
    )
    try:
        review_config = make_review_config()
        orchestrator.run(review_config)

        extract_call_kwargs = mocks["extract"].call_args.kwargs
        assert extract_call_kwargs["screening_output"] is screening_out, (
            "ExtractionPipeline must receive the ScreeningOutput from ScreeningPipeline"
        )
    finally:
        for p in patches.values():
            p.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Checkpoints are created after each stage
# ─────────────────────────────────────────────────────────────────────────────

def test_checkpoints_created_after_each_stage(tmp_path):
    """A JSON checkpoint file must exist for each completed stage."""
    search_out = make_search_output()
    screening_out = make_screening_output()

    orchestrator, mocks, patches, checkpoint_dir = _make_orchestrator(
        tmp_path, search_out, screening_out, {}
    )
    try:
        review_config = make_review_config()
        orchestrator.run(review_config)

        pmid = review_config.pmid
        for stage in ("search", "screening", "extraction"):
            cp_path = os.path.join(checkpoint_dir, f"{pmid}_{stage}.json")
            assert os.path.exists(cp_path), f"Missing checkpoint: {cp_path}"
    finally:
        for p in patches.values():
            p.stop()


def test_checkpoint_json_is_valid(tmp_path):
    """Each checkpoint file must be a valid Checkpoint JSON (loadable by Pydantic)."""
    search_out = make_search_output()
    screening_out = make_screening_output()

    orchestrator, mocks, patches, checkpoint_dir = _make_orchestrator(
        tmp_path, search_out, screening_out, {}
    )
    try:
        review_config = make_review_config()
        orchestrator.run(review_config)

        pmid = review_config.pmid
        for stage in ("search", "screening", "extraction"):
            cp_path = os.path.join(checkpoint_dir, f"{pmid}_{stage}.json")
            with open(cp_path, encoding="utf-8") as f:
                raw = json.load(f)

            cp = Checkpoint.model_validate(raw)
            assert cp.stage_completed == stage
            assert cp.app_state.review_config.pmid == pmid
    finally:
        for p in patches.values():
            p.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Resumption from checkpoint skips completed stages
# ─────────────────────────────────────────────────────────────────────────────

def test_resume_from_search_checkpoint_skips_search(tmp_path):
    """
    When a 'search' checkpoint already exists, SearchPipeline.run() must NOT
    be called again — only Screening and Extraction should execute.
    """
    search_out = make_search_output()
    screening_out = make_screening_output()

    # First: run full pipeline to create the search checkpoint.
    orchestrator, mocks, patches, checkpoint_dir = _make_orchestrator(
        tmp_path, search_out, screening_out, {}
    )
    try:
        review_config = make_review_config()
        orchestrator.run(review_config)
    finally:
        for p in patches.values():
            p.stop()

    # Second: create a NEW orchestrator and run again — search checkpoint exists.
    orchestrator2, mocks2, patches2, _ = _make_orchestrator(
        tmp_path, search_out, screening_out, {}
    )
    try:
        orchestrator2.run(review_config)
        # SearchPipeline.run() should NOT have been called in the second run.
        mocks2["search"].return_value.run.assert_not_called()
    finally:
        for p in patches2.values():
            p.stop()


def test_resume_from_screening_checkpoint_skips_search_and_screening(tmp_path):
    """
    When a 'screening' checkpoint already exists, neither SearchPipeline nor
    ScreeningPipeline should run — only Extraction.
    """
    search_out = make_search_output()
    screening_out = make_screening_output()

    # Seed: write a screening checkpoint directly.
    checkpoint_dir = str(tmp_path / "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    app_state = AppState(
        review_config=make_review_config(),
        search_output=search_out.model_dump(mode="json"),
        screening_output=screening_out.model_dump(mode="json"),
        current_stage="screening",
        skill_generation_complete=True,
    )
    checkpoint = Checkpoint(
        timestamp=datetime.now(timezone.utc),
        stage_completed="screening",
        app_state=app_state,
        metadata={},
    )
    cp_path = os.path.join(checkpoint_dir, "TEST001_screening.json")
    with open(cp_path, "w", encoding="utf-8") as f:
        f.write(checkpoint.model_dump_json())

    orchestrator, mocks, patches, _ = _make_orchestrator(
        tmp_path, search_out, screening_out, {}
    )
    try:
        review_config = make_review_config()
        orchestrator.run(review_config)

        mocks["search"].return_value.run.assert_not_called()
        mocks["screen"].return_value.run.assert_not_called()
        # Extraction SHOULD have been called
        mocks["extract"].return_value.run.assert_called_once()
    finally:
        for p in patches.values():
            p.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Final AppState has correct current_stage
# ─────────────────────────────────────────────────────────────────────────────

def test_final_app_state_stage(tmp_path):
    """After a full run, app_state.current_stage must be 'extraction'."""
    search_out = make_search_output()
    screening_out = make_screening_output()

    orchestrator, mocks, patches, _ = _make_orchestrator(
        tmp_path, search_out, screening_out, {}
    )
    try:
        review_config = make_review_config()
        final_state = orchestrator.run(review_config)

        assert final_state.current_stage == "extraction"
    finally:
        for p in patches.values():
            p.stop()
