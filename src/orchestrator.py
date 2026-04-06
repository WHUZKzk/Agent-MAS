"""
SystematicReviewOrchestrator — master controller for AutoSR.

Spec: docs/01_MASTER_BLUEPRINT.md §6 (step 8)

Responsibilities:
  1. Load ReviewConfig from bench_review.json.
  2. Run SkillGenerator (once, idempotent).
  3. Execute Search → Screening → Extraction pipelines in order.
  4. Checkpoint AppState to JSON after each stage.
  5. Support resumption: detect existing checkpoints and skip completed stages.

Checkpoints are written to:
  {checkpoint_dir}/{pmid}_{stage}.json
  where stage ∈ {"search", "screening", "extraction"}
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.clients.pubmed_client import PubMedClient
from src.engine.context_manager import ContextManager
from src.engine.model_registry import ModelRegistry
from src.schemas.common import AppState, Checkpoint, ReviewConfig
from src.schemas.screening import ScreeningOutput
from src.schemas.search import SearchOutput
from src.skill_generator import SkillGenerator
from src.stages.extraction_pipeline import ExtractionPipeline
from src.stages.screening_pipeline import ScreeningPipeline
from src.stages.search_pipeline import SearchPipeline

logger = logging.getLogger("autosr.orchestrator")


def _coerce(value: Any, schema_class: type) -> Any:
    """Ensure value is an instance of schema_class, validating from dict if needed."""
    if isinstance(value, dict):
        return schema_class.model_validate(value)
    return value


class SystematicReviewOrchestrator:
    """
    Wires the three AutoSR pipeline DAGs together and manages checkpointing.

    Usage:
        orchestrator = SystematicReviewOrchestrator()
        final_state = orchestrator.run(review_config)
    """

    def __init__(
        self,
        config_path: str = "configs/models.yaml",
        checkpoint_dir: str = "data/checkpoints",
        uploads_dir: str = "data/uploads",
        outputs_dir: str = "data/outputs",
        progress_callback: Optional[Any] = None,
    ) -> None:
        self._checkpoint_dir = checkpoint_dir
        self._uploads_dir = uploads_dir
        self._outputs_dir = outputs_dir
        self._progress_callback = progress_callback
        os.makedirs(checkpoint_dir, exist_ok=True)

        self._registry = ModelRegistry(config_path)
        self._cm = ContextManager()

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, review_config: ReviewConfig) -> AppState:
        """
        Execute all three pipeline stages, resuming from checkpoint if available.

        Returns the final AppState with current_stage == "extraction".
        """
        app_state = self._resume_or_init(review_config)

        if app_state.current_stage == "init":
            if not app_state.skill_generation_complete:
                self._run_skill_generation(app_state.review_config)
                app_state = app_state.model_copy(
                    update={"skill_generation_complete": True}
                )
            app_state = self._run_search(app_state)

        if app_state.current_stage == "search":
            app_state = self._run_screening(app_state)

        if app_state.current_stage == "screening":
            app_state = self._run_extraction(app_state)

        logger.info(
            "[Orchestrator] Pipeline complete for PMID %s. Stage: %s",
            app_state.review_config.pmid,
            app_state.current_stage,
        )
        return app_state

    # ─────────────────────────────────────────────────────────────────────────
    # Progress helper
    # ─────────────────────────────────────────────────────────────────────────

    def _emit(self, event_type: str, stage: str, message: str) -> None:
        if self._progress_callback is None:
            return
        try:
            self._progress_callback({
                "type": event_type,
                "stage": stage,
                "message": message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # Stage runners
    # ─────────────────────────────────────────────────────────────────────────

    def _run_skill_generation(self, review_config: ReviewConfig) -> None:
        self._emit("log", "init", "Running SkillGenerator…")
        logger.info("[Orchestrator] Running SkillGenerator for PMID %s.", review_config.pmid)
        gen = SkillGenerator(
            review_config=review_config,
            context_manager=self._cm,
            model_registry=self._registry,
        )
        skill_ids = gen.generate()
        logger.info("[Orchestrator] SkillGenerator produced %d skills.", len(skill_ids))
        self._emit("log", "init", f"SkillGenerator complete — {len(skill_ids)} skills.")

    def _run_search(self, app_state: AppState) -> AppState:
        self._emit("stage_start", "search", "Search stage starting.")
        logger.info("[Orchestrator] Starting Search stage.")
        t0 = time.time()
        pipeline = SearchPipeline(
            review_config=app_state.review_config,
            context_manager=self._cm,
            model_registry=self._registry,
            pubmed_client=PubMedClient(),
            progress_callback=self._progress_callback,
        )
        search_output = pipeline.run()
        elapsed = time.time() - t0

        new_state = app_state.model_copy(update={
            "search_output": search_output,
            "current_stage": "search",
        })
        self._save_checkpoint("search", new_state, {"elapsed_seconds": round(elapsed, 2)})
        self._emit("stage_complete", "search",
                   f"Search complete in {elapsed:.1f}s — "
                   f"{len(search_output.pmids)} candidates found.")
        logger.info("[Orchestrator] Search complete in %.1fs.", elapsed)
        return new_state

    def _run_screening(self, app_state: AppState) -> AppState:
        self._emit("stage_start", "screening", "Screening stage starting.")
        logger.info("[Orchestrator] Starting Screening stage.")
        t0 = time.time()
        search_output: SearchOutput = _coerce(app_state.search_output, SearchOutput)

        pipeline = ScreeningPipeline(
            review_config=app_state.review_config,
            context_manager=self._cm,
            model_registry=self._registry,
            search_output=search_output,
            progress_callback=self._progress_callback,
        )
        screening_output = pipeline.run()
        elapsed = time.time() - t0

        new_state = app_state.model_copy(update={
            "screening_output": screening_output,
            "current_stage": "screening",
        })
        self._save_checkpoint("screening", new_state, {"elapsed_seconds": round(elapsed, 2)})
        self._emit("stage_complete", "screening",
                   f"Screening complete in {elapsed:.1f}s — "
                   f"{len(screening_output.included_pmids)} papers included, "
                   f"κ={screening_output.cohens_kappa:.3f}.")
        logger.info("[Orchestrator] Screening complete in %.1fs.", elapsed)
        return new_state

    def _run_extraction(self, app_state: AppState) -> AppState:
        self._emit("stage_start", "extraction", "Extraction stage starting.")
        logger.info("[Orchestrator] Starting Extraction stage.")
        t0 = time.time()
        screening_output: ScreeningOutput = _coerce(
            app_state.screening_output, ScreeningOutput
        )

        pipeline = ExtractionPipeline(
            review_config=app_state.review_config,
            context_manager=self._cm,
            model_registry=self._registry,
            screening_output=screening_output,
            uploads_dir=self._uploads_dir,
            outputs_dir=self._outputs_dir,
            progress_callback=self._progress_callback,
        )
        extraction_results: Dict[str, Any] = pipeline.run()
        elapsed = time.time() - t0

        # Serialize ExtractionOutput objects for checkpoint storage.
        extraction_dicts = {
            pmid: (out.model_dump(mode="json") if hasattr(out, "model_dump") else out)
            for pmid, out in extraction_results.items()
        }
        new_state = app_state.model_copy(update={
            "extraction_outputs": extraction_dicts,
            "current_stage": "extraction",
        })
        self._save_checkpoint("extraction", new_state, {"elapsed_seconds": round(elapsed, 2)})
        self._emit("stage_complete", "extraction",
                   f"Extraction complete in {elapsed:.1f}s — "
                   f"{len(extraction_results)} papers processed.")
        logger.info(
            "[Orchestrator] Extraction complete in %.1fs. Processed %d papers.",
            elapsed, len(extraction_results),
        )
        return new_state

    # ─────────────────────────────────────────────────────────────────────────
    # Checkpoint management
    # ─────────────────────────────────────────────────────────────────────────

    def _resume_or_init(self, review_config: ReviewConfig) -> AppState:
        """
        Try to load the most recent checkpoint for this review.
        Stages are checked from most complete to least, so we resume at the
        furthest stage already finished.
        """
        pmid = review_config.pmid
        for stage in ("extraction", "screening", "search"):
            cp = self._load_checkpoint(pmid, stage)
            if cp is not None:
                logger.info(
                    "[Orchestrator] Resuming from '%s' checkpoint for PMID %s.",
                    stage, pmid,
                )
                return self._reconstruct(cp.app_state)

        logger.info("[Orchestrator] No checkpoint found. Starting fresh for PMID %s.", pmid)
        return AppState(review_config=review_config)

    def _save_checkpoint(
        self,
        stage: str,
        app_state: AppState,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        checkpoint = Checkpoint(
            timestamp=datetime.now(timezone.utc),
            stage_completed=stage,
            app_state=app_state,
            metadata=metadata or {},
        )
        path = self._checkpoint_path(app_state.review_config.pmid, stage)
        with open(path, "w", encoding="utf-8") as f:
            f.write(checkpoint.model_dump_json())
        logger.info("[Orchestrator] Checkpoint saved → %s", path)

    def _load_checkpoint(self, pmid: str, stage: str) -> Optional[Checkpoint]:
        path = self._checkpoint_path(pmid, stage)
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return Checkpoint.model_validate(data)
        except Exception as exc:
            logger.error(
                "[Orchestrator] Failed to load checkpoint %s: %s. Ignoring.", path, exc
            )
            return None

    def _checkpoint_path(self, pmid: str, stage: str) -> str:
        return os.path.join(self._checkpoint_dir, f"{pmid}_{stage}.json")

    def _reconstruct(self, app_state: AppState) -> AppState:
        """
        After loading from JSON, stage outputs are plain dicts.
        Reconstruct them as typed Pydantic objects so downstream pipelines
        receive the correct types.
        """
        updates: Dict[str, Any] = {}

        if isinstance(app_state.search_output, dict):
            try:
                updates["search_output"] = SearchOutput.model_validate(
                    app_state.search_output
                )
            except Exception as exc:
                logger.error("[Orchestrator] Failed to reconstruct SearchOutput: %s", exc)

        if isinstance(app_state.screening_output, dict):
            try:
                updates["screening_output"] = ScreeningOutput.model_validate(
                    app_state.screening_output
                )
            except Exception as exc:
                logger.error("[Orchestrator] Failed to reconstruct ScreeningOutput: %s", exc)

        if updates:
            return app_state.model_copy(update=updates)
        return app_state
