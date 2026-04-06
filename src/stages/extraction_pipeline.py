"""
Extraction Pipeline — 6-node linear DAG implementation.

Spec: docs/07_EXTRACTION_STAGE.md

Nodes (linear, no branches):
  3.1  (Hard+Soft) Document Cartography
  3.2  (Soft)      Study Characteristics Extraction
  3.3a (Soft)      Chunk Relevance Scoring (two-level DCR, Level 1)
  3.3b (Soft)      Focused Outcome Extraction (two-level DCR, Level 2)
  3.4  (Hard)      Neuro-Symbolic Math Sandbox (standardization)
  3.5  (Hard)      Effect Size Computation (Hedges' g / OR)
  3.6  (Hard)      Benchmark Output Alignment (CSV export)

Critical: LLMs are transcriptionists. ALL math is in math_sandbox.py.
"""
from __future__ import annotations

import csv
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from src.engine.agents import ExecutorAgent
from src.engine.context_manager import ContextManager, MountedContext
from src.engine.dag import DAGRunner
from src.engine.model_registry import ModelRegistry
from src.math_sandbox import (
    standardize,
    compute_hedges_g,
    compute_odds_ratio,
    compute_or_from_reported,
)
from src.schemas.common import (
    DAGDefinition, EdgeDefinition, NodeDefinition, ReviewConfig,
)
from src.schemas.extraction import (
    ArmInfo,
    CharacteristicsExtraction,
    DocumentMap,
    EffectSizeData,
    ExtractionOutput,
    RawDataPoint,
    RawOutcomeExtraction,
    StandardizedDataPoint,
    StandardizedOutcome,
    TextChunk,
    TimepointInfo,
)
from src.schemas.screening import ScreeningOutput
from src.text_parser import parse_document, first_sentence

logger = logging.getLogger("autosr.extraction_pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# Node 3.1: Document Cartography
# ─────────────────────────────────────────────────────────────────────────────

class _Node31_DocumentCartography:
    """
    Composite node: Hard sub-step A (parse file → chunks) +
                    Soft sub-step B (LLM: arms + timepoints).
    """

    def __init__(
        self,
        context_manager: ContextManager,
        agent: ExecutorAgent,
        vision_agent: Optional[ExecutorAgent] = None,
    ) -> None:
        self._cm = context_manager
        self._agent = agent
        self._vision_agent = vision_agent

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pmid: str        = state["current_pmid"]
        file_path: str   = state["current_file_path"]

        # ── Sub-step A: Parse document into chunks ───────────────────────────
        doc_map: DocumentMap = parse_document(
            pmid, file_path, vision_agent=self._vision_agent)

        # ── Sub-step B: LLM identifies arms and timepoints ───────────────────
        abstract_text = "\n\n".join(
            c.content for c in doc_map.chunks if c.section == "Abstract"
        )
        methods_text = "\n\n".join(
            c.content for c in doc_map.chunks if c.section == "Methods"
        )

        cartography_state = {
            **state,
            "abstract_chunks_text": abstract_text or "(no abstract extracted)",
            "methods_chunks_text":  methods_text  or "(no methods extracted)",
        }

        self._cm.mount("extraction.document_cartography", cartography_state)
        mounted: MountedContext = self._cm._current_mount
        raw = self._agent.call(mounted)
        self._cm.unmount()

        try:
            parsed = json.loads(raw)
            arms = [ArmInfo(**a) for a in parsed.get("arms", [])]
            timepoints = [TimepointInfo(**t) for t in parsed.get("timepoints", [])]
        except Exception as exc:
            logger.warning(
                "[Node 3.1] Failed to parse cartography output: %s. "
                "Using defaults.", exc)
            arms = [
                ArmInfo(arm_id="arm_intervention", arm_name="Intervention"),
                ArmInfo(arm_id="arm_control", arm_name="Control"),
            ]
            timepoints = [TimepointInfo(timepoint_id="tp_post", label="Post-intervention")]

        doc_map = DocumentMap(
            pmid=doc_map.pmid,
            source_type=doc_map.source_type,
            chunks=doc_map.chunks,
            arms=arms,
            timepoints=timepoints,
        )

        logger.info(
            "[Node 3.1] pmid=%s: %d chunks, %d arms, %d timepoints",
            pmid, len(doc_map.chunks), len(arms), len(timepoints),
        )

        return {**state, "document_map": doc_map}


# ─────────────────────────────────────────────────────────────────────────────
# Node 3.2: Study Characteristics Extraction
# ─────────────────────────────────────────────────────────────────────────────

class _Node32_CharacteristicsExtraction:
    """
    Soft Node 3.2 — DCR-filtered context: Abstract + Methods + Table 1.
    LLM extracts values for review_config.target_characteristics.
    """

    def __init__(self, context_manager: ContextManager, agent: ExecutorAgent) -> None:
        self._cm = context_manager
        self._agent = agent

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pmid: str          = state["current_pmid"]
        doc_map: DocumentMap = state["document_map"]
        review_config: ReviewConfig = state["review_config"]

        # DCR: Abstract + Methods + first Table
        abstract_chunks = [c for c in doc_map.chunks if c.section == "Abstract"]
        methods_chunks  = [c for c in doc_map.chunks if c.section == "Methods"]
        table_chunks    = [c for c in doc_map.chunks if c.chunk_type == "table"][:1]
        selected_chunks = abstract_chunks + methods_chunks + table_chunks

        context_text = "\n\n---\n\n".join(
            f"[{c.chunk_id} | {c.section}]\n{c.content}" for c in selected_chunks
        )
        target_text = "\n".join(
            f"- {ch}" for ch in review_config.target_characteristics
        )

        char_state = {
            **state,
            "target_characteristics_text": target_text,
            "characteristics_context_text": context_text,
        }

        self._cm.mount("extraction.characteristics_extraction", char_state)
        mounted = self._cm._current_mount
        raw = self._agent.call(mounted)
        self._cm.unmount()

        try:
            parsed = json.loads(raw)
            chars = CharacteristicsExtraction(
                pmid=pmid,
                values=parsed.get("values", {}),
                source_chunks=parsed.get("source_chunks", [c.chunk_id for c in selected_chunks]),
            )
        except Exception as exc:
            logger.warning("[Node 3.2] Parse error: %s", exc)
            chars = CharacteristicsExtraction(
                pmid=pmid,
                values={ch: None for ch in review_config.target_characteristics},
                source_chunks=[],
            )

        logger.info("[Node 3.2] pmid=%s: extracted %d characteristics.",
                    pmid, len(chars.values))
        return {**state, "characteristics": chars}


# ─────────────────────────────────────────────────────────────────────────────
# Node 3.3a: Chunk Relevance Scoring (Two-Level DCR, Level 1)
# ─────────────────────────────────────────────────────────────────────────────

class _Node33a_ChunkRelevanceScoring:
    """
    Soft Node 3.3a — Lightweight LLM classification.
    Sends ONLY chunk metadata (ID + section + first sentence).
    ALL table chunks are always included regardless of LLM decision.
    """

    def __init__(self, context_manager: ContextManager, agent: ExecutorAgent) -> None:
        self._cm = context_manager
        self._agent = agent

    def score_outcome(
        self,
        outcome: str,
        doc_map: DocumentMap,
        state: Dict[str, Any],
    ) -> List[str]:
        """Return list of relevant chunk_ids for a given outcome."""
        # Build lightweight chunk index
        chunk_index_lines = [
            f"[{c.chunk_id}] ({c.section}, {c.chunk_type}) {first_sentence(c.content)}"
            for c in doc_map.chunks
        ]
        chunk_index_text = "\n".join(chunk_index_lines)

        scoring_state = {
            **state,
            "current_outcome_name": outcome,
            "chunk_index_text": chunk_index_text,
        }

        self._cm.mount("extraction.chunk_relevance_scoring", scoring_state)
        mounted = self._cm._current_mount
        raw = self._agent.call(mounted)
        self._cm.unmount()

        try:
            parsed = json.loads(raw)
            relevant_ids: List[str] = parsed.get("relevant_chunk_ids", [])
        except Exception as exc:
            logger.warning("[Node 3.3a] Parse error for outcome '%s': %s", outcome, exc)
            relevant_ids = []

        # Hard override: ALL table chunks must be included
        table_ids = {c.chunk_id for c in doc_map.chunks if c.chunk_type == "table"}
        relevant_set = set(relevant_ids) | table_ids
        return list(relevant_set)

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        doc_map: DocumentMap       = state["document_map"]
        review_config: ReviewConfig = state["review_config"]

        relevance_map: Dict[str, List[str]] = {}
        for outcome in review_config.target_outcomes:
            relevant_ids = self.score_outcome(outcome, doc_map, state)
            relevance_map[outcome] = relevant_ids
            logger.info(
                "[Node 3.3a] outcome=%s: %d relevant chunks.", outcome, len(relevant_ids)
            )

        return {**state, "relevance_map": relevance_map}


# ─────────────────────────────────────────────────────────────────────────────
# Node 3.3b: Focused Outcome Extraction (Two-Level DCR, Level 2)
# ─────────────────────────────────────────────────────────────────────────────

class _Node33b_FocusedOutcomeExtraction:
    """
    Soft Node 3.3b — Nested loop over outcomes × timepoints.
    Uses dynamically generated per-outcome Skill YAMLs.
    LLM transcribes numbers verbatim. NO computation.
    """

    def __init__(self, context_manager: ContextManager, agent: ExecutorAgent) -> None:
        self._cm = context_manager
        self._agent = agent

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        doc_map: DocumentMap       = state["document_map"]
        review_config: ReviewConfig = state["review_config"]
        relevance_map: Dict[str, List[str]] = state["relevance_map"]

        raw_extractions: List[RawOutcomeExtraction] = []

        for outcome in review_config.target_outcomes:
            relevant_ids = relevance_map.get(outcome, [])
            relevant_chunks = [
                c for c in doc_map.chunks if c.chunk_id in relevant_ids
            ]
            relevant_text = "\n\n---\n\n".join(
                f"[{c.chunk_id}]\n{c.content}" for c in relevant_chunks
            )
            arms_text = "\n".join(
                f"- {a.arm_id}: {a.arm_name} (N={a.total_n})"
                for a in doc_map.arms
            )

            for timepoint in doc_map.timepoints:
                tp_state = {
                    **state,
                    "relevant_outcome_chunks": relevant_text,
                    "arms_info_text": arms_text,
                    "current_timepoint_label": timepoint.label,
                    "current_outcome_name": outcome,
                }

                skill_id = f"extraction.outcome.{outcome.replace(' ', '_').lower()}"
                self._cm.mount(skill_id, tp_state)
                mounted = self._cm._current_mount
                raw = self._agent.call(mounted)
                self._cm.unmount()

                try:
                    parsed = json.loads(raw)
                    arms_data: Dict[str, RawDataPoint] = {}
                    for arm_id, arm_vals in parsed.get("arms", {}).items():
                        arms_data[arm_id] = RawDataPoint(**arm_vals)

                    raw_ext = RawOutcomeExtraction(
                        outcome=outcome,
                        timepoint=timepoint.label,
                        arms=arms_data,
                        relevant_chunk_ids=parsed.get("relevant_chunk_ids", relevant_ids),
                        extractor_confidence=parsed.get("extractor_confidence"),
                    )
                except Exception as exc:
                    logger.warning(
                        "[Node 3.3b] Parse error outcome=%s tp=%s: %s",
                        outcome, timepoint.label, exc,
                    )
                    # not_reported fallback
                    raw_ext = RawOutcomeExtraction(
                        outcome=outcome,
                        timepoint=timepoint.label,
                        arms={
                            a.arm_id: RawDataPoint(data_type="not_reported")
                            for a in doc_map.arms
                        },
                        relevant_chunk_ids=relevant_ids,
                    )

                raw_extractions.append(raw_ext)
                logger.info(
                    "[Node 3.3b] pmid=%s outcome=%s tp=%s: extracted %d arms.",
                    state["current_pmid"], outcome, timepoint.label, len(arms_data),
                )

        return {**state, "raw_extractions": raw_extractions}


# ─────────────────────────────────────────────────────────────────────────────
# Node 3.4: Neuro-Symbolic Math Sandbox
# ─────────────────────────────────────────────────────────────────────────────

class _Node34_MathSandbox:
    """
    Hard Node 3.4 — Pure Python standardization. No LLM calls.
    Routes each RawDataPoint through math_sandbox.standardize().
    """

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        raw_extractions: List[RawOutcomeExtraction] = state["raw_extractions"]
        pmid: str = state["current_pmid"]
        standardized_outcomes: List[StandardizedOutcome] = []

        for raw_ext in raw_extractions:
            std_arms: Dict[str, StandardizedDataPoint] = {}
            for arm_id, raw_dp in raw_ext.arms.items():
                std_dp = standardize(raw_dp)
                if not std_dp.is_valid:
                    logger.warning(
                        "[Node 3.4] PMID %s, outcome=%s, arm=%s: invalid — %s",
                        pmid, raw_ext.outcome, arm_id, std_dp.validation_notes,
                    )
                else:
                    logger.info(
                        "[Node 3.4] PMID %s, outcome=%s, arm=%s: method=%s",
                        pmid, raw_ext.outcome, arm_id,
                        std_dp.standardization_method,
                    )
                std_arms[arm_id] = std_dp

            standardized_outcomes.append(StandardizedOutcome(
                outcome=raw_ext.outcome,
                timepoint=raw_ext.timepoint,
                arms=std_arms,
            ))

        return {**state, "standardized_outcomes": standardized_outcomes}


# ─────────────────────────────────────────────────────────────────────────────
# Node 3.5: Effect Size Computation
# ─────────────────────────────────────────────────────────────────────────────

class _Node35_EffectSize:
    """
    Hard Node 3.5 — Compute Hedges' g (continuous) or OR (dichotomous).
    Requires exactly one intervention arm and one control arm.
    """

    _IG_IDS = {"arm_intervention", "arm_ig", "intervention", "experimental"}
    _CG_IDS = {"arm_control", "arm_cg", "control"}

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        standardized: List[StandardizedOutcome] = state["standardized_outcomes"]
        raw_extractions: List[RawOutcomeExtraction] = state["raw_extractions"]
        pmid: str = state["current_pmid"]
        effect_sizes: List[EffectSizeData] = []

        # Build lookup: (outcome, timepoint) → RawOutcomeExtraction
        raw_lookup: Dict[Tuple[str, str], RawOutcomeExtraction] = {
            (r.outcome, r.timepoint): r for r in raw_extractions
        }

        for std_out in standardized:
            ig_dp, cg_dp = self._find_arms(std_out.arms)
            if ig_dp is None or cg_dp is None:
                logger.warning(
                    "[Node 3.5] pmid=%s, outcome=%s: could not identify IG/CG arms.",
                    pmid, std_out.outcome,
                )
                continue
            if not ig_dp.is_valid or not cg_dp.is_valid:
                continue

            es = self._compute(std_out.outcome, std_out.timepoint,
                               ig_dp, cg_dp,
                               raw_lookup.get((std_out.outcome, std_out.timepoint)))
            if es:
                effect_sizes.append(es)

        logger.info(
            "[Node 3.5] pmid=%s: computed %d effect sizes.",
            pmid, len(effect_sizes),
        )
        return {**state, "effect_sizes": effect_sizes}

    def _find_arms(
        self, arms: Dict[str, StandardizedDataPoint]
    ) -> Tuple[Optional[StandardizedDataPoint], Optional[StandardizedDataPoint]]:
        ig, cg = None, None
        for arm_id, dp in arms.items():
            if any(kw in arm_id.lower() for kw in self._IG_IDS):
                ig = dp
            elif any(kw in arm_id.lower() for kw in self._CG_IDS):
                cg = dp
        return ig, cg

    def _compute(
        self,
        outcome: str,
        timepoint: str,
        ig: StandardizedDataPoint,
        cg: StandardizedDataPoint,
        raw_ext: Optional[RawOutcomeExtraction],
    ) -> Optional[EffectSizeData]:
        try:
            # or_ci passthrough — paper already reported OR
            if ig.original_type == "or_ci" and raw_ext:
                raw_ig = raw_ext.arms.get("arm_intervention")
                if raw_ig:
                    res = compute_or_from_reported(raw_ig.val1, raw_ig.val2, raw_ig.val3)
                    return EffectSizeData(
                        outcome=outcome, timepoint=timepoint,
                        **{k: v for k, v in res.items() if k in EffectSizeData.model_fields},
                    )

            # Dichotomous
            if ig.events is not None and cg.events is not None:
                res = compute_odds_ratio(ig.events, ig.total, cg.events, cg.total)
                return EffectSizeData(
                    outcome=outcome, timepoint=timepoint,
                    effect_measure=res["effect_measure"],
                    effect_value=res["effect_value"],
                    ci_lower=res["ci_lower"],
                    ci_upper=res["ci_upper"],
                    se=res["se"],
                    computation_method=res["computation_method"],
                    is_valid=res["is_valid"],
                )

            # Continuous
            if (ig.mean is not None and ig.sd is not None and ig.n is not None and
                    cg.mean is not None and cg.sd is not None and cg.n is not None):
                res = compute_hedges_g(ig.mean, ig.sd, ig.n,
                                       cg.mean, cg.sd, cg.n)
                return EffectSizeData(
                    outcome=outcome, timepoint=timepoint,
                    effect_measure=res["effect_measure"],
                    effect_value=res["effect_value"],
                    ci_lower=res["ci_lower"],
                    ci_upper=res["ci_upper"],
                    se=res["se"],
                    computation_method=res["computation_method"],
                    is_valid=res["is_valid"],
                )

            logger.warning(
                "[Node 3.5] outcome=%s tp=%s: insufficient data for effect size.",
                outcome, timepoint,
            )
            return None

        except Exception as exc:
            logger.error(
                "[Node 3.5] Effect size computation failed for outcome=%s: %s",
                outcome, exc,
            )
            return EffectSizeData(
                outcome=outcome, timepoint=timepoint,
                effect_measure="SMD",
                computation_method="failed",
                is_valid=False,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Node 3.6: Benchmark Output Alignment
# ─────────────────────────────────────────────────────────────────────────────

class _Node36_BenchmarkOutput:
    """
    Hard Node 3.6 — Write CSV files to data/outputs/.
    Produces:
      data/outputs/study_characteristics/{pmid}.csv
      data/outputs/study_results/{pmid}.csv
      data/outputs/study_raw/{pmid}_{outcome}.csv
    """

    def __init__(self, output_dir: str = "data/outputs") -> None:
        self._output_dir = output_dir

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pmid: str = state["current_pmid"]
        characteristics: CharacteristicsExtraction = state["characteristics"]
        standardized: List[StandardizedOutcome] = state["standardized_outcomes"]
        effect_sizes: List[EffectSizeData] = state["effect_sizes"]
        doc_map: DocumentMap = state["document_map"]
        review_config: ReviewConfig = state["review_config"]

        self._write_characteristics(pmid, characteristics, review_config)
        self._write_results(pmid, effect_sizes)
        self._write_raw(pmid, standardized)

        extraction_output = ExtractionOutput(
            pmid=pmid,
            document_map=doc_map,
            characteristics=characteristics,
            raw_outcomes=state["raw_extractions"],
            standardized_outcomes=standardized,
            effect_sizes=effect_sizes,
        )
        return {**state, "extraction_output": extraction_output}

    def _ensure_dir(self, *parts: str) -> str:
        path = os.path.join(self._output_dir, *parts)
        os.makedirs(path, exist_ok=True)
        return path

    def _write_characteristics(
        self,
        pmid: str,
        chars: CharacteristicsExtraction,
        review_config: ReviewConfig,
    ) -> None:
        out_dir = self._ensure_dir("study_characteristics")
        path = os.path.join(out_dir, f"{pmid}.csv")
        fieldnames = ["PMID"] + list(chars.values.keys())
        row = {"PMID": pmid, **{k: v for k, v in chars.values.items()}}
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
        logger.info("[Node 3.6] Wrote characteristics: %s", path)

    def _write_results(
        self,
        pmid: str,
        effect_sizes: List[EffectSizeData],
    ) -> None:
        out_dir = self._ensure_dir("study_results")
        path = os.path.join(out_dir, f"{pmid}.csv")
        fieldnames = ["PMID", "Outcome", "Timepoint", "Effect Measure",
                      "Effect Value", "95% CI Lower", "95% CI Upper", "SE"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for es in effect_sizes:
                if not es.is_valid:
                    continue
                writer.writerow({
                    "PMID": pmid,
                    "Outcome": es.outcome,
                    "Timepoint": es.timepoint,
                    "Effect Measure": es.effect_measure,
                    "Effect Value": round(es.effect_value, 4) if es.effect_value else "",
                    "95% CI Lower": round(es.ci_lower, 4) if es.ci_lower else "",
                    "95% CI Upper": round(es.ci_upper, 4) if es.ci_upper else "",
                    "SE": round(es.se, 4) if es.se else "",
                })
        logger.info("[Node 3.6] Wrote results: %s", path)

    def _write_raw(
        self,
        pmid: str,
        standardized: List[StandardizedOutcome],
    ) -> None:
        out_dir = self._ensure_dir("study_raw")
        outcomes_seen: Dict[str, List[StandardizedOutcome]] = {}
        for so in standardized:
            outcomes_seen.setdefault(so.outcome, []).append(so)

        for outcome, rows in outcomes_seen.items():
            safe_name = outcome.replace(" ", "_").replace("/", "-")
            path = os.path.join(out_dir, f"{pmid}_{safe_name}.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Arm", "Timepoint", "Mean", "SD", "N",
                                  "Events", "Total", "Method"])
                for so in rows:
                    for arm_id, dp in so.arms.items():
                        writer.writerow([
                            arm_id, so.timepoint,
                            dp.mean, dp.sd, dp.n,
                            dp.events, dp.total,
                            dp.standardization_method,
                        ])
        logger.info("[Node 3.6] Wrote raw data for %d outcomes.", len(outcomes_seen))


# ─────────────────────────────────────────────────────────────────────────────
# DAG Definition (linear)
# ─────────────────────────────────────────────────────────────────────────────

EXTRACTION_DAG = DAGDefinition(
    dag_id="extraction_pipeline",
    entry_node="e3_1",
    terminal_nodes=["e3_6"],
    nodes=[
        NodeDefinition(
            node_id="e3_1", node_type="soft",
            skill_id="extraction.document_cartography",
            implementation="stages.extraction_pipeline._Node31_DocumentCartography",
            description="Parse full-text document + LLM identifies arms/timepoints",
        ),
        NodeDefinition(
            node_id="e3_2", node_type="soft",
            skill_id="extraction.characteristics_extraction",
            implementation="stages.extraction_pipeline._Node32_CharacteristicsExtraction",
            description="Extract study characteristics (DCR-filtered to Abs+Methods+Table1)",
        ),
        NodeDefinition(
            node_id="e3_3a", node_type="soft",
            skill_id="extraction.chunk_relevance_scoring",
            implementation="stages.extraction_pipeline._Node33a_ChunkRelevanceScoring",
            description="Score chunk relevance per outcome (two-level DCR, Level 1)",
        ),
        NodeDefinition(
            node_id="e3_3b", node_type="soft",
            skill_id="extraction.chunk_relevance_scoring",  # placeholder — per-outcome skill
            implementation="stages.extraction_pipeline._Node33b_FocusedOutcomeExtraction",
            description="Focused outcome extraction with relevant chunks (two-level DCR, Level 2)",
        ),
        NodeDefinition(
            node_id="e3_4", node_type="hard",
            implementation="stages.extraction_pipeline._Node34_MathSandbox",
            description="Neuro-Symbolic Math Sandbox — all 7 standardization routes",
        ),
        NodeDefinition(
            node_id="e3_5", node_type="hard",
            implementation="stages.extraction_pipeline._Node35_EffectSize",
            description="Effect size computation — Hedges' g / OR with zero-cell correction",
        ),
        NodeDefinition(
            node_id="e3_6", node_type="hard",
            implementation="stages.extraction_pipeline._Node36_BenchmarkOutput",
            description="Write CSV outputs aligned with benchmark format",
        ),
    ],
    edges=[
        EdgeDefinition(from_node="e3_1",  to_node="e3_2"),
        EdgeDefinition(from_node="e3_2",  to_node="e3_3a"),
        EdgeDefinition(from_node="e3_3a", to_node="e3_3b"),
        EdgeDefinition(from_node="e3_3b", to_node="e3_4"),
        EdgeDefinition(from_node="e3_4",  to_node="e3_5"),
        EdgeDefinition(from_node="e3_5",  to_node="e3_6"),
    ],
)


# ─────────────────────────────────────────────────────────────────────────────
# ExtractionPipeline
# ─────────────────────────────────────────────────────────────────────────────

class ExtractionPipeline:
    """
    Runs the 6-node linear Extraction DAG for each included paper.

    Usage:
        pipeline = ExtractionPipeline(review_config, context_manager,
                                      model_registry, screening_output,
                                      uploads_dir="data/uploads")
        extraction_outputs = pipeline.run()
    """

    def __init__(
        self,
        review_config: ReviewConfig,
        context_manager: ContextManager,
        model_registry: ModelRegistry,
        screening_output: ScreeningOutput,
        uploads_dir: str = "data/uploads",
        outputs_dir: str = "data/outputs",
        progress_callback: Optional[Any] = None,
    ) -> None:
        self._review_config = review_config
        self._cm = context_manager
        self._screening_output = screening_output
        self._uploads_dir = uploads_dir
        self._progress_callback = progress_callback

        exec_cfg  = model_registry.get_default("executor")
        exec_name = model_registry.default_name("executor")
        self._executor = ExecutorAgent(model_id=exec_name, model_config=exec_cfg)

        # Node instances (shared across papers)
        self._node31 = _Node31_DocumentCartography(context_manager, self._executor)
        self._node32 = _Node32_CharacteristicsExtraction(context_manager, self._executor)
        self._node33a = _Node33a_ChunkRelevanceScoring(context_manager, self._executor)
        self._node33b = _Node33b_FocusedOutcomeExtraction(context_manager, self._executor)
        self._node34  = _Node34_MathSandbox()
        self._node35  = _Node35_EffectSize()
        self._node36  = _Node36_BenchmarkOutput(output_dir=outputs_dir)

        self._node_registry = {
            "e3_1":  self._node31,
            "e3_2":  self._node32,
            "e3_3a": self._node33a,
            "e3_3b": self._node33b,
            "e3_4":  self._node34,
            "e3_5":  self._node35,
            "e3_6":  self._node36,
        }

    def run(self) -> Dict[str, ExtractionOutput]:
        """Process all included PMIDs. Returns dict of pmid → ExtractionOutput."""
        results: Dict[str, ExtractionOutput] = {}
        total = len(self._screening_output.included_pmids)

        for idx, pmid in enumerate(self._screening_output.included_pmids):
            if self._progress_callback:
                try:
                    self._progress_callback({
                        "type": "log",
                        "stage": "extraction",
                        "node_id": "e3_1",
                        "item_id": f"PMID:{pmid}",
                        "progress": {"current": idx + 1, "total": total},
                        "message": f"Extracting paper {idx + 1}/{total}: PMID {pmid}",
                        "timestamp": "",
                    })
                except Exception:
                    pass

            file_path = self._resolve_upload(pmid)
            if file_path is None:
                logger.warning(
                    "[Extraction] No upload found for pmid=%s — skipping.", pmid
                )
                continue

            logger.info("[Extraction] Processing pmid=%s from %s", pmid, file_path)
            try:
                output = self._process_paper(pmid, file_path)
                results[pmid] = output
            except Exception as exc:
                logger.error(
                    "[Extraction] Failed for pmid=%s: %s", pmid, exc, exc_info=True
                )

        return results

    def _resolve_upload(self, pmid: str) -> Optional[str]:
        """Find the uploaded full-text file for a given PMID."""
        for ext in (".xml", ".pdf"):
            path = os.path.join(self._uploads_dir, f"{pmid}{ext}")
            if os.path.exists(path):
                return path
        return None

    def _process_paper(self, pmid: str, file_path: str) -> ExtractionOutput:
        initial_state: Dict[str, Any] = {
            "current_pmid": pmid,
            "current_file_path": file_path,
            "review_config": self._review_config,
        }
        runner = DAGRunner(
            dag=EXTRACTION_DAG,
            context_manager=self._cm,
            node_registry=self._node_registry,
            progress_callback=self._progress_callback,
        )
        final_state = runner.run(initial_state)
        return final_state["extraction_output"]
