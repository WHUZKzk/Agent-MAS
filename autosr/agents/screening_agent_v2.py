"""
ScreeningAgentV2 — four-stage PICOS-based literature screening pipeline.

Stage 0  Rule-based pre-filtering   (publication type + study design filter)
Stage 1  PICOS structured extraction (LLM extracts P/I/C/O/S from title+abstract)
         + study-design cross-validation
Stage 2  Criteria-based matching     (generate criteria once → per-paper CoT matching)
         + high-recall decision rule
Stage 3  Uncertain-paper review      (stronger model re-evaluates borderline papers,
                                      optionally with user-uploaded full-text PDF)
"""

import json
import logging
from typing import Dict, Generator, List, Optional

from autosr.agents.base_agent import BaseAgent
from autosr.schemas.models import (
    PICODefinition,
    Paper,
    StudyDesignFilter,
    PICOSProfile,
    DimensionCriteria,
    StudyDesignCriteria,
    MatchingCriteria,
    DimensionResult,
    ReviewResult,
    PaperDecisionV2,
    ScreeningSummaryV2,
    ScreeningResultV2,
)
from autosr.tools.llm import call_llm, batch_function_call_llm
from autosr.prompts.picos_extraction import (
    PICOS_EXTRACTION_PROMPT,
    PICOS_EXTRACTION_TOOL,
)
from autosr.prompts.criteria_generation import (
    CRITERIA_GENERATION_V2_PROMPT,
    CRITERIA_GENERATION_V2_TOOL,
)
from autosr.prompts.picos_matching import (
    PICOS_MATCHING_PROMPT,
    PICOS_MATCHING_TOOL,
)
from autosr.prompts.uncertain_review import (
    UNCERTAIN_REVIEW_PROMPT,
    UNCERTAIN_REVIEW_TOOL,
)
from configs.settings import settings

logger = logging.getLogger(__name__)

# ===========================================================================
# Stage 0 constants
# ===========================================================================

EXCLUDED_PUB_TYPES = {
    "review", "systematic review", "meta-analysis",
    "guideline", "practice guideline",
    "editorial", "letter", "comment",
    "case reports", "news",
    "biography", "published erratum",
    "retracted publication", "retraction of publication",
}

RCT_PUB_TYPES = {
    "randomized controlled trial", "clinical trial",
    "clinical trial, phase i", "clinical trial, phase ii",
    "clinical trial, phase iii", "clinical trial, phase iv",
    "controlled clinical trial", "pragmatic clinical trial",
    "equivalence trial",
}

OBSERVATIONAL_PUB_TYPES = {
    "observational study", "cohort study",
    "case-control study", "cross-sectional study",
    "comparative study", "multicenter study",
    "twin study", "validation study", "longitudinal study",
}

# Study designs (from LLM extraction) that map to RCT / observational
_RCT_DESIGNS = {"RCT", "Quasi-experimental"}
_OBS_DESIGNS = {"Cohort", "Case-control", "Cross-sectional", "Before-after"}


# ===========================================================================
# Decision rule (deterministic — no LLM involvement)
# ===========================================================================

def _decide_v2(dimensions: Dict[str, str]) -> str:
    """
    High-recall decision rule.

    - Only P or I MISMATCH → EXCLUDE
    - P+I MATCH with no MISMATCH elsewhere → INCLUDE
    - Everything else → UNCERTAIN (sent to Stage 3)
    """
    p = dimensions.get("P", "UNCERTAIN")
    i = dimensions.get("I", "UNCERTAIN")
    c = dimensions.get("C", "UNCERTAIN")
    o = dimensions.get("O", "UNCERTAIN")
    s = dimensions.get("S", "UNCERTAIN")

    # Hard exclude: P or I clearly mismatched
    if p == "MISMATCH" or i == "MISMATCH":
        return "EXCLUDE"

    # Strong include: P+I match and nothing else mismatched
    if p == "MATCH" and i == "MATCH" and "MISMATCH" not in [c, o, s]:
        return "INCLUDE"

    # Everything else → borderline, needs Stage 3
    return "UNCERTAIN"


# ===========================================================================
# Helper: parse publication types from PubMed metadata
# ===========================================================================

def _parse_pub_types(raw: Optional[str]) -> set:
    """Parse semicolon/comma-separated publication_type string into a lowercase set."""
    if not raw:
        return set()
    separators_replaced = raw.replace(";", ",")
    return {t.strip().lower() for t in separators_replaced.split(",") if t.strip()}


# ===========================================================================
# ScreeningAgentV2
# ===========================================================================

class ScreeningAgentV2(BaseAgent):
    """
    Four-stage literature screening agent.

    Usage::

        agent = ScreeningAgentV2()

        # Full screening (Stages 0-2, returns UNCERTAIN papers for optional Stage 3)
        result = agent.run(papers, pico, study_design_filter)

        # Stage 3 review of uncertain papers
        reviewed = agent.review(uncertain_decisions, pico, criteria, pdf_map)
    """

    def __init__(self):
        super().__init__("ScreeningAgentV2")

    # ------------------------------------------------------------------
    # Public: full screening (Stage 0 → 1 → 2)
    # ------------------------------------------------------------------

    def run(
        self,
        papers: List[Paper],
        pico: PICODefinition,
        study_design_filter: StudyDesignFilter = StudyDesignFilter.BOTH,
        max_concurrency: int = 50,
    ) -> ScreeningResultV2:
        self.reset()

        if not papers:
            empty_criteria = MatchingCriteria(
                P_criteria=DimensionCriteria(core="", acceptable_variations="", exclusion_boundary=""),
                I_criteria=DimensionCriteria(core="", acceptable_variations="", exclusion_boundary=""),
                C_criteria=DimensionCriteria(core="", acceptable_variations="", exclusion_boundary=""),
                O_criteria=DimensionCriteria(core="", acceptable_variations="", exclusion_boundary=""),
                S_criteria=StudyDesignCriteria(),
            )
            return ScreeningResultV2(
                criteria=empty_criteria,
                decisions=[],
                summary=ScreeningSummaryV2(
                    total=0, stage0_excluded=0, stage1_excluded=0,
                    stage2_included=0, stage2_excluded=0,
                    stage3_reviewed=0, stage3_included=0, stage3_excluded=0,
                    final_included=0, final_excluded=0,
                ),
            )

        pico_dict = {"P": pico.P, "I": pico.I, "C": pico.C, "O": pico.O}

        # Stage 0 — rule-based pre-filtering
        kept_papers, stage0_decisions = self._run_step(
            "stage0_filter",
            self._stage0_filter, papers, study_design_filter,
        )

        # Stage 1 — PICOS extraction
        picos_list = self._run_step(
            "stage1_extract_picos",
            self._stage1_extract_picos, kept_papers, max_concurrency,
        )

        # Stage 1 cross-validation: study design check
        s1_kept, s1_kept_picos, s1_excluded_decisions = self._run_step(
            "stage1_design_check",
            self._stage1_design_check,
            kept_papers, picos_list, study_design_filter,
        )

        # Stage 2a — generate matching criteria (once)
        criteria = self._run_step(
            "stage2_generate_criteria",
            self._stage2_generate_criteria, pico_dict, study_design_filter,
        )

        # Stage 2b — per-paper dimension matching
        dim_results = self._run_step(
            "stage2_match",
            self._stage2_match,
            s1_kept, s1_kept_picos, criteria, max_concurrency,
        )

        # Stage 2c — apply decision rule & build decisions
        stage2_decisions = self._run_step(
            "stage2_decide",
            self._stage2_build_decisions,
            s1_kept, s1_kept_picos, dim_results,
        )

        # Merge all decisions
        all_decisions = stage0_decisions + s1_excluded_decisions + stage2_decisions

        # Compute summary
        summary = self._compute_summary(all_decisions)

        logger.info(
            "[ScreeningAgentV2] Done: %d papers → %d included / %d excluded / %d uncertain (%.1fs)",
            len(papers), summary.final_included, summary.final_excluded,
            summary.stage3_reviewed, self.state.elapsed,
        )

        return ScreeningResultV2(
            criteria=criteria, decisions=all_decisions, summary=summary,
        )

    # ------------------------------------------------------------------
    # Public: streaming entry (Stage 0 → 1 → 2)
    # ------------------------------------------------------------------

    def run_stream(
        self,
        papers: List[Paper],
        pico: PICODefinition,
        study_design_filter: StudyDesignFilter = StudyDesignFilter.BOTH,
        max_concurrency: int = 50,
    ) -> Generator[dict, None, None]:
        """Yields SSE-friendly progress events."""
        self.reset()

        if not papers:
            yield {"type": "summary", "data": {
                "total": 0, "final_included": 0, "final_excluded": 0,
            }}
            yield {"type": "done"}
            return

        pico_dict = {"P": pico.P, "I": pico.I, "C": pico.C, "O": pico.O}

        # Stage 0
        try:
            kept_papers, stage0_decisions = self._stage0_filter(papers, study_design_filter)
        except Exception as exc:
            logger.exception("run_stream: stage0 failed")
            yield {"type": "error", "data": str(exc)}
            return

        yield {
            "type": "stage0_done",
            "data": {
                "total": len(papers),
                "kept": len(kept_papers),
                "excluded": len(stage0_decisions),
            },
        }

        # Stage 1 — PICOS extraction
        try:
            picos_list = self._stage1_extract_picos(kept_papers, max_concurrency)
        except Exception as exc:
            logger.exception("run_stream: stage1 extraction failed")
            yield {"type": "error", "data": str(exc)}
            return

        # Stage 1 cross-validation
        try:
            s1_kept, s1_kept_picos, s1_excluded_decisions = self._stage1_design_check(
                kept_papers, picos_list, study_design_filter,
            )
        except Exception as exc:
            logger.exception("run_stream: stage1 design check failed")
            yield {"type": "error", "data": str(exc)}
            return

        yield {
            "type": "stage1_done",
            "data": {
                "total": len(kept_papers),
                "kept": len(s1_kept),
                "design_excluded": len(s1_excluded_decisions),
            },
        }

        # Stage 2a — criteria generation
        try:
            criteria = self._stage2_generate_criteria(pico_dict, study_design_filter)
        except Exception as exc:
            logger.exception("run_stream: criteria generation failed")
            yield {"type": "error", "data": str(exc)}
            return

        yield {"type": "criteria_generated", "data": criteria.model_dump()}

        # Stage 2b — dimension matching
        try:
            dim_results = self._stage2_match(
                s1_kept, s1_kept_picos, criteria, max_concurrency,
            )
        except Exception as exc:
            logger.exception("run_stream: stage2 matching failed")
            yield {"type": "error", "data": str(exc)}
            return

        # Stage 2c — build decisions & yield per-paper events
        stage2_decisions = self._stage2_build_decisions(
            s1_kept, s1_kept_picos, dim_results,
        )

        for dec in stage0_decisions + s1_excluded_decisions:
            yield {"type": "paper_decided", "data": dec.model_dump()}
        for dec in stage2_decisions:
            yield {"type": "paper_decided", "data": dec.model_dump()}

        all_decisions = stage0_decisions + s1_excluded_decisions + stage2_decisions
        summary = self._compute_summary(all_decisions)

        yield {"type": "summary", "data": summary.model_dump()}
        yield {"type": "done"}

    # ------------------------------------------------------------------
    # Public: Stage 3 review of uncertain papers
    # ------------------------------------------------------------------

    def review(
        self,
        uncertain_decisions: List[PaperDecisionV2],
        papers_map: Dict[str, Paper],
        pico: PICODefinition,
        criteria: MatchingCriteria,
        pdf_map: Optional[Dict[str, str]] = None,
        max_concurrency: int = 10,
    ) -> List[PaperDecisionV2]:
        """
        Re-evaluate UNCERTAIN papers with a stronger model.

        Args:
            uncertain_decisions: PaperDecisionV2 items with decision_stage still pending.
            papers_map: {pmid: Paper} for original title+abstract.
            pico: Review PICO definition.
            criteria: Matching criteria from Stage 2a.
            pdf_map: Optional {pmid: fulltext_content_string} for papers with uploaded PDFs.
            max_concurrency: Parallel review requests.

        Returns:
            Updated PaperDecisionV2 list with final_decision set to INCLUDE or EXCLUDE.
        """
        if not uncertain_decisions:
            return []

        pdf_map = pdf_map or {}
        pico_dict = {"P": pico.P, "I": pico.I, "C": pico.C, "O": pico.O}
        criteria_json = json.dumps(criteria.model_dump(), indent=2, ensure_ascii=False)

        review_results = self._run_step(
            "stage3_review",
            self._stage3_review_batch,
            uncertain_decisions, papers_map, pico_dict,
            criteria_json, pdf_map, max_concurrency,
        )

        updated = []
        for dec, rr in zip(uncertain_decisions, review_results):
            dec.review_result = rr
            dec.final_decision = rr.final_decision
            dec.decision_stage = "stage3"
            updated.append(dec)
        return updated

    def review_stream(
        self,
        uncertain_decisions: List[PaperDecisionV2],
        papers_map: Dict[str, Paper],
        pico: PICODefinition,
        criteria: MatchingCriteria,
        pdf_map: Optional[Dict[str, str]] = None,
        max_concurrency: int = 10,
    ) -> Generator[dict, None, None]:
        """Streaming version of review()."""
        if not uncertain_decisions:
            yield {"type": "review_done", "data": {"included": 0, "excluded": 0}}
            return

        pdf_map = pdf_map or {}
        pico_dict = {"P": pico.P, "I": pico.I, "C": pico.C, "O": pico.O}
        criteria_json = json.dumps(criteria.model_dump(), indent=2, ensure_ascii=False)

        try:
            review_results = self._stage3_review_batch(
                uncertain_decisions, papers_map, pico_dict,
                criteria_json, pdf_map, max_concurrency,
            )
        except Exception as exc:
            logger.exception("review_stream: stage3 failed")
            yield {"type": "error", "data": str(exc)}
            return

        included = 0
        excluded = 0
        for dec, rr in zip(uncertain_decisions, review_results):
            dec.review_result = rr
            dec.final_decision = rr.final_decision
            dec.decision_stage = "stage3"
            if rr.final_decision == "INCLUDE":
                included += 1
            else:
                excluded += 1
            yield {"type": "review_decided", "data": dec.model_dump()}

        yield {"type": "review_done", "data": {"included": included, "excluded": excluded}}

    # ==================================================================
    # Stage 0: Rule-based pre-filtering
    # ==================================================================

    def _stage0_filter(
        self,
        papers: List[Paper],
        study_design_filter: StudyDesignFilter,
    ) -> tuple:
        """
        Returns:
            (kept_papers, excluded_decisions)
        """
        kept = []
        excluded_decisions = []

        for paper in papers:
            result = self._stage0_classify(paper, study_design_filter)
            if result == "KEEP":
                kept.append(paper)
            else:
                excluded_decisions.append(PaperDecisionV2(
                    pmid=paper.pmid,
                    title=paper.title,
                    stage0_result=result,
                    final_decision="EXCLUDE",
                    decision_stage="stage0",
                ))

        logger.info(
            "[Stage 0] %d/%d papers kept, %d excluded by rules",
            len(kept), len(papers), len(excluded_decisions),
        )
        return kept, excluded_decisions

    @staticmethod
    def _stage0_classify(paper: Paper, sdf: StudyDesignFilter) -> str:
        pub_types = _parse_pub_types(paper.publication_type)

        # Hard exclusion
        if pub_types & EXCLUDED_PUB_TYPES:
            return "EXCLUDED_pub_type"

        if sdf == StudyDesignFilter.BOTH:
            return "KEEP"

        is_rct = bool(pub_types & RCT_PUB_TYPES)
        is_obs = bool(pub_types & OBSERVATIONAL_PUB_TYPES)

        if sdf == StudyDesignFilter.RCT_ONLY:
            if is_obs and not is_rct:
                return "EXCLUDED_study_design"
            return "KEEP"

        if sdf == StudyDesignFilter.OBSERVATIONAL_ONLY:
            if is_rct and not is_obs:
                return "EXCLUDED_study_design"
            return "KEEP"

        return "KEEP"

    # ==================================================================
    # Stage 1: PICOS extraction
    # ==================================================================

    def _stage1_extract_picos(
        self,
        papers: List[Paper],
        max_concurrency: int,
    ) -> List[PICOSProfile]:
        batch_inputs = [
            {"title": p.title, "abstract": p.abstract or ""}
            for p in papers
        ]

        raw_results = batch_function_call_llm(
            PICOS_EXTRACTION_PROMPT,
            batch_inputs,
            tool=PICOS_EXTRACTION_TOOL,
            max_concurrency=max_concurrency,
        )

        profiles = []
        for raw in raw_results:
            profiles.append(PICOSProfile(
                P_population=raw.get("P_population", "Not reported"),
                I_intervention=raw.get("I_intervention", "Not reported"),
                C_comparison=raw.get("C_comparison", "Not reported"),
                O_outcome=raw.get("O_outcome", "Not reported"),
                S_study_design=raw.get("S_study_design", "Not reported"),
                sample_size=raw.get("sample_size", "Not reported"),
                duration=raw.get("duration", "Not reported"),
            ))

        logger.info("[Stage 1] PICOS extraction done for %d papers", len(profiles))
        return profiles

    def _stage1_design_check(
        self,
        papers: List[Paper],
        picos_list: List[PICOSProfile],
        study_design_filter: StudyDesignFilter,
    ) -> tuple:
        """
        Cross-validate extracted study design against user's filter.
        Only excludes papers whose pub_type was empty (passed Stage 0 by default)
        but whose extracted design clearly contradicts the filter.

        Returns:
            (kept_papers, kept_picos, excluded_decisions)
        """
        if study_design_filter == StudyDesignFilter.BOTH:
            return papers, picos_list, []

        kept_papers = []
        kept_picos = []
        excluded_decisions = []

        for paper, picos in zip(papers, picos_list):
            pub_types = _parse_pub_types(paper.publication_type)
            # Only apply cross-validation if Stage 0 couldn't determine design
            has_known_type = bool(pub_types & (RCT_PUB_TYPES | OBSERVATIONAL_PUB_TYPES))

            if has_known_type:
                # Stage 0 already made the call; keep this paper
                kept_papers.append(paper)
                kept_picos.append(picos)
                continue

            design = picos.S_study_design
            if design in {"Not reported", "Other", "Mixed methods", "Qualitative"}:
                # Can't determine → keep (conservative)
                kept_papers.append(paper)
                kept_picos.append(picos)
                continue

            if study_design_filter == StudyDesignFilter.RCT_ONLY and design in _OBS_DESIGNS:
                excluded_decisions.append(PaperDecisionV2(
                    pmid=paper.pmid,
                    title=paper.title,
                    stage0_result="KEEP",
                    picos_profile=picos,
                    final_decision="EXCLUDE",
                    decision_stage="stage1",
                ))
                continue

            if study_design_filter == StudyDesignFilter.OBSERVATIONAL_ONLY and design in _RCT_DESIGNS:
                excluded_decisions.append(PaperDecisionV2(
                    pmid=paper.pmid,
                    title=paper.title,
                    stage0_result="KEEP",
                    picos_profile=picos,
                    final_decision="EXCLUDE",
                    decision_stage="stage1",
                ))
                continue

            kept_papers.append(paper)
            kept_picos.append(picos)

        logger.info(
            "[Stage 1 cross-val] %d kept, %d excluded by design check",
            len(kept_papers), len(excluded_decisions),
        )
        return kept_papers, kept_picos, excluded_decisions

    # ==================================================================
    # Stage 2: Criteria generation + matching + decision
    # ==================================================================

    def _stage2_generate_criteria(
        self,
        pico_dict: dict,
        study_design_filter: StudyDesignFilter,
    ) -> MatchingCriteria:
        design_desc = {
            StudyDesignFilter.RCT_ONLY: "Randomized Controlled Trials (RCTs) only",
            StudyDesignFilter.OBSERVATIONAL_ONLY: "Observational studies only (cohort, case-control, cross-sectional, etc.)",
            StudyDesignFilter.BOTH: "Both RCTs and observational studies",
        }

        inputs = {
            **pico_dict,
            "study_design_description": design_desc[study_design_filter],
        }

        raw = batch_function_call_llm(
            CRITERIA_GENERATION_V2_PROMPT,
            [inputs],
            tool=CRITERIA_GENERATION_V2_TOOL,
            max_concurrency=1,
        )[0]

        criteria = MatchingCriteria(
            P_criteria=DimensionCriteria(**raw.get("P_criteria", {
                "core": "", "acceptable_variations": "", "exclusion_boundary": "",
            })),
            I_criteria=DimensionCriteria(**raw.get("I_criteria", {
                "core": "", "acceptable_variations": "", "exclusion_boundary": "",
            })),
            C_criteria=DimensionCriteria(**raw.get("C_criteria", {
                "core": "", "acceptable_variations": "", "exclusion_boundary": "",
            })),
            O_criteria=DimensionCriteria(**raw.get("O_criteria", {
                "core": "", "acceptable_variations": "", "exclusion_boundary": "",
            })),
            S_criteria=StudyDesignCriteria(**raw.get("S_criteria", {
                "acceptable_designs": [], "excluded_designs": [],
            })),
        )

        logger.info("[Stage 2a] Matching criteria generated")
        return criteria

    def _stage2_match(
        self,
        papers: List[Paper],
        picos_list: List[PICOSProfile],
        criteria: MatchingCriteria,
        max_concurrency: int,
    ) -> List[DimensionResult]:
        criteria_json = json.dumps(criteria.model_dump(), indent=2, ensure_ascii=False)

        batch_inputs = []
        for paper, picos in zip(papers, picos_list):
            batch_inputs.append({
                "criteria_json": criteria_json,
                "study_P": picos.P_population,
                "study_I": picos.I_intervention,
                "study_C": picos.C_comparison,
                "study_O": picos.O_outcome,
                "study_S": picos.S_study_design,
                "study_sample_size": picos.sample_size,
                "study_duration": picos.duration,
                "title": paper.title,
            })

        raw_results = batch_function_call_llm(
            PICOS_MATCHING_PROMPT,
            batch_inputs,
            tool=PICOS_MATCHING_TOOL,
            max_concurrency=max_concurrency,
        )

        dim_results = []
        for raw in raw_results:
            reasoning = raw.get("reasoning", {})
            dimensions = raw.get("dimensions", {})
            # Normalize missing dimensions to UNCERTAIN
            for dim in ("P", "I", "C", "O", "S"):
                if dim not in reasoning:
                    reasoning[dim] = "No reasoning provided"
                if dim not in dimensions or dimensions[dim] not in ("MATCH", "MISMATCH", "UNCERTAIN"):
                    dimensions[dim] = "UNCERTAIN"

            dim_results.append(DimensionResult(
                reasoning=reasoning,
                dimensions=dimensions,
                overall_decision=raw.get("overall_decision", "UNCERTAIN"),
            ))

        logger.info("[Stage 2b] Dimension matching done for %d papers", len(dim_results))
        return dim_results

    def _stage2_build_decisions(
        self,
        papers: List[Paper],
        picos_list: List[PICOSProfile],
        dim_results: List[DimensionResult],
    ) -> List[PaperDecisionV2]:
        decisions = []
        for paper, picos, dr in zip(papers, picos_list, dim_results):
            # Apply deterministic decision rule (overrides LLM's overall_decision)
            rule_decision = _decide_v2(dr.dimensions)

            if rule_decision == "UNCERTAIN":
                # Will be resolved in Stage 3; mark as UNCERTAIN for now
                final = "UNCERTAIN"
                stage = "stage2"
            else:
                final = rule_decision
                stage = "stage2"

            decisions.append(PaperDecisionV2(
                pmid=paper.pmid,
                title=paper.title,
                stage0_result="KEEP",
                picos_profile=picos,
                dimension_result=dr,
                final_decision=final,
                decision_stage=stage,
            ))

        inc = sum(1 for d in decisions if d.final_decision == "INCLUDE")
        exc = sum(1 for d in decisions if d.final_decision == "EXCLUDE")
        unc = sum(1 for d in decisions if d.final_decision == "UNCERTAIN")
        logger.info(
            "[Stage 2c] Decisions: %d included, %d excluded, %d uncertain",
            inc, exc, unc,
        )
        return decisions

    # ==================================================================
    # Stage 3: Uncertain-paper review (stronger model)
    # ==================================================================

    def _stage3_review_batch(
        self,
        uncertain_decisions: List[PaperDecisionV2],
        papers_map: Dict[str, Paper],
        pico_dict: dict,
        criteria_json: str,
        pdf_map: Dict[str, str],
        max_concurrency: int,
    ) -> List[ReviewResult]:
        review_model = settings.review_model_name

        batch_inputs = []
        for dec in uncertain_decisions:
            paper = papers_map.get(dec.pmid)
            picos = dec.picos_profile
            dr = dec.dimension_result

            # Identify which dimensions are uncertain
            uncertain_dims = []
            if dr:
                for dim_key, dim_val in dr.dimensions.items():
                    if dim_val != "MATCH":
                        reason = dr.reasoning.get(dim_key, "No details")
                        uncertain_dims.append(f"- {dim_key}: {dim_val} — {reason}")
            uncertain_detail = "\n".join(uncertain_dims) if uncertain_dims else "No specific uncertain dimensions recorded."

            # Full text section (optional)
            fulltext = pdf_map.get(dec.pmid, "")
            fulltext_section = ""
            if fulltext:
                fulltext_section = (
                    "\n# FULL TEXT — RELEVANT SECTIONS (from uploaded PDF)\n"
                    f"{fulltext}\n"
                )

            batch_inputs.append({
                **pico_dict,
                "criteria_json": criteria_json,
                "title": paper.title if paper else dec.title,
                "abstract": paper.abstract if paper else "",
                "study_P": picos.P_population if picos else "Not reported",
                "study_I": picos.I_intervention if picos else "Not reported",
                "study_C": picos.C_comparison if picos else "Not reported",
                "study_O": picos.O_outcome if picos else "Not reported",
                "study_S": picos.S_study_design if picos else "Not reported",
                "study_sample_size": picos.sample_size if picos else "Not reported",
                "study_duration": picos.duration if picos else "Not reported",
                "stage2_reasoning_json": json.dumps(
                    dr.reasoning if dr else {}, indent=2, ensure_ascii=False,
                ),
                "uncertain_dimensions_detail": uncertain_detail,
                "fulltext_section": fulltext_section,
            })

        raw_results = batch_function_call_llm(
            UNCERTAIN_REVIEW_PROMPT,
            batch_inputs,
            tool=UNCERTAIN_REVIEW_TOOL,
            max_concurrency=max_concurrency,
            model=review_model,
        )

        review_results = []
        for raw in raw_results:
            resolved = raw.get("resolved_dimensions", {})
            for dim in ("P", "I", "C", "O", "S"):
                if dim not in resolved:
                    resolved[dim] = "STILL_UNCERTAIN"

            final = raw.get("final_decision", "INCLUDE")
            if final not in ("INCLUDE", "EXCLUDE"):
                final = "INCLUDE"  # Default to INCLUDE per Cochrane principle

            review_results.append(ReviewResult(
                review_reasoning=raw.get("review_reasoning", ""),
                resolved_dimensions=resolved,
                final_decision=final,
                confidence=raw.get("confidence", "LOW"),
            ))

        logger.info(
            "[Stage 3] Reviewed %d papers: %d included, %d excluded",
            len(review_results),
            sum(1 for r in review_results if r.final_decision == "INCLUDE"),
            sum(1 for r in review_results if r.final_decision == "EXCLUDE"),
        )
        return review_results

    # ==================================================================
    # Summary computation
    # ==================================================================

    @staticmethod
    def _compute_summary(decisions: List[PaperDecisionV2]) -> ScreeningSummaryV2:
        s0_exc = sum(1 for d in decisions if d.decision_stage == "stage0")
        s1_exc = sum(1 for d in decisions if d.decision_stage == "stage1")
        s2_inc = sum(1 for d in decisions
                     if d.decision_stage == "stage2" and d.final_decision == "INCLUDE")
        s2_exc = sum(1 for d in decisions
                     if d.decision_stage == "stage2" and d.final_decision == "EXCLUDE")
        s2_unc = sum(1 for d in decisions
                     if d.decision_stage == "stage2" and d.final_decision == "UNCERTAIN")
        s3_inc = sum(1 for d in decisions
                     if d.decision_stage == "stage3" and d.final_decision == "INCLUDE")
        s3_exc = sum(1 for d in decisions
                     if d.decision_stage == "stage3" and d.final_decision == "EXCLUDE")

        final_inc = s2_inc + s3_inc
        # UNCERTAIN papers not yet reviewed count as included (will go to Stage 3)
        final_inc += s2_unc
        final_exc = s0_exc + s1_exc + s2_exc + s3_exc

        return ScreeningSummaryV2(
            total=len(decisions),
            stage0_excluded=s0_exc,
            stage1_excluded=s1_exc,
            stage2_included=s2_inc,
            stage2_excluded=s2_exc,
            stage3_reviewed=s3_inc + s3_exc,
            stage3_included=s3_inc,
            stage3_excluded=s3_exc,
            final_included=final_inc,
            final_excluded=final_exc,
        )
