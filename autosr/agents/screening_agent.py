"""
ScreeningAgent — orchestrates literature screening based on PICO-derived criteria.

Two-phase screening pipeline:
  Phase 0. Generate eligibility criteria from PICO           (call_llm → SCREENING_CRITERIA_GENERATION)
  Phase 1. Title-only screening (all papers)                 (batch_function_call_llm, title_criteria only)
           → Papers with any NO are immediately EXCLUDED (skip Phase 2)
  Phase 2. Abstract screening (surviving papers only)        (batch_function_call_llm, content_criteria)
  Phase 3. Merge results + apply deterministic decision rule
"""

import json
import re
import logging
from typing import Generator, List, Optional

from autosr.agents.base_agent import BaseAgent
from autosr.schemas.models import (
    PICODefinition, Paper,
    CriteriaSet, PaperDecision, ScreeningSummary, ScreeningResult,
)
from autosr.tools.llm import call_llm, batch_function_call_llm
from autosr.prompts.screen_criteria import SCREENING_CRITERIA_GENERATION
from autosr.prompts.screening import LITERATURE_SCREENING_FC

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decision rule (deterministic — no LLM involvement)
# "SKIPPED" means the paper was excluded at Phase 1 and Phase 2 was not run.
# ---------------------------------------------------------------------------

def _decide(evaluations: List[str]) -> str:
    """
    INCLUDE  : all non-SKIPPED criteria are YES
    EXCLUDE  : at least one criterion is NO
    UNCERTAIN: some UNCERTAIN (no NO, no SKIPPED blocking)
    """
    evals = [e.upper() for e in evaluations if e.upper() != "SKIPPED"]
    if not evals:
        return "UNCERTAIN"
    if "NO" in evals:
        return "EXCLUDE"
    if all(e == "YES" for e in evals):
        return "INCLUDE"
    return "UNCERTAIN"


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> Optional[dict]:
    m = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Screening tool schema builder
# ---------------------------------------------------------------------------

def _build_screening_tool(n_criteria: int) -> dict:
    return {
        "type": "function",
        "function": {
            "name": "submit_evaluations",
            "description": (
                f"Submit paper evaluation results for all {n_criteria} criteria. "
                "Return exactly one decision per criterion in the order listed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "evaluations": {
                        "type": "array",
                        "description": (
                            f"Exactly {n_criteria} decisions (YES / NO / UNCERTAIN), "
                            "one per criterion in the order listed."
                        ),
                        "items": {
                            "type": "string",
                            "enum": ["YES", "NO", "UNCERTAIN"],
                        },
                        "minItems": n_criteria,
                        "maxItems": n_criteria,
                    }
                },
                "required": ["evaluations"],
            },
        },
    }


class ScreeningAgent(BaseAgent):
    """
    Screens a list of papers against PICO-derived eligibility criteria
    using a two-phase approach: title-first, abstract-second.

    Usage::

        agent = ScreeningAgent()
        result = agent.run(papers, pico, num_title_criteria=3, num_content_criteria=3)
        # result.decisions  → list of PaperDecision
        # result.summary    → included / excluded / uncertain counts
    """

    def __init__(self):
        super().__init__("ScreeningAgent")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        papers: List[Paper],
        pico: PICODefinition,
        num_title_criteria: int = 3,
        num_content_criteria: int = 3,
        max_concurrency: int = 50,
    ) -> ScreeningResult:
        self.reset()

        if not papers:
            return ScreeningResult(
                criteria=CriteriaSet(),
                decisions=[],
                summary=ScreeningSummary(total=0, included=0, excluded=0, uncertain=0),
            )

        pico_dict = {"P": pico.P, "I": pico.I, "C": pico.C, "O": pico.O}

        # Phase 0 – generate eligibility criteria
        criteria = self._run_step(
            "generate_criteria",
            self._generate_criteria,
            pico_dict,
            num_title_criteria,
            num_content_criteria,
        )

        title_criteria = criteria.title_criteria
        content_criteria = criteria.content_criteria
        n_title = len(title_criteria)
        n_content = len(content_criteria)

        # Phase 1 – title-only screening (all papers)
        phase1_raw = self._run_step(
            "phase1_title_screen",
            self._phase1_screen,
            papers,
            pico_dict,
            title_criteria,
            max_concurrency,
        )

        # Parse Phase 1 results; collect survivors for Phase 2
        phase1_evals_list: List[List[str]] = []
        phase2_indices: List[int] = []
        for i, raw in enumerate(phase1_raw):
            evals = self._parse_evals(raw, n_title)
            phase1_evals_list.append(evals)
            if "NO" not in evals:
                phase2_indices.append(i)

        logger.info(
            "[ScreeningAgent] Phase 1 done: %d/%d papers pass to Phase 2",
            len(phase2_indices), len(papers),
        )

        # Phase 2 – abstract screening (surviving papers only)
        phase2_papers = [papers[i] for i in phase2_indices]
        if phase2_papers and n_content > 0:
            phase2_raw = self._run_step(
                "phase2_content_screen",
                self._phase2_screen,
                phase2_papers,
                pico_dict,
                content_criteria,
                max_concurrency,
            )
        else:
            phase2_raw = []

        phase2_evals_map: dict = {}
        for idx, raw in zip(phase2_indices, phase2_raw):
            phase2_evals_map[idx] = self._parse_evals(raw, n_content)

        # Phase 3 – merge + apply decision rule
        decisions = self._run_step(
            "apply_decision_rule",
            self._merge_decisions,
            papers,
            phase1_evals_list,
            phase2_evals_map,
            n_content,
        )

        summary = ScreeningSummary(
            total=len(decisions),
            included=sum(1 for d in decisions if d.decision == "INCLUDE"),
            excluded=sum(1 for d in decisions if d.decision == "EXCLUDE"),
            uncertain=sum(1 for d in decisions if d.decision == "UNCERTAIN"),
        )

        logger.info(
            "[ScreeningAgent] Done: %d papers → %d included / %d excluded / %d uncertain (%.1fs)",
            len(papers), summary.included, summary.excluded, summary.uncertain,
            self.state.elapsed,
        )

        return ScreeningResult(criteria=criteria, decisions=decisions, summary=summary)

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    def _generate_criteria(
        self,
        pico_dict: dict,
        num_title: int,
        num_content: int,
    ) -> CriteriaSet:
        inputs = {
            **pico_dict,
            "num_title_criteria": num_title,
            "num_abstract_criteria": num_content,
        }
        raw = call_llm(SCREENING_CRITERIA_GENERATION, inputs)
        parsed = _extract_json(raw)

        if not parsed:
            logger.warning("SCREENING_CRITERIA_GENERATION returned unparseable JSON")
            return CriteriaSet()

        title_criteria   = parsed.get("TITLE_CRITERIA", [])[:num_title]
        content_criteria = parsed.get("CONTENT_CRITERIA", [])[:num_content]
        eligibility      = parsed.get("ELIGIBILITY_ANALYSIS", [])

        logger.info(
            "Criteria generated: %d title + %d content",
            len(title_criteria), len(content_criteria),
        )
        return CriteriaSet(
            title_criteria=title_criteria,
            content_criteria=content_criteria,
            eligibility_analysis=eligibility,
        )

    def _phase1_screen(
        self,
        papers: List[Paper],
        pico_dict: dict,
        title_criteria: List[str],
        max_concurrency: int,
    ) -> List[dict]:
        """Phase 1: evaluate all papers using title ONLY + title_criteria."""
        n = len(title_criteria)
        criteria_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(title_criteria))
        tool = _build_screening_tool(n)

        batch_inputs = [
            {
                **pico_dict,
                "paper_content": f"Title: {paper.title}",
                "criteria_text": criteria_text,
                "num_criteria": n,
            }
            for paper in papers
        ]

        return batch_function_call_llm(
            LITERATURE_SCREENING_FC,
            batch_inputs,
            tool=tool,
            max_concurrency=max_concurrency,
        )

    def _phase2_screen(
        self,
        papers: List[Paper],
        pico_dict: dict,
        content_criteria: List[str],
        max_concurrency: int,
    ) -> List[dict]:
        """Phase 2: evaluate surviving papers using title+abstract + content_criteria."""
        n = len(content_criteria)
        criteria_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(content_criteria))
        tool = _build_screening_tool(n)

        batch_inputs = [
            {
                **pico_dict,
                "paper_content": f"Title: {paper.title}\n\nAbstract: {paper.abstract}",
                "criteria_text": criteria_text,
                "num_criteria": n,
            }
            for paper in papers
        ]

        return batch_function_call_llm(
            LITERATURE_SCREENING_FC,
            batch_inputs,
            tool=tool,
            max_concurrency=max_concurrency,
        )

    def _parse_evals(self, raw: dict, n: int) -> List[str]:
        """Validate and normalise LLM evaluation output to a list of length n."""
        evals = raw.get("evaluations", [])
        evals = [str(e).upper() for e in evals]
        evals = [e if e in ("YES", "NO", "UNCERTAIN") else "UNCERTAIN" for e in evals]
        if len(evals) != n:
            evals = ["UNCERTAIN"] * n
        return evals

    def _merge_decisions(
        self,
        papers: List[Paper],
        phase1_evals_list: List[List[str]],
        phase2_evals_map: dict,
        n_content: int,
    ) -> List[PaperDecision]:
        decisions = []
        for i, paper in enumerate(papers):
            title_evals = phase1_evals_list[i]
            content_evals = phase2_evals_map.get(i, ["SKIPPED"] * n_content)
            all_evals = title_evals + content_evals
            decisions.append(
                PaperDecision(
                    pmid=paper.pmid,
                    title=paper.title,
                    evaluations=all_evals,
                    decision=_decide(all_evals),
                )
            )
        return decisions

    # ------------------------------------------------------------------
    # Streaming entry point (yields SSE-friendly dicts)
    # ------------------------------------------------------------------

    def run_stream(
        self,
        papers: List[Paper],
        pico: PICODefinition,
        num_title_criteria: int = 3,
        num_content_criteria: int = 3,
        max_concurrency: int = 50,
    ) -> Generator[dict, None, None]:
        """
        Generator that yields progress events for SSE streaming.

        Event types:
          {"type": "criteria",       "data": CriteriaSet.model_dump()}
          {"type": "phase1_done",    "data": {"total": N, "passed": M}}
          {"type": "paper_done",     "data": PaperDecision.model_dump()}
          {"type": "summary",        "data": ScreeningSummary.model_dump()}
          {"type": "done"}
          {"type": "error",          "data": "<message>"}
        """
        self.reset()

        if not papers:
            yield {"type": "summary", "data": {"total": 0, "included": 0, "excluded": 0, "uncertain": 0}}
            yield {"type": "done"}
            return

        pico_dict = {"P": pico.P, "I": pico.I, "C": pico.C, "O": pico.O}

        # Phase 0 – generate criteria
        try:
            criteria = self._generate_criteria(pico_dict, num_title_criteria, num_content_criteria)
        except Exception as exc:
            logger.exception("run_stream: _generate_criteria failed")
            yield {"type": "error", "data": str(exc)}
            return

        yield {"type": "criteria", "data": criteria.model_dump()}

        title_criteria = criteria.title_criteria
        content_criteria = criteria.content_criteria
        n_title = len(title_criteria)
        n_content = len(content_criteria)

        # Phase 1 – title-only screening (all papers)
        try:
            phase1_raw = self._phase1_screen(papers, pico_dict, title_criteria, max_concurrency)
        except Exception as exc:
            logger.exception("run_stream: phase1 failed")
            yield {"type": "error", "data": str(exc)}
            return

        phase1_evals_list: List[List[str]] = []
        phase2_indices: List[int] = []
        for i, raw in enumerate(phase1_raw):
            evals = self._parse_evals(raw, n_title)
            phase1_evals_list.append(evals)
            if "NO" not in evals:
                phase2_indices.append(i)

        yield {
            "type": "phase1_done",
            "data": {"total": len(papers), "passed": len(phase2_indices)},
        }

        # Phase 2 – abstract screening (surviving papers only)
        phase2_papers = [papers[i] for i in phase2_indices]
        try:
            phase2_raw = (
                self._phase2_screen(phase2_papers, pico_dict, content_criteria, max_concurrency)
                if phase2_papers and n_content > 0
                else []
            )
        except Exception as exc:
            logger.exception("run_stream: phase2 failed")
            yield {"type": "error", "data": str(exc)}
            return

        phase2_evals_map: dict = {}
        for idx, raw in zip(phase2_indices, phase2_raw):
            phase2_evals_map[idx] = self._parse_evals(raw, n_content)

        # Yield per-paper decisions
        all_decisions: List[PaperDecision] = []
        for i, paper in enumerate(papers):
            title_evals = phase1_evals_list[i]
            content_evals = phase2_evals_map.get(i, ["SKIPPED"] * n_content)
            all_evals = title_evals + content_evals
            decision = PaperDecision(
                pmid=paper.pmid,
                title=paper.title,
                evaluations=all_evals,
                decision=_decide(all_evals),
            )
            all_decisions.append(decision)
            yield {"type": "paper_done", "data": decision.model_dump()}

        # Summary
        summary = ScreeningSummary(
            total=len(all_decisions),
            included=sum(1 for d in all_decisions if d.decision == "INCLUDE"),
            excluded=sum(1 for d in all_decisions if d.decision == "EXCLUDE"),
            uncertain=sum(1 for d in all_decisions if d.decision == "UNCERTAIN"),
        )
        yield {"type": "summary", "data": summary.model_dump()}
        yield {"type": "done"}
