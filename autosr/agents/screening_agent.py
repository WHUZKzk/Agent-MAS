"""
ScreeningAgent — orchestrates literature screening based on PICO-derived criteria.

Mirrors TrialMind's ScreeningCriteriaGeneration + LiteratureScreening but wrapped
in an agent class with structured output.

Pipeline:
  1. Generate eligibility criteria from PICO           (call_llm → SCREENING_CRITERIA_GENERATION)
  2. Format paper content (title + abstract)
  3. Batch-screen all papers against all criteria      (batch_function_call_llm → LITERATURE_SCREENING_FC)
  4. Decide INCLUDE / EXCLUDE / UNCERTAIN per paper    (deterministic rule)
  5. Return structured ScreeningResult
"""

import json
import re
import logging
from typing import List, Optional

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
# ---------------------------------------------------------------------------

def _decide(evaluations: List[str]) -> str:
    """
    INCLUDE  : all criteria are YES
    EXCLUDE  : at least one criterion is NO
    UNCERTAIN: some UNCERTAIN, no NO
    """
    evals = [e.upper() for e in evaluations]
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
    Screens a list of papers against PICO-derived eligibility criteria.

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
        batch_size: int = 10,
    ) -> ScreeningResult:
        self.reset()

        if not papers:
            return ScreeningResult(
                criteria=CriteriaSet(),
                decisions=[],
                summary=ScreeningSummary(total=0, included=0, excluded=0, uncertain=0),
            )

        pico_dict = {"P": pico.P, "I": pico.I, "C": pico.C, "O": pico.O}

        # Step 1 – generate eligibility criteria
        criteria = self._run_step(
            "generate_criteria",
            self._generate_criteria,
            pico_dict,
            num_title_criteria,
            num_content_criteria,
        )

        # Step 2 – batch screen papers
        all_criteria = criteria.title_criteria + criteria.content_criteria
        n_criteria = len(all_criteria)

        raw_decisions = self._run_step(
            "batch_screen_papers",
            self._batch_screen,
            papers,
            pico_dict,
            all_criteria,
            n_criteria,
            batch_size,
        )

        # Step 3 – apply decision rule (deterministic)
        decisions = self._run_step(
            "apply_decision_rule",
            self._apply_decisions,
            papers,
            raw_decisions,
            n_criteria,
        )

        # Summary
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

    def _batch_screen(
        self,
        papers: List[Paper],
        pico_dict: dict,
        all_criteria: List[str],
        n_criteria: int,
        batch_size: int,
    ) -> List[dict]:
        # Build criteria text (numbered list for the prompt)
        criteria_text = "\n".join(
            f"{i+1}. {c}" for i, c in enumerate(all_criteria)
        )
        # Build batch inputs
        batch_inputs = []
        for paper in papers:
            paper_content = f"Title: {paper.title}\n\nAbstract: {paper.abstract}"
            batch_inputs.append({
                **pico_dict,
                "paper_content": paper_content,
                "criteria_text": criteria_text,
                "num_criteria": n_criteria,
            })

        tool = _build_screening_tool(n_criteria)
        results = batch_function_call_llm(
            LITERATURE_SCREENING_FC,
            batch_inputs,
            tool=tool,
            batch_size=batch_size,
        )
        return results

    def _apply_decisions(
        self,
        papers: List[Paper],
        raw_decisions: List[dict],
        n_criteria: int,
    ) -> List[PaperDecision]:
        decisions = []
        for paper, raw in zip(papers, raw_decisions):
            evals = raw.get("evaluations", [])
            # Validate / fix length
            evals = [str(e).upper() for e in evals]
            evals = [e if e in ("YES", "NO", "UNCERTAIN") else "UNCERTAIN" for e in evals]
            if len(evals) != n_criteria:
                evals = ["UNCERTAIN"] * n_criteria

            decisions.append(
                PaperDecision(
                    pmid=paper.pmid,
                    title=paper.title,
                    evaluations=evals,
                    decision=_decide(evals),
                )
            )
        return decisions
