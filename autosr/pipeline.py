"""
AutoSR pipeline — chains SearchAgent and ScreeningAgentV2 for a complete run.
"""

import logging
from typing import Optional

from autosr.agents.search_agent import SearchAgent
from autosr.agents.screening_agent_v2 import ScreeningAgentV2
from autosr.schemas.models import (
    PICODefinition, StudyDesignFilter,
    SearchResult, ScreeningResultV2,
)

logger = logging.getLogger(__name__)


class AutoSRPipeline:
    """
    Run the full AutoSR pipeline: Search → Screen (Stages 0-2).

    Usage::

        pipeline = AutoSRPipeline()
        search_result, screen_result = pipeline.run(pico)
    """

    def __init__(self):
        self.search_agent = SearchAgent()
        self.screening_agent = ScreeningAgentV2()

    def run(
        self,
        pico: PICODefinition,
        retmax: int = 1000,
        study_design_filter: StudyDesignFilter = StudyDesignFilter.BOTH,
        max_concurrency: int = 50,
    ) -> tuple[SearchResult, ScreeningResultV2]:
        logger.info("=== AutoSR Pipeline START ===")

        search_result = self.search_agent.run(pico, retmax=retmax)
        logger.info("Search complete: %d papers", len(search_result.papers))

        screen_result = self.screening_agent.run(
            papers=search_result.papers,
            pico=pico,
            study_design_filter=study_design_filter,
            max_concurrency=max_concurrency,
        )
        logger.info(
            "Screening complete: %d included / %d excluded",
            screen_result.summary.final_included,
            screen_result.summary.final_excluded,
        )

        logger.info("=== AutoSR Pipeline DONE ===")
        return search_result, screen_result
