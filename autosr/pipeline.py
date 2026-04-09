"""
AutoSR pipeline — chains SearchAgent and ScreeningAgent for a complete run.
"""

import logging
from typing import Optional

from autosr.agents.search_agent import SearchAgent
from autosr.agents.screening_agent import ScreeningAgent
from autosr.schemas.models import PICODefinition, SearchResult, ScreeningResult

logger = logging.getLogger(__name__)


class AutoSRPipeline:
    """
    Run the full AutoSR pipeline: Search → Screen.

    Usage::

        pipeline = AutoSRPipeline()
        search_result, screen_result = pipeline.run(pico)
    """

    def __init__(self):
        self.search_agent = SearchAgent()
        self.screening_agent = ScreeningAgent()

    def run(
        self,
        pico: PICODefinition,
        retmax: int = 1000,
        num_title_criteria: int = 3,
        num_content_criteria: int = 3,
        batch_size: int = 10,
    ) -> tuple[SearchResult, ScreeningResult]:
        logger.info("=== AutoSR Pipeline START ===")

        search_result = self.search_agent.run(pico, retmax=retmax)
        logger.info("Search complete: %d papers", len(search_result.papers))

        screen_result = self.screening_agent.run(
            papers=search_result.papers,
            pico=pico,
            num_title_criteria=num_title_criteria,
            num_content_criteria=num_content_criteria,
            batch_size=batch_size,
        )
        logger.info(
            "Screening complete: %d included / %d excluded / %d uncertain",
            screen_result.summary.included,
            screen_result.summary.excluded,
            screen_result.summary.uncertain,
        )

        logger.info("=== AutoSR Pipeline DONE ===")
        return search_result, screen_result
