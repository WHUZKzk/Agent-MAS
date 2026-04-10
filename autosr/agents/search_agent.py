"""
SearchAgent — orchestrates the full literature search pipeline.

Mirrors TrialMind's SearchQueryGeneration + PubmedAPIWrapper flow but wrapped
in an agent class with state tracking and structured output.

Pipeline:
  1. Generate initial search terms from PICO          (call_llm → PRIMARY_TERM_EXTRACTION)
  2. Fetch 7 reference papers from PubMed             (ReqPubmedID + ReqPubmedFull)
  3. Refine + expand terms using reference context    (call_llm → SEARCH_TERM_EXTRACTION)
  4. Build keyword_map and run main PubMed search     (PubmedAPIWrapper)
  5. Fetch full metadata for retrieved PMIDs          (pmid2papers)
  6. Return structured SearchResult
"""

import json
import re
import logging
from typing import Optional, Tuple

from autosr.agents.base_agent import BaseAgent
from autosr.schemas.models import PICODefinition, Paper, SearchResult, SearchTerms
from autosr.tools.llm import call_llm
from autosr.tools.pubmed import ReqPubmedID, ReqPubmedFull, PubmedAPIWrapper, pmid2papers
from autosr.prompts.search_query import PRIMARY_TERM_EXTRACTION, SEARCH_TERM_EXTRACTION
from configs.settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON parsing helper (adapted from TrialMind's parse_json_outputs)
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> Optional[dict]:
    """Try to extract a JSON object from LLM response text."""
    # 1. ```json ... ```
    m = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 2. First { … } block
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    # 3. Whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


class SearchAgent(BaseAgent):
    """
    Runs the full literature search pipeline for a given PICO definition.

    Usage::

        agent = SearchAgent()
        result = agent.run(pico, retmax=1000)
        # result.papers  → list of Paper objects
        # result.search_terms → populations / interventions / outcomes
    """

    def __init__(self):
        super().__init__("SearchAgent")
        self._req_id = ReqPubmedID()
        self._req_full = ReqPubmedFull()
        self._wrapper = PubmedAPIWrapper()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        pico: PICODefinition,
        retmax: int = 1000,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        fetch_all: bool = False,
    ) -> SearchResult:
        self.reset()
        pico_dict = {"P": pico.P, "I": pico.I, "C": pico.C, "O": pico.O}

        # Step 1 – generate initial terms
        initial_terms = self._run_step(
            "generate_initial_terms",
            self._generate_initial_terms,
            pico_dict,
        )

        # Step 2 – fetch reference papers
        ref_pmids = self._run_step(
            "fetch_reference_ids",
            self._fetch_reference_ids,
            initial_terms,
        )
        ref_text = self._run_step(
            "fetch_reference_texts",
            self._fetch_reference_texts,
            ref_pmids,
        )

        # Step 3 – generate refined + expanded terms
        search_terms = self._run_step(
            "generate_refined_terms",
            self._generate_refined_terms,
            pico_dict,
            ref_text,
        )

        # Step 4 – build query and search PubMed
        pmid_list, query_url, total_count = self._run_step(
            "pubmed_search",
            self._pubmed_search,
            search_terms,
            retmax,
            min_year,
            max_year,
            fetch_all,
        )

        # Step 5 – fetch paper metadata
        papers = self._run_step(
            "fetch_paper_metadata",
            self._fetch_papers,
            pmid_list,
        )

        logger.info(
            "[SearchAgent] Done: %d papers retrieved (total in PubMed: %d) in %.1fs",
            len(papers),
            total_count,
            self.state.elapsed,
        )

        return SearchResult(
            query_url=query_url,
            total_count=total_count,
            retrieved_count=len(papers),
            papers=papers,
            search_terms=search_terms,
        )

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    def _generate_initial_terms(self, pico_dict: dict) -> list:
        raw = call_llm(PRIMARY_TERM_EXTRACTION, pico_dict)
        parsed = _extract_json(raw)
        terms = parsed.get("terms", []) if parsed else []
        if not terms:
            # Fallback: use PICO elements as rough terms
            logger.warning("PRIMARY_TERM_EXTRACTION returned no terms; using fallback")
            terms = [pico_dict["I"].split()[0]] if pico_dict.get("I") else ["systematic review"]
        logger.info("Initial search terms: %s", terms)
        return terms

    def _fetch_reference_ids(self, terms: list) -> list:
        combined = "+AND+".join(f"({t})" for t in terms)
        pmids = self._req_id.run(term=combined, retmax=7)
        logger.info("Reference PMIDs: %s", pmids)
        return pmids

    def _fetch_reference_texts(self, pmids: list) -> str:
        if not pmids:
            return "(No reference papers retrieved)"
        papers = self._req_full.run(pmids)
        lines = [
            f"{i+1}. {p['title']}\nAbstract: {p['abstract']}"
            for i, p in enumerate(papers)
        ]
        return "\n\n".join(lines)

    def _generate_refined_terms(self, pico_dict: dict, ref_text: str) -> SearchTerms:
        inputs = {**pico_dict, "pubmed_reference_text": ref_text}
        raw = call_llm(SEARCH_TERM_EXTRACTION, inputs)
        parsed = _extract_json(raw)

        if not parsed:
            logger.warning("SEARCH_TERM_EXTRACTION returned unparseable JSON; using empty terms")
            return SearchTerms()

        step2 = parsed.get("step 2", {})
        step3 = parsed.get("step 3", {})

        populations   = list(set(step2.get("CORE_POPULATION",   []) + step3.get("EXPAND_POPULATION",   [])))
        interventions = list(set(step2.get("CORE_INTERVENTION",  []) + step3.get("EXPAND_INTERVENTION",  [])))
        outcomes      = list(set(step2.get("CORE_OUTCOME",       []) + step3.get("EXPAND_OUTCOME",       [])))

        logger.info(
            "Refined terms — pop: %d, int: %d, out: %d",
            len(populations), len(interventions), len(outcomes),
        )
        return SearchTerms(
            populations=populations,
            interventions=interventions,
            outcomes=outcomes,
        )

    def _pubmed_search(
        self,
        search_terms: SearchTerms,
        retmax: int,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        fetch_all: bool = False,
    ):
        keyword_map = {}
        if search_terms.populations:
            keyword_map["population"] = search_terms.populations
        if search_terms.interventions:
            keyword_map["intervention"] = search_terms.interventions
        if search_terms.outcomes:
            keyword_map["outcome"] = search_terms.outcomes

        inputs: dict = {"keyword_map": keyword_map, "page_size": retmax}
        if min_year:
            inputs["min_date"] = str(min_year)
        if max_year:
            inputs["max_date"] = str(max_year)

        if fetch_all:
            logger.info("fetch_all=True: retrieving all PMIDs via pagination")
            pmid_list, query_url, total_count = self._wrapper.search_all(inputs)
        else:
            pmid_list, query_url, total_count = self._wrapper.search(inputs)

        return pmid_list, query_url, total_count

    def _fetch_papers(self, pmid_list: list) -> list:
        if not pmid_list:
            return []
        api_key = settings.pubmed_api_key
        df = pmid2papers(pmid_list, api_key=api_key)
        if df.empty:
            return []

        papers = []
        for _, row in df.iterrows():
            papers.append(
                Paper(
                    pmid=str(row.get("PMID", "")),
                    title=str(row.get("Title", "")),
                    abstract=str(row.get("Abstract", "")),
                    authors=str(row.get("Authors", "")),
                    year=str(row.get("Year", "")),
                    journal=str(row.get("Journal", "")),
                    publication_type=str(row.get("PublicationType", "")),
                )
            )
        return papers
