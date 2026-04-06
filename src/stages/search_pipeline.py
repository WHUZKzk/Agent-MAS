"""
Search Pipeline — 7-node DAG implementation.

Spec: docs/05_SEARCH_STAGE.md

Nodes:
  1.1 (Soft)  PICO Term Generation
  1.2 (Hard)  MeSH Validation & Alignment
  1.3 (Hard)  Boolean Query Construction
  1.4 (Hard)  PubMed Search Execution + TF-IDF seed selection
  1.5 (Hard+Soft) Pearl Growing
  1.6 (Hard)  Final Search Execution
  1.7 (Hard)  Deduplication & PRISMA

Public module-level functions (used by nodes and tested independently):
  align_pico_terms(pico_term_set, pubmed_client) → PICOTermSet   [async]
  build_boolean_query(pico_term_set) → str                        [sync]
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz import fuzz

from src.clients.pubmed_client import MeSHResult, PubMedClient
from src.engine.agents import ExecutorAgent, ReviewerAdjudicatorAgent
from src.engine.context_manager import ContextManager, MountedContext
from src.engine.dag import DAGRunner
from src.engine.model_registry import ModelRegistry
from src.engine.nodes import HardNode, SoftNode
from src.schemas.common import (
    DAGDefinition, EdgeDefinition, NodeDefinition, PaperMetadata, ReviewConfig,
)
from src.schemas.search import (
    AugmentedTerm, PearlGrowingOutput, PICOTerm, PICOTermSet,
    PRISMASearchData, QueryRecord, SearchOutput,
)

logger = logging.getLogger("autosr.search_pipeline")

_FUZZY_THRESHOLD = 80.0


# ─────────────────────────────────────────────────────────────────────────────
# Module-level pure functions (testable independently)
# ─────────────────────────────────────────────────────────────────────────────

async def _align_single_term(
    term: PICOTerm,
    client: PubMedClient,
) -> PICOTerm:
    """
    Three-level MeSH alignment for a single PICOTerm.
    Returns a NEW PICOTerm (does not mutate the input).

    Level 1 – Exact MeSH match:
        validate_mesh(original) → found → valid_mesh
    Level 2 – Spell-corrected match:
        spell_check → corrected → validate_mesh(corrected) → found → mapped
    Level 3 – Fuzzy match (≥ threshold) against NCBI-returned descriptors:
        token_sort_ratio(original, descriptor/entry_terms) ≥ 80 → fuzzy_mapped
    Fallback – not_found, search_field=[tiab]
    """
    original = term.original

    # ── Level 1 ──────────────────────────────────────────────────────────────
    result1: MeSHResult = await client.validate_mesh(original)
    if result1.found and result1.descriptor_name:
        return term.model_copy(update={
            "status": "valid_mesh",
            "normalized": result1.descriptor_name,
            "search_field": "[MeSH Terms]",
        })

    # Keep any NCBI candidates from Level 1 for Level 3 fuzzy matching
    ncbi_candidates: List[str] = []
    if result1.descriptor_name:
        ncbi_candidates.append(result1.descriptor_name)
    ncbi_candidates.extend(result1.entry_terms)

    # ── Level 2 ──────────────────────────────────────────────────────────────
    corrected: Optional[str] = await client.spell_check(original)
    if corrected:
        result2: MeSHResult = await client.validate_mesh(corrected)
        if result2.found and result2.descriptor_name:
            return term.model_copy(update={
                "status": "mapped",
                "normalized": result2.descriptor_name,
                "search_field": "[MeSH Terms]",
            })
        # Accumulate candidates from the corrected lookup too
        if result2.descriptor_name:
            ncbi_candidates.append(result2.descriptor_name)
        ncbi_candidates.extend(result2.entry_terms)

    # ── Level 3 ──────────────────────────────────────────────────────────────
    if ncbi_candidates:
        best_name, best_score = _best_fuzzy_match(original, ncbi_candidates)
        if best_score >= _FUZZY_THRESHOLD:
            return term.model_copy(update={
                "status": "fuzzy_mapped",
                "normalized": best_name,
                "search_field": "[MeSH Terms]",
                "similarity_score": best_score,
            })

    # ── Fallback ─────────────────────────────────────────────────────────────
    return term.model_copy(update={
        "status": "not_found",
        "search_field": "[tiab]",
    })


def _best_fuzzy_match(
    query: str, candidates: List[str]
) -> Tuple[str, float]:
    """Return (best_candidate, score) using token_sort_ratio."""
    best = max(
        candidates,
        key=lambda c: fuzz.token_sort_ratio(query.lower(), c.lower()),
    )
    score = float(fuzz.token_sort_ratio(query.lower(), best.lower()))
    return best, score


async def align_pico_terms(
    pico_term_set: PICOTermSet,
    pubmed_client: PubMedClient,
) -> PICOTermSet:
    """
    Run three-level MeSH alignment on every term in all PICO dimensions.
    Returns a new PICOTermSet (original is never mutated).

    MUST: No unverified term gets [MeSH Terms] search_field.
    """
    async def _align_list(terms: List[PICOTerm]) -> List[PICOTerm]:
        return [await _align_single_term(t, pubmed_client) for t in terms]

    return PICOTermSet(
        P=await _align_list(pico_term_set.P),
        I=await _align_list(pico_term_set.I),
        C=await _align_list(pico_term_set.C),
        O=await _align_list(pico_term_set.O),
    )


def build_boolean_query(pico_term_set: PICOTermSet) -> str:
    """
    Construct a PubMed Boolean query from a validated PICOTermSet.

    Rules (spec §4 Node 1.3):
    - MeSH term:  "normalized"[MeSH Terms]
    - tiab term:  "original"[tiab]
    - Within dimension: (term1 OR term2 OR …)
    - Between dimensions: (P) AND (I) AND (C) AND (O)
    """
    def _format_term(t: PICOTerm) -> str:
        if t.search_field == "[MeSH Terms]":
            name = t.normalized or t.original
            return f'"{name}"[MeSH Terms]'
        return f'"{t.original}"[tiab]'

    def _dim_query(terms: List[PICOTerm]) -> str:
        parts = [_format_term(t) for t in terms]
        return "(" + " OR ".join(parts) + ")"

    dims = [
        _dim_query(pico_term_set.P),
        _dim_query(pico_term_set.I),
        _dim_query(pico_term_set.C),
        _dim_query(pico_term_set.O),
    ]
    return " AND ".join(dims)


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF Top-K seed selection
# ─────────────────────────────────────────────────────────────────────────────

def _select_seed_papers(
    papers: List[PaperMetadata],
    review_abstract: str,
    top_k: int = 20,
) -> List[PaperMetadata]:
    """
    Select the top-K papers most similar to the review abstract using
    local TF-IDF cosine similarity.

    MUST use local scoring, NOT PubMed's sort=relevance (spec §5).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    if not papers:
        return []

    docs = [p.abstract or p.title for p in papers]
    query_doc = [review_abstract]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
    try:
        tfidf_matrix = vectorizer.fit_transform(docs + query_doc)
    except ValueError:
        return papers[:top_k]

    query_vec = tfidf_matrix[-1]
    paper_vecs = tfidf_matrix[:-1]
    scores = cosine_similarity(query_vec, paper_vecs).flatten()

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [papers[i] for i in top_indices]


# ─────────────────────────────────────────────────────────────────────────────
# MeSH gap analysis (Node 1.5 Sub-step A)
# ─────────────────────────────────────────────────────────────────────────────

def _find_mesh_gaps(
    seed_papers: List[PaperMetadata],
    current_pico: PICOTermSet,
    min_frequency: int = 3,
) -> List[str]:
    """
    Find MeSH terms appearing in ≥ min_frequency seed papers but NOT already
    in the current PICOTermSet.
    """
    all_current = set()
    for dim in (current_pico.P, current_pico.I, current_pico.C, current_pico.O):
        for t in dim:
            all_current.add((t.normalized or t.original).lower())
            all_current.add(t.original.lower())

    counts: Counter = Counter()
    for paper in seed_papers:
        for term in paper.mesh_terms:
            if term.lower() not in all_current:
                counts[term] += 1

    return [term for term, count in counts.most_common() if count >= min_frequency]


# ─────────────────────────────────────────────────────────────────────────────
# Node implementations (callables registered in the DAGRunner node_registry)
# ─────────────────────────────────────────────────────────────────────────────

class _Node11_PicoGeneration:
    """
    Node 1.1 — Soft Node: PICO Term Generation.
    The DAGRunner mounts the skill and calls us; we pull the mounted context,
    call the agent, parse the response, and return updated state.
    """
    def __init__(self, context_manager: ContextManager, agent: ExecutorAgent) -> None:
        self._cm = context_manager
        self._agent = agent

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        mounted: MountedContext = self._cm._current_mount
        raw = self._agent.call(mounted)
        pico_terms = PICOTermSet.model_validate_json(raw)
        return {**state, "pico_terms": pico_terms}


class _Node12_MeSHValidation:
    """Node 1.2 — Hard Node: three-level MeSH alignment."""
    def __init__(self, pubmed_client: PubMedClient) -> None:
        self._client = pubmed_client

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pico_terms: PICOTermSet = state["pico_terms"]
        validated = asyncio.run(align_pico_terms(pico_terms, self._client))
        logger.info(
            "[Node 1.2] MeSH alignment complete. "
            "valid_mesh=%d, mapped=%d, fuzzy=%d, not_found=%d",
            *_count_statuses(validated),
        )
        return {**state, "pico_terms": validated}


class _Node13_BooleanQuery:
    """Node 1.3 — Hard Node: Boolean query construction."""
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pico_terms: PICOTermSet = state["pico_terms"]
        query_str = build_boolean_query(pico_terms)
        logger.info("[Node 1.3] Query: %s", query_str[:200])
        return {**state, "initial_query_string": query_str}


class _Node14_SearchExecution:
    """
    Node 1.4 — Hard Node: PubMed search execution + TF-IDF seed selection.
    assert search_result.total_count > 0 (DAGRunner catches AssertionError → FAILED).
    """
    def __init__(self, pubmed_client: PubMedClient, review_config: ReviewConfig) -> None:
        self._client = pubmed_client
        self._review_config = review_config

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query_str: str = state["initial_query_string"]

        async def _run():
            search_res = await self._client.search(query_str)
            assert search_res.total_count > 0, (
                f"PubMed search returned 0 results for query: {query_str[:100]}"
            )
            all_papers = await self._client.fetch_all(
                search_res.webenv, search_res.query_key, search_res.total_count
            )
            return search_res, all_papers

        search_res, all_papers = asyncio.run(_run())

        papers_dict: Dict[str, PaperMetadata] = {p.pmid: p for p in all_papers}
        initial_pmids = list(papers_dict)

        record = QueryRecord(
            query_string=query_str,
            stage="initial",
            result_count=search_res.total_count,
            webenv=search_res.webenv,
            query_key=search_res.query_key,
        )

        seed_papers = _select_seed_papers(
            all_papers, self._review_config.abstract, top_k=20
        )
        logger.info(
            "[Node 1.4] Found %d papers; selected %d seeds.",
            len(all_papers), len(seed_papers),
        )

        return {
            **state,
            "papers": papers_dict,
            "initial_pmids": initial_pmids,
            "query_history": [record],
            "seed_papers": seed_papers,
        }


class _Node15_PearlGrowing:
    """
    Node 1.5 — Composite (Hard sub-step A + Soft sub-step B).
    Sub-step A: extract MeSH gaps from seed papers.
    Sub-step B: LLM call to identify additional keywords.
    Sub-step C: run new terms through MeSH alignment.
    """
    def __init__(
        self,
        context_manager: ContextManager,
        agent: ExecutorAgent,
        pubmed_client: PubMedClient,
    ) -> None:
        self._cm = context_manager
        self._agent = agent
        self._client = pubmed_client

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        seed_papers: List[PaperMetadata] = state.get("seed_papers", [])
        pico_terms: PICOTermSet = state["pico_terms"]

        # Sub-step A (Hard): MeSH gap analysis
        mesh_gaps = _find_mesh_gaps(seed_papers, pico_terms, min_frequency=3)
        logger.info("[Node 1.5-A] Found %d MeSH gaps.", len(mesh_gaps))

        # Prepare LLM input context
        top10_abstracts = "\n\n---\n\n".join(
            f"[{p.pmid}] {p.title}\n{p.abstract}"
            for p in seed_papers[:10]
        )
        state_with_pearl = {
            **state,
            "top_seed_abstracts": top10_abstracts,
            "mesh_gap_terms": json.dumps(mesh_gaps[:30]),
        }

        # Sub-step B (Soft): LLM keyword augmentation
        mounted: MountedContext = self._cm._current_mount
        raw = self._agent.call(mounted)

        try:
            pearl_output = PearlGrowingOutput.model_validate_json(raw)
        except Exception as exc:
            logger.error("[Node 1.5-B] Failed to parse LLM output: %s", exc)
            return {**state, "augmented_terms": []}

        # Sub-step C (Hard): MeSH align new terms before adding to query
        new_pico_terms = self._align_new_terms(pearl_output.new_terms, pico_terms)
        logger.info(
            "[Node 1.5-C] %d augmented terms after MeSH alignment.",
            sum(len(v) for v in new_pico_terms.values()),
        )

        return {**state, "augmented_terms": new_pico_terms}

    def _align_new_terms(
        self,
        new_terms: List[AugmentedTerm],
        current_pico: PICOTermSet,
    ) -> Dict[str, List[PICOTerm]]:
        """Align new terms and return them grouped by PICO dimension."""
        by_dim: Dict[str, List[AugmentedTerm]] = {d: [] for d in "PICO"}
        for t in new_terms:
            by_dim[t.dimension].append(t)

        result: Dict[str, List[PICOTerm]] = {}
        for dim, terms in by_dim.items():
            if not terms:
                continue
            raw_terms = [
                PICOTerm(original=t.term, status="not_found", search_field="[tiab]")
                for t in terms
            ]

            async def _run(raw=raw_terms):
                return [await _align_single_term(t, self._client) for t in raw]

            result[dim] = asyncio.run(_run())
        return result


class _Node16_FinalSearch:
    """
    Node 1.6 — Hard Node: rebuild query with augmented terms, re-execute.
    Constraint: 10 < result_count < 50,000 (log warning if out of range).
    """
    def __init__(self, pubmed_client: PubMedClient) -> None:
        self._client = pubmed_client

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pico_terms: PICOTermSet = state["pico_terms"]
        augmented: Dict[str, List[PICOTerm]] = state.get("augmented_terms", {})

        # Merge augmented terms into existing PICOTermSet
        merged_pico = _merge_pico_terms(pico_terms, augmented)
        final_query = build_boolean_query(merged_pico)

        async def _run():
            search_res = await self._client.search(final_query)
            if search_res.total_count <= 10:
                logger.warning("[Node 1.6] Very few results: %d", search_res.total_count)
            elif search_res.total_count >= 50000:
                logger.warning("[Node 1.6] Very many results: %d — consider refining query",
                               search_res.total_count)
            new_papers = await self._client.fetch_all(
                search_res.webenv, search_res.query_key, search_res.total_count
            )
            return search_res, new_papers

        search_res, new_papers = asyncio.run(_run())

        # Merge with existing papers dict
        papers_dict: Dict[str, PaperMetadata] = dict(state.get("papers", {}))
        final_pmids_set = set(papers_dict)
        for p in new_papers:
            if p.pmid not in papers_dict:
                papers_dict[p.pmid] = p
        final_pmids_set.update(p.pmid for p in new_papers)

        record = QueryRecord(
            query_string=final_query,
            stage="augmented",
            result_count=search_res.total_count,
            webenv=search_res.webenv,
            query_key=search_res.query_key,
        )
        history = list(state.get("query_history", [])) + [record]

        return {
            **state,
            "final_query_string": final_query,
            "augmented_pmids": [p.pmid for p in new_papers],
            "papers": papers_dict,
            "query_history": history,
            "final_pico_terms": merged_pico,
        }


class _Node17_Deduplication:
    """
    Node 1.7 — Hard Node: deduplicate, finalize PRISMA numbers, produce SearchOutput.
    """
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        initial_pmids: List[str] = state.get("initial_pmids", [])
        augmented_pmids: List[str] = state.get("augmented_pmids", [])
        papers: Dict[str, PaperMetadata] = state.get("papers", {})

        # Merge & deduplicate
        all_pmids = list(dict.fromkeys(initial_pmids + augmented_pmids))
        deduped_papers = {pmid: papers[pmid] for pmid in all_pmids if pmid in papers}

        prisma = PRISMASearchData(
            initial_query_results=len(initial_pmids),
            augmented_query_results=len(augmented_pmids),
            after_deduplication=len(all_pmids),
            final_candidate_count=len(deduped_papers),
        )

        search_output = SearchOutput(
            pmids=all_pmids,
            papers=deduped_papers,
            query_history=state.get("query_history", []),
            pico_terms=state.get("final_pico_terms", state["pico_terms"]),
            prisma_numbers=prisma,
        )

        logger.info(
            "[Node 1.7] Deduplication complete: %d initial + %d augmented "
            "→ %d after dedup → %d with metadata",
            len(initial_pmids), len(augmented_pmids),
            len(all_pmids), len(deduped_papers),
        )

        return {**state, "search_output": search_output}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _merge_pico_terms(
    base: PICOTermSet,
    augmented: Dict[str, List[PICOTerm]],
) -> PICOTermSet:
    """Merge augmented terms into a PICOTermSet, deduplicating by original."""
    def _merge_dim(base_terms: List[PICOTerm], new_terms: List[PICOTerm]) -> List[PICOTerm]:
        existing = {t.original.lower() for t in base_terms}
        deduped = [t for t in new_terms if t.original.lower() not in existing]
        return base_terms + deduped

    return PICOTermSet(
        P=_merge_dim(base.P, augmented.get("P", [])),
        I=_merge_dim(base.I, augmented.get("I", [])),
        C=_merge_dim(base.C, augmented.get("C", [])),
        O=_merge_dim(base.O, augmented.get("O", [])),
    )


def _count_statuses(pico: PICOTermSet) -> Tuple[int, int, int, int]:
    """Return (valid_mesh, mapped, fuzzy_mapped, not_found) counts."""
    counts = Counter(
        t.status
        for dim in (pico.P, pico.I, pico.C, pico.O)
        for t in dim
    )
    return (
        counts["valid_mesh"],
        counts["mapped"],
        counts["fuzzy_mapped"],
        counts["not_found"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# DAG definition
# ─────────────────────────────────────────────────────────────────────────────

SEARCH_DAG = DAGDefinition(
    dag_id="search_pipeline",
    entry_node="s1_1",
    terminal_nodes=["s1_7"],
    nodes=[
        NodeDefinition(
            node_id="s1_1", node_type="soft",
            skill_id="search.pico_generation",
            implementation="stages.search_pipeline._Node11_PicoGeneration",
            description="Generate PICO search terms from review question",
        ),
        NodeDefinition(
            node_id="s1_2", node_type="hard",
            implementation="stages.search_pipeline._Node12_MeSHValidation",
            description="Three-level MeSH alignment for all PICO terms",
        ),
        NodeDefinition(
            node_id="s1_3", node_type="hard",
            implementation="stages.search_pipeline._Node13_BooleanQuery",
            description="Construct PubMed Boolean query from validated PICO terms",
        ),
        NodeDefinition(
            node_id="s1_4", node_type="hard",
            implementation="stages.search_pipeline._Node14_SearchExecution",
            description="Execute initial PubMed search + TF-IDF seed selection",
        ),
        NodeDefinition(
            node_id="s1_5", node_type="soft",
            skill_id="search.pearl_growing",
            implementation="stages.search_pipeline._Node15_PearlGrowing",
            description="Pearl Growing: MeSH gap analysis + LLM keyword augmentation",
        ),
        NodeDefinition(
            node_id="s1_6", node_type="hard",
            implementation="stages.search_pipeline._Node16_FinalSearch",
            description="Re-execute search with augmented terms",
        ),
        NodeDefinition(
            node_id="s1_7", node_type="hard",
            implementation="stages.search_pipeline._Node17_Deduplication",
            description="Deduplicate, finalize PRISMA numbers, produce SearchOutput",
        ),
    ],
    edges=[
        EdgeDefinition(from_node="s1_1", to_node="s1_2"),
        EdgeDefinition(from_node="s1_2", to_node="s1_3"),
        EdgeDefinition(from_node="s1_3", to_node="s1_4"),
        EdgeDefinition(from_node="s1_4", to_node="s1_5"),
        EdgeDefinition(from_node="s1_5", to_node="s1_6"),
        EdgeDefinition(from_node="s1_6", to_node="s1_7"),
    ],
)


# ─────────────────────────────────────────────────────────────────────────────
# SearchPipeline class
# ─────────────────────────────────────────────────────────────────────────────

class SearchPipeline:
    """
    Wires the 7-node Search DAG and exposes a single run() method.

    Usage:
        pipeline = SearchPipeline(review_config, context_manager,
                                  model_registry, pubmed_client)
        search_output = pipeline.run()
    """

    def __init__(
        self,
        review_config: ReviewConfig,
        context_manager: ContextManager,
        model_registry: ModelRegistry,
        pubmed_client: PubMedClient,
        progress_callback: Optional[Any] = None,
    ) -> None:
        self._review_config = review_config
        self._cm = context_manager
        self._progress_callback = progress_callback

        # Build executor agent (for PICO gen and Pearl Growing)
        exec_cfg = model_registry.get_default("executor")
        exec_name = model_registry.default_name("executor")
        self._executor = ExecutorAgent(model_id=exec_name, model_config=exec_cfg)

        # Node instances
        node11 = _Node11_PicoGeneration(context_manager, self._executor)
        node12 = _Node12_MeSHValidation(pubmed_client)
        node13 = _Node13_BooleanQuery()
        node14 = _Node14_SearchExecution(pubmed_client, review_config)
        node15 = _Node15_PearlGrowing(context_manager, self._executor, pubmed_client)
        node16 = _Node16_FinalSearch(pubmed_client)
        node17 = _Node17_Deduplication()

        self._node_registry = {
            "s1_1": node11,
            "s1_2": node12,
            "s1_3": node13,
            "s1_4": node14,
            "s1_5": node15,
            "s1_6": node16,
            "s1_7": node17,
        }

    def run(self) -> SearchOutput:
        """Execute the full Search DAG and return SearchOutput."""
        runner = DAGRunner(
            dag=SEARCH_DAG,
            context_manager=self._cm,
            node_registry=self._node_registry,
            progress_callback=self._progress_callback,
        )
        initial_state = {"review_config": self._review_config}
        final_state = runner.run(initial_state)
        return final_state["search_output"]
