"""
TDD tests for Search Pipeline — Node 1.2 (MeSH Alignment) and Node 1.3 (Boolean Query).

Written BEFORE implementation. Must fail (ImportError) initially.

Imports tested:
  src.clients.pubmed_client   → MeSHResult, SearchResult
  src.stages.search_pipeline  → align_pico_terms, build_boolean_query
"""
import asyncio
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.schemas.search import PICOTerm, PICOTermSet

# ── TDD imports ─────────────────────────────────────────────────────────────
from src.clients.pubmed_client import MeSHResult                    # noqa: E402
from src.stages.search_pipeline import (                             # noqa: E402
    align_pico_terms,
    build_boolean_query,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_term(original: str) -> PICOTerm:
    """Create a pre-validation term (conservative defaults)."""
    return PICOTerm(
        original=original,
        status="not_found",
        search_field="[tiab]",
    )


def make_pico(
    P: List[str] = None,
    I: List[str] = None,
    C: List[str] = None,
    O: List[str] = None,
) -> PICOTermSet:
    return PICOTermSet(
        P=[make_term(t) for t in (P or ["adults"])],
        I=[make_term(t) for t in (I or ["exercise"])],
        C=[make_term(t) for t in (C or ["control"])],
        O=[make_term(t) for t in (O or ["IL-6"])],
    )


def mesh_result(
    found: bool,
    descriptor_name: Optional[str] = None,
    entry_terms: List[str] = None,
) -> MeSHResult:
    return MeSHResult(
        found=found,
        descriptor_name=descriptor_name,
        entry_terms=entry_terms or [],
    )


def make_mock_client(
    validate_mesh_by_term: dict = None,
    spell_check_by_term: dict = None,
):
    """
    Build a mock PubMedClient that dispatches by the term argument.

    Keys are the exact term strings passed to validate_mesh / spell_check.
    Unknown terms fall back to: mesh_result(False) / None respectively.
    This prevents mock exhaustion when align_pico_terms processes all 4 dimensions.
    """
    vm_map = validate_mesh_by_term or {}
    sc_map = spell_check_by_term or {}

    async def vm_effect(term):
        return vm_map.get(term, mesh_result(False))

    async def sc_effect(term):
        return sc_map.get(term, None)

    client = MagicMock()
    client.validate_mesh = AsyncMock(side_effect=vm_effect)
    client.spell_check   = AsyncMock(side_effect=sc_effect)
    return client


# ─────────────────────────────────────────────────────────────────────────────
# Node 1.2: MeSH Alignment — Three-Level Logic
# ─────────────────────────────────────────────────────────────────────────────

class TestMeSHAlignment:
    """
    Tests for align_pico_terms(pico_term_set, pubmed_client) → PICOTermSet.

    Spec: docs/05_SEARCH_STAGE.md §4 Node 1.2
    """

    # ── Level 1: Exact MeSH match ───────────────────────────────────────────

    def test_level1_exact_match_sets_valid_mesh(self):
        """
        When validate_mesh returns found=True, term gets:
          status="valid_mesh", search_field="[MeSH Terms]", normalized=descriptor_name
        """
        client = make_mock_client(
            validate_mesh_side_effect=[
                mesh_result(True, descriptor_name="Exercise"),
            ],
            spell_check_side_effect=[],
        )
        pico = make_pico(I=["Exercise"])
        result = asyncio.run(align_pico_terms(pico, client))

        term = result.I[0]
        assert term.status == "valid_mesh"
        assert term.search_field == "[MeSH Terms]"
        assert term.normalized == "Exercise"
        # spell_check must NOT be called (short-circuit after Level 1)
        client.spell_check.assert_not_called()

    def test_level1_normalized_uses_descriptor_not_original(self):
        """Normalized stores the official MeSH descriptor name, not the original."""
        client = make_mock_client(
            validate_mesh_side_effect=[
                mesh_result(True, descriptor_name="Diabetes Mellitus, Type 2"),
            ],
            spell_check_side_effect=[],
        )
        pico = make_pico(P=["type 2 diabetes"])
        result = asyncio.run(align_pico_terms(pico, client))
        assert result.P[0].normalized == "Diabetes Mellitus, Type 2"
        assert result.P[0].original == "type 2 diabetes"  # original unchanged

    # ── Level 2a: Spelling correction → found ───────────────────────────────

    def test_level2_spell_correction_sets_mapped(self):
        """
        When Level 1 fails but spelling correction leads to a MeSH hit:
          status="mapped", search_field="[MeSH Terms]"
        """
        client = make_mock_client(
            validate_mesh_side_effect=[
                mesh_result(False),                               # Level 1 miss
                mesh_result(True, descriptor_name="Hypertension"),# Level 2 hit
            ],
            spell_check_side_effect=["hypertension"],             # correction available
        )
        pico = make_pico(P=["hpertension"])   # typo
        result = asyncio.run(align_pico_terms(pico, client))

        term = result.P[0]
        assert term.status == "mapped"
        assert term.search_field == "[MeSH Terms]"
        assert term.normalized == "Hypertension"

    def test_level2_no_spell_correction_skips_to_level3(self):
        """When spell_check returns None, go directly to Level 3."""
        client = make_mock_client(
            validate_mesh_side_effect=[
                mesh_result(False, descriptor_name="Exercise Therapy",
                            entry_terms=["exercise therapy", "Exercise Therapy"]),
            ],
            spell_check_side_effect=[None],   # no correction
        )
        pico = make_pico(I=["exercise therap"])
        result = asyncio.run(align_pico_terms(pico, client))

        # "exercise therap" vs "Exercise Therapy" — token_sort_ratio is high
        term = result.I[0]
        assert term.status in ("fuzzy_mapped", "not_found")

    # ── Level 3: Fuzzy match ─────────────────────────────────────────────────

    def test_level3_high_score_sets_fuzzy_mapped(self):
        """
        When fuzzy score ≥ 80 against NCBI-returned descriptor/entry_terms:
          status="fuzzy_mapped", search_field="[MeSH Terms]", similarity_score set
        """
        # "Physycal Activity" → fuzzy matches "Physical Activity" (entry term)
        client = make_mock_client(
            validate_mesh_side_effect=[
                mesh_result(
                    False,
                    descriptor_name="Exercise",
                    entry_terms=["Physical Activity", "Motor Activity"],
                ),
            ],
            spell_check_side_effect=[None],   # no correction
        )
        pico = make_pico(I=["Physical Activty"])   # single typo
        result = asyncio.run(align_pico_terms(pico, client))

        term = result.I[0]
        assert term.status == "fuzzy_mapped"
        assert term.search_field == "[MeSH Terms]"
        assert term.similarity_score is not None
        assert term.similarity_score >= 80

    def test_level3_low_score_sets_not_found_tiab(self):
        """
        When fuzzy score < 80 across all NCBI candidates:
          status="not_found", search_field="[tiab]"
        """
        client = make_mock_client(
            validate_mesh_side_effect=[
                mesh_result(
                    False,
                    descriptor_name="Astrophysics",
                    entry_terms=["cosmic rays", "dark matter"],
                ),
            ],
            spell_check_side_effect=[None],
        )
        pico = make_pico(I=["exergame"])   # no close MeSH match
        result = asyncio.run(align_pico_terms(pico, client))

        term = result.I[0]
        assert term.status == "not_found"
        assert term.search_field == "[tiab]"

    def test_level3_no_ncbi_candidates_is_not_found(self):
        """When validate_mesh returns nothing at all → not_found."""
        client = make_mock_client(
            validate_mesh_side_effect=[
                mesh_result(False),    # no descriptor, no entry_terms
            ],
            spell_check_side_effect=[None],
        )
        pico = make_pico(O=["somecompletelyunknownterm"])
        result = asyncio.run(align_pico_terms(pico, client))

        term = result.O[0]
        assert term.status == "not_found"
        assert term.search_field == "[tiab]"

    # ── All dimensions processed ─────────────────────────────────────────────

    def test_all_pico_dimensions_are_processed(self):
        """align_pico_terms processes every term in every PICO dimension."""
        found_result = mesh_result(True, descriptor_name="X")
        client = make_mock_client(
            validate_mesh_side_effect=[found_result] * 4,  # 4 terms (one per dim)
            spell_check_side_effect=[],
        )
        pico = make_pico(P=["p1"], I=["i1"], C=["c1"], O=["o1"])
        result = asyncio.run(align_pico_terms(pico, client))

        for dim in (result.P, result.I, result.C, result.O):
            assert dim[0].status == "valid_mesh"

    def test_multiple_terms_per_dimension_all_processed(self):
        """Multiple terms per dimension are each independently aligned."""
        client = make_mock_client(
            validate_mesh_side_effect=[
                mesh_result(True, descriptor_name="Diabetes Mellitus, Type 2"),
                mesh_result(False),
                mesh_result(False),  # for spell check second call
            ],
            spell_check_side_effect=[None, None],
        )
        pico = make_pico(P=["type 2 diabetes", "unknownterm"])
        result = asyncio.run(align_pico_terms(pico, client))

        assert result.P[0].status == "valid_mesh"
        # Second term: no match at any level → not_found or fuzzy
        assert result.P[1].status in ("not_found", "fuzzy_mapped")

    def test_returns_new_picoterm_set_not_mutated_original(self):
        """The original PICOTermSet must not be mutated; a new one is returned."""
        client = make_mock_client(
            validate_mesh_side_effect=[mesh_result(True, descriptor_name="Exercise")],
            spell_check_side_effect=[],
        )
        original = make_pico(I=["Exercise"])
        original_status = original.I[0].status   # "not_found"

        result = asyncio.run(align_pico_terms(original, client))

        # Original unchanged
        assert original.I[0].status == original_status
        # Result updated
        assert result.I[0].status == "valid_mesh"

    # ── Search field invariant ───────────────────────────────────────────────

    def test_only_validated_terms_get_mesh_search_field(self):
        """
        MUST: No unverified term enters the query as [MeSH Terms].
        Only valid_mesh, mapped, fuzzy_mapped terms get [MeSH Terms].
        not_found always gets [tiab].
        """
        client = make_mock_client(
            validate_mesh_side_effect=[mesh_result(False)],
            spell_check_side_effect=[None],
        )
        pico = make_pico(O=["unknownbiomarker"])
        result = asyncio.run(align_pico_terms(pico, client))

        assert result.O[0].search_field == "[tiab]"


# ─────────────────────────────────────────────────────────────────────────────
# Node 1.3: Boolean Query Construction
# ─────────────────────────────────────────────────────────────────────────────

class TestBooleanQueryConstruction:
    """
    Tests for build_boolean_query(pico_term_set) → str.

    Spec: docs/05_SEARCH_STAGE.md §4 Node 1.3
    Construction rules:
      - MeSH terms:  "normalized"[MeSH Terms]
      - tiab terms:  "original"[tiab]
      - Within dimension: joined with OR inside parentheses
      - Between dimensions: joined with AND
    """

    def _mesh_term(self, original: str, normalized: str) -> PICOTerm:
        return PICOTerm(
            original=original,
            normalized=normalized,
            status="valid_mesh",
            search_field="[MeSH Terms]",
        )

    def _tiab_term(self, original: str) -> PICOTerm:
        return PICOTerm(
            original=original,
            status="not_found",
            search_field="[tiab]",
        )

    def _pico_from_terms(self, P, I, C, O) -> PICOTermSet:
        return PICOTermSet(P=P, I=I, C=C, O=O)

    # ── Single term per dimension ────────────────────────────────────────────

    def test_mesh_term_uses_normalized_and_mesh_tag(self):
        """A validated MeSH term → "normalized_name"[MeSH Terms]"""
        pico = self._pico_from_terms(
            P=[self._mesh_term("adults", "Adult")],
            I=[self._mesh_term("exercise", "Exercise")],
            C=[self._tiab_term("usual care")],
            O=[self._mesh_term("il-6", "Interleukin-6")],
        )
        query = build_boolean_query(pico)
        assert '"Adult"[MeSH Terms]' in query
        assert '"Exercise"[MeSH Terms]' in query
        assert '"Interleukin-6"[MeSH Terms]' in query

    def test_tiab_term_uses_original_and_tiab_tag(self):
        """A [tiab] term → "original"[tiab]"""
        pico = self._pico_from_terms(
            P=[self._tiab_term("usual care")],
            I=[self._tiab_term("smartphone app")],
            C=[self._tiab_term("control group")],
            O=[self._tiab_term("step count")],
        )
        query = build_boolean_query(pico)
        assert '"usual care"[tiab]' in query
        assert '"smartphone app"[tiab]' in query
        assert '"step count"[tiab]' in query

    # ── OR within dimension ──────────────────────────────────────────────────

    def test_multiple_terms_joined_with_or_in_parentheses(self):
        """P: [A, B] → ("A"[...] OR "B"[...])"""
        pico = self._pico_from_terms(
            P=[self._mesh_term("diabetes", "Diabetes Mellitus"),
               self._tiab_term("type 2 diabetes")],
            I=[self._tiab_term("exercise")],
            C=[self._tiab_term("control")],
            O=[self._tiab_term("HbA1c")],
        )
        query = build_boolean_query(pico)
        # Both P terms should be in the query with OR
        assert '"Diabetes Mellitus"[MeSH Terms]' in query
        assert '"type 2 diabetes"[tiab]' in query
        assert " OR " in query

    def test_dimension_group_wrapped_in_parentheses(self):
        """Each dimension's terms must be wrapped: (term1 OR term2)"""
        pico = self._pico_from_terms(
            P=[self._tiab_term("adult"), self._tiab_term("elderly")],
            I=[self._tiab_term("exercise")],
            C=[self._tiab_term("control")],
            O=[self._tiab_term("outcome")],
        )
        query = build_boolean_query(pico)
        assert '("adult"[tiab] OR "elderly"[tiab])' in query

    # ── AND between dimensions ───────────────────────────────────────────────

    def test_four_dimensions_joined_with_and(self):
        """Final query: (P) AND (I) AND (C) AND (O)"""
        pico = self._pico_from_terms(
            P=[self._tiab_term("A")],
            I=[self._tiab_term("B")],
            C=[self._tiab_term("C")],
            O=[self._tiab_term("D")],
        )
        query = build_boolean_query(pico)
        assert " AND " in query
        # Count AND occurrences: should be 3 (between 4 groups)
        assert query.count(" AND ") == 3

    def test_query_structure_outer_parentheses_per_dimension(self):
        """Each dimension block is wrapped in parentheses."""
        pico = self._pico_from_terms(
            P=[self._tiab_term("p")],
            I=[self._tiab_term("i")],
            C=[self._tiab_term("c")],
            O=[self._tiab_term("o")],
        )
        query = build_boolean_query(pico)
        # Should have 4 opening parens (one per dimension)
        assert query.count("(") >= 4

    # ── Full integration: mixed MeSH + tiab ─────────────────────────────────

    def test_full_query_structure_smoke(self):
        """End-to-end structure check for a realistic PICO."""
        pico = self._pico_from_terms(
            P=[self._mesh_term("adults", "Adult"),
               self._tiab_term("older adults")],
            I=[self._mesh_term("exercise", "Exercise"),
               self._tiab_term("physical activity")],
            C=[self._tiab_term("usual care"),
               self._tiab_term("control")],
            O=[self._mesh_term("il-6", "Interleukin-6"),
               self._tiab_term("CRP")],
        )
        query = build_boolean_query(pico)

        # Must be a non-empty string
        assert isinstance(query, str)
        assert len(query) > 0

        # Structure: (P) AND (I) AND (C) AND (O)
        parts = query.split(" AND ")
        assert len(parts) == 4
        for part in parts:
            assert part.strip().startswith("(")
            assert part.strip().endswith(")")

    def test_single_term_dimension_still_wrapped_in_parens(self):
        """Even a single-term dimension must be wrapped: ("x"[tiab])"""
        pico = self._pico_from_terms(
            P=[self._tiab_term("adults")],
            I=[self._tiab_term("exercise")],
            C=[self._tiab_term("control")],
            O=[self._tiab_term("outcome")],
        )
        query = build_boolean_query(pico)
        assert '("adults"[tiab])' in query

    # ── Correctness: normalized vs original precedence ───────────────────────

    def test_mesh_term_uses_normalized_not_original(self):
        """
        For a valid_mesh term, the query must use normalized (the official
        descriptor name), NOT the original user-supplied term.
        """
        pico = self._pico_from_terms(
            P=[self._mesh_term("t2dm", "Diabetes Mellitus, Type 2")],
            I=[self._tiab_term("exercise")],
            C=[self._tiab_term("control")],
            O=[self._tiab_term("hba1c")],
        )
        query = build_boolean_query(pico)
        assert '"Diabetes Mellitus, Type 2"[MeSH Terms]' in query
        assert '"t2dm"' not in query  # original must NOT appear in MeSH position

    def test_fuzzy_mapped_term_uses_normalized_and_mesh_tag(self):
        """fuzzy_mapped terms also get [MeSH Terms] with normalized name."""
        term = PICOTerm(
            original="exrcise",
            normalized="Exercise",
            status="fuzzy_mapped",
            search_field="[MeSH Terms]",
            similarity_score=88.5,
        )
        pico = self._pico_from_terms(
            P=[self._tiab_term("adults")],
            I=[term],
            C=[self._tiab_term("control")],
            O=[self._tiab_term("outcome")],
        )
        query = build_boolean_query(pico)
        assert '"Exercise"[MeSH Terms]' in query
