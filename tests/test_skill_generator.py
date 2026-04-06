"""
TDD tests for SkillGenerator.

Written BEFORE implementation. All tests must fail initially, then pass
once SkillGenerator is implemented.

Test plan:
  A. Type inference (pure Python, HardNode logic) — no LLM mock needed.
  B. Skill compilation (LLM call mocked) — verifies file creation + content.
  C. Integration: generate() end-to-end with mock LLM.
  D. Edge cases: empty target_outcomes, unrecognised model, YAML validation.
"""
import json
import os
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch, call

import pytest
import yaml

from src.schemas.common import PICODefinition, ReviewConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_review_config(
    target_outcomes: List[str],
    target_characteristics: List[str] = None,
    pico_o: str = "",
) -> ReviewConfig:
    return ReviewConfig(
        pmid="12345678",
        title="Test Systematic Review",
        abstract="A test abstract.",
        pico=PICODefinition(
            P="Adults with hypertension",
            I="Exercise intervention",
            C="Usual care",
            O=pico_o or "Change in IL-6 levels, events rate, smoking cessation rate",
        ),
        target_characteristics=target_characteristics or ["Mean Age", "Sample Size"],
        target_outcomes=target_outcomes,
    )


# ---------------------------------------------------------------------------
# Import target (will fail until implemented)
# ---------------------------------------------------------------------------

from src.skill_generator import SkillGenerator  # noqa: E402


# ---------------------------------------------------------------------------
# A. Type inference tests (no LLM, pure Python HardNode)
# ---------------------------------------------------------------------------

class TestTypeInference:
    def test_continuous_keywords_infer_continuous(self):
        """Words like 'level', 'score', 'mean', 'change', 'count' → continuous."""
        sg = SkillGenerator.__new__(SkillGenerator)
        assert sg._infer_data_type("IL-6 level", "") == "continuous"
        assert sg._infer_data_type("PANAS score", "") == "continuous"
        assert sg._infer_data_type("mean step count", "") == "continuous"
        assert sg._infer_data_type("change in BMI", "") == "continuous"

    def test_dichotomous_keywords_infer_dichotomous(self):
        """Words like 'rate', 'incidence', 'events', 'odds', 'risk' → dichotomous."""
        sg = SkillGenerator.__new__(SkillGenerator)
        assert sg._infer_data_type("smoking cessation rate", "") == "dichotomous"
        assert sg._infer_data_type("incidence of adverse events", "") == "dichotomous"
        assert sg._infer_data_type("events per 100 patient-years", "") == "dichotomous"
        assert sg._infer_data_type("odds ratio", "") == "dichotomous"
        assert sg._infer_data_type("relative risk", "") == "dichotomous"

    def test_ambiguous_defaults_to_continuous(self):
        """No keyword match → default to continuous (safer)."""
        sg = SkillGenerator.__new__(SkillGenerator)
        assert sg._infer_data_type("physical fitness", "") == "continuous"
        assert sg._infer_data_type("", "") == "continuous"

    def test_pico_o_context_breaks_ambiguity(self):
        """PICO.O context can tip ambiguous outcome names."""
        sg = SkillGenerator.__new__(SkillGenerator)
        # "Global relief" is ambiguous on its own, but PICO.O mentions events
        result = sg._infer_data_type(
            "Global Relief",
            "reported as events/total (binary response)",
        )
        assert result == "dichotomous"

    def test_case_insensitive_matching(self):
        """Keyword matching must be case-insensitive."""
        sg = SkillGenerator.__new__(SkillGenerator)
        assert sg._infer_data_type("SMOKING CESSATION RATE", "") == "dichotomous"
        assert sg._infer_data_type("Inflammatory LEVEL", "") == "continuous"

    def test_build_extraction_plan_maps_all_outcomes(self):
        """_build_extraction_plan returns one entry per target_outcome."""
        sg = SkillGenerator.__new__(SkillGenerator)
        rc = make_review_config(
            target_outcomes=["IL-6", "CRP", "smoking rate"],
            pico_o="Change in IL-6 and CRP levels; smoking cessation rate",
        )
        sg.review_config = rc
        plan = sg._build_extraction_plan()
        assert len(plan) == 3
        outcome_names = [p["outcome"] for p in plan]
        assert "IL-6" in outcome_names
        assert "CRP" in outcome_names
        assert "smoking rate" in outcome_names

    def test_build_extraction_plan_assigns_correct_types(self):
        sg = SkillGenerator.__new__(SkillGenerator)
        rc = make_review_config(
            target_outcomes=["IL-6", "event rate"],
            pico_o="IL-6 level change; event rate",
        )
        sg.review_config = rc
        plan = sg._build_extraction_plan()
        type_map = {p["outcome"]: p["inferred_type"] for p in plan}
        assert type_map["IL-6"] == "continuous"
        assert type_map["event rate"] == "dichotomous"


# ---------------------------------------------------------------------------
# B. Outcome name normalisation
# ---------------------------------------------------------------------------

class TestOutcomeNormalisation:
    def test_normalise_outcome_name_for_filename(self):
        """Special chars → underscores, spaces → underscores, lowercase."""
        from src.skill_generator import normalise_outcome_name
        assert normalise_outcome_name("IL-6") == "IL_6"
        assert normalise_outcome_name("8-OHdG") == "8_OHdG"
        assert normalise_outcome_name("Step Count") == "Step_Count"
        assert normalise_outcome_name("IBS-SSS Score") == "IBS_SSS_Score"

    def test_no_leading_or_trailing_underscores(self):
        from src.skill_generator import normalise_outcome_name
        result = normalise_outcome_name(" IL-6 ")
        assert not result.startswith("_")
        assert not result.endswith("_")


# ---------------------------------------------------------------------------
# C. Skill compilation — mock LLM, verify file creation
# ---------------------------------------------------------------------------

LLM_DYNAMIC_RESPONSE = {
    "primary_search_terms": "IL-6, interleukin-6, interleukin 6, IL6",
    "expected_data_format": "Typically reported as mean ± SD in pg/mL.",
    "common_reporting_patterns": "Usually in Results table or Methods section.",
}


class TestSkillCompilation:
    @pytest.fixture
    def tmp_skills_dir(self, tmp_path):
        extraction_dir = tmp_path / "extraction"
        extraction_dir.mkdir()
        return str(tmp_path)

    @pytest.fixture
    def mock_model_registry(self):
        reg = MagicMock()
        cfg = MagicMock()
        cfg.model_id = "qwen/qwen3.6-plus:free"
        cfg.provider = "openrouter"
        cfg.api_base = "https://openrouter.ai/api/v1"
        reg.get_default.return_value = cfg
        reg.default_name.return_value = "model_general"
        return reg

    @pytest.fixture
    def mock_context_manager(self):
        cm = MagicMock()
        cm.mount.return_value = MagicMock(
            system_message="sys",
            user_message="usr",
            model_id="model_general",
            temperature=0.0,
            response_format="json",
            metadata={},
        )
        return cm

    @pytest.fixture
    def mock_agent(self):
        agent = MagicMock()
        agent.call.return_value = json.dumps(LLM_DYNAMIC_RESPONSE)
        return agent

    def _make_generator(
        self, tmp_skills_dir, mock_context_manager, mock_model_registry, mock_agent,
        outcomes=None
    ):
        rc = make_review_config(
            target_outcomes=outcomes or ["IL-6", "CRP"],
        )
        sg = SkillGenerator(
            review_config=rc,
            context_manager=mock_context_manager,
            model_registry=mock_model_registry,
            skills_dir=tmp_skills_dir,
            _agent_override=mock_agent,   # injected for testing
        )
        return sg

    def test_generate_creates_yaml_file_per_outcome(
        self, tmp_skills_dir, mock_context_manager, mock_model_registry, mock_agent
    ):
        sg = self._make_generator(
            tmp_skills_dir, mock_context_manager, mock_model_registry, mock_agent,
            outcomes=["IL-6", "CRP"],
        )
        skill_ids = sg.generate()

        assert len(skill_ids) == 2
        extraction_dir = Path(tmp_skills_dir) / "extraction"
        yaml_files = list(extraction_dir.glob("outcome_*.yaml"))
        assert len(yaml_files) == 2

    def test_generated_yaml_has_correct_skill_id(
        self, tmp_skills_dir, mock_context_manager, mock_model_registry, mock_agent
    ):
        sg = self._make_generator(
            tmp_skills_dir, mock_context_manager, mock_model_registry, mock_agent,
            outcomes=["IL-6"],
        )
        sg.generate()

        yaml_path = Path(tmp_skills_dir) / "extraction" / "outcome_IL_6.yaml"
        assert yaml_path.exists()
        with open(yaml_path, encoding="utf-8") as f:
            content = yaml.safe_load(f)
        assert content["skill_id"] == "extraction.outcome_IL_6"

    def test_generated_yaml_passes_skill_definition_validation(
        self, tmp_skills_dir, mock_context_manager, mock_model_registry, mock_agent
    ):
        """Generated YAML must be loadable by SkillRegistry (schema validation)."""
        from src.engine.context_manager import SkillRegistry, SkillDefinition

        sg = self._make_generator(
            tmp_skills_dir, mock_context_manager, mock_model_registry, mock_agent,
            outcomes=["IL-6"],
        )
        sg.generate()

        yaml_path = Path(tmp_skills_dir) / "extraction" / "outcome_IL_6.yaml"
        with open(yaml_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        # Must not raise
        skill = SkillDefinition.model_validate(raw)
        assert skill.output_schema == "RawOutcomeExtraction"
        assert skill.model_requirement == "executor"

    def test_generated_yaml_contains_dynamic_fields(
        self, tmp_skills_dir, mock_context_manager, mock_model_registry, mock_agent
    ):
        """LLM-generated fields appear in the role/system message area."""
        sg = self._make_generator(
            tmp_skills_dir, mock_context_manager, mock_model_registry, mock_agent,
            outcomes=["IL-6"],
        )
        sg.generate()

        yaml_path = Path(tmp_skills_dir) / "extraction" / "outcome_IL_6.yaml"
        with open(yaml_path, encoding="utf-8") as f:
            content = yaml.safe_load(f)

        role_text = content["context_template"]["role"]
        assert "interleukin-6" in role_text or "IL-6" in role_text

    def test_generate_returns_correct_skill_ids(
        self, tmp_skills_dir, mock_context_manager, mock_model_registry, mock_agent
    ):
        sg = self._make_generator(
            tmp_skills_dir, mock_context_manager, mock_model_registry, mock_agent,
            outcomes=["IL-6", "CRP"],
        )
        skill_ids = sg.generate()
        assert "extraction.outcome_IL_6" in skill_ids
        assert "extraction.outcome_CRP" in skill_ids

    def test_hard_constraints_are_fixed_not_llm_generated(
        self, tmp_skills_dir, mock_context_manager, mock_model_registry, mock_agent
    ):
        """hard_constraints must be the fixed template values, never altered by LLM."""
        sg = self._make_generator(
            tmp_skills_dir, mock_context_manager, mock_model_registry, mock_agent,
            outcomes=["IL-6"],
        )
        sg.generate()

        yaml_path = Path(tmp_skills_dir) / "extraction" / "outcome_IL_6.yaml"
        with open(yaml_path, encoding="utf-8") as f:
            content = yaml.safe_load(f)

        constraints = content.get("hard_constraints", [])
        assert len(constraints) >= 4
        # The transcriptionist constraint must always be present
        assert any("TRANSCRIPTIONIST" in c.upper() for c in constraints)

    def test_llm_call_uses_skill_generator_model_role(
        self, tmp_skills_dir, mock_context_manager, mock_model_registry, mock_agent
    ):
        """SkillGenerator MUST use model_registry.get_default('skill_generator')."""
        sg = self._make_generator(
            tmp_skills_dir, mock_context_manager, mock_model_registry, mock_agent,
            outcomes=["IL-6"],
        )
        sg.generate()
        mock_model_registry.get_default.assert_any_call("skill_generator")


# ---------------------------------------------------------------------------
# D. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_target_outcomes_returns_empty_list(self, tmp_path):
        rc = make_review_config(target_outcomes=[])
        sg = SkillGenerator(
            review_config=rc,
            context_manager=MagicMock(),
            model_registry=MagicMock(),
            skills_dir=str(tmp_path),
        )
        result = sg.generate()
        assert result == []

    def test_generate_does_not_call_llm_for_empty_outcomes(self, tmp_path):
        rc = make_review_config(target_outcomes=[])
        mock_agent = MagicMock()
        sg = SkillGenerator(
            review_config=rc,
            context_manager=MagicMock(),
            model_registry=MagicMock(),
            skills_dir=str(tmp_path),
            _agent_override=mock_agent,
        )
        sg.generate()
        mock_agent.call.assert_not_called()

    def test_invalid_llm_json_response_marks_outcome_failed_continues(
        self, tmp_path, capsys
    ):
        """If LLM returns invalid JSON, that outcome is skipped; others continue."""
        rc = make_review_config(target_outcomes=["IL-6", "CRP"])

        # First call → invalid JSON; second call → valid
        mock_agent = MagicMock()
        mock_agent.call.side_effect = [
            "THIS IS NOT JSON",
            json.dumps(LLM_DYNAMIC_RESPONSE),
        ]

        mock_reg = MagicMock()
        cfg = MagicMock()
        cfg.model_id = "qwen/qwen3.6-plus:free"
        cfg.provider = "openrouter"
        cfg.api_base = "https://openrouter.ai/api/v1"
        mock_reg.get_default.return_value = cfg
        mock_reg.default_name.return_value = "model_general"

        (tmp_path / "extraction").mkdir()
        sg = SkillGenerator(
            review_config=rc,
            context_manager=MagicMock(),
            model_registry=mock_reg,
            skills_dir=str(tmp_path),
            _agent_override=mock_agent,
        )
        result = sg.generate()
        # Only 1 outcome succeeded (CRP); IL-6 was skipped
        assert len(result) == 1

    def test_outcome_with_special_chars_produces_safe_filename(self, tmp_path):
        rc = make_review_config(target_outcomes=["8-OHdG"])
        mock_agent = MagicMock()
        mock_agent.call.return_value = json.dumps(LLM_DYNAMIC_RESPONSE)

        mock_reg = MagicMock()
        cfg = MagicMock()
        cfg.model_id = "qwen"; cfg.provider = "openrouter"; cfg.api_base = "http://x"
        mock_reg.get_default.return_value = cfg
        mock_reg.default_name.return_value = "model_general"

        (tmp_path / "extraction").mkdir()
        sg = SkillGenerator(
            review_config=rc,
            context_manager=MagicMock(),
            model_registry=mock_reg,
            skills_dir=str(tmp_path),
            _agent_override=mock_agent,
        )
        result = sg.generate()
        assert len(result) == 1
        yaml_path = tmp_path / "extraction" / "outcome_8_OHdG.yaml"
        assert yaml_path.exists()

    def test_generate_reloads_skill_registry_after_writing(self, tmp_path):
        """After generating, SkillRegistry cache should be cleared."""
        rc = make_review_config(target_outcomes=["IL-6"])
        mock_cm = MagicMock()
        mock_reg = MagicMock()
        cfg = MagicMock()
        cfg.model_id = "qwen"; cfg.provider = "openrouter"; cfg.api_base = "http://x"
        mock_reg.get_default.return_value = cfg
        mock_reg.default_name.return_value = "model_general"
        mock_agent = MagicMock()
        mock_agent.call.return_value = json.dumps(LLM_DYNAMIC_RESPONSE)

        (tmp_path / "extraction").mkdir()
        sg = SkillGenerator(
            review_config=rc,
            context_manager=mock_cm,
            model_registry=mock_reg,
            skills_dir=str(tmp_path),
            _agent_override=mock_agent,
        )
        sg.generate()
        # SkillRegistry.clear_cache() must have been called on the registry
        # (the ContextManager's skill_registry should be refreshed)
        mock_cm._skill_registry.clear_cache.assert_called_once()
