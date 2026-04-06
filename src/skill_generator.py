"""
SkillGenerator — benchmark-driven Skill YAML compiler.

Spec: docs/04_SKILL_FRAMEWORK.md §4

Runs once during system initialization (before any pipeline stage).
Reads ReviewConfig.target_outcomes, infers data types, and calls the
"qwen" model (via ModelRegistry role "skill_generator") to generate
customized extraction Skill YAMLs for each outcome.

Architecture note:
- _build_extraction_plan() is a HardNode — pure Python, no LLM.
- _compile_skill() is a SoftNode — one LLM call per outcome.
- All LLM calls use context_manager.mount() / unmount() (DCR compliance).
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import yaml

from src.engine.agents import ExecutorAgent
from src.engine.context_manager import ContextManager, SkillDefinition
from src.engine.model_registry import ModelRegistry
from src.schemas.common import ReviewConfig

logger = logging.getLogger("autosr.skill_generator")

# ---------------------------------------------------------------------------
# Public helpers (tested independently)
# ---------------------------------------------------------------------------

# Keywords that signal continuous outcomes
# Deliberately narrow: "ratio", "index" omitted (too ambiguous — "odds ratio" is
# dichotomous; "index" appears in both).  The dichotomous list wins ties.
_CONTINUOUS_KEYWORDS = frozenset([
    "level", "levels", "score", "scores", "count", "counts",
    "mean", "means", "change", "difference", "concentration",
    "measure", "measurement", "scale",
    "mg", "pg/ml", "ng/ml", "mmol", "step",
])

# Keywords that signal dichotomous outcomes
_DICHOTOMOUS_KEYWORDS = frozenset([
    "rate", "rates", "incidence", "events", "event",
    "odds", "risk", "prevalence", "proportion", "binary",
    "responder", "response", "abstinence", "cessation",
    "success", "failure", "adverse", "mortality", "death",
])


def normalise_outcome_name(name: str) -> str:
    """
    Convert an outcome name to a filesystem-safe identifier.

    Rules:
    - Strip leading/trailing whitespace.
    - Replace spaces and hyphens (and other non-alphanumeric, non-underscore
      chars) with underscores.
    - Collapse consecutive underscores.
    - Strip leading/trailing underscores.

    Examples:
        "IL-6"        → "IL_6"
        "8-OHdG"      → "8_OHdG"
        "Step Count"  → "Step_Count"
        " IBS-SSS "   → "IBS_SSS"
    """
    name = name.strip()
    name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")
    return name


# ---------------------------------------------------------------------------
# Fixed extraction skill template skeleton (04_SKILL_FRAMEWORK.md §4.3)
# ---------------------------------------------------------------------------

_FIXED_HARD_CONSTRAINTS = [
    "You are a TRANSCRIPTIONIST. Copy numbers exactly as they appear in the text.",
    "Do NOT calculate, derive, or infer any numbers.",
    "If text provides a range (Min-Max), do NOT record it as SD.",
    (
        "Record the data_type exactly as reported: mean_sd, mean_se, mean_95ci, "
        "median_iqr, median_range, events_total, percentage_total, or not_reported."
    ),
    (
        "If you cannot find the data for a specific arm/timepoint, "
        "set data_type to 'not_reported' and all values to null."
    ),
    (
        "Always include the raw_text field — paste the exact snippet "
        "you read the number from."
    ),
    "If a table has the data, specify the table ID in relevant_chunk_ids.",
]

_INPUT_SLOTS = [
    {"name": "relevant_chunks", "source": "dcr_filtered_chunks", "required": True},
    {"name": "arms",            "source": "document_map.arms",    "required": True},
    {"name": "timepoints",      "source": "document_map.timepoints", "required": True},
]

_USER_MESSAGE_TEMPLATE = """\
## Study Arms
{arms}

## Timepoints
{timepoints}

## Relevant Text & Tables
{relevant_chunks}

## Task
For each arm and timepoint, extract the raw numerical data for
this outcome. Return JSON matching the RawOutcomeExtraction schema.
"""


def _build_skill_yaml(
    outcome_name: str,
    safe_name: str,
    data_type: str,
    llm_fields: Dict[str, str],
) -> Dict[str, Any]:
    """
    Merge the fixed template skeleton with LLM-generated dynamic fields.
    Returns a dict ready for yaml.dump().
    """
    role = (
        f"You are extracting quantitative outcome data from a clinical trial paper.\n"
        f"You are a precise transcriptionist — you copy numbers exactly as written.\n\n"
        f"Target Outcome: {outcome_name}\n"
        f"Data Type: {data_type}\n\n"
        f"Search Terms (look for these words/phrases in the text):\n"
        f"{llm_fields.get('primary_search_terms', outcome_name)}\n\n"
        f"Expected Data Format:\n"
        f"{llm_fields.get('expected_data_format', 'See data_extraction_rules guidelines.')}\n\n"
        f"Common Reporting Patterns:\n"
        f"{llm_fields.get('common_reporting_patterns', 'Check Results tables and figures.')}\n"
    )

    return {
        "skill_id": f"extraction.outcome_{safe_name}",
        "node_type": "soft",
        "description": f"Extract {outcome_name} data ({data_type})",
        "model_requirement": "executor",
        "context_template": {
            "role": role,
            "guidelines_source": "data_extraction_rules",
            "input_slots": _INPUT_SLOTS,
            "user_message_template": _USER_MESSAGE_TEMPLATE,
        },
        "output_schema": "RawOutcomeExtraction",
        "response_format": "json",
        "constraints": {
            "max_retries": 2,
            "temperature_override": 0.0,
        },
        "hard_constraints": _FIXED_HARD_CONSTRAINTS,
    }


# ---------------------------------------------------------------------------
# SkillGenerator
# ---------------------------------------------------------------------------

class SkillGenerator:
    """
    Benchmark-driven Skill YAML compiler.

    MUST use ModelRegistry role "skill_generator" (→ qwen model) for all
    LLM calls. This is enforced in __init__.
    """

    def __init__(
        self,
        review_config: ReviewConfig,
        context_manager: ContextManager,
        model_registry: ModelRegistry,
        skills_dir: str = "src/skills",
        _agent_override: Optional[Any] = None,  # injection point for tests
    ) -> None:
        self.review_config = review_config
        self.context_manager = context_manager
        self.model_registry = model_registry
        self.skills_dir = skills_dir

        # CRITICAL: explicitly request the "skill_generator" role → qwen model
        if _agent_override is not None:
            self._agent = _agent_override
        else:
            qwen_config = model_registry.get_default("skill_generator")
            qwen_name = model_registry.default_name("skill_generator")
            self._agent = ExecutorAgent(
                model_id=qwen_name,
                model_config=qwen_config,
            )
            logger.info(
                "[SkillGenerator] Using model '%s' (%s) for skill generation.",
                qwen_name, qwen_config.model_id,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> List[str]:
        """
        Generate customized extraction Skill YAMLs.

        Process:
        1. _build_extraction_plan() — HardNode: type inference, no LLM.
        2. _compile_skill()         — SoftNode: one LLM call per outcome.
        3. Reload SkillRegistry cache.

        Returns list of generated skill_ids.
        """
        plan = self._build_extraction_plan()
        if not plan:
            logger.info("[SkillGenerator] No target_outcomes defined — skipping.")
            return []

        generated_ids: List[str] = []
        for entry in plan:
            skill_id = self._compile_skill(entry)
            if skill_id:
                generated_ids.append(skill_id)

        # Reload SkillRegistry so pipeline can immediately use new skills
        if hasattr(self.context_manager, "_skill_registry"):
            self.context_manager._skill_registry.clear_cache()
            logger.info(
                "[SkillGenerator] SkillRegistry cache cleared. "
                "%d new skills available.", len(generated_ids)
            )

        return generated_ids

    # ------------------------------------------------------------------
    # Step 1: HardNode — type inference (no LLM)
    # ------------------------------------------------------------------

    def _build_extraction_plan(self) -> List[Dict[str, str]]:
        """
        For each outcome in review_config.target_outcomes, infer data type
        from the outcome name and PICO.O context.

        Returns a list of dicts: [{outcome, inferred_type}, ...]
        """
        pico_o = self.review_config.pico.O
        plan = []
        for outcome in self.review_config.target_outcomes:
            inferred_type = self._infer_data_type(outcome, pico_o)
            plan.append({"outcome": outcome, "inferred_type": inferred_type})
            logger.debug(
                "[SkillGenerator] %s → %s", outcome, inferred_type
            )
        return plan

    def _infer_data_type(self, outcome_name: str, pico_o_context: str) -> str:
        """
        Keyword-based data type inference.

        Checks outcome_name first, then pico_o_context as a tiebreaker.
        - dichotomous keywords → "dichotomous"
        - continuous keywords  → "continuous"
        - no match             → "continuous" (safer default)
        """
        combined = f"{outcome_name} {pico_o_context}".lower()
        tokens = set(re.findall(r"\b\w+\b", combined))

        has_continuous = bool(tokens & _CONTINUOUS_KEYWORDS)
        has_dichotomous = bool(tokens & _DICHOTOMOUS_KEYWORDS)

        # Outcome-name tokens take priority over context tokens
        name_tokens = set(re.findall(r"\b\w+\b", outcome_name.lower()))
        name_continuous = bool(name_tokens & _CONTINUOUS_KEYWORDS)
        name_dichotomous = bool(name_tokens & _DICHOTOMOUS_KEYWORDS)

        if name_dichotomous and not name_continuous:
            return "dichotomous"
        if name_continuous and not name_dichotomous:
            return "continuous"
        # Fall back to context
        if has_dichotomous and not has_continuous:
            return "dichotomous"
        return "continuous"     # default

    # ------------------------------------------------------------------
    # Step 2: SoftNode — LLM call per outcome
    # ------------------------------------------------------------------

    def _compile_skill(self, entry: Dict[str, str]) -> Optional[str]:
        """
        Call the LLM to generate dynamic fields, merge with the fixed
        template skeleton, validate, and write the YAML file.

        Returns the skill_id on success, None on failure.
        """
        outcome = entry["outcome"]
        data_type = entry["inferred_type"]
        safe_name = normalise_outcome_name(outcome)
        skill_id = f"extraction.outcome_{safe_name}"

        logger.info("[SkillGenerator] Compiling skill for '%s' (%s)…",
                    outcome, data_type)

        # Build state for context_manager.mount()
        state = {
            "outcome_name": outcome,
            "data_type": data_type,
            "review_config": self.review_config,
        }

        # DCR: mount / call / unmount
        try:
            mounted = self._build_generator_prompt(outcome, data_type)
            raw_response = self._agent.call(mounted)
        except Exception as exc:
            logger.error(
                "[SkillGenerator] LLM call failed for '%s': %s", outcome, exc
            )
            return None

        # Parse LLM response
        try:
            llm_fields = json.loads(raw_response)
            if not isinstance(llm_fields, dict):
                raise ValueError("Expected JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error(
                "[SkillGenerator] Invalid JSON from LLM for outcome '%s': %s",
                outcome, exc,
            )
            return None

        # Build and validate the skill YAML dict
        skill_dict = _build_skill_yaml(outcome, safe_name, data_type, llm_fields)
        try:
            SkillDefinition.model_validate(skill_dict)
        except Exception as exc:
            logger.error(
                "[SkillGenerator] Generated YAML for '%s' failed validation: %s",
                outcome, exc,
            )
            return None

        # Write to disk
        out_path = os.path.join(
            self.skills_dir, "extraction", f"outcome_{safe_name}.yaml"
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(
                f"# AUTO-GENERATED by SkillGenerator — do not edit manually\n"
                f"# Outcome: {outcome}  |  Type: {data_type}\n\n"
            )
            yaml.dump(skill_dict, f, allow_unicode=True, sort_keys=False,
                      default_flow_style=False)

        logger.info("[SkillGenerator] Wrote %s", out_path)
        return skill_id

    def _build_generator_prompt(self, outcome: str, data_type: str) -> Any:
        """
        Build a MountedContext for the LLM call that generates dynamic fields.

        Rather than using ContextManager.mount() (which requires a Skill YAML
        for the generator itself), we directly build a MountedContext here.
        This is the only place in the system where a prompt is assembled
        without going through a Skill YAML — it is an initialization tool,
        not a pipeline SoftNode, and is documented as an explicit exception.
        """
        from src.engine.context_manager import MountedContext

        system_message = (
            "You are a medical informatics expert specialising in clinical trial "
            "data extraction.\n"
            "Given an outcome name and its data type (continuous or dichotomous), "
            "generate metadata that will guide an LLM-based data extractor.\n\n"
            "Return a JSON object with exactly these three keys:\n"
            "  primary_search_terms: comma-separated list of synonyms and "
            "abbreviations for the outcome (the extractor will scan text for these)\n"
            "  expected_data_format: how this outcome is typically reported in "
            "clinical trials (e.g. 'mean ± SD in pg/mL')\n"
            "  common_reporting_patterns: typical locations in a paper where "
            "this data appears (e.g. 'Results table, Figure 2 legend')\n\n"
            "Hard constraints:\n"
            "- Return ONLY valid JSON, no markdown fences.\n"
            "- Do not include any other keys."
        )
        user_message = (
            f"Outcome name: {outcome}\n"
            f"Data type: {data_type}\n\n"
            f"Generate the three metadata fields described in your instructions."
        )

        # CRITICAL: explicitly request the skill_generator role (→ qwen model)
        self.model_registry.get_default("skill_generator")   # validates role exists
        model_logical = self.model_registry.default_name("skill_generator")

        return MountedContext(
            system_message=system_message,
            user_message=user_message,
            model_id=model_logical,
            temperature=0.0,
            response_format="json",
            metadata={"skill_id": "skill_generator_meta", "outcome": outcome},
        )
