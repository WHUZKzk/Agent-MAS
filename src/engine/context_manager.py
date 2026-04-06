"""
Dynamic Context Routing (DCR) — ContextManager and SkillRegistry.

Spec: docs/03_CORE_ENGINE_SPEC.md §4, docs/04_SKILL_FRAMEWORK.md §2

The ContextManager is the implementation of DCR. Its job:
- mount(): assemble a node-specific prompt from Skill YAML + guidelines + state.
- unmount(): clear context and log call metadata.

MUST: guidelines_source file must exist. Silently missing guidelines = error.
MUST: Skill content MUST NOT be hardcoded — only loaded from YAML files.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel

logger = logging.getLogger("autosr.context_manager")


# ---------------------------------------------------------------------------
# Skill YAML schema  (matches 04_SKILL_FRAMEWORK.md §2 exactly)
# ---------------------------------------------------------------------------

class InputSlot(BaseModel):
    name: str           # Variable name referenced in user_message_template
    source: str         # Dot-path into pipeline state, e.g. "current_paper.abstract"
    required: bool = True


class SkillConstraints(BaseModel):
    max_retries: int = 2
    temperature_override: Optional[float] = None


class ContextTemplate(BaseModel):
    role: str                               # System message / persona template
    guidelines_source: Optional[str] = None # Filename (no .md) under src/guidelines/
    input_slots: List[InputSlot] = []
    user_message_template: str              # User message template; refs {slot_name}


class SkillDefinition(BaseModel):
    skill_id: str                           # "{stage}.{skill_name}"
    node_type: Literal["soft"] = "soft"
    description: str
    model_requirement: Literal["any", "executor", "reviewer", "adjudicator"] = "executor"
    context_template: ContextTemplate
    output_schema: str                      # Pydantic model name, e.g. "ReviewerOutput"
    response_format: Literal["json", "text"] = "json"
    constraints: SkillConstraints = SkillConstraints()
    hard_constraints: List[str] = []


# ---------------------------------------------------------------------------
# MountedContext
# ---------------------------------------------------------------------------

class MountedContext(BaseModel):
    """Fully assembled LLM call payload produced by ContextManager.mount()."""
    system_message: str
    user_message: str
    model_id: str
    temperature: float
    response_format: Literal["json", "text"]
    metadata: Dict[str, Any] = {}   # skill_id, node_id, resolved_slots — for logging


# ---------------------------------------------------------------------------
# SkillRegistry
# ---------------------------------------------------------------------------

class SkillRegistry:
    """Loads and caches Skill YAML files from src/skills/."""

    def __init__(self, skills_dir: str) -> None:
        self.skills_dir = skills_dir
        self._cache: Dict[str, SkillDefinition] = {}

    def _skill_path(self, skill_id: str) -> str:
        """
        skill_id format: "{stage}.{skill_name}"  (may have dots after the first)
        Maps to: {skills_dir}/{stage}/{skill_name}.yaml

        Multi-segment names like "extraction.outcome.IL_6" are joined:
        → {skills_dir}/extraction/outcome.IL_6.yaml
        """
        parts = skill_id.split(".", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid skill_id '{skill_id}'. "
                "Expected format: '<stage>.<skill_name>'"
            )
        stage, name = parts
        return os.path.join(self.skills_dir, stage, f"{name}.yaml")

    def load(self, skill_id: str) -> SkillDefinition:
        if skill_id in self._cache:
            return self._cache[skill_id]
        return self.reload(skill_id)

    def reload(self, skill_id: str) -> SkillDefinition:
        path = self._skill_path(skill_id)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Skill YAML not found for '{skill_id}' at: {path}"
            )
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        skill = SkillDefinition.model_validate(raw)
        self._cache[skill_id] = skill
        logger.debug("SkillRegistry: loaded '%s' from %s", skill_id, path)
        return skill

    def clear_cache(self) -> None:
        """Force reload of all skills on next access (used after SkillGenerator runs)."""
        self._cache.clear()


# ---------------------------------------------------------------------------
# ContextManager
# ---------------------------------------------------------------------------

class ContextManager:
    def __init__(
        self,
        skill_registry: SkillRegistry,
        guidelines_dir: str,
        model_registry: Any,        # ModelRegistry — avoid circular import
    ) -> None:
        self._skill_registry = skill_registry
        self._guidelines_dir = guidelines_dir
        self._model_registry = model_registry
        self._current_mount: Optional[MountedContext] = None

    def mount(self, skill_id: str, state: Dict[str, Any]) -> MountedContext:
        """
        Assemble the full LLM prompt payload for a SoftNode.

        Steps (per 03 §4.3 + 04 §2):
        1. Load Skill YAML.
        2. Load guidelines text (if guidelines_source is set).
        3. Resolve input_slots from state.
        4. Render system_message = role + guidelines + hard_constraints.
        5. Render user_message from user_message_template.
        6. Look up model config via model_requirement.
        7. Return MountedContext; store in self._current_mount.
        """
        # 1. Skill YAML
        skill = self._skill_registry.load(skill_id)
        ct = skill.context_template

        # 2. Guidelines (mandatory when specified)
        guidelines_text = ""
        if ct.guidelines_source:
            guidelines_path = os.path.join(
                self._guidelines_dir, f"{ct.guidelines_source}.md"
            )
            if not os.path.exists(guidelines_path):
                raise FileNotFoundError(
                    f"[DCR] Guidelines file missing: {guidelines_path}\n"
                    f"Skill '{skill_id}' requires: {ct.guidelines_source}.md\n"
                    "Create the file in src/guidelines/ before running this node."
                )
            with open(guidelines_path, "r", encoding="utf-8") as f:
                guidelines_text = f.read()

        # 3. Resolve input_slots
        slot_values: Dict[str, str] = {}
        for slot in ct.input_slots:
            value = self._resolve_slot(slot.source, state)
            if value is None and slot.required:
                logger.warning(
                    "[DCR] Required slot '%s' (source='%s') resolved to None "
                    "for skill '%s'.",
                    slot.name, slot.source, skill_id,
                )
            slot_values[slot.name] = str(value) if value is not None else ""

        # 4. Render system_message
        try:
            role_rendered = ct.role.format(**slot_values)
        except KeyError:
            role_rendered = ct.role

        parts = [role_rendered.strip()]
        if guidelines_text:
            parts.append(f"---\n\n## Methodological Guidelines\n\n{guidelines_text.strip()}")
        if skill.hard_constraints:
            constraints_block = "\n".join(
                f"- {c}" for c in skill.hard_constraints
            )
            parts.append(f"---\n\n## Hard Constraints (MUST follow)\n\n{constraints_block}")
        system_message = "\n\n".join(parts)

        # 5. Render user_message
        try:
            user_message = ct.user_message_template.format(**slot_values)
        except KeyError:
            user_message = ct.user_message_template

        # 6. Look up model via model_requirement
        role = self._requirement_to_role(skill.model_requirement, skill_id)
        model_logical = self._model_registry.default_name(role)

        ctx = MountedContext(
            system_message=system_message,
            user_message=user_message,
            model_id=model_logical,
            temperature=skill.constraints.temperature_override
            if skill.constraints.temperature_override is not None
            else 0.0,
            response_format=skill.response_format,
            metadata={
                "skill_id": skill_id,
                "model_requirement": skill.model_requirement,
                "resolved_slots": list(slot_values),
            },
        )
        self._current_mount = ctx
        logger.debug("[DCR] Mounted skill='%s' model='%s'", skill_id, model_logical)
        return ctx

    def unmount(self) -> None:
        """Clear context after SoftNode execution. Called in try/finally."""
        if self._current_mount:
            logger.info(
                "[DCR] Unmounted skill='%s' model='%s' at %s",
                self._current_mount.metadata.get("skill_id"),
                self._current_mount.model_id,
                datetime.now(timezone.utc).isoformat(),
            )
        self._current_mount = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _requirement_to_role(model_requirement: str, skill_id: str) -> str:
        """Map Skill model_requirement → ModelRegistry role key."""
        mapping = {
            "executor":    "executor",
            "reviewer":    "reviewer_a",    # default; caller swaps for reviewer_b
            "adjudicator": "adjudicator",
            "any":         "executor",
        }
        if model_requirement not in mapping:
            raise ValueError(
                f"Unknown model_requirement '{model_requirement}' "
                f"in skill '{skill_id}'"
            )
        return mapping[model_requirement]

    @staticmethod
    def _resolve_slot(source: str, state: Dict[str, Any]) -> Any:
        """
        Walk a dotted path through the state dict (or Pydantic objects).
        e.g. "current_paper.abstract" → state["current_paper"]["abstract"]
        """
        parts = source.split(".")
        value: Any = state
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            elif hasattr(value, part):
                value = getattr(value, part)
            else:
                return None
        return value
