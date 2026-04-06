"""
Model Registry — loads configs/models.yaml and dispatches logical model
names to their API configurations.

Spec: docs/03_CORE_ENGINE_SPEC.md §5
All models are routed through OpenRouter (provider="openrouter"), which
exposes an OpenAI-compatible Chat Completions API.

INVARIANT (checked at __init__):
    defaults.reviewer_a != defaults.reviewer_b
"""
from __future__ import annotations

import logging
from typing import Dict, Literal

import yaml
from pydantic import BaseModel, model_validator

logger = logging.getLogger("autosr.model_registry")


# ---------------------------------------------------------------------------
# Config schemas
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel):
    provider: Literal["anthropic", "openai", "google", "openrouter"]
    model_id: str           # Provider-specific string, e.g. "deepseek/deepseek-v3.2"
    api_base: str
    max_context_tokens: int = 32768
    supports_vision: bool = False


class ModelRegistryConfig(BaseModel):
    models: Dict[str, ModelConfig]   # logical_name → config
    defaults: Dict[str, str]         # role → logical_name

    @model_validator(mode="after")
    def reviewer_heterogeneity_invariant(self) -> "ModelRegistryConfig":
        ra = self.defaults.get("reviewer_a")
        rb = self.defaults.get("reviewer_b")
        if ra and rb and ra == rb:
            raise ValueError(
                "INVARIANT VIOLATED: defaults.reviewer_a and defaults.reviewer_b "
                "MUST point to different model entries to ensure architectural "
                "heterogeneity in dual-blind screening."
            )
        return self


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    def __init__(self, config_path: str = "configs/models.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        self._config = ModelRegistryConfig.model_validate(raw)
        logger.info(
            "ModelRegistry loaded: %d models, defaults=%s",
            len(self._config.models),
            self._config.defaults,
        )

    def get_model(self, logical_name: str) -> ModelConfig:
        """Look up a model config by its logical name."""
        if logical_name not in self._config.models:
            raise KeyError(
                f"Model '{logical_name}' not found in registry. "
                f"Available: {list(self._config.models)}"
            )
        return self._config.models[logical_name]

    def get_default(self, role: str) -> ModelConfig:
        """Look up the default model config for a pipeline role."""
        if role not in self._config.defaults:
            raise KeyError(
                f"No default model for role '{role}'. "
                f"Configured roles: {list(self._config.defaults)}"
            )
        return self.get_model(self._config.defaults[role])

    def default_name(self, role: str) -> str:
        """Return the logical model name for a role (not the config object)."""
        return self._config.defaults[role]

    @property
    def all_model_names(self) -> list[str]:
        return list(self._config.models)
