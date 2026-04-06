"""
Node system — BaseNode, HardNode, SoftNode.

Spec: docs/03_CORE_ENGINE_SPEC.md §2

Design contract:
- HardNode: deterministic Python. Uses assert for invariants. AssertionError is
  caught by DAGRunner (marks item FAILED, continues). All other exceptions propagate.
- SoftNode: wraps an LLM call. Calls context_manager.mount() / unmount() via
  try/finally. Retries on Pydantic ValidationError up to max_retries.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Type

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from .agents import BaseAgent
    from .context_manager import ContextManager


class BaseNode(ABC):
    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self.logger = logging.getLogger(f"autosr.{node_id}")

    @abstractmethod
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the node. Returns updated state dict."""


# ---------------------------------------------------------------------------
# HardNode
# ---------------------------------------------------------------------------

class HardNode(BaseNode):
    """
    Deterministic Python logic. No LLM calls.

    AssertionError → caught by DAGRunner, item marked FAILED.
    All other exceptions propagate.
    """

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info("[HardNode] Executing %s", self.node_id)
        result = self.execute(state)
        self.logger.info("[HardNode] Completed %s", self.node_id)
        return result

    @abstractmethod
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pass


# ---------------------------------------------------------------------------
# SoftNode
# ---------------------------------------------------------------------------

class SoftNode(BaseNode):
    """
    Wraps an LLM call via ContextManager + Agent.

    - MUST call context_manager.mount() before invocation.
    - MUST call context_manager.unmount() after (even on failure) via try/finally.
    - MUST validate output against output_schema.
    - Retries up to max_retries on ValidationError.
    - After all retries exhausted: marks state["_failed"] = True.
    """

    def __init__(
        self,
        node_id: str,
        skill_id: str,
        context_manager: "ContextManager",
        agent: "BaseAgent",
        output_schema: Type[BaseModel],
        max_retries: int = 2,
    ) -> None:
        super().__init__(node_id)
        self.skill_id = skill_id
        self.context_manager = context_manager
        self.agent = agent
        self.output_schema = output_schema
        self.max_retries = max_retries

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info("[SoftNode] Executing %s", self.node_id)
        mounted_context = self.context_manager.mount(self.skill_id, state)
        try:
            for attempt in range(1, self.max_retries + 2):
                raw_output = self.agent.call(mounted_context)
                try:
                    parsed = self.output_schema.model_validate_json(raw_output)
                    self.logger.info(
                        "[SoftNode] %s succeeded on attempt %d", self.node_id, attempt
                    )
                    return self._update_state(state, parsed)
                except ValidationError as exc:
                    self.logger.warning(
                        "[SoftNode] %s attempt %d validation failed: %s",
                        self.node_id, attempt, exc,
                    )
                    if attempt > self.max_retries:
                        self.logger.error(
                            "[SoftNode] %s FAILED after %d attempts",
                            self.node_id, self.max_retries + 1,
                        )
                        return self._mark_failed(state)
        finally:
            self.context_manager.unmount()

        return self._mark_failed(state)  # unreachable, but satisfies type checkers

    def _update_state(
        self, state: Dict[str, Any], parsed: BaseModel
    ) -> Dict[str, Any]:
        """Merge parsed output into state under a key named after the node."""
        return {**state, self.node_id: parsed}

    def _mark_failed(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {**state, "_failed": True}
