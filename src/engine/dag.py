"""
DAG system — DAGDefinition, DAGRunner, DAGTraversalError.

Spec: docs/03_CORE_ENGINE_SPEC.md §1

PI-CAG innovation: the SR methodology is encoded as an explicit declarative
graph, not procedural step1(); step2() calls. The DAGRunner interprets it.

Guard evaluation:
- Guards are Python expressions evaluated with `state` bound in the namespace.
- Example: 'state["has_conflicts"] == True'
- None guard = unconditional (always taken).

Edge priority: edges are evaluated in declaration order; first matching edge wins.

Loop protection:
- Per-node visit counter. If any node is visited > max_iterations, force exit
  via the first available edge and log a warning.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from src.schemas.common import DAGDefinition, NodeDefinition

if TYPE_CHECKING:
    from .context_manager import ContextManager

logger = logging.getLogger("autosr.dag")


class DAGTraversalError(Exception):
    """Raised when no edge guard matches and the DAG cannot continue."""


class DAGRunner:
    def __init__(
        self,
        dag: DAGDefinition,
        context_manager: "ContextManager",
        node_registry: Dict[str, Callable],
        max_iterations: int = 5,
    ) -> None:
        self.dag = dag
        self.context_manager = context_manager
        self.node_registry = node_registry
        self.max_iterations = max_iterations

        # Pre-build lookup structures
        self._node_map: Dict[str, NodeDefinition] = {
            n.node_id: n for n in dag.nodes
        }
        # edges indexed by from_node
        self._edges: Dict[str, list] = defaultdict(list)
        for edge in dag.edges:
            self._edges[edge.from_node].append(edge)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Traverse the DAG from entry_node to a terminal_node.
        Returns the final state dict.
        """
        state = dict(initial_state)
        current_id = self.dag.entry_node
        visit_counts: Dict[str, int] = defaultdict(int)

        while current_id not in self.dag.terminal_nodes:
            node_def = self._node_map[current_id]
            visit_counts[current_id] += 1

            logger.info(
                "[%s] Entering node '%s' (visit #%d)",
                self.dag.dag_id, current_id, visit_counts[current_id],
            )

            # Execute the node
            state = self._execute_node(node_def, state)

            # Determine next node
            next_id = self._select_next(current_id, state, visit_counts)

            logger.info("[%s] %s → %s", self.dag.dag_id, current_id, next_id)
            current_id = next_id

        # Execute the terminal node
        node_def = self._node_map[current_id]
        logger.info("[%s] Entering terminal node '%s'", self.dag.dag_id, current_id)
        state = self._execute_node(node_def, state)
        return state

    # ------------------------------------------------------------------
    # Node execution
    # ------------------------------------------------------------------

    def _execute_node(
        self, node_def: NodeDefinition, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        impl = self.node_registry[node_def.node_id]   # KeyError propagates

        if node_def.node_type == "soft":
            return self._execute_soft(node_def, impl, state)
        else:
            return self._execute_hard(node_def, impl, state)

    def _execute_hard(
        self,
        node_def: NodeDefinition,
        impl: Callable,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run a HardNode implementation.
        AssertionError → mark _failed, continue.
        All other exceptions propagate.
        """
        try:
            return impl(state)
        except AssertionError as exc:
            logger.error(
                "[%s] HardNode '%s' AssertionError: %s — marking item FAILED",
                self.dag.dag_id, node_def.node_id, exc,
            )
            return {**state, "_failed": True}

    def _execute_soft(
        self,
        node_def: NodeDefinition,
        impl: Callable,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run a SoftNode implementation.
        Wraps call with context_manager.mount() / unmount() (try/finally).
        AssertionError → mark _failed.
        """
        self.context_manager.mount(node_def.skill_id, state)
        try:
            result = impl(state)
            return result
        except AssertionError as exc:
            logger.error(
                "[%s] SoftNode '%s' AssertionError: %s — marking item FAILED",
                self.dag.dag_id, node_def.node_id, exc,
            )
            return {**state, "_failed": True}
        finally:
            self.context_manager.unmount()

    # ------------------------------------------------------------------
    # Edge selection
    # ------------------------------------------------------------------

    def _select_next(
        self,
        current_id: str,
        state: Dict[str, Any],
        visit_counts: Dict[str, int],
    ) -> str:
        """
        Evaluate outgoing edges in declaration order.
        Returns the node_id of the first matching edge.

        Loop protection: if current_id has been visited > max_iterations,
        force the first available edge regardless of guard and log a warning.
        """
        edges = self._edges.get(current_id, [])
        if not edges:
            raise DAGTraversalError(
                f"[{self.dag.dag_id}] Node '{current_id}' has no outgoing edges "
                "but is not a terminal node."
            )

        # Loop protection
        if visit_counts.get(current_id, 0) > self.max_iterations:
            logger.warning(
                "[%s] Loop protection triggered at '%s' (visited %d times). "
                "Forcing least-visited outgoing edge.",
                self.dag.dag_id, current_id, visit_counts[current_id],
            )
            # Pick the edge whose destination has been visited fewest times,
            # breaking ties by declaration order. This always breaks cycles
            # because the exit (terminal-direction) node has fewer visits.
            return min(edges, key=lambda e: visit_counts.get(e.to_node, 0)).to_node

        # Normal evaluation
        for edge in edges:
            if self._evaluate_guard(edge.guard, state):
                return edge.to_node

        raise DAGTraversalError(
            f"[{self.dag.dag_id}] No matching edge from node '{current_id}'. "
            f"State keys: {list(state)}. "
            f"Guards: {[e.guard for e in edges]}"
        )

    @staticmethod
    def _evaluate_guard(guard: Optional[str], state: Dict[str, Any]) -> bool:
        """
        Evaluate a guard expression string against state.
        None guard = unconditional (always True).
        """
        if guard is None:
            return True
        try:
            return bool(eval(guard, {"state": state, "__builtins__": {}}))  # noqa: S307
        except Exception as exc:
            logger.debug("Guard eval failed ('%s'): %s", guard, exc)
            return False
