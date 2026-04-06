"""
TDD tests for DAGRunner.

Written BEFORE implementation. All tests must fail initially, then pass
once DAGRunner is implemented.

Uses a mock Screening DAG that mirrors the real 6-node structure from
docs/03_CORE_ENGINE_SPEC.md §1.4, with lightweight stub implementations.
"""
import pytest
from unittest.mock import MagicMock, patch, call
from typing import Dict, Any

from src.schemas.common import DAGDefinition, NodeDefinition, EdgeDefinition


# ---------------------------------------------------------------------------
# Helpers: build test DAGs
# ---------------------------------------------------------------------------

def make_linear_dag() -> DAGDefinition:
    """3-node linear DAG: A → B → C (all HardNodes for simplicity)."""
    return DAGDefinition(
        dag_id="test_linear",
        entry_node="A",
        terminal_nodes=["C"],
        nodes=[
            NodeDefinition(node_id="A", node_type="hard",
                           implementation="stub.node_a", description="Node A"),
            NodeDefinition(node_id="B", node_type="hard",
                           implementation="stub.node_b", description="Node B"),
            NodeDefinition(node_id="C", node_type="hard",
                           implementation="stub.node_c", description="Node C"),
        ],
        edges=[
            EdgeDefinition(from_node="A", to_node="B"),
            EdgeDefinition(from_node="B", to_node="C"),
        ],
    )


def make_branching_dag() -> DAGDefinition:
    """
    4-node DAG with conditional branch:

        A → B → C  (guard: state["branch"] == "yes")
              ↘ D  (guard: state["branch"] == "no")

    C and D are both terminal nodes.
    """
    return DAGDefinition(
        dag_id="test_branching",
        entry_node="A",
        terminal_nodes=["C", "D"],
        nodes=[
            NodeDefinition(node_id="A", node_type="hard",
                           implementation="stub.node_a", description="A"),
            NodeDefinition(node_id="B", node_type="hard",
                           implementation="stub.node_b", description="B"),
            NodeDefinition(node_id="C", node_type="hard",
                           implementation="stub.node_c", description="C — branch yes"),
            NodeDefinition(node_id="D", node_type="hard",
                           implementation="stub.node_d", description="D — branch no"),
        ],
        edges=[
            EdgeDefinition(from_node="A", to_node="B"),
            EdgeDefinition(from_node="B", to_node="C",
                           guard='state["branch"] == "yes"'),
            EdgeDefinition(from_node="B", to_node="D",
                           guard='state["branch"] == "no"'),
        ],
    )


def make_loop_dag() -> DAGDefinition:
    """
    Mock Screening DAG with the 2.4 ↔ 2.5 cycle:

        s2_1 → s2_2 → s2_3 → s2_4 → s2_5 (guard: has_conflicts)
                                    ↓ s2_6   (guard: not has_conflicts)
                              ← s2_5 (back to s2_4)
    """
    return DAGDefinition(
        dag_id="test_loop",
        entry_node="s2_1",
        terminal_nodes=["s2_6"],
        nodes=[
            NodeDefinition(node_id="s2_1", node_type="hard",
                           implementation="stub.s2_1", description="Criteria binarization"),
            NodeDefinition(node_id="s2_2", node_type="hard",
                           implementation="stub.s2_2", description="Metadata prefilter"),
            NodeDefinition(node_id="s2_3", node_type="hard",
                           implementation="stub.s2_3", description="Dual review"),
            NodeDefinition(node_id="s2_4", node_type="hard",
                           implementation="stub.s2_4", description="Logic gate"),
            NodeDefinition(node_id="s2_5", node_type="hard",
                           implementation="stub.s2_5", description="Adjudication"),
            NodeDefinition(node_id="s2_6", node_type="hard",
                           implementation="stub.s2_6", description="PRISMA reporting"),
        ],
        edges=[
            EdgeDefinition(from_node="s2_1", to_node="s2_2"),
            EdgeDefinition(from_node="s2_2", to_node="s2_3"),
            EdgeDefinition(from_node="s2_3", to_node="s2_4"),
            EdgeDefinition(from_node="s2_4", to_node="s2_5",
                           guard='state["has_conflicts"] == True'),
            EdgeDefinition(from_node="s2_4", to_node="s2_6",
                           guard='state["has_conflicts"] == False'),
            EdgeDefinition(from_node="s2_5", to_node="s2_4",
                           guard='state["adjudication_complete"] == True'),
        ],
    )


def make_soft_node_dag() -> DAGDefinition:
    """2-node DAG with one SoftNode to test ContextManager integration."""
    return DAGDefinition(
        dag_id="test_soft",
        entry_node="soft_1",
        terminal_nodes=["hard_2"],
        nodes=[
            NodeDefinition(node_id="soft_1", node_type="soft",
                           skill_id="screening.reviewer_screening",
                           implementation="stub.soft_1", description="LLM node"),
            NodeDefinition(node_id="hard_2", node_type="hard",
                           implementation="stub.hard_2", description="Hard node"),
        ],
        edges=[
            EdgeDefinition(from_node="soft_1", to_node="hard_2"),
        ],
    )


# ---------------------------------------------------------------------------
# Import target (will fail until implemented)
# ---------------------------------------------------------------------------

from src.engine.dag import DAGRunner, DAGTraversalError  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def make_registry():
    """Factory: returns a node_registry dict from a mapping of {node_id: fn}."""
    def _make(mapping: Dict[str, Any]) -> Dict[str, Any]:
        return mapping
    return _make


@pytest.fixture
def dummy_cm():
    """A minimal ContextManager mock."""
    cm = MagicMock()
    cm.mount.return_value = MagicMock()   # MountedContext
    return cm


# ---------------------------------------------------------------------------
# Test 1: Linear traversal — visits all nodes in order
# ---------------------------------------------------------------------------

class TestLinearTraversal:
    def test_all_nodes_visited_in_order(self, make_registry, dummy_cm):
        visited = []

        def node_a(state):
            visited.append("A")
            return {**state, "a_done": True}

        def node_b(state):
            visited.append("B")
            return {**state, "b_done": True}

        def node_c(state):
            visited.append("C")
            return {**state, "c_done": True}

        dag = make_linear_dag()
        registry = make_registry({"A": node_a, "B": node_b, "C": node_c})
        runner = DAGRunner(dag=dag, context_manager=dummy_cm, node_registry=registry)

        result = runner.run({"state": {}})

        assert visited == ["A", "B", "C"]

    def test_state_accumulates_across_nodes(self, make_registry, dummy_cm):
        dag = make_linear_dag()
        registry = make_registry({
            "A": lambda s: {**s, "a": 1},
            "B": lambda s: {**s, "b": 2},
            "C": lambda s: {**s, "c": 3},
        })
        runner = DAGRunner(dag=dag, context_manager=dummy_cm, node_registry=registry)
        result = runner.run({})

        assert result["a"] == 1
        assert result["b"] == 2
        assert result["c"] == 3

    def test_returns_final_state(self, make_registry, dummy_cm):
        dag = make_linear_dag()
        registry = make_registry({
            "A": lambda s: {**s, "x": 42},
            "B": lambda s: s,
            "C": lambda s: {**s, "y": 99},
        })
        runner = DAGRunner(dag=dag, context_manager=dummy_cm, node_registry=registry)
        result = runner.run({})
        assert result == {"x": 42, "y": 99}


# ---------------------------------------------------------------------------
# Test 2: Conditional edges (branching)
# ---------------------------------------------------------------------------

class TestConditionalEdges:
    def test_takes_yes_branch(self, make_registry, dummy_cm):
        visited = []
        dag = make_branching_dag()
        registry = make_registry({
            "A": lambda s: {**s, "branch": "yes"},
            "B": lambda s: (visited.append("B"), s)[1],
            "C": lambda s: (visited.append("C"), s)[1],
            "D": lambda s: (visited.append("D"), s)[1],
        })
        runner = DAGRunner(dag=dag, context_manager=dummy_cm, node_registry=registry)
        runner.run({"branch": "yes"})
        assert "C" in visited
        assert "D" not in visited

    def test_takes_no_branch(self, make_registry, dummy_cm):
        visited = []
        dag = make_branching_dag()
        registry = make_registry({
            "A": lambda s: {**s, "branch": "no"},
            "B": lambda s: s,
            "C": lambda s: (visited.append("C"), s)[1],
            "D": lambda s: (visited.append("D"), s)[1],
        })
        runner = DAGRunner(dag=dag, context_manager=dummy_cm, node_registry=registry)
        runner.run({"branch": "no"})
        assert "D" in visited
        assert "C" not in visited

    def test_no_matching_guard_raises(self, make_registry, dummy_cm):
        dag = make_branching_dag()
        registry = make_registry({
            "A": lambda s: {**s, "branch": "maybe"},  # matches neither guard
            "B": lambda s: s,
            "C": lambda s: s,
            "D": lambda s: s,
        })
        runner = DAGRunner(dag=dag, context_manager=dummy_cm, node_registry=registry)
        with pytest.raises(DAGTraversalError):
            runner.run({"branch": "maybe"})


# ---------------------------------------------------------------------------
# Test 3: Cycle with adjudication loop (mock Screening DAG)
# ---------------------------------------------------------------------------

class TestAdjudicationLoop:
    def test_loop_resolves_after_one_adjudication(self, make_registry, dummy_cm):
        """
        s2_4 sees conflicts=True on first pass → s2_5 runs → sets
        adjudication_complete=True and has_conflicts=False → s2_4 runs
        again → routes to s2_6.
        """
        s2_4_call_count = [0]

        def s2_4(state):
            s2_4_call_count[0] += 1
            if s2_4_call_count[0] == 1:
                return {**state, "has_conflicts": True, "adjudication_complete": False}
            else:
                return {**state, "has_conflicts": False}

        def s2_5(state):
            return {**state, "adjudication_complete": True, "has_conflicts": False}

        dag = make_loop_dag()
        registry = make_registry({
            "s2_1": lambda s: s,
            "s2_2": lambda s: s,
            "s2_3": lambda s: {**s, "has_conflicts": True, "adjudication_complete": False},
            "s2_4": s2_4,
            "s2_5": s2_5,
            "s2_6": lambda s: {**s, "done": True},
        })
        runner = DAGRunner(dag=dag, context_manager=dummy_cm, node_registry=registry)
        result = runner.run({})

        assert result.get("done") is True
        assert s2_4_call_count[0] == 2   # visited twice: before and after adjudication

    def test_loop_protection_triggers_on_excessive_cycles(self, make_registry, dummy_cm):
        """
        If adjudication never resolves conflicts, loop protection kicks in
        after max_iterations and forces forward progress with a warning.
        """
        dag = make_loop_dag()
        registry = make_registry({
            "s2_1": lambda s: s,
            "s2_2": lambda s: s,
            "s2_3": lambda s: {**s, "has_conflicts": True, "adjudication_complete": True},
            # s2_4 always reports conflicts → infinite loop without protection
            "s2_4": lambda s: {**s, "has_conflicts": True},
            "s2_5": lambda s: {**s, "adjudication_complete": True},
            "s2_6": lambda s: {**s, "done": True},
        })
        runner = DAGRunner(dag=dag, context_manager=dummy_cm, node_registry=registry,
                           max_iterations=5)
        # Should NOT raise — loop protection forces exit
        result = runner.run({})
        assert result.get("done") is True


# ---------------------------------------------------------------------------
# Test 4: HardNode AssertionError handling
# ---------------------------------------------------------------------------

class TestHardNodeAssertionHandling:
    def test_assertion_error_marks_item_failed_does_not_halt(
        self, make_registry, dummy_cm
    ):
        """
        A HardNode that raises AssertionError should not propagate —
        DAGRunner catches it, marks state["_failed"] = True, and continues
        traversal (or returns the failed state, as the pipeline will handle it).
        """
        dag = make_linear_dag()
        registry = make_registry({
            "A": lambda s: s,
            "B": lambda s: (_ for _ in ()).throw(AssertionError("SD <= 0")),
            "C": lambda s: {**s, "c_done": True},
        })
        runner = DAGRunner(dag=dag, context_manager=dummy_cm, node_registry=registry)
        result = runner.run({})

        # Pipeline continues to C even after B fails
        assert result.get("c_done") is True
        # Failure is recorded
        assert result.get("_failed") is True

    def test_non_assertion_exception_propagates(self, make_registry, dummy_cm):
        """
        A HardNode that raises a non-AssertionError exception (e.g. TypeError)
        MUST propagate and halt the pipeline.
        """
        dag = make_linear_dag()

        def broken_node(state):
            raise TypeError("Unexpected internal error")

        registry = make_registry({
            "A": lambda s: s,
            "B": broken_node,
            "C": lambda s: s,
        })
        runner = DAGRunner(dag=dag, context_manager=dummy_cm, node_registry=registry)
        with pytest.raises(TypeError):
            runner.run({})


# ---------------------------------------------------------------------------
# Test 5: SoftNode — ContextManager mount/unmount contract
# ---------------------------------------------------------------------------

class TestSoftNodeContextContract:
    def test_mount_and_unmount_called_for_soft_node(self, make_registry, dummy_cm):
        """
        For a SoftNode, DAGRunner MUST call context_manager.mount() before
        execution and unmount() after — even if the node succeeds.
        """
        from src.engine.dag import DAGRunner

        dag = make_soft_node_dag()
        registry = make_registry({
            "soft_1": lambda s: s,
            "hard_2": lambda s: s,
        })
        runner = DAGRunner(dag=dag, context_manager=dummy_cm, node_registry=registry)
        runner.run({})

        dummy_cm.mount.assert_called_once()
        dummy_cm.unmount.assert_called_once()

    def test_unmount_called_even_on_soft_node_assertion_failure(
        self, make_registry, dummy_cm
    ):
        """unmount() MUST be called in the finally block."""
        dag = make_soft_node_dag()
        registry = make_registry({
            "soft_1": lambda s: (_ for _ in ()).throw(AssertionError("parse fail")),
            "hard_2": lambda s: s,
        })
        runner = DAGRunner(dag=dag, context_manager=dummy_cm, node_registry=registry)
        runner.run({})  # should not raise

        dummy_cm.unmount.assert_called_once()


# ---------------------------------------------------------------------------
# Test 6: Node registry lookup
# ---------------------------------------------------------------------------

class TestNodeRegistryLookup:
    def test_missing_node_in_registry_raises_key_error(self, dummy_cm):
        dag = make_linear_dag()
        runner = DAGRunner(dag=dag, context_manager=dummy_cm,
                           node_registry={"A": lambda s: s})  # B and C missing
        with pytest.raises(KeyError):
            runner.run({})

    def test_unconditional_edge_always_taken(self, make_registry, dummy_cm):
        """An edge with guard=None is always taken."""
        visited = []
        dag = make_linear_dag()
        registry = make_registry({
            "A": lambda s: (visited.append("A"), s)[1],
            "B": lambda s: (visited.append("B"), s)[1],
            "C": lambda s: (visited.append("C"), s)[1],
        })
        runner = DAGRunner(dag=dag, context_manager=dummy_cm, node_registry=registry)
        runner.run({})
        assert visited == ["A", "B", "C"]
