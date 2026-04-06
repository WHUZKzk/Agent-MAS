# 03 — CORE ENGINE SPEC

> **Dependencies:** You MUST have read `01_MASTER_BLUEPRINT.md` and
> `02_SCHEMA_CONTRACT.md` before implementing this module.
>
> **Purpose:** This document specifies the foundational infrastructure —
> the DAG runner, node system, agent classes, DCR context manager, model
> registry, and orchestrator. Build these BEFORE any stage pipeline.

---

## 1. DAG System (`src/engine/dag.py`)

### 1.1 Design Rationale

The PI-CAG innovation requires that the SR methodology is encoded as an
explicit graph data structure — not as procedural `step1(); step2(); step3()`
calls. The DAG definition is declarative; the DAGRunner interprets it.

### 1.2 DAGDefinition

Use the `DAGDefinition`, `NodeDefinition`, and `EdgeDefinition` schemas from
`02_SCHEMA_CONTRACT.md`. Each Stage pipeline declares its DAG at initialization.

### 1.3 DAGRunner

```
class DAGRunner:
    def __init__(self, dag: DAGDefinition, context_manager: ContextManager,
                 node_registry: Dict[str, Callable])

    def run(self, initial_state: dict) -> dict:
        """
        Traverse the DAG from entry_node to a terminal_node.

        Algorithm:
        1. current_node = dag.entry_node
        2. While current_node not in dag.terminal_nodes:
            a. Look up implementation from node_registry
            b. If SoftNode:
               - context_manager.mount(node.skill_id, state)
               - result = agent.execute(mounted_context)
               - validate result against output Pydantic schema
               - if validation fails: retry up to node's max_retries
               - context_manager.unmount()
            c. If HardNode:
               - result = implementation(state)
               - catch AssertionError → mark item as FAILED
            d. Update state with result
            e. Evaluate outgoing edges:
               - For each edge from current_node:
                 - If edge.guard is None → candidate
                 - If eval(edge.guard, state) is True → candidate
               - Select first matching edge (edges are ordered by priority)
               - If no edge matches → raise DAGTraversalError
            f. current_node = selected_edge.to_node
            g. Log: "[{dag_id}] {current_node} → {next_node}"
        3. Return final state

        Guard condition format:
        - Guards are Python expressions as strings.
        - They are evaluated with `state` as the namespace.
        - Examples:
          - "state.has_conflicts == True"
          - "state.adjudication_complete == True"
          - "len(state.conflicts) == 0"

        Loop protection:
        - Track visited nodes with a counter per node_id.
        - If any node is visited > max_iterations (default 5), force
          exit via the first available edge and log a warning.
        """
```

### 1.4 Example: Screening DAG Declaration

```python
screening_dag = DAGDefinition(
    dag_id="screening_pipeline",
    entry_node="s2_1",
    terminal_nodes=["s2_6"],
    nodes=[
        NodeDefinition(node_id="s2_1", node_type="soft",
                       skill_id="screening.criteria_binarization",
                       implementation="stages.screening.criteria_binarization",
                       description="Convert PICO into binary screening questions"),
        NodeDefinition(node_id="s2_2", node_type="hard",
                       implementation="stages.screening.metadata_prefilter",
                       description="Exclude non-primary research by metadata"),
        NodeDefinition(node_id="s2_3", node_type="soft",
                       skill_id="screening.reviewer_screening",
                       implementation="stages.screening.dual_review",
                       description="Heterogeneous dual-blind T/A screening"),
        NodeDefinition(node_id="s2_4", node_type="hard",
                       implementation="stages.screening.logic_gate",
                       description="Symbolic decision logic + conflict detection"),
        NodeDefinition(node_id="s2_5", node_type="soft",
                       skill_id="screening.adjudicator_resolution",
                       implementation="stages.screening.adjudication",
                       description="Blinded CoT adjudication of conflicts"),
        NodeDefinition(node_id="s2_6", node_type="hard",
                       implementation="stages.screening.prisma_reporting",
                       description="Compute Kappa, tally exclusions, produce output"),
    ],
    edges=[
        EdgeDefinition(from_node="s2_1", to_node="s2_2"),
        EdgeDefinition(from_node="s2_2", to_node="s2_3"),
        EdgeDefinition(from_node="s2_3", to_node="s2_4"),
        EdgeDefinition(from_node="s2_4", to_node="s2_5",
                       guard="state.has_conflicts == True"),
        EdgeDefinition(from_node="s2_4", to_node="s2_6",
                       guard="state.has_conflicts == False"),
        EdgeDefinition(from_node="s2_5", to_node="s2_4",
                       guard="state.adjudication_complete == True"),
    ],
)
```

> **Note on the 2.4 → 2.5 → 2.4 loop:** This is the only cycle in any DAG.
> After adjudication (2.5), the logic gate (2.4) re-evaluates with the
> adjudicated answers merged in. If all conflicts are resolved,
> `has_conflicts` becomes `False` and the DAG proceeds to 2.6.
> The loop protection counter prevents infinite cycling.

---

## 2. Node System (`src/engine/nodes.py`)

### 2.1 BaseNode

```
class BaseNode(ABC):
    node_id: str
    logger: logging.Logger  # Named logger: f"autosr.{node_id}"

    @abstractmethod
    def run(self, state: dict) -> dict:
        """Execute the node logic. Returns updated state."""
        pass
```

### 2.2 HardNode

```
class HardNode(BaseNode):
    """Deterministic Python logic. No LLM calls.

    Implementation contract:
    - Use `assert` for critical invariants.
    - AssertionError is caught by DAGRunner, which marks the
      current item as FAILED and continues.
    - All other exceptions propagate and halt the pipeline.
    """

    def run(self, state: dict) -> dict:
        self.logger.info(f"[HardNode] Executing {self.node_id}")
        result = self.execute(state)
        self.logger.info(f"[HardNode] Completed {self.node_id}")
        return result

    @abstractmethod
    def execute(self, state: dict) -> dict:
        pass
```

### 2.3 SoftNode

```
class SoftNode(BaseNode):
    """Wraps an LLM call via ContextManager + Agent.

    Implementation contract:
    - MUST call context_manager.mount() before LLM invocation.
    - MUST call context_manager.unmount() after (even on failure).
    - MUST validate output against the Pydantic schema specified in
      the Skill YAML's output_schema field.
    - On validation failure: retry up to max_retries (from Skill YAML).
    - After all retries exhausted: mark item as FAILED.
    """

    def __init__(self, node_id: str, context_manager: ContextManager,
                 agent: BaseAgent, output_schema: Type[BaseModel],
                 max_retries: int = 2):
        ...

    def run(self, state: dict) -> dict:
        self.logger.info(f"[SoftNode] Executing {self.node_id}")

        mounted_context = self.context_manager.mount(
            skill_id=self.skill_id,
            state=state
        )

        try:
            for attempt in range(1, self.max_retries + 1):
                raw_output = self.agent.call(mounted_context)
                try:
                    parsed = self.output_schema.model_validate_json(raw_output)
                    self.logger.info(f"[SoftNode] {self.node_id} succeeded on attempt {attempt}")
                    return self._update_state(state, parsed)
                except ValidationError as e:
                    self.logger.warning(f"[SoftNode] {self.node_id} attempt {attempt} "
                                        f"validation failed: {e}")
                    if attempt == self.max_retries:
                        self.logger.error(f"[SoftNode] {self.node_id} FAILED after "
                                          f"{self.max_retries} attempts")
                        return self._mark_failed(state)
        finally:
            self.context_manager.unmount()
```

---

## 3. Agent System (`src/engine/agents.py`)

### 3.1 BaseAgent

```
class BaseAgent(ABC):
    model_id: str               # Logical name from ModelRegistry
    model_config: ModelConfig    # Resolved config from ModelRegistry

    @abstractmethod
    def call(self, context: MountedContext) -> str:
        """Send the mounted context to the LLM and return raw string output."""
        pass
```

### 3.2 ExecutorAgent

```
class ExecutorAgent(BaseAgent):
    """For deterministic extraction tasks. Temperature = 0.0.

    Used by: Search (PICO generation, Pearl Growing), Extraction (all Soft Nodes).
    """
    temperature: float = 0.0
```

### 3.3 ReviewerAdjudicatorAgent

```
class ReviewerAdjudicatorAgent(BaseAgent):
    """For screening review and adjudication.

    Behavior changes based on `role`:
    - role="reviewer": Standard screening behavior. Temperature = 0.0.
      (Heterogeneity is provided by different base models, not temperature.)
    - role="adjudicator": Uses specialized blinding prompt.
      Temperature = 0.0.

    The `model_id` determines which base model to use, enabling true
    architectural heterogeneity between Reviewer A and Reviewer B.
    """
    role: Literal["reviewer", "adjudicator"]
    temperature: float = 0.0
```

### 3.4 LLM API Call Implementation

```
All agents share a common LLM calling mechanism:

def _call_llm(self, messages: List[dict], model_config: ModelConfig,
              temperature: float) -> str:
    """
    Route to the correct provider based on model_config.provider.

    Providers:
    - "anthropic": Use anthropic SDK. Messages API.
    - "openai": Use openai SDK. Chat Completions API.
    - "google": Use google-generativeai SDK.

    MUST:
    - Set response_format to JSON when the Skill requires it.
    - Handle rate limit errors (HTTP 429) with exponential backoff.
    - Log: model_id, input_tokens, output_tokens, latency.

    For vision-capable models processing PDF pages:
    - Convert PDF pages to images (PNG).
    - Include images as base64 in the message content array.
    """
```

---

## 4. Context Manager / DCR (`src/engine/context_manager.py`)

### 4.1 Core Concept

The ContextManager is the implementation of Dynamic Context Routing. Its job
is to ensure every SoftNode sees *only* the information relevant to its task.

### 4.2 MountedContext

```
class MountedContext(BaseModel):
    """The fully assembled LLM call payload."""
    system_message: str           # Role + guidelines
    user_message: str             # Input data
    model_id: str                 # Target model logical name
    temperature: float
    response_format: Literal["json", "text"]
    metadata: dict                # For logging: node_id, skill_id, etc.
```

### 4.3 ContextManager

```
class ContextManager:
    def __init__(self, skill_registry: SkillRegistry,
                 guidelines_dir: str,
                 model_registry: ModelRegistry):
        self._current_mount: Optional[MountedContext] = None

    def mount(self, skill_id: str, state: dict) -> MountedContext:
        """
        Assemble the prompt for a SoftNode.

        Steps:
        1. Load Skill YAML from SkillRegistry by skill_id.
        2. Load guidelines text:
           - Read the file referenced by skill.guidelines_source
             from src/guidelines/ directory.
        3. Resolve input_slots:
           - For each input_slot in the Skill YAML, extract the
             corresponding value from `state` using the `source` path.
           - Example: source="current_paper.abstract" → state["current_paper"]["abstract"]
        4. Render the system_message:
           - Combine skill.context_template.role (with variable substitution)
             + loaded guidelines text.
        5. Render the user_message:
           - Combine all resolved input_slot values into a structured
             user message.
        6. Look up model configuration:
           - Use the model_id appropriate for this node (from Skill YAML
             or agent configuration).
        7. Construct and return MountedContext.
        8. Store the MountedContext in self._current_mount (for unmount logging).

        MUST: If guidelines_source file does not exist, raise an error.
              Do NOT silently proceed without guidelines.
        """

    def unmount(self) -> None:
        """
        Finalize and clean up after a SoftNode execution.

        Steps:
        1. Log call metadata from self._current_mount:
           - node_id, skill_id, model_id
           - timestamp
           - (token counts are logged by the Agent, not here)
        2. Set self._current_mount = None.
        3. Clear any internal caches (loaded YAML, guidelines text).

        This method MUST be called even if the SoftNode fails.
        Use try/finally in SoftNode.run() to guarantee this.
        """
```

### 4.4 SkillRegistry

```
class SkillRegistry:
    """Loads and caches Skill YAML files from src/skills/."""

    def __init__(self, skills_dir: str):
        self._cache: Dict[str, SkillDefinition] = {}

    def load(self, skill_id: str) -> SkillDefinition:
        """
        Load a Skill YAML by its ID.

        skill_id format: "{stage}.{skill_name}"
        Maps to file: src/skills/{stage}/{skill_name}.yaml

        Validates the YAML against the Skill schema (see 04_SKILL_FRAMEWORK.md).
        Caches after first load.
        """

    def reload(self, skill_id: str) -> SkillDefinition:
        """Force reload from disk. Used after SkillGenerator creates new skills."""
```

---

## 5. Model Registry (`src/engine/model_registry.py`)

### 5.1 Configuration

```
class ModelConfig(BaseModel):
    provider: Literal["anthropic", "openai", "google"]
    model_id: str                 # Provider-specific model string
    api_base: str
    max_context_tokens: int
    supports_vision: bool = False

class ModelRegistryConfig(BaseModel):
    models: Dict[str, ModelConfig]     # logical_name → config
    defaults: Dict[str, str]           # role → logical_name
```

### 5.2 ModelRegistry

```
class ModelRegistry:
    def __init__(self, config_path: str = "configs/models.yaml"):
        ...

    def get_model(self, logical_name: str) -> ModelConfig:
        """Look up a model by logical name."""

    def get_default(self, role: str) -> ModelConfig:
        """Look up the default model for a role (executor, reviewer_a, etc.)."""

    MUST: Validate on initialization that defaults.reviewer_a and
          defaults.reviewer_b point to different model entries.
```

---

## 6. Orchestrator (`src/orchestrator.py`)

### 6.1 SystematicReviewOrchestrator

```
class SystematicReviewOrchestrator:
    """Master controller for the entire systematic review lifecycle."""

    def __init__(self, bench_review_path: str, config_dir: str = "configs/"):
        """
        1. Parse bench_review.json → ReviewConfig.
        2. Initialize ModelRegistry from configs/models.yaml.
        3. Initialize SkillRegistry from src/skills/.
        4. Initialize ContextManager.
        5. Check for existing checkpoints in data/checkpoints/.
        """

    def run(self) -> AppState:
        """
        Execute the full pipeline:

        1. Check checkpoint:
           - If checkpoint_3_extraction.json exists → skip to reporting.
           - If checkpoint_2_screening.json exists → skip to Extraction.
           - If checkpoint_1_search.json exists → skip to Screening.
           - Otherwise → start from SkillGenerator.

        2. SkillGenerator:
           - Run skill_generator.generate(review_config)
           - Reload SkillRegistry to pick up new files.

        3. Search Stage:
           - Instantiate SearchPipeline with its DAG definition.
           - pipeline.run(review_config) → SearchOutput.
           - Save checkpoint_1_search.json.

        4. Screening Stage:
           - Instantiate ScreeningPipeline with its DAG definition.
           - For each paper in SearchOutput.papers (sequential loop):
             - pipeline.run_single(paper, screening_criteria)
           - Aggregate results → ScreeningOutput.
           - Save checkpoint_2_screening.json.

        5. [User uploads full-text files]
           - Wait for user to place files in data/uploads/.
           - Validate that all included PMIDs have corresponding files.

        6. Extraction Stage:
           - Instantiate ExtractionPipeline with its DAG definition.
           - For each pmid in ScreeningOutput.included_pmids (sequential loop):
             - pipeline.run_single(pmid, uploaded_file_path)
           - Aggregate results → Dict[str, ExtractionOutput].
           - Save checkpoint_3_extraction.json.

        7. Output Generation:
           - Export CSVs to data/outputs/ matching benchmark format.

        Logging:
        - INFO at each stage boundary: "[Orchestrator] Starting Search Stage"
        - INFO per paper: "[Screening] Processing PMID: 12345 [5/100]"
        - ERROR on failures: "[Extraction] PMID 12345 FAILED at node 3.4: SD <= 0"
        """
```

### 6.2 Checkpointing

```
def save_checkpoint(self, stage: str, app_state: AppState) -> None:
    """
    Serialize AppState to data/checkpoints/checkpoint_{stage}.json.

    MUST: Use Pydantic's .model_dump_json(indent=2) for human-readable output.
    MUST: Include metadata (timestamp, runtime duration, error count).
    SHOULD: Write to a temp file first, then rename (atomic write).
    """

def load_checkpoint(self) -> Optional[Tuple[str, AppState]]:
    """
    Check for existing checkpoints in reverse order:
    extraction → screening → search.

    Return the (stage_name, AppState) of the most recent checkpoint,
    or None if no checkpoints exist.
    """
```

---

## 7. Implementation Checklist

Build in this order:

1. [ ] `src/schemas/common.py` — PaperMetadata, PICODefinition, ReviewConfig, AppState
2. [ ] `src/schemas/search.py` — All search schemas
3. [ ] `src/schemas/screening.py` — All screening schemas
4. [ ] `src/schemas/extraction.py` — All extraction schemas
5. [ ] `configs/models.yaml` — Model registry configuration
6. [ ] `src/engine/model_registry.py` — ModelRegistry class
7. [ ] `src/engine/nodes.py` — BaseNode, HardNode, SoftNode
8. [ ] `src/engine/agents.py` — ExecutorAgent, ReviewerAdjudicatorAgent
9. [ ] `src/engine/context_manager.py` — ContextManager, SkillRegistry, MountedContext
10. [ ] `src/engine/dag.py` — DAGDefinition, DAGRunner
11. [ ] `src/orchestrator.py` — SystematicReviewOrchestrator + Checkpointing
12. [ ] `src/main.py` — CLI entry point
