# 01 — MASTER BLUEPRINT

> **Read this document FIRST.** It provides the complete system overview, directory
> structure, data-flow, and global constraints. Every subsequent spec assumes you
> have fully internalized this file.

---

## 1. System Identity

**Name:** AutoSR — Automated Systematic Review & Meta-Analysis Agent System

**Purpose:** Given a structured review question (`bench_review.json`), AutoSR
autonomously executes the three core stages of a systematic review:

1. **Search** — construct and execute PubMed queries; produce a candidate pool.
2. **Screening** — title/abstract dual-blind screening with adjudication;
   produce a final included-study list.
3. **Extraction** — parse full-text documents; extract, standardize, and compute
   effect sizes for quantitative outcomes.

**Research Context:** The system is designed for an EMNLP submission. Its three
core innovations are:

| # | Innovation | Key Idea |
|---|-----------|----------|
| 1 | **Dynamic Context Routing (DCR)** | Each DAG node receives *only* the Skill (prompt + guidelines) relevant to its task. Context is assembled on mount, discarded on unmount. |
| 2 | **Protocol-Isomorphic Cognitive Action Graph (PI-CAG)** | The SR methodology is explicitly encoded as a DAG. Soft Nodes (LLM) handle semantic reasoning; Hard Nodes (Python) enforce deterministic rules. Edge traversal requires validation. |
| 3 | **Epistemic-Aware Heterogeneous Dual-Adjudication** | In Screening, two *different* base LLMs independently answer the same binary questions. Conflicts are resolved by a blinded Adjudicator that sees only anonymous reasoning traces. |

---

## 2. Directory Structure

```
autosr/
├── src/
│   ├── engine/                      # Core infrastructure
│   │   ├── dag.py                   # DAGDefinition, DAGRunner
│   │   ├── nodes.py                 # BaseNode, SoftNode, HardNode
│   │   ├── agents.py                # ExecutorAgent, ReviewerAdjudicatorAgent
│   │   ├── context_manager.py       # DCR mount / unmount
│   │   └── model_registry.py        # Multi-model config & dispatch
│   │
│   ├── schemas/                     # Unified Pydantic schemas (cross-stage)
│   │   ├── common.py                # Shared types (PaperMetadata, etc.)
│   │   ├── search.py
│   │   ├── screening.py
│   │   └── extraction.py
│   │
│   ├── skills/                      # Skill YAML files (one per Soft Node)
│   │   ├── _schema.yaml             # Skill YAML meta-schema (validation)
│   │   ├── search/
│   │   │   ├── pico_generation.yaml
│   │   │   ├── pearl_growing.yaml
│   │   │   └── ...
│   │   ├── screening/
│   │   │   ├── criteria_binarization.yaml
│   │   │   ├── reviewer_screening.yaml
│   │   │   ├── adjudicator_resolution.yaml
│   │   │   └── ...
│   │   └── extraction/
│   │       ├── document_cartography.yaml
│   │       ├── characteristics_extraction.yaml
│   │       ├── chunk_relevance_scoring.yaml
│   │       ├── outcome_extraction.yaml
│   │       └── ...                  # + dynamically generated skills
│   │
│   ├── guidelines/                  # Methodological guideline fragments
│   │   ├── cochrane_ch4_pico.md
│   │   ├── cochrane_ch4_search.md
│   │   ├── cochrane_ch7_screening.md
│   │   ├── data_extraction_rules.md
│   │   └── ...                      # Stub files — content to be filled later
│   │
│   ├── stages/                      # Stage pipelines (one per stage)
│   │   ├── search_pipeline.py
│   │   ├── screening_pipeline.py
│   │   └── extraction_pipeline.py
│   │
│   ├── clients/                     # External API clients
│   │   └── pubmed_client.py         # Async PubMed E-utilities wrapper
│   │
│   ├── math_sandbox.py              # Deterministic data standardization
│   ├── skill_generator.py           # Benchmark-driven Skill compiler
│   ├── orchestrator.py              # Master controller
│   └── main.py                      # CLI entry point
│
├── data/
│   ├── checkpoints/                 # Stage checkpoints (JSON)
│   ├── benchmarks/                  # bench_review.json, ground-truth CSVs
│   ├── uploads/                     # User-uploaded full-text files (XML/PDF)
│   └── outputs/                     # Final CSVs, reports
│
├── configs/
│   └── models.yaml                  # Model Registry configuration
│
├── tests/                           # Unit & integration tests
└── requirements.txt
```

---

## 3. End-to-End Data Flow

```
bench_review.json
       │
       ▼
┌──────────────────┐
│  SkillGenerator   │  Reads PICO & target outcomes → generates
│  (initialization) │  customized extraction Skill YAMLs
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Search Pipeline  │  Input:  review_question + PICO
│  (7-node DAG)     │  Output: SearchOutput (PMIDs + full metadata)
└────────┬─────────┘
         │  → checkpoint_1_search.json
         ▼
┌──────────────────┐
│ Screening Pipeline│  Input:  SearchOutput
│ (6-node DAG)      │  Output: ScreeningOutput (included PMIDs + decisions)
└────────┬─────────┘
         │  → checkpoint_2_screening.json
         ▼
   [User uploads full-text files for included studies]
         │
         ▼
┌──────────────────┐
│Extraction Pipeline│  Input:  ScreeningOutput + uploaded full texts
│ (6-node DAG)      │  Output: ExtractionOutput (raw + standardized + effect sizes)
└────────┬─────────┘
         │  → checkpoint_3_extraction.json
         ▼
   Final CSVs in data/outputs/
```

---

## 4. Global Constraints

These rules apply across ALL modules. Violating any MUST constraint is a bug.

### 4.1 Neuro-Symbolic Decoupling

| Rule | Level |
|------|-------|
| LLMs MUST NOT perform mathematical calculations. All math belongs in Hard Nodes. | MUST |
| LLMs MUST NOT make final inclusion/exclusion decisions. They extract features only. | MUST |
| All SoftNode outputs MUST be validated against Pydantic schemas. | MUST |
| All SoftNode outputs MUST be JSON format. | MUST |
| Hard Nodes MUST use `assert` statements for critical invariants. A failed assert marks the current item as failed but MUST NOT crash the pipeline. | MUST |

### 4.2 Dynamic Context Routing (DCR)

| Rule | Level |
|------|-------|
| Every LLM call MUST go through `ContextManager.mount()` / `unmount()`. | MUST |
| `mount()` MUST load only the Skill YAML for the current node. | MUST |
| `unmount()` MUST clear all temporary context and log the call metadata. | MUST |
| Methodological guidelines MUST NOT be hardcoded in Python files. Load from `src/guidelines/`. | MUST |
| Skill content MUST NOT be hardcoded. Use Skill YAML files in `src/skills/`. | MUST |

### 4.3 State & Checkpointing

| Rule | Level |
|------|-------|
| After each Stage completes, the Orchestrator MUST dump the full `AppState` to a checkpoint JSON. | MUST |
| On startup, the Orchestrator MUST check for existing checkpoints and resume from the latest. | MUST |
| Checkpoint files MUST be human-readable JSON. | MUST |

### 4.4 Concurrency

| Rule | Level |
|------|-------|
| Within a single paper's processing, independent Soft Nodes (e.g., Reviewer A and B) MAY run in parallel via `asyncio`. | MAY |
| NCBI API calls MUST respect rate limits: `Semaphore(10)` with API key, `Semaphore(3)` without. | MUST |
| Batch fetching from NCBI SHOULD use `retmax=500` pages with History Server (`usehistory=y`). | SHOULD |

### 4.5 Error Handling

| Rule | Level |
|------|-------|
| SoftNode JSON parse failure → retry up to N times (defined per Skill YAML). If still failing, mark item as `FAILED`. | MUST |
| HardNode assertion failure → mark item as `FAILED`, log error, continue to next item. | MUST |
| NCBI API timeout → retry 3 times with exponential backoff. | SHOULD |
| A single item failure MUST NOT halt the entire pipeline. | MUST |

### 4.6 Logging

| Rule | Level |
|------|-------|
| Use Python `logging` module throughout. | MUST |
| Log every DAG node entry/exit with node_id and item_id (e.g., `[Screening][Node 2.3] Processing PMID: 12345 [5/100]`). | MUST |
| Log every Adjudication event with full conflict details. | MUST |

---

## 5. Key Architectural Components (Summary)

Detailed specs are in separate documents. This section provides a quick reference.

### 5.1 DAG System (`src/engine/dag.py`)

The DAG is declared as a data structure (node list + edge list), not procedural
code. A `DAGRunner` traverses the graph, evaluating guard conditions on edges
to determine the next node.

→ Full spec: `02_SCHEMA_CONTRACT.md` (data structures), `03_CORE_ENGINE_SPEC.md` (implementation)

### 5.2 Node System (`src/engine/nodes.py`)

Two node types:
- **HardNode** — deterministic Python. Uses `assert` for validation.
- **SoftNode** — wraps an LLM call. Output validated by Pydantic. Retries on parse failure.

→ Full spec: `03_CORE_ENGINE_SPEC.md`

### 5.3 Agent System (`src/engine/agents.py`)

Two agent classes, differentiated by injected role:
- **ExecutorAgent** — temperature 0.0, for deterministic extraction.
- **ReviewerAdjudicatorAgent** — accepts a `role` parameter and a `model_id`
  for heterogeneous deployment.

→ Full spec: `03_CORE_ENGINE_SPEC.md`

### 5.4 Context Manager / DCR (`src/engine/context_manager.py`)

Assembles prompts for SoftNodes by:
1. Loading the Skill YAML for the current node.
2. Resolving `input_slots` from pipeline state.
3. Loading `guidelines_source` from `src/guidelines/`.
4. Assembling the final `messages` array for the target model.

→ Full spec: `03_CORE_ENGINE_SPEC.md`, `04_SKILL_FRAMEWORK.md`

### 5.5 Model Registry (`src/engine/model_registry.py`, `configs/models.yaml`)

A configuration-driven registry of available LLM backends. Each model entry
specifies: provider, model_id, api_base, max_tokens, supports_vision.
Agents reference models by a logical name (e.g., `"model_a"`, `"model_b"`).

→ Full spec: `03_CORE_ENGINE_SPEC.md`

### 5.6 Skill Generator (`src/skill_generator.py`)

Runs at system initialization, before any pipeline. Reads `bench_review.json`,
infers outcome data types, and uses an LLM to generate customized extraction
Skill YAMLs.

→ Full spec: `04_SKILL_FRAMEWORK.md`

### 5.7 PubMed Client (`src/clients/pubmed_client.py`)

Async wrapper for NCBI E-utilities (esearch, efetch, espell, elink). Uses
History Server, chunked batch fetching, and rate-limited concurrency.

→ Full spec: `05_SEARCH_STAGE.md`

---

## 6. Implementation Order

Build the system in this exact sequence. Each step depends on the previous.

| Step | Module | Spec Document |
|------|--------|--------------|
| 1 | Pydantic Schemas (`src/schemas/`) | `02_SCHEMA_CONTRACT.md` |
| 2 | Core Engine (DAG, Nodes, Agents, ContextManager, ModelRegistry) | `03_CORE_ENGINE_SPEC.md` |
| 3 | Skill Framework + SkillGenerator | `04_SKILL_FRAMEWORK.md` |
| 4 | PubMed Client | `05_SEARCH_STAGE.md` §3 |
| 5 | Search Pipeline | `05_SEARCH_STAGE.md` |
| 6 | Screening Pipeline | `06_SCREENING_STAGE.md` |
| 7 | Extraction Pipeline (incl. MathSandbox) | `07_EXTRACTION_STAGE.md` |
| 8 | Orchestrator + Checkpointing + main.py | `03_CORE_ENGINE_SPEC.md` §6 |

---

## 7. Dependencies

```
# Python 3.11+
pydantic>=2.0
aiohttp
rapidfuzz          # MeSH fuzzy matching
numpy
scipy
pandas
scikit-learn       # TF-IDF for local relevance scoring (Search stage)
pyyaml             # Skill YAML loading
lxml               # XML parsing
```

Optional (for PDF full-text fallback):
```
pymupdf            # PDF → image rendering (for multimodal LLM parsing)
```
