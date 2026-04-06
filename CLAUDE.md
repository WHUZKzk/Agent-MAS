# AutoSR — Automated Systematic Review Agent System

## Project Overview
This is an automated systematic review and meta-analysis system designed
for an EMNLP submission. It implements three core innovations:
1. Dynamic Context Routing (DCR)
2. Protocol-Isomorphic Cognitive Action Graph (PI-CAG)
3. Epistemic-Aware Heterogeneous Dual-Adjudication

## Architecture Documents
All design specifications are in `docs/`. Read them in numbered order.
The master blueprint is `docs/01_MASTER_BLUEPRINT.md`.

## Key Principles
- LLMs MUST NOT do math. All computation in Hard Nodes (Python).
- LLMs MUST NOT make inclusion/exclusion decisions. They extract features only.
- All LLM calls go through ContextManager (DCR).
- Pipeline state is checkpointed after each stage.
- Skill YAMLs define what context each node needs — never hardcode prompts.

## Tech Stack
- Python 3.11+, Pydantic v2, FastAPI, aiohttp
- React + TypeScript + Tailwind (frontend)
- SQLite (state persistence for web layer)

## Benchmark
Ground-truth data is in `data/benchmarks/`. The system input is
`bench_review.json`. Study-level ground truth CSVs are in subdirectories.

## Implementation Order
Follow `docs/01_MASTER_BLUEPRINT.md` §6 strictly:
1. Schemas → 2. Core Engine → 3. Skills → 4. PubMed Client →
5. Search → 6. Screening → 7. Extraction → 8. Orchestrator → 9. Web UI