# 05 — SEARCH STAGE SPEC

> **Dependencies:** You MUST have implemented the Core Engine
> (`03_CORE_ENGINE_SPEC.md`) and Skill Framework (`04_SKILL_FRAMEWORK.md`).
>
> **Purpose:** Implement the 7-node Search DAG. Takes a natural language
> review question and outputs a deduplicated candidate PMID list with full
> metadata.

---

## 1. Stage Overview

**Input:** `ReviewConfig` from `bench_review.json`
**Output:** `SearchOutput` (see `02_SCHEMA_CONTRACT.md`)

The Search stage constructs a PubMed Boolean query from the review question,
executes it, augments it via Pearl Growing, and produces a final candidate pool.

---

## 2. DAG Definition

```
Nodes: [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
Entry: 1.1
Terminal: 1.7

Edges (all unconditional):
  1.1 → 1.2 → 1.3 → 1.4 → 1.5 → 1.6 → 1.7
```

This is a strictly linear DAG with no branches or loops.

---

## 3. PubMed Client (`src/clients/pubmed_client.py`)

Build this BEFORE implementing the DAG nodes.

### 3.1 Interface

```
class PubMedClient:
    def __init__(self, api_key: Optional[str] = None):
        """
        api_key: From environment variable NCBI_API_KEY.
        If present: rate limit = 10 req/sec.
        If absent: rate limit = 3 req/sec.
        """

    async def search(self, query: str, use_history: bool = True
                     ) -> SearchResult:
        """
        Execute esearch.
        - db="pubmed"
        - usehistory="y" (always)
        - Returns: {webenv, query_key, total_count, pmid_list}
        """

    async def fetch_batch(self, webenv: str, query_key: str,
                          retstart: int, retmax: int = 500
                          ) -> List[PaperMetadata]:
        """
        Execute efetch for a single page.
        - db="pubmed"
        - rettype="xml"
        - Parse XML to extract: pmid, title, abstract,
          publication_types, mesh_terms.
        """

    async def fetch_all(self, webenv: str, query_key: str,
                        total_count: int) -> List[PaperMetadata]:
        """
        Fetch all records using paginated batch requests.
        - Pages of retmax=500.
        - Concurrent requests controlled by asyncio.Semaphore.
        - Rate limited: 0.1s minimum between requests.
        """

    async def validate_mesh(self, term: str) -> MeSHResult:
        """
        Query NCBI MeSH database for a single term.
        - db="mesh"
        - Returns: {found: bool, descriptor_name: str | None,
                     entry_terms: List[str]}
        """

    async def spell_check(self, term: str) -> Optional[str]:
        """
        Use espell to get NCBI's spelling suggestion.
        """
```

### 3.2 Rate Limiting

```
class AsyncRateLimiter:
    """Token-bucket rate limiter for async HTTP calls."""

    def __init__(self, rate: int):
        """rate: max requests per second."""
        self._semaphore = asyncio.Semaphore(rate)
        self._min_interval = 1.0 / rate

    async def acquire(self):
        """Wait for a slot. Enforces minimum interval between requests."""

    MUST: Handle HTTP 429 responses by backing off for 1 second and retrying.
    MUST: Handle HTTP 500/503 by retrying up to 3 times with exponential backoff.
```

### 3.3 Implementation Notes

- Use `aiohttp.ClientSession` for all HTTP calls.
- Base URL: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/`
- Always include `api_key` parameter if available.
- Parse XML responses with `lxml.etree`.
- MUST: Close the `aiohttp` session when the client is no longer needed.

---

## 4. Node Specifications

### Node 1.1: PICO Term Generation

| Field | Value |
|-------|-------|
| **Type** | Soft Node |
| **Skill** | `search.pico_generation` |
| **Input** | `review_config.title`, `review_config.pico` |
| **Output** | `PICOTermSet` (pre-validation, terms have status="pending") |
| **Constraints** | MUST: Output valid JSON. MUST: ≥1 term per dimension. SHOULD: 3–5 terms per dimension including synonyms and MeSH candidates. |
| **Error Handling** | Parse failure → retry up to 2 times. |

### Node 1.2: MeSH Validation & Alignment

| Field | Value |
|-------|-------|
| **Type** | Hard Node |
| **Input** | `PICOTermSet` from Node 1.1 |
| **Logic** | Three-level alignment for each term (see below) |
| **Output** | `PICOTermSet` (all terms now have validated status + search_field) |
| **Constraints** | MUST: No unverified term enters the Boolean query as `[MeSH Terms]`. |
| **Error Handling** | NCBI API timeout → retry 3× with exponential backoff. |

**Three-Level MeSH Alignment Logic:**

```
for each term in pico_terms (all dimensions):
    # Level 1: Exact match
    result = await pubmed_client.validate_mesh(term.original)
    if result.found:
        term.status = "valid_mesh"
        term.normalized = result.descriptor_name
        term.search_field = "[MeSH Terms]"
        continue

    # Level 2: NCBI spelling correction + Entry Term lookup
    corrected = await pubmed_client.spell_check(term.original)
    if corrected:
        result2 = await pubmed_client.validate_mesh(corrected)
        if result2.found:
            term.status = "mapped"
            term.normalized = result2.descriptor_name
            term.search_field = "[MeSH Terms]"
            continue
    # Also check if original is an Entry Term of any MeSH descriptor
    # (This is embedded in the validate_mesh response — check entry_terms)

    # Level 3: Fuzzy matching
    # Use rapidfuzz.fuzz.token_sort_ratio against the NCBI-returned
    # descriptor names and entry terms.
    best_match, score = fuzzy_match_mesh(term.original)
    if score >= 80:
        term.status = "fuzzy_mapped"
        term.normalized = best_match
        term.search_field = "[MeSH Terms]"
        term.similarity_score = score
    else:
        term.status = "not_found"
        term.search_field = "[tiab]"
```

### Node 1.3: Boolean Query Construction

| Field | Value |
|-------|-------|
| **Type** | Hard Node |
| **Input** | Validated `PICOTermSet` from Node 1.2 |
| **Logic** | Pure Python string assembly |
| **Output** | `QueryRecord` with the Boolean query string |
| **Constraints** | MUST: Intra-dimension terms joined with OR. MUST: Inter-dimension groups joined with AND. MUST: Each term tagged with its search_field. |

**Construction rules:**

```
For each dimension D in [P, I, C, O]:
    group_parts = []
    for term in pico_terms[D]:
        if term.search_field == "[MeSH Terms]":
            group_parts.append(f'"{term.normalized}"[MeSH Terms]')
        else:
            group_parts.append(f'"{term.original}"[tiab]')
    dimension_query = "(" + " OR ".join(group_parts) + ")"

final_query = " AND ".join([P_query, I_query, C_query, O_query])
```

### Node 1.4: PubMed Search Execution

| Field | Value |
|-------|-------|
| **Type** | Hard Node |
| **Input** | Query string from Node 1.3 |
| **Logic** | Execute esearch + efetch |
| **Output** | Updated `SearchOutput` with initial candidates |
| **Constraints** | MUST: Use History Server (usehistory=y). MUST: result count > 0. |
| **Error Handling** | Zero results → log error, halt pipeline. |

**Execution logic:**

```
1. search_result = await pubmed_client.search(query_string)
2. assert search_result.total_count > 0, "Search returned zero results"
3. all_papers = await pubmed_client.fetch_all(
       search_result.webenv,
       search_result.query_key,
       search_result.total_count
   )
4. Store all papers in state.search_output.papers

5. Select Top-K seed papers for Pearl Growing:
   - Use scikit-learn's TfidfVectorizer on all abstracts.
   - Compute cosine similarity between each abstract and the
     review_config.abstract (the review's own abstract).
   - Select top 20 by similarity score.
   - Store as state.seed_papers
```

### Node 1.5: Pearl Growing (Search Augmentation)

| Field | Value |
|-------|-------|
| **Type** | Soft Node + Hard Node (composite) |
| **Skill** | `search.pearl_growing` |
| **Input** | Current PICO terms + Top-20 seed paper abstracts |
| **Output** | List of augmented terms |

**This node has two sub-steps:**

**Sub-step A (Hard Node):** Extract MeSH terms from seed papers.

```
For each seed paper in top-20:
    extract its mesh_terms from PaperMetadata
Compute frequency of all MeSH terms across seed papers.
Identify MeSH terms that appear in ≥3 seed papers but are NOT
already in the current PICOTermSet.
These are "MeSH gaps" — high-value missing terms.
```

**Sub-step B (Soft Node):** LLM reads seed abstracts for keyword augmentation.

```
Input to LLM:
  - Current PICO terms
  - Top-10 seed abstracts (not all 20 — context efficiency)
  - The MeSH gaps identified in Sub-step A

LLM task:
  - Identify critical missing keywords, synonyms, specific tool names,
    population descriptors not covered by current terms.
  - Prioritize the MeSH gaps — confirm which are relevant.
  - Output: JSON list of new terms with PICO dimension assignment.
```

**After Sub-step B:** New terms go through Node 1.2's MeSH alignment logic
(call it as a function, not a DAG re-traversal) before entering the query.

### Node 1.6: Final Search Execution

| Field | Value |
|-------|-------|
| **Type** | Hard Node |
| **Input** | Augmented terms from Node 1.5 + original query |
| **Logic** | Rebuild Boolean query with augmented terms. Re-execute full search. |
| **Output** | Updated `SearchOutput` with expanded candidate pool |
| **Constraints** | MUST: result count > 10 and < 50,000. |
| **Error Handling** | Out of range → log warning but continue with whatever results exist. |

### Node 1.7: Deduplication & Metadata Prefetch

| Field | Value |
|-------|-------|
| **Type** | Hard Node |
| **Input** | All candidate PMIDs from Node 1.6 |
| **Logic** | Deduplicate, finalize metadata, produce output |
| **Output** | Final `SearchOutput` |

**Logic:**

```
1. Merge all PMIDs from initial search and augmented search.
2. Remove duplicates (by PMID string).
3. Ensure all PMIDs have complete metadata (title, abstract,
   publication_types, mesh_terms). If any are missing,
   batch-fetch from NCBI.
4. Compute PRISMA numbers:
   - initial_query_results
   - augmented_query_results
   - after_deduplication
   - final_candidate_count
5. Produce final SearchOutput.
```

---

## 5. Stage-Level Constraints

| Rule | Level |
|------|-------|
| The entire Search pipeline runs ONCE per review (not per paper). | MUST |
| All NCBI API calls go through PubMedClient with rate limiting. | MUST |
| The Top-K selection in Node 1.4 MUST use local TF-IDF scoring, NOT PubMed's sort=relevance. | MUST |
| Pearl Growing (Node 1.5) MUST include both Hard (MeSH gap analysis) and Soft (LLM keyword augmentation) sub-steps. | MUST |
| New terms from Pearl Growing MUST go through MeSH alignment before entering the query. | MUST |

---

## 6. Skill Files to Create

| Skill ID | File |
|----------|------|
| `search.pico_generation` | `src/skills/search/pico_generation.yaml` |
| `search.pearl_growing` | `src/skills/search/pearl_growing.yaml` |

(All other Search nodes are Hard Nodes and do not need Skills.)

---

## 7. Implementation Checklist

1. [ ] `src/clients/pubmed_client.py` — PubMedClient + AsyncRateLimiter
2. [ ] `src/skills/search/pico_generation.yaml`
3. [ ] `src/skills/search/pearl_growing.yaml`
4. [ ] `src/guidelines/cochrane_ch4_pico.md` (stub)
5. [ ] `src/guidelines/pearl_growing_method.md` (stub)
6. [ ] `src/stages/search_pipeline.py` — SearchPipeline class + all 7 nodes
7. [ ] Unit tests for MeSH alignment logic (Node 1.2)
8. [ ] Unit tests for Boolean query construction (Node 1.3)
