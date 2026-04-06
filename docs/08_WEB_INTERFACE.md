# 08 — WEB INTERFACE SPEC

> **Dependencies:** Read `01_MASTER_BLUEPRINT.md` for system architecture.
>
> **Purpose:** Define the web interface that allows users to configure, execute,
> monitor, and interact with the AutoSR pipeline. The UI is the human-facing
> layer on top of the core engine.

---

## 1. Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Frontend | React + TypeScript + Tailwind CSS | Modern, component-driven, rapid UI |
| Backend | FastAPI (Python) | Shares runtime with core engine; async native |
| Real-time | WebSocket (FastAPI) | Stream pipeline progress to frontend |
| State | SQLite (via SQLAlchemy) | Lightweight, no external DB dependency |
| File Storage | Local filesystem (`data/uploads/`, `data/outputs/`) | Simple, direct |

The backend wraps the existing `SystematicReviewOrchestrator` — it does NOT
reimplement pipeline logic. The web layer is a thin control plane.

---

## 2. Page Structure & User Flow

The interface follows the natural SR workflow as a **step-by-step wizard**
with a persistent sidebar showing progress.

```
┌─────────────────────────────────────────────────┐
│  Sidebar (persistent)     │  Main Content Area  │
│                           │                     │
│  ● Step 1: Configure      │  [Current step      │
│  ○ Step 2: Search         │   content renders   │
│  ○ Step 3: Screening      │   here]             │
│  ○ Step 4: Upload PDFs    │                     │
│  ○ Step 5: Extraction     │                     │
│  ○ Step 6: Results        │                     │
│                           │                     │
│  [Progress bar]           │                     │
│  [Pipeline status]        │                     │
└─────────────────────────────────────────────────┘
```

### Page 1: Configure Review

**Purpose:** User provides the review question and PICO definition.

**UI Elements:**
- Text input: Review Title
- Text area: Review Question (natural language)
- Four text areas: P, I, C, O definitions
- Multi-tag input: Target Characteristics (e.g., "Mean Age", "Sample Size")
- Multi-tag input: Target Outcomes (e.g., "IL-6", "CRP")
- Dropdown: Select models for Reviewer A / Reviewer B / Executor
- Button: "Start Review" → triggers SkillGenerator + Search

**Alternative:** File upload for `bench_review.json` to auto-fill all fields.

**Backend action:** POST `/api/reviews` → creates a ReviewConfig, runs
SkillGenerator, then begins Search pipeline.

### Page 2: Search Progress & Results

**Purpose:** Monitor search execution and review the candidate pool.

**UI Elements:**
- Real-time log stream (WebSocket): shows DAG node transitions, query strings,
  result counts as they happen.
- After completion:
  - Summary card: total candidates found, PRISMA numbers
  - Table: generated Boolean query strings (initial + augmented)
  - Table: validated PICO terms with MeSH alignment status
  - Button: "Proceed to Screening"

**Backend action:** GET `/api/reviews/{id}/search` for results.
WebSocket `/ws/reviews/{id}/progress` for live updates.

### Page 3: Screening Dashboard

**Purpose:** Monitor dual-review screening with real-time progress.

**UI Elements:**
- Progress indicator: "Screening paper 15 / 200"
- Real-time stats:
  - Papers included / excluded / in adjudication
  - Running Cohen's Kappa
- After completion:
  - Summary card: PRISMA screening numbers
  - Expandable table: each paper's decision, reviewer answers, conflicts
  - Filter/sort: by status (Included / Excluded / Adjudicated)
  - Highlight rows that went to adjudication (these are interesting for the paper)
  - Button: "Proceed to Upload Full Texts"

### Page 4: Full-Text Upload

**Purpose:** User uploads full-text files for included studies.

**UI Elements:**
- List of included PMIDs with title (from screening output)
- Each row has:
  - Status indicator: ✅ Uploaded / ❌ Missing
  - Upload button (accepts .xml or .pdf)
  - Auto-detect: if file is XML or PDF, show badge
- Drag-and-drop zone for batch upload (match files to PMIDs by filename)
- Validation: warn if any included PMID is missing a file
- Button: "Start Extraction" (enabled only when all files uploaded, or user
  explicitly opts to skip missing papers)

**Backend action:** POST `/api/reviews/{id}/uploads` (multipart file upload).
Files stored in `data/uploads/{review_id}/{pmid}.xml` or `.pdf`.

### Page 5: Extraction Progress & Results

**Purpose:** Monitor extraction, review raw and standardized data.

**UI Elements:**
- Progress: "Extracting paper 3 / 12"
- Per-paper accordion:
  - Document Map visualization (sections, tables, arms, timepoints)
  - Characteristics extraction table
  - Outcome extraction table:
    - Raw values (as extracted by LLM)
    - Standardized values (after Math Sandbox)
    - Effect sizes (computed)
    - Validation flags (any failed sanity checks highlighted in red)
- After completion:
  - Button: "Download All CSVs" (zip file)
  - Button: "View Results Summary"

### Page 6: Results & Export

**Purpose:** Final overview and data export.

**UI Elements:**
- Summary statistics:
  - Total studies included
  - Outcomes extracted
  - Data completeness percentage
- Downloadable files:
  - `study_characteristics.csv` (merged across all studies)
  - Per-outcome `study_results_{outcome}.csv`
  - PRISMA flow data (JSON)
  - Full pipeline log
- Optional: simple forest plot visualization per outcome
  (using a React charting library — not a formal meta-analysis tool)

---

## 3. Backend API Design

### 3.1 REST Endpoints

```
POST   /api/reviews                    Create new review (accepts ReviewConfig JSON or bench_review.json upload)
GET    /api/reviews/{id}               Get review status and current stage
POST   /api/reviews/{id}/start         Start/resume pipeline from current checkpoint
GET    /api/reviews/{id}/search        Get search results
GET    /api/reviews/{id}/screening     Get screening results
POST   /api/reviews/{id}/uploads       Upload full-text files (multipart)
GET    /api/reviews/{id}/extraction    Get extraction results
GET    /api/reviews/{id}/export        Download all output CSVs as zip
DELETE /api/reviews/{id}               Delete review and all associated data
```

### 3.2 WebSocket

```
WS /ws/reviews/{id}/progress

Pushes JSON messages:
{
  "type": "node_start" | "node_complete" | "stage_complete" | "error" | "log",
  "stage": "search" | "screening" | "extraction",
  "node_id": "s2_3",
  "item_id": "PMID:12345",           // for per-paper nodes
  "progress": {"current": 15, "total": 200},
  "message": "Processing PMID 12345...",
  "timestamp": "2025-..."
}
```

### 3.3 Pipeline Integration

The backend does NOT re-implement the pipeline. It wraps the Orchestrator:

```python
# In the FastAPI background task:
from src.orchestrator import SystematicReviewOrchestrator

async def run_pipeline(review_id: str):
    orchestrator = SystematicReviewOrchestrator(
        bench_review_path=f"data/reviews/{review_id}/config.json",
        config_dir="configs/",
        progress_callback=lambda msg: websocket_broadcast(review_id, msg)
    )
    result = orchestrator.run()
    save_to_db(review_id, result)
```

**MUST:** The Orchestrator needs a `progress_callback` parameter added to
its interface. Each node entry/exit and each stage completion should invoke
this callback with a structured progress message. This is the only
modification needed to the core engine to support the web UI.

---

## 4. Key Design Constraints

| Rule | Level |
|------|-------|
| The web UI is a THIN layer — all logic stays in the core engine | MUST |
| Pipeline runs as a background task; UI polls/subscribes for progress | MUST |
| User can close browser and return — pipeline continues, state preserved in checkpoints | MUST |
| Full-text files are uploaded by the user, never fetched automatically | MUST |
| Multiple reviews can be created but only one pipeline runs at a time (queue others) | SHOULD |
| All API responses use the same Pydantic schemas from `src/schemas/` | SHOULD |

---

## 5. File Structure Addition

```
autosr/
├── web/
│   ├── api/
│   │   ├── main.py              # FastAPI app
│   │   ├── routes/
│   │   │   ├── reviews.py       # REST endpoints
│   │   │   └── websocket.py     # WS endpoint
│   │   ├── models.py            # API-specific request/response models
│   │   └── dependencies.py      # DB session, auth, etc.
│   │
│   └── frontend/                # React app (Vite + TypeScript)
│       ├── src/
│       │   ├── pages/
│       │   │   ├── ConfigureReview.tsx
│       │   │   ├── SearchResults.tsx
│       │   │   ├── ScreeningDashboard.tsx
│       │   │   ├── UploadFullTexts.tsx
│       │   │   ├── ExtractionResults.tsx
│       │   │   └── FinalResults.tsx
│       │   ├── components/
│       │   │   ├── Sidebar.tsx
│       │   │   ├── ProgressStream.tsx
│       │   │   ├── PaperTable.tsx
│       │   │   └── FileUploader.tsx
│       │   └── hooks/
│       │       └── useWebSocket.ts
│       └── package.json
```

---

## 6. Implementation Order

1. FastAPI app skeleton + review CRUD endpoints
2. progress_callback integration into Orchestrator
3. WebSocket progress streaming
4. Frontend: Configure page + API integration
5. Frontend: Search results page
6. Frontend: Screening dashboard
7. Frontend: File upload page
8. Frontend: Extraction results page
9. Frontend: Final export page
