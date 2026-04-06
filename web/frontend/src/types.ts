// ─────────────────────────────────────────────────────────────────────────────
// Shared type definitions (mirroring Python Pydantic models)
// ─────────────────────────────────────────────────────────────────────────────

export type ReviewStatus =
  | 'created'
  | 'running'
  | 'search_complete'
  | 'screening_complete'
  | 'extraction_complete'
  | 'failed'

export type ReviewStage = 'init' | 'search' | 'screening' | 'extraction'

export interface PICODefinition {
  P: string
  I: string
  C: string
  O: string
}

export interface Review {
  id: string
  title: string
  status: ReviewStatus
  current_stage: ReviewStage
  error_message?: string | null
  created_at: string
  updated_at: string
}

export interface ReviewListResponse {
  reviews: Review[]
  total: number
}

export interface ReviewCreateRequest {
  title: string
  abstract: string
  pico: PICODefinition
  target_characteristics: string[]
  target_outcomes: string[]
}

// ─────────────────────────────────────────────────────────────────────────────
// WebSocket / Progress
// ─────────────────────────────────────────────────────────────────────────────

export type ProgressEventType =
  | 'node_start'
  | 'node_complete'
  | 'stage_start'
  | 'stage_complete'
  | 'error'
  | 'log'

export interface ProgressMessage {
  type: ProgressEventType
  stage?: string | null
  node_id?: string | null
  item_id?: string | null
  progress?: { current: number; total: number } | null
  message: string
  timestamp: string
}

// ─────────────────────────────────────────────────────────────────────────────
// API results
// ─────────────────────────────────────────────────────────────────────────────

export interface PRISMASearchSummary {
  initial_query_results: number
  augmented_query_results: number
  after_deduplication: number
  final_candidate_count: number
}

export interface SearchResultsSummary {
  total_candidates: number
  prisma: PRISMASearchSummary
  query_history: Array<{ query_string: string; stage: string; result_count: number }>
  pico_terms?: Record<string, unknown> | null
}

export interface PRISMAScreeningSummary {
  total_screened: number
  excluded_by_metadata: number
  excluded_by_dual_review: number
  sent_to_adjudication: number
  included_after_screening: number
}

export interface ScreeningDecisionSummary {
  pmid: string
  final_status: string
  individual_status_a: string
  individual_status_b: string
  exclusion_reasons: string[]
  conflicts: unknown[]
}

export interface ScreeningResultsSummary {
  cohens_kappa: number
  prisma: PRISMAScreeningSummary
  included_pmids: string[]
  decisions: ScreeningDecisionSummary[]
}

export interface ExtractionResultsSummary {
  papers_processed: number
  outcomes_extracted: number
  effect_sizes_computed: number
}

export interface UploadResponse {
  pmid: string
  filename: string
  size_bytes: number
  message: string
}
