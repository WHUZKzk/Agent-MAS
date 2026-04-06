/**
 * API client — wraps all backend REST calls.
 * All requests go through Vite's dev proxy (/api → localhost:8000).
 */
import axios from 'axios'
import type {
  ExtractionResultsSummary,
  Review,
  ReviewCreateRequest,
  ReviewListResponse,
  ScreeningResultsSummary,
  SearchResultsSummary,
  UploadResponse,
} from './types'

const http = axios.create({ baseURL: '/api' })

export const api = {
  // ── Reviews ──────────────────────────────────────────────────────────────

  createReview: (data: ReviewCreateRequest): Promise<Review> =>
    http.post<Review>('/reviews', data).then(r => r.data),

  createReviewFromBench: (file: File, index = 0): Promise<Review> => {
    const form = new FormData()
    form.append('file', file)
    return http
      .post<Review>(`/reviews/upload-bench?index=${index}`, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      .then(r => r.data)
  },

  listReviews: (skip = 0, limit = 50): Promise<ReviewListResponse> =>
    http.get<ReviewListResponse>('/reviews', { params: { skip, limit } }).then(r => r.data),

  getReview: (id: string): Promise<Review> =>
    http.get<Review>(`/reviews/${id}`).then(r => r.data),

  startPipeline: (id: string): Promise<Review> =>
    http.post<Review>(`/reviews/${id}/start`).then(r => r.data),

  deleteReview: (id: string): Promise<void> =>
    http.delete(`/reviews/${id}`).then(() => undefined),

  // ── Stage results ─────────────────────────────────────────────────────────

  getSearchResults: (id: string): Promise<SearchResultsSummary> =>
    http.get<SearchResultsSummary>(`/reviews/${id}/search`).then(r => r.data),

  getScreeningResults: (id: string): Promise<ScreeningResultsSummary> =>
    http.get<ScreeningResultsSummary>(`/reviews/${id}/screening`).then(r => r.data),

  getExtractionResults: (id: string): Promise<ExtractionResultsSummary> =>
    http.get<ExtractionResultsSummary>(`/reviews/${id}/extraction`).then(r => r.data),

  // ── File upload ───────────────────────────────────────────────────────────

  uploadFullText: (reviewId: string, pmid: string, file: File): Promise<UploadResponse> => {
    const form = new FormData()
    form.append('file', file)
    return http
      .post<UploadResponse>(`/reviews/${reviewId}/uploads?pmid=${pmid}`, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      .then(r => r.data)
  },

  // ── Export ────────────────────────────────────────────────────────────────

  getExportUrl: (id: string): string => `/api/reviews/${id}/export`,
}
