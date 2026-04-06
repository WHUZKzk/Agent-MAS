/**
 * Step 2 — Search Progress & Results
 *
 * Shows real-time DAG node log while search is running.
 * After completion shows PRISMA numbers, query history, and PICO terms.
 */
import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { api } from '../api'
import { ProgressStream } from '../components/ProgressStream'
import { useWebSocket } from '../hooks/useWebSocket'
import type { Review, SearchResultsSummary } from '../types'

export function SearchResults() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()

  const [review, setReview] = useState<Review | null>(null)
  const [results, setResults] = useState<SearchResultsSummary | null>(null)
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const { messages, status: wsStatus } = useWebSocket(id)

  // Poll review status while running
  useEffect(() => {
    if (!id) return
    const poll = async () => {
      try {
        const r = await api.getReview(id)
        setReview(r)
        if (r.status === 'search_complete' || r.status === 'screening_complete' || r.status === 'extraction_complete') {
          const res = await api.getSearchResults(id)
          setResults(res)
        }
      } catch { /* ignore */ }
    }
    poll()
    const interval = setInterval(poll, 3000)
    return () => clearInterval(interval)
  }, [id])

  const handleStart = async () => {
    if (!id) return
    setStarting(true)
    setError(null)
    try {
      await api.startPipeline(id)
    } catch {
      setError('Failed to start pipeline.')
    } finally {
      setStarting(false)
    }
  }

  const isRunning = review?.status === 'running'
  const isDone = review?.status !== undefined &&
    ['search_complete', 'screening_complete', 'extraction_complete'].includes(review.status)

  return (
    <div className="max-w-4xl mx-auto py-10 px-6 space-y-8">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Search</h2>
          <p className="text-gray-500 mt-1">
            PubMed query construction, MeSH alignment, pearl growing, deduplication.
          </p>
        </div>
        {!isRunning && !isDone && (
          <button className="btn-primary" onClick={handleStart} disabled={starting}>
            {starting ? 'Starting…' : 'Start Search ▶'}
          </button>
        )}
        {isDone && (
          <button className="btn-primary" onClick={() => navigate(`/reviews/${id}/screening`)}>
            Proceed to Screening →
          </button>
        )}
      </div>

      {error && (
        <div className="rounded-lg bg-red-50 border border-red-200 px-4 py-3 text-sm text-red-700">{error}</div>
      )}

      {/* Live log */}
      <ProgressStream messages={messages} status={wsStatus} maxHeight="20rem" />

      {/* Results (after completion) */}
      {results && (
        <div className="space-y-6">
          {/* PRISMA card */}
          <div className="card p-5">
            <h3 className="font-semibold text-gray-800 mb-4">PRISMA Search Numbers</h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              {[
                { label: 'Initial Query', value: results.prisma.initial_query_results },
                { label: 'Augmented Query', value: results.prisma.augmented_query_results },
                { label: 'After Dedup', value: results.prisma.after_deduplication },
                { label: 'Final Candidates', value: results.prisma.final_candidate_count },
              ].map(s => (
                <div key={s.label} className="text-center">
                  <div className="text-3xl font-bold text-brand-600">{s.value.toLocaleString()}</div>
                  <div className="text-xs text-gray-500 mt-1">{s.label}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Query history */}
          {results.query_history.length > 0 && (
            <div className="card p-5">
              <h3 className="font-semibold text-gray-800 mb-3">Query History</h3>
              <div className="space-y-3">
                {results.query_history.map((q, i) => (
                  <div key={i} className="bg-gray-50 rounded-lg p-3">
                    <div className="flex items-center gap-2 mb-1">
                      <span className={`badge ${q.stage === 'initial' ? 'badge-blue' : 'badge-green'}`}>
                        {q.stage}
                      </span>
                      <span className="text-xs text-gray-500">{q.result_count.toLocaleString()} results</span>
                    </div>
                    <code className="text-xs text-gray-700 break-all leading-relaxed">{q.query_string}</code>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
