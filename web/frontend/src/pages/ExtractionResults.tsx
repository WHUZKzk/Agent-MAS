/**
 * Step 5 — Extraction Results
 *
 * Real-time extraction log, then a summary of processed papers + download link.
 */
import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { api } from '../api'
import { ProgressStream } from '../components/ProgressStream'
import { useWebSocket } from '../hooks/useWebSocket'
import type { ExtractionResultsSummary, Review } from '../types'

export function ExtractionResults() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()

  const [review, setReview] = useState<Review | null>(null)
  const [results, setResults] = useState<ExtractionResultsSummary | null>(null)

  const { messages, status: wsStatus } = useWebSocket(id)

  const liveProgress = useMemo(() => {
    const pm = messages.filter(m => m.progress && m.stage === 'extraction')
    return pm[pm.length - 1]?.progress ?? null
  }, [messages])

  useEffect(() => {
    if (!id) return
    const poll = async () => {
      try {
        const r = await api.getReview(id)
        setReview(r)
        if (r.status === 'extraction_complete') {
          const res = await api.getExtractionResults(id)
          setResults(res)
        }
      } catch { /* ignore */ }
    }
    poll()
    const interval = setInterval(poll, 3000)
    return () => clearInterval(interval)
  }, [id])

  const isDone = review?.status === 'extraction_complete'

  return (
    <div className="max-w-4xl mx-auto py-10 px-6 space-y-8">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Extraction</h2>
          <p className="text-gray-500 mt-1">
            Document cartography, characteristic & outcome extraction, effect sizes.
          </p>
        </div>
        {isDone && (
          <button className="btn-primary" onClick={() => navigate(`/reviews/${id}/results`)}>
            View Final Results →
          </button>
        )}
      </div>

      {/* Live progress */}
      {liveProgress && (
        <div>
          <div className="flex justify-between text-xs text-gray-500 mb-1">
            <span>Extracting paper {liveProgress.current} of {liveProgress.total}</span>
            <span>{Math.round((liveProgress.current / liveProgress.total) * 100)}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-brand-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${(liveProgress.current / liveProgress.total) * 100}%` }}
            />
          </div>
        </div>
      )}

      {/* Log */}
      <ProgressStream messages={messages} status={wsStatus} maxHeight="20rem" />

      {/* Summary */}
      {results && (
        <div className="card p-5">
          <h3 className="font-semibold text-gray-800 mb-4">Extraction Summary</h3>
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <div className="text-3xl font-bold text-brand-600">{results.papers_processed}</div>
              <div className="text-xs text-gray-500 mt-1">Papers Processed</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-emerald-600">{results.outcomes_extracted}</div>
              <div className="text-xs text-gray-500 mt-1">Outcomes Extracted</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-violet-600">{results.effect_sizes_computed}</div>
              <div className="text-xs text-gray-500 mt-1">Effect Sizes Computed</div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
