/**
 * Step 3 — Screening Dashboard
 *
 * Real-time progress per paper, live Cohen's κ, final table of decisions.
 */
import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { api } from '../api'
import { PaperTable } from '../components/PaperTable'
import { ProgressStream } from '../components/ProgressStream'
import { useWebSocket } from '../hooks/useWebSocket'
import type { Review, ScreeningResultsSummary } from '../types'

export function ScreeningDashboard() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()

  const [review, setReview] = useState<Review | null>(null)
  const [results, setResults] = useState<ScreeningResultsSummary | null>(null)

  const { messages, status: wsStatus } = useWebSocket(id)

  // Derive live progress from WS messages
  const liveProgress = useMemo(() => {
    const progressMsgs = messages.filter(m => m.progress)
    const last = progressMsgs[progressMsgs.length - 1]
    return last?.progress ?? null
  }, [messages])

  useEffect(() => {
    if (!id) return
    const poll = async () => {
      try {
        const r = await api.getReview(id)
        setReview(r)
        if (['screening_complete', 'extraction_complete'].includes(r.status)) {
          const res = await api.getScreeningResults(id)
          setResults(res)
        }
      } catch { /* ignore */ }
    }
    poll()
    const interval = setInterval(poll, 3000)
    return () => clearInterval(interval)
  }, [id])

  const isDone = review?.status === 'screening_complete' || review?.status === 'extraction_complete'

  return (
    <div className="max-w-5xl mx-auto py-10 px-6 space-y-8">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Screening Dashboard</h2>
          <p className="text-gray-500 mt-1">
            Dual-blind heterogeneous review with epistemic adjudication.
          </p>
        </div>
        {isDone && (
          <button className="btn-primary" onClick={() => navigate(`/reviews/${id}/upload`)}>
            Proceed to Upload →
          </button>
        )}
      </div>

      {/* Live progress bar */}
      {liveProgress && (
        <div>
          <div className="flex justify-between text-xs text-gray-500 mb-1">
            <span>Screening paper {liveProgress.current} of {liveProgress.total}</span>
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

      {/* Real-time log */}
      <ProgressStream messages={messages} status={wsStatus} maxHeight="16rem" />

      {/* Results */}
      {results && (
        <div className="space-y-6">
          {/* Summary cards */}
          <div className="grid grid-cols-3 gap-4">
            <div className="card p-4 text-center">
              <div className="text-3xl font-bold text-emerald-600">
                {results.included_pmids.length}
              </div>
              <div className="text-xs text-gray-500 mt-1">Papers Included</div>
            </div>
            <div className="card p-4 text-center">
              <div className="text-3xl font-bold text-gray-600">
                {results.prisma.total_screened}
              </div>
              <div className="text-xs text-gray-500 mt-1">Total Screened</div>
            </div>
            <div className="card p-4 text-center">
              <div className="text-3xl font-bold text-violet-600">
                κ {results.cohens_kappa.toFixed(3)}
              </div>
              <div className="text-xs text-gray-500 mt-1">Cohen's Kappa</div>
            </div>
          </div>

          {/* PRISMA detail */}
          <div className="card p-5">
            <h3 className="font-semibold text-gray-800 mb-3">PRISMA Screening Numbers</h3>
            <dl className="grid grid-cols-2 gap-3 text-sm">
              {[
                ['Total screened', results.prisma.total_screened],
                ['Excluded by metadata', results.prisma.excluded_by_metadata],
                ['Excluded by dual review', results.prisma.excluded_by_dual_review],
                ['Sent to adjudication', results.prisma.sent_to_adjudication],
                ['Included after screening', results.prisma.included_after_screening],
              ].map(([label, value]) => (
                <div key={String(label)} className="flex justify-between border-b border-gray-100 pb-2">
                  <dt className="text-gray-600">{label}</dt>
                  <dd className="font-semibold">{String(value)}</dd>
                </div>
              ))}
            </dl>
          </div>

          {/* Decision table */}
          <div className="card p-5">
            <h3 className="font-semibold text-gray-800 mb-4">Paper Decisions</h3>
            <PaperTable decisions={results.decisions} />
          </div>
        </div>
      )}
    </div>
  )
}
