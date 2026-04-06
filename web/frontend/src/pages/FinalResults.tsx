/**
 * Step 6 — Final Results & Export
 *
 * Summary stats and download buttons for all output CSVs.
 */
import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { api } from '../api'
import type { ExtractionResultsSummary, ScreeningResultsSummary } from '../types'

export function FinalResults() {
  const { id } = useParams<{ id: string }>()

  const [screening, setScreening] = useState<ScreeningResultsSummary | null>(null)
  const [extraction, setExtraction] = useState<ExtractionResultsSummary | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!id) return
    Promise.all([
      api.getScreeningResults(id).catch(() => null),
      api.getExtractionResults(id).catch(() => null),
    ]).then(([sc, ex]) => {
      setScreening(sc)
      setExtraction(ex)
    }).finally(() => setLoading(false))
  }, [id])

  if (loading) return <div className="p-10 text-gray-400">Loading results…</div>

  const totalIncluded = screening?.included_pmids.length ?? '—'
  const completeness = extraction
    ? Math.round((extraction.outcomes_extracted / Math.max(extraction.papers_processed * 4, 1)) * 100)
    : '—'

  return (
    <div className="max-w-3xl mx-auto py-10 px-6 space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-gray-900">Results & Export</h2>
        <p className="text-gray-500 mt-1">
          Pipeline complete. Download all output CSV files below.
        </p>
      </div>

      {/* Summary stats */}
      <div className="grid grid-cols-3 gap-4">
        {[
          { label: 'Studies Included', value: totalIncluded, cls: 'text-emerald-600' },
          { label: 'Outcomes Extracted', value: extraction?.outcomes_extracted ?? '—', cls: 'text-brand-600' },
          { label: 'Data Completeness', value: `${completeness}%`, cls: 'text-violet-600' },
        ].map(s => (
          <div key={s.label} className="card p-4 text-center">
            <div className={`text-3xl font-bold ${s.cls}`}>{s.value}</div>
            <div className="text-xs text-gray-500 mt-1">{s.label}</div>
          </div>
        ))}
      </div>

      {/* Cohen's κ */}
      {screening && (
        <div className="card p-5">
          <h3 className="font-semibold text-gray-800 mb-3">Screening Quality</h3>
          <div className="flex items-center gap-6 text-sm">
            <div>
              <span className="text-gray-500">Cohen's κ: </span>
              <span className="font-bold text-violet-600">{screening.cohens_kappa.toFixed(3)}</span>
            </div>
            <div>
              <span className="text-gray-500">Total screened: </span>
              <span className="font-bold">{screening.prisma.total_screened}</span>
            </div>
            <div>
              <span className="text-gray-500">Adjudicated: </span>
              <span className="font-bold">{screening.prisma.sent_to_adjudication}</span>
            </div>
          </div>
        </div>
      )}

      {/* Downloads */}
      <div className="card p-5">
        <h3 className="font-semibold text-gray-800 mb-4">Download Outputs</h3>
        <div className="space-y-3">
          <DownloadRow
            label="All Output Files (ZIP)"
            description="study_characteristics.csv, study_results.csv, raw data"
            href={api.getExportUrl(id!)}
            primary
          />
        </div>
        <p className="text-xs text-gray-400 mt-4">
          Individual CSVs are in <code>data/outputs/{id}/</code> on the server.
        </p>
      </div>

      {/* PRISMA snapshot */}
      {screening && (
        <div className="card p-5">
          <h3 className="font-semibold text-gray-800 mb-3">PRISMA Flow Snapshot</h3>
          <div className="space-y-2 text-sm">
            {[
              ['Records identified (initial query)', screening.prisma.total_screened],
              ['Excluded by publication type / metadata', screening.prisma.excluded_by_metadata],
              ['Excluded by dual-blind review', screening.prisma.excluded_by_dual_review],
              ['Conflicts adjudicated', screening.prisma.sent_to_adjudication],
              ['Studies included in synthesis', screening.prisma.included_after_screening],
            ].map(([label, val]) => (
              <div key={String(label)} className="flex justify-between border-b border-gray-100 pb-2">
                <span className="text-gray-600">{label}</span>
                <span className="font-semibold">{String(val)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function DownloadRow({
  label,
  description,
  href,
  primary = false,
}: {
  label: string
  description: string
  href: string
  primary?: boolean
}) {
  return (
    <div className="flex items-center justify-between gap-4 rounded-lg bg-gray-50 px-4 py-3">
      <div>
        <div className="text-sm font-medium text-gray-800">{label}</div>
        <div className="text-xs text-gray-500">{description}</div>
      </div>
      <a
        href={href}
        download
        className={primary ? 'btn-primary' : 'btn-secondary'}
      >
        ↓ Download
      </a>
    </div>
  )
}
