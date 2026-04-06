/**
 * Step 4 — Upload Full Texts
 *
 * Shows included PMIDs from screening, allows uploading .xml or .pdf per paper.
 * "Start Extraction" is enabled when all files are uploaded (or user overrides).
 */
import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { api } from '../api'
import { FileUploader } from '../components/FileUploader'
import type { ScreeningResultsSummary } from '../types'

interface UploadState {
  [pmid: string]: string | null  // null = not uploaded, string = filename
}

export function UploadFullTexts() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()

  const [screening, setScreening] = useState<ScreeningResultsSummary | null>(null)
  const [uploads, setUploads] = useState<UploadState>({})
  const [loading, setLoading] = useState(true)
  const [starting, setStarting] = useState(false)
  const [skipMissing, setSkipMissing] = useState(false)

  useEffect(() => {
    if (!id) return
    api.getScreeningResults(id)
      .then(res => {
        setScreening(res)
        const initial: UploadState = {}
        res.included_pmids.forEach(p => { initial[p] = null })
        setUploads(initial)
      })
      .catch(() => {/* no results yet */})
      .finally(() => setLoading(false))
  }, [id])

  const handleUploaded = (pmid: string, filename: string) => {
    setUploads(prev => ({ ...prev, [pmid]: filename }))
  }

  const allUploaded = Object.values(uploads).every(v => v !== null)
  const uploadedCount = Object.values(uploads).filter(v => v !== null).length
  const total = Object.keys(uploads).length

  const handleStartExtraction = async () => {
    if (!id) return
    setStarting(true)
    try {
      await api.startPipeline(id)
      navigate(`/reviews/${id}/extraction`)
    } catch {
      alert('Failed to start extraction.')
    } finally {
      setStarting(false)
    }
  }

  if (loading) {
    return <div className="p-10 text-gray-400">Loading screening results…</div>
  }

  if (!screening) {
    return (
      <div className="max-w-2xl mx-auto py-10 px-6">
        <div className="card p-6 text-center text-gray-500">
          Screening results not yet available. Complete screening before uploading full texts.
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-3xl mx-auto py-10 px-6 space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-gray-900">Upload Full Texts</h2>
        <p className="text-gray-500 mt-1">
          Upload full-text files for each included study. Accepted formats: .xml, .pdf.
        </p>
      </div>

      {/* Progress summary */}
      <div className="card p-4 flex items-center gap-4">
        <div className="flex-1">
          <div className="flex justify-between text-sm mb-1">
            <span className="text-gray-600">{uploadedCount} / {total} files uploaded</span>
            <span className="text-gray-400">{Math.round((uploadedCount / Math.max(total, 1)) * 100)}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-emerald-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${(uploadedCount / Math.max(total, 1)) * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* Per-PMID upload rows */}
      <div className="card divide-y divide-gray-100">
        {screening.included_pmids.length === 0 ? (
          <p className="px-4 py-6 text-sm text-center text-gray-400">
            No papers were included in screening.
          </p>
        ) : (
          screening.included_pmids.map(pmid => (
            <div key={pmid} className="flex items-center gap-4 px-4 py-3">
              <code className="text-xs font-mono text-gray-600 w-28 shrink-0">{pmid}</code>
              <div className="flex-1">
                {uploads[pmid] ? (
                  <span className="badge-green">✓ {uploads[pmid]}</span>
                ) : (
                  <FileUploader
                    reviewId={id!}
                    pmid={pmid}
                    onUploaded={handleUploaded}
                  />
                )}
              </div>
            </div>
          ))
        )}
      </div>

      {/* Skip option + start button */}
      {!allUploaded && uploadedCount > 0 && (
        <label className="flex items-center gap-2 text-sm text-gray-600 cursor-pointer">
          <input
            type="checkbox"
            checked={skipMissing}
            onChange={e => setSkipMissing(e.target.checked)}
            className="rounded border-gray-300 text-brand-600 focus:ring-brand-500"
          />
          Skip papers without uploaded files and start extraction anyway
        </label>
      )}

      <button
        className="btn-primary w-full py-3 text-base"
        disabled={(!allUploaded && !skipMissing) || starting || total === 0}
        onClick={handleStartExtraction}
      >
        {starting ? 'Starting Extraction…' : 'Start Extraction →'}
      </button>
    </div>
  )
}
