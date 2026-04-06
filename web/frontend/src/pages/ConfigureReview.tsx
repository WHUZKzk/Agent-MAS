/**
 * Step 1 — Configure Review
 *
 * User fills in PICO form and target outcomes/characteristics,
 * or uploads a bench_review.json to auto-fill.
 *
 * On submit → POST /api/reviews → navigate to /reviews/{id}/search.
 */
import { useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import type { PICODefinition } from '../types'

interface TagInputProps {
  label: string
  placeholder: string
  tags: string[]
  onChange: (tags: string[]) => void
}

function TagInput({ label, placeholder, tags, onChange }: TagInputProps) {
  const [input, setInput] = useState('')

  const add = () => {
    const val = input.trim()
    if (val && !tags.includes(val)) onChange([...tags, val])
    setInput('')
  }

  const remove = (t: string) => onChange(tags.filter(x => x !== t))

  return (
    <div>
      <label className="label">{label}</label>
      <div className="flex flex-wrap gap-1.5 mb-2 min-h-[2rem]">
        {tags.map(t => (
          <span key={t} className="badge-blue flex items-center gap-1">
            {t}
            <button onClick={() => remove(t)} className="ml-0.5 text-blue-500 hover:text-blue-700">×</button>
          </span>
        ))}
      </div>
      <div className="flex gap-2">
        <input
          className="input flex-1"
          placeholder={placeholder}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); add() } }}
        />
        <button type="button" className="btn-secondary" onClick={add}>Add</button>
      </div>
    </div>
  )
}

export function ConfigureReview() {
  const navigate = useNavigate()
  const fileRef = useRef<HTMLInputElement>(null)

  const [title, setTitle] = useState('')
  const [abstract, setAbstract] = useState('')
  const [pico, setPico] = useState<PICODefinition>({ P: '', I: '', C: '', O: '' })
  const [chars, setChars] = useState<string[]>(['Mean Age', '% Female', 'Sample Size'])
  const [outcomes, setOutcomes] = useState<string[]>(['Physical Activity', 'Step Count', 'BMI'])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleBenchUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      const review = await api.createReviewFromBench(file, 0)
      navigate(`/reviews/${review.id}/search`)
    } catch {
      setError('Failed to parse bench_review.json. Make sure it is a valid file.')
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!title || !pico.P || !pico.I || !pico.C || !pico.O) {
      setError('Please fill in the Review Title and all four PICO fields.')
      return
    }
    setLoading(true)
    setError(null)
    try {
      const review = await api.createReview({
        title,
        abstract,
        pico,
        target_characteristics: chars,
        target_outcomes: outcomes,
      })
      navigate(`/reviews/${review.id}/search`)
    } catch {
      setError('Failed to create review. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-3xl mx-auto py-10 px-6">
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-gray-900">Configure Systematic Review</h2>
        <p className="text-gray-500 mt-1">
          Define your PICO question and target outcomes, or upload a{' '}
          <code className="text-xs bg-gray-100 px-1 rounded">bench_review.json</code> to auto-fill.
        </p>
      </div>

      {/* Quick-load from bench_review.json */}
      <div className="card p-4 mb-8 flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-800">Auto-fill from bench_review.json</p>
          <p className="text-xs text-gray-500 mt-0.5">Upload the benchmark file to populate all fields instantly.</p>
        </div>
        <button className="btn-secondary" onClick={() => fileRef.current?.click()} disabled={loading}>
          Upload File
        </button>
        <input ref={fileRef} type="file" accept=".json" className="hidden" onChange={handleBenchUpload} />
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Title */}
        <div>
          <label className="label">Review Title *</label>
          <input
            className="input"
            placeholder="e.g. Exergame interventions for physical activity in older adults"
            value={title}
            onChange={e => setTitle(e.target.value)}
            required
          />
        </div>

        {/* Abstract / Question */}
        <div>
          <label className="label">Review Question / Abstract</label>
          <textarea
            className="input h-28 resize-y"
            placeholder="Describe the research question and background…"
            value={abstract}
            onChange={e => setAbstract(e.target.value)}
          />
        </div>

        {/* PICO */}
        <div className="card p-5 space-y-4">
          <h3 className="font-semibold text-gray-800">PICO Definition</h3>
          {(['P', 'I', 'C', 'O'] as const).map(dim => {
            const labels: Record<typeof dim, string> = {
              P: 'Population *',
              I: 'Intervention *',
              C: 'Comparison / Control *',
              O: 'Outcome(s) *',
            }
            return (
              <div key={dim}>
                <label className="label">{labels[dim]}</label>
                <textarea
                  className="input h-20 resize-y"
                  placeholder={`Describe the ${dim} element…`}
                  value={pico[dim]}
                  onChange={e => setPico(p => ({ ...p, [dim]: e.target.value }))}
                  required
                />
              </div>
            )
          })}
        </div>

        {/* Target characteristics + outcomes */}
        <div className="card p-5 space-y-5">
          <h3 className="font-semibold text-gray-800">Extraction Targets</h3>
          <TagInput
            label="Target Characteristics"
            placeholder="e.g. Mean Age"
            tags={chars}
            onChange={setChars}
          />
          <TagInput
            label="Target Outcomes"
            placeholder="e.g. IL-6"
            tags={outcomes}
            onChange={setOutcomes}
          />
        </div>

        {error && (
          <div className="rounded-lg bg-red-50 border border-red-200 px-4 py-3 text-sm text-red-700">
            {error}
          </div>
        )}

        <button type="submit" className="btn-primary w-full py-3 text-base" disabled={loading}>
          {loading ? 'Creating Review…' : 'Start Review →'}
        </button>
      </form>
    </div>
  )
}
