/**
 * Sidebar — persistent 6-step wizard navigation.
 */
import { Link, useLocation } from 'react-router-dom'
import type { Review } from '../types'

interface Step {
  index: number
  label: string
  path: (id: string) => string
  requiredStatus: Review['status'][]
}

const STEPS: Step[] = [
  {
    index: 1,
    label: 'Configure Review',
    path: () => '/reviews/new',
    requiredStatus: ['created', 'running', 'search_complete', 'screening_complete', 'extraction_complete', 'failed'],
  },
  {
    index: 2,
    label: 'Search',
    path: (id) => `/reviews/${id}/search`,
    requiredStatus: ['running', 'search_complete', 'screening_complete', 'extraction_complete'],
  },
  {
    index: 3,
    label: 'Screening',
    path: (id) => `/reviews/${id}/screening`,
    requiredStatus: ['search_complete', 'screening_complete', 'extraction_complete'],
  },
  {
    index: 4,
    label: 'Upload Full Texts',
    path: (id) => `/reviews/${id}/upload`,
    requiredStatus: ['search_complete', 'screening_complete', 'extraction_complete'],
  },
  {
    index: 5,
    label: 'Extraction',
    path: (id) => `/reviews/${id}/extraction`,
    requiredStatus: ['screening_complete', 'extraction_complete'],
  },
  {
    index: 6,
    label: 'Results & Export',
    path: (id) => `/reviews/${id}/results`,
    requiredStatus: ['extraction_complete'],
  },
]

interface Props {
  review?: Review | null
}

function isStepEnabled(step: Step, review: Review | null | undefined): boolean {
  if (step.index === 1) return true
  if (!review) return false
  return step.requiredStatus.includes(review.status)
}

function isStepActive(step: Step, pathname: string, review: Review | null | undefined): boolean {
  if (!review) return pathname === '/reviews/new' && step.index === 1
  return pathname.includes(step.path(review.id).split('/').slice(-1)[0])
}

function stepStatus(step: Step, review: Review | null | undefined): 'complete' | 'active' | 'locked' {
  if (!review) return step.index === 1 ? 'active' : 'locked'
  const statusOrder = ['created', 'running', 'search_complete', 'screening_complete', 'extraction_complete']
  const reviewIdx = statusOrder.indexOf(review.status)

  const completedByStage: Record<number, number> = { 2: 2, 3: 3, 4: 3, 5: 4, 6: 4 }
  if (step.index === 1) return reviewIdx >= 0 ? 'complete' : 'active'
  const neededIdx = completedByStage[step.index] ?? 99
  if (reviewIdx >= neededIdx) return 'complete'
  if (isStepEnabled(step, review)) return 'active'
  return 'locked'
}

export function Sidebar({ review }: Props) {
  const { pathname } = useLocation()

  return (
    <aside className="w-64 shrink-0 flex flex-col bg-gray-900 text-white min-h-screen">
      {/* Logo / title */}
      <div className="px-6 py-5 border-b border-gray-700">
        <h1 className="text-lg font-bold tracking-tight text-white">AutoSR</h1>
        <p className="text-xs text-gray-400 mt-0.5">Systematic Review Engine</p>
      </div>

      {/* Steps */}
      <nav className="flex-1 px-4 py-6 space-y-1">
        {STEPS.map((step) => {
          const enabled = isStepEnabled(step, review)
          const st = stepStatus(step, review)
          const active = review
            ? pathname === step.path(review.id)
            : pathname === '/reviews/new' && step.index === 1

          const bg = active
            ? 'bg-brand-600 text-white'
            : enabled
            ? 'text-gray-300 hover:bg-gray-800 hover:text-white'
            : 'text-gray-600 cursor-not-allowed'

          const to = review && enabled ? step.path(review.id) : '#'

          return (
            <Link
              key={step.index}
              to={to}
              onClick={e => !enabled && e.preventDefault()}
              className={`flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors ${bg}`}
            >
              {/* Step circle */}
              <span
                className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-bold
                  ${st === 'complete' ? 'bg-emerald-500 text-white' : ''}
                  ${st === 'active' && !active ? 'bg-gray-700 text-gray-300' : ''}
                  ${active ? 'bg-white text-brand-700' : ''}
                  ${st === 'locked' ? 'bg-gray-800 text-gray-600' : ''}
                `}
              >
                {st === 'complete' && !active ? '✓' : step.index}
              </span>
              {step.label}
            </Link>
          )
        })}
      </nav>

      {/* Review status footer */}
      {review && (
        <div className="px-4 py-4 border-t border-gray-700 text-xs text-gray-400 space-y-1">
          <div className="font-medium text-gray-300 truncate" title={review.title}>
            {review.title}
          </div>
          <div className="flex items-center gap-1.5">
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                review.status === 'running'
                  ? 'bg-yellow-400 animate-pulse'
                  : review.status === 'extraction_complete'
                  ? 'bg-emerald-400'
                  : review.status === 'failed'
                  ? 'bg-red-400'
                  : 'bg-gray-500'
              }`}
            />
            {review.status.replace(/_/g, ' ')}
          </div>
        </div>
      )}
    </aside>
  )
}
