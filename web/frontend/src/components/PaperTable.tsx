/**
 * PaperTable — a sortable / filterable table of screening decisions.
 */
import { useState } from 'react'
import type { ScreeningDecisionSummary } from '../types'

interface Props {
  decisions: ScreeningDecisionSummary[]
}

type Filter = 'all' | 'INCLUDED' | 'EXCLUDED' | 'EXCLUDED_BY_METADATA'

function statusBadge(status: string) {
  const map: Record<string, string> = {
    INCLUDED:            'badge-green',
    EXCLUDED:            'badge-red',
    EXCLUDED_BY_METADATA:'badge-gray',
    FAILED:              'badge-yellow',
  }
  return <span className={map[status] ?? 'badge-gray'}>{status}</span>
}

export function PaperTable({ decisions }: Props) {
  const [filter, setFilter] = useState<Filter>('all')
  const [query, setQuery] = useState('')

  const filtered = decisions.filter(d => {
    if (filter !== 'all' && d.final_status !== filter) return false
    if (query && !d.pmid.includes(query)) return false
    return true
  })

  const counts = {
    INCLUDED: decisions.filter(d => d.final_status === 'INCLUDED').length,
    EXCLUDED: decisions.filter(d => d.final_status === 'EXCLUDED').length,
    EXCLUDED_BY_METADATA: decisions.filter(d => d.final_status === 'EXCLUDED_BY_METADATA').length,
    adjudicated: decisions.filter(d => d.conflicts && d.conflicts.length > 0).length,
  }

  return (
    <div className="space-y-4">
      {/* Stats row */}
      <div className="grid grid-cols-4 gap-3">
        {[
          { label: 'Included', value: counts.INCLUDED, cls: 'text-emerald-600' },
          { label: 'Excluded', value: counts.EXCLUDED, cls: 'text-red-600' },
          { label: 'Excl. by Metadata', value: counts.EXCLUDED_BY_METADATA, cls: 'text-gray-600' },
          { label: 'Adjudicated', value: counts.adjudicated, cls: 'text-yellow-600' },
        ].map(s => (
          <div key={s.label} className="card p-3 text-center">
            <div className={`text-2xl font-bold ${s.cls}`}>{s.value}</div>
            <div className="text-xs text-gray-500 mt-0.5">{s.label}</div>
          </div>
        ))}
      </div>

      {/* Filter / search bar */}
      <div className="flex gap-3">
        <input
          type="text"
          className="input w-48"
          placeholder="Search PMID…"
          value={query}
          onChange={e => setQuery(e.target.value)}
        />
        {(['all', 'INCLUDED', 'EXCLUDED', 'EXCLUDED_BY_METADATA'] as Filter[]).map(f => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
              filter === f
                ? 'bg-brand-600 border-brand-600 text-white'
                : 'border-gray-300 text-gray-600 hover:bg-gray-50'
            }`}
          >
            {f === 'all' ? 'All' : f.replace(/_/g, ' ')}
          </button>
        ))}
      </div>

      {/* Table */}
      <div className="overflow-x-auto rounded-xl border border-gray-200">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 border-b border-gray-200">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase">PMID</th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase">Status</th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase">Reviewer A</th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase">Reviewer B</th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase">Conflicts</th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase">Exclusion Reason</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {filtered.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-4 py-8 text-center text-sm text-gray-400">
                  No decisions match the current filter.
                </td>
              </tr>
            ) : (
              filtered.map(d => (
                <tr
                  key={d.pmid}
                  className={`hover:bg-gray-50 transition-colors ${
                    d.conflicts && d.conflicts.length > 0 ? 'bg-yellow-50' : ''
                  }`}
                >
                  <td className="px-4 py-3 font-mono text-xs text-gray-600">{d.pmid}</td>
                  <td className="px-4 py-3">{statusBadge(d.final_status)}</td>
                  <td className="px-4 py-3 text-xs">{d.individual_status_a}</td>
                  <td className="px-4 py-3 text-xs">{d.individual_status_b}</td>
                  <td className="px-4 py-3 text-center">
                    {d.conflicts && d.conflicts.length > 0 ? (
                      <span className="badge-yellow">{d.conflicts.length}</span>
                    ) : (
                      <span className="text-gray-300">—</span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-xs text-gray-500 max-w-xs truncate">
                    {d.exclusion_reasons?.join('; ') || '—'}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
