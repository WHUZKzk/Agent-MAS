/**
 * ProgressStream — renders a scrolling terminal-style log of
 * WebSocket ProgressMessage events from the DAGRunner.
 */
import { useEffect, useRef } from 'react'
import type { ProgressMessage } from '../types'
import type { WsStatus } from '../hooks/useWebSocket'

interface Props {
  messages: ProgressMessage[]
  status: WsStatus
  maxHeight?: string
}

function colorForType(type: string): string {
  switch (type) {
    case 'node_start':     return 'text-sky-400'
    case 'node_complete':  return 'text-emerald-400'
    case 'stage_start':    return 'text-violet-400 font-semibold'
    case 'stage_complete': return 'text-violet-300 font-semibold'
    case 'error':          return 'text-red-400'
    default:               return 'text-gray-300'
  }
}

function typeLabel(type: string): string {
  switch (type) {
    case 'node_start':     return '▶ node'
    case 'node_complete':  return '✓ node'
    case 'stage_start':    return '━ stage'
    case 'stage_complete': return '✔ stage'
    case 'error':          return '✗ error'
    default:               return '·'
  }
}

function statusDot(s: WsStatus): JSX.Element {
  const base = 'inline-block w-2 h-2 rounded-full mr-2'
  switch (s) {
    case 'connected':    return <span className={`${base} bg-emerald-400`} />
    case 'connecting':   return <span className={`${base} bg-yellow-400 animate-pulse`} />
    case 'disconnected': return <span className={`${base} bg-gray-500`} />
    case 'error':        return <span className={`${base} bg-red-500`} />
    default:             return <span className={`${base} bg-gray-600`} />
  }
}

export function ProgressStream({ messages, status, maxHeight = '18rem' }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  return (
    <div className="rounded-xl border border-gray-700 bg-gray-900 overflow-hidden">
      {/* Header bar */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-gray-700">
        <span className="text-xs font-mono font-semibold text-gray-400 uppercase tracking-wider">
          Pipeline Log
        </span>
        <span className="flex items-center text-xs text-gray-400">
          {statusDot(status)}
          {status}
        </span>
      </div>

      {/* Log area */}
      <div
        className="overflow-y-auto p-4 font-mono text-xs leading-relaxed"
        style={{ maxHeight }}
      >
        {messages.length === 0 ? (
          <span className="text-gray-500 italic">Waiting for pipeline events…</span>
        ) : (
          messages.map((msg, i) => (
            <div key={i} className={`flex gap-2 mb-0.5 ${colorForType(msg.type)}`}>
              {/* Timestamp */}
              <span className="shrink-0 text-gray-600 w-20">
                {msg.timestamp ? msg.timestamp.slice(11, 23) : ''}
              </span>
              {/* Type badge */}
              <span className="shrink-0 w-16 text-right opacity-70">
                {typeLabel(msg.type)}
              </span>
              {/* Node id */}
              {msg.node_id && (
                <span className="shrink-0 text-yellow-400 w-14">{msg.node_id}</span>
              )}
              {/* Progress */}
              {msg.progress && (
                <span className="shrink-0 text-gray-500">
                  [{msg.progress.current}/{msg.progress.total}]
                </span>
              )}
              {/* Message */}
              <span className="truncate">{msg.message}</span>
            </div>
          ))
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
