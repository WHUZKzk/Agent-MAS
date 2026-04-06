/**
 * useWebSocket — connects to /ws/reviews/{reviewId}/progress and
 * accumulates ProgressMessage objects in state.
 *
 * Auto-reconnects on disconnect (up to maxRetries times).
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import type { ProgressMessage } from '../types'

export type WsStatus = 'idle' | 'connecting' | 'connected' | 'disconnected' | 'error'

interface UseWebSocketReturn {
  messages: ProgressMessage[]
  status: WsStatus
  clearMessages: () => void
}

const MAX_RETRIES = 5
const RETRY_DELAY_MS = 2000

export function useWebSocket(reviewId: string | null | undefined): UseWebSocketReturn {
  const [messages, setMessages] = useState<ProgressMessage[]>([])
  const [status, setStatus] = useState<WsStatus>('idle')
  const wsRef = useRef<WebSocket | null>(null)
  const retriesRef = useRef(0)
  const unmountedRef = useRef(false)

  const clearMessages = useCallback(() => setMessages([]), [])

  const connect = useCallback(() => {
    if (!reviewId || unmountedRef.current) return
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return

    setStatus('connecting')

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    // In dev, Vite proxies /ws → backend. In prod, same origin.
    const url = `${protocol}//${window.location.host}/ws/reviews/${reviewId}/progress`
    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => {
      if (unmountedRef.current) { ws.close(); return }
      setStatus('connected')
      retriesRef.current = 0
    }

    ws.onmessage = (event) => {
      if (unmountedRef.current) return
      try {
        const msg: ProgressMessage = JSON.parse(event.data as string)
        // Filter out bare heartbeats to avoid log noise
        if (msg.type === 'log' && msg.message === 'heartbeat') return
        setMessages(prev => [...prev, msg])
      } catch {
        // Ignore unparseable frames
      }
    }

    ws.onclose = () => {
      if (unmountedRef.current) return
      setStatus('disconnected')
      wsRef.current = null
      // Auto-reconnect
      if (retriesRef.current < MAX_RETRIES) {
        retriesRef.current += 1
        setTimeout(connect, RETRY_DELAY_MS)
      }
    }

    ws.onerror = () => {
      setStatus('error')
      ws.close()
    }
  }, [reviewId])

  useEffect(() => {
    unmountedRef.current = false
    retriesRef.current = 0
    connect()
    return () => {
      unmountedRef.current = true
      wsRef.current?.close()
      wsRef.current = null
    }
  }, [connect])

  return { messages, status, clearMessages }
}
