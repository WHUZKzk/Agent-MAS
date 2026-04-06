/**
 * App — root component with react-router-dom routing and persistent Sidebar.
 *
 * Layout:
 *   ┌────────────────┬────────────────────────────────┐
 *   │  Sidebar       │  <Outlet> (page content)       │
 *   └────────────────┴────────────────────────────────┘
 *
 * The Sidebar receives the current Review object so it can show step status.
 * The review is loaded from the URL param :id on every page inside ReviewLayout.
 */
import { useEffect, useState } from 'react'
import { Navigate, Route, Routes, useParams } from 'react-router-dom'
import { api } from './api'
import { Sidebar } from './components/Sidebar'
import { ConfigureReview } from './pages/ConfigureReview'
import { ExtractionResults } from './pages/ExtractionResults'
import { FinalResults } from './pages/FinalResults'
import { ScreeningDashboard } from './pages/ScreeningDashboard'
import { SearchResults } from './pages/SearchResults'
import { UploadFullTexts } from './pages/UploadFullTexts'
import type { Review } from './types'

// ─────────────────────────────────────────────────────────────────────────────
// Review layout — loads review by :id and passes it to Sidebar
// ─────────────────────────────────────────────────────────────────────────────
function ReviewLayout({ children }: { children: React.ReactNode }) {
  const { id } = useParams<{ id: string }>()
  const [review, setReview] = useState<Review | null>(null)

  useEffect(() => {
    if (!id) return
    api.getReview(id).then(setReview).catch(() => null)
    const interval = setInterval(() => {
      api.getReview(id).then(setReview).catch(() => null)
    }, 5000)
    return () => clearInterval(interval)
  }, [id])

  return (
    <div className="flex min-h-screen">
      <Sidebar review={review} />
      <main className="flex-1 overflow-auto">{children}</main>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// New-review layout (no :id yet)
// ─────────────────────────────────────────────────────────────────────────────
function NewReviewLayout() {
  return (
    <div className="flex min-h-screen">
      <Sidebar review={null} />
      <main className="flex-1 overflow-auto">
        <ConfigureReview />
      </main>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Root App
// ─────────────────────────────────────────────────────────────────────────────
export default function App() {
  return (
    <Routes>
      {/* Default → configure page */}
      <Route path="/" element={<Navigate to="/reviews/new" replace />} />
      <Route path="/reviews/new" element={<NewReviewLayout />} />

      {/* Review-scoped pages */}
      <Route
        path="/reviews/:id/search"
        element={
          <ReviewLayout>
            <SearchResults />
          </ReviewLayout>
        }
      />
      <Route
        path="/reviews/:id/screening"
        element={
          <ReviewLayout>
            <ScreeningDashboard />
          </ReviewLayout>
        }
      />
      <Route
        path="/reviews/:id/upload"
        element={
          <ReviewLayout>
            <UploadFullTexts />
          </ReviewLayout>
        }
      />
      <Route
        path="/reviews/:id/extraction"
        element={
          <ReviewLayout>
            <ExtractionResults />
          </ReviewLayout>
        }
      />
      <Route
        path="/reviews/:id/results"
        element={
          <ReviewLayout>
            <FinalResults />
          </ReviewLayout>
        }
      />

      {/* Catch-all */}
      <Route path="*" element={<Navigate to="/reviews/new" replace />} />
    </Routes>
  )
}
