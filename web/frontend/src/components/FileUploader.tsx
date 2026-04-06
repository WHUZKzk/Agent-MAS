/**
 * FileUploader — drag-and-drop or click-to-upload for a single PMID's full text.
 */
import { useRef, useState } from 'react'
import { api } from '../api'

interface Props {
  reviewId: string
  pmid: string
  onUploaded: (pmid: string, filename: string) => void
}

export function FileUploader({ reviewId, pmid, onUploaded }: Props) {
  const [dragging, setDragging] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFile = async (file: File) => {
    const ext = file.name.split('.').pop()?.toLowerCase()
    if (ext !== 'xml' && ext !== 'pdf') {
      setError('Only .xml or .pdf files are accepted.')
      return
    }
    setLoading(true)
    setError(null)
    try {
      const res = await api.uploadFullText(reviewId, pmid, file)
      onUploaded(pmid, res.filename)
    } catch (e: unknown) {
      setError('Upload failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
  }

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  return (
    <div>
      <div
        className={`border-2 border-dashed rounded-lg px-4 py-3 text-center cursor-pointer transition-colors ${
          dragging ? 'border-brand-500 bg-brand-50' : 'border-gray-300 hover:border-gray-400'
        }`}
        onClick={() => inputRef.current?.click()}
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        {loading ? (
          <span className="text-xs text-gray-500">Uploading…</span>
        ) : (
          <span className="text-xs text-gray-500">
            Drop .xml / .pdf or <span className="text-brand-600 underline">browse</span>
          </span>
        )}
      </div>
      <input
        ref={inputRef}
        type="file"
        accept=".xml,.pdf"
        className="hidden"
        onChange={onInputChange}
      />
      {error && <p className="mt-1 text-xs text-red-500">{error}</p>}
    </div>
  )
}
