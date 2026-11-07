// The library search box. Typing updates the URL after a pause, Enter updates it at once.
// A ref remembers what this box pushed so a value coming from outside (Back, a link) replaces the text.
import { useEffect, useRef, useState, type FormEvent } from 'react'
import { Search, X } from 'lucide-react'

export const MAX_QUERY = 200
const DEBOUNCE_MS = 400

export default function SearchBox({
  value,
  onCommit,
  placeholder = 'Search photos, videos and documents',
}: {
  value: string
  onCommit: (q: string) => void
  placeholder?: string
}) {
  const [text, setText] = useState(value)
  const pushed = useRef(value)
  const commitRef = useRef(onCommit)
  commitRef.current = onCommit

  useEffect(() => {
    if (value !== pushed.current) {
      pushed.current = value
      setText(value)
    }
  }, [value])

  const commit = (raw: string) => {
    const next = raw.trim()
    if (next === pushed.current) return
    pushed.current = next
    commitRef.current(next)
  }

  useEffect(() => {
    const timer = window.setTimeout(() => commit(text), DEBOUNCE_MS)
    return () => window.clearTimeout(timer)
    // The timer restarts on typing only: commit reads refs at fire time.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text])

  const submit = (e: FormEvent) => {
    e.preventDefault()
    commit(text)
  }

  return (
    <form role="search" onSubmit={submit} className="relative w-full">
      <label htmlFor="library-search" className="sr-only">
        {placeholder}
      </label>
      <Search className="pointer-events-none absolute left-4 top-1/2 size-5 -translate-y-1/2 text-ink-faint" aria-hidden />
      <input
        id="library-search"
        type="search"
        value={text}
        onChange={(e) => setText(e.target.value)}
        maxLength={MAX_QUERY}
        placeholder={placeholder}
        autoComplete="off"
        enterKeyHint="search"
        className="h-12 w-full rounded-full border border-line bg-surface pl-12 pr-12 text-base text-ink placeholder:text-ink-faint hover:border-ink-faint focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/20"
      />
      {text && (
        <button
          type="button"
          onClick={() => {
            setText('')
            commit('')
          }}
          className="absolute right-1 top-1/2 flex size-10 -translate-y-1/2 items-center justify-center rounded-full text-ink-soft hover:bg-surface-2"
          aria-label="Clear search"
        >
          <X className="size-4" aria-hidden />
        </button>
      )}
    </form>
  )
}
