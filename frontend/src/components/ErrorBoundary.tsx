// Catches render errors below it, so a broken page or viewer shows what went wrong and a way out
// instead of a blank screen (React unmounts the whole tree on an uncaught render error, and a lazy
// chunk that fails to load throws the same way). `resetKey` clears the error when it changes (the
// route, the viewed item); `fallback` draws the error, by default ErrorPanel.
import { Component, type ErrorInfo, type ReactNode } from 'react'
import { Button } from './ui'

export type ErrorFallback = (error: Error, reset: () => void) => ReactNode

interface Props {
  children: ReactNode
  fallback?: ErrorFallback
  resetKey?: unknown
}

interface State {
  error: Error | null
}

export function errorMessage(error: unknown): string {
  if (error instanceof Error) return error.message || error.name
  if (typeof error === 'string' && error) return error
  return 'Unknown error'
}

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: unknown): State {
    return { error: error instanceof Error ? error : new Error(errorMessage(error)) }
  }

  componentDidCatch(error: unknown, info: ErrorInfo) {
    console.error('Render error', error, info.componentStack)
  }

  componentDidUpdate(prev: Props) {
    if (this.state.error && prev.resetKey !== this.props.resetKey) this.setState({ error: null })
  }

  reset = () => this.setState({ error: null })

  render() {
    const { error } = this.state
    if (!error) return this.props.children
    if (this.props.fallback) return this.props.fallback(error, this.reset)
    return <ErrorPanel error={error} onRetry={this.reset} />
  }
}

/** The error, plain advice and the ways out: reload, try again, close (when given), back to the library. */
export function ErrorPanel({
  error,
  title = 'This page hit an error',
  onRetry,
  onClose,
}: {
  error: Error
  title?: string
  onRetry?: () => void
  onClose?: () => void
}) {
  return (
    <div role="alert" className="mx-auto flex max-w-md flex-col items-center gap-3 px-4 py-10 text-center">
      <h2 className="text-base font-semibold">{title}</h2>
      <p className="text-sm text-ink-soft">
        The rest of the app still works. Reloading usually fixes it. If it happens again after a reload, the
        server is probably running an older version than this site: restart it from the portal.
      </p>
      <code className="max-w-full overflow-x-auto rounded-lg bg-surface-2 px-3 py-2 text-left text-xs text-ink">
        {errorMessage(error)}
      </code>
      <div className="flex flex-wrap justify-center gap-2">
        <Button onClick={() => window.location.reload()}>Reload the page</Button>
        {onRetry && (
          <Button variant="secondary" onClick={onRetry}>
            Try again
          </Button>
        )}
        {onClose && (
          <Button variant="secondary" onClick={onClose}>
            Close
          </Button>
        )}
      </div>
      <a href="/" className="text-sm text-accent underline-offset-2 hover:underline">
        Back to the library
      </a>
    </div>
  )
}
