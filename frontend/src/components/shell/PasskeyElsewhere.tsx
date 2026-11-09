// Shown in place of a passkey button that would fail (docs/ARCHITECTURE.md section 4): on a custom domain, in a
// browser without WebAuthn related origins (usePasskeyPlace() says 'elsewhere'). It links to the same page at the
// server's own address, where the passkeys work. `lead` is the sentence before the link, for example "Sign in with
// one at".
import { Info } from 'lucide-react'
import type { HomeAddress } from '../../lib/auth'

export default function PasskeyElsewhere({ home, lead }: { home: HomeAddress; lead: string }) {
  const here = window.location
  const href = `${home.origin}${here.pathname}${here.search}`
  return (
    <p role="note" className="flex gap-2 rounded-xl bg-surface-2 px-3 py-2 text-left text-sm text-ink-soft">
      <Info className="mt-0.5 size-4 shrink-0" aria-hidden />
      <span className="min-w-0">
        This browser can't use passkeys on {here.hostname}. {lead}{' '}
        <a href={href} className="font-medium text-accent underline underline-offset-2 [overflow-wrap:anywhere]">
          {home.host}
        </a>
        .
      </span>
    </p>
  )
}
