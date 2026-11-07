// Sticky top bar: name of the cloud, the main navigation on desktop, the vault countdown while it is
// unlocked, the theme toggle and (desktop) sign out. On phones the navigation lives in the bottom tab
// bar instead (Nav.tsx).
import { Link } from 'react-router'
import { Cloud, LogOut } from 'lucide-react'
import { useAuth } from '../../lib/auth'
import { DesktopNav } from './Nav'
import ThemeToggle from './ThemeToggle'
import VaultBadge from './VaultBadge'
import { DEFAULT_SITE_TITLE } from './title'

export default function Header({ onLogout, loggingOut }: { onLogout: () => void; loggingOut: boolean }) {
  const { state } = useAuth()
  const title = state?.site_title || DEFAULT_SITE_TITLE
  return (
    <header className="sticky top-0 z-30 border-b border-line bg-bg/90 pt-[env(safe-area-inset-top)] backdrop-blur supports-[backdrop-filter]:bg-bg/75">
      <div className="mx-auto flex h-14 max-w-7xl items-center gap-2 px-4">
        <Link to="/" className="mr-2 flex min-w-0 items-center gap-2.5 rounded-lg" aria-label={`${title}, library`}>
          <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-ink text-bg">
            <Cloud className="size-[18px]" aria-hidden />
          </span>
          <span className="truncate text-[15px] font-bold tracking-tight md:max-lg:sr-only">{title}</span>
        </Link>
        <div className="ml-auto flex min-w-0 items-center gap-1">
          <DesktopNav />
          <VaultBadge className="ml-1" />
          <ThemeToggle />
          <button
            type="button"
            onClick={onLogout}
            disabled={loggingOut}
            aria-label="Sign out"
            title={state?.username ? `Sign out ${state.username}` : 'Sign out'}
            className="hidden size-10 shrink-0 items-center justify-center rounded-full text-ink-soft transition-colors hover:bg-ink/5 hover:text-ink disabled:opacity-50 md:inline-flex"
          >
            <LogOut className="size-5" aria-hidden />
          </button>
        </div>
      </div>
    </header>
  )
}
