// Main navigation (docs/ARCHITECTURE.md section 14). Desktop (md and up): a row of links in the header,
// icon only on md and lg, icon and label from xl. Phones: a bottom tab bar with the four most used sections
// and a More sheet for the rest, theme and sign out.
import { useEffect, useState, type ComponentType } from 'react'
import { Link, useLocation } from 'react-router'
import { useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import { ClipboardCheck, FileText, Images, Lock, LockOpen, LogOut, Menu, Settings, Trash2, Upload, Users, BookImage, Map } from 'lucide-react'
import { api } from '../../lib/api'
import { useAuth } from '../../lib/auth'
import { THEME_CHOICES, useTheme } from '../../lib/theme'
import { Modal } from '../ui'

type Icon = ComponentType<{ className?: string; 'aria-hidden'?: boolean }>

export interface NavItem {
  to: string
  label: string
  icon: Icon
  /** Extra path prefixes that mark this item active. */
  match?: (pathname: string) => boolean
}

export const NAV_ITEMS: NavItem[] = [
  { to: '/', label: 'Library', icon: Images, match: (p) => p === '/' || p.startsWith('/m/') },
  { to: '/people', label: 'People', icon: Users },
  { to: '/documents', label: 'Documents', icon: FileText },
  { to: '/albums', label: 'Albums', icon: BookImage },
  { to: '/map', label: 'Map', icon: Map },
  { to: '/review', label: 'Review', icon: ClipboardCheck },
  { to: '/vault', label: 'Vault', icon: Lock },
  { to: '/upload', label: 'Upload', icon: Upload },
  { to: '/trash', label: 'Trash', icon: Trash2 },
  { to: '/settings', label: 'Settings', icon: Settings },
]

const TAB_PATHS = ['/', '/people', '/upload', '/documents']

export function isActive(item: NavItem, pathname: string): boolean {
  if (item.match) return item.match(pathname)
  return pathname === item.to || pathname.startsWith(`${item.to}/`)
}

/** Items waiting in Review (mis-clicks plus guesses), from the shared ['stats'] query. */
function useReviewCount(): number {
  const q = useQuery({ queryKey: ['stats'], queryFn: api.stats, staleTime: 60_000 })
  return q.data ? q.data.misclicks + q.data.guesses : 0
}

function CountBadge({ n, className }: { n: number; className?: string }) {
  if (!n) return null
  return (
    <span
      className={clsx(
        'min-w-5 rounded-full bg-accent px-1.5 text-center text-[11px] font-semibold leading-5 text-white tabular-nums',
        className,
      )}
    >
      {n > 999 ? '999+' : n}
      <span className="sr-only"> waiting</span>
    </span>
  )
}

function itemIcon(item: NavItem, vaultUnlocked: boolean): Icon {
  return item.to === '/vault' && vaultUnlocked ? LockOpen : item.icon
}

export function DesktopNav() {
  const { pathname } = useLocation()
  const { vaultUnlocked } = useAuth()
  const review = useReviewCount()
  return (
    <nav aria-label="Main" className="hidden items-center gap-0.5 md:flex">
      {NAV_ITEMS.map((item) => {
        const active = isActive(item, pathname)
        const Icon = itemIcon(item, vaultUnlocked)
        return (
          <Link
            key={item.to}
            to={item.to}
            aria-current={active ? 'page' : undefined}
            title={item.label}
            className={clsx(
              'relative inline-flex h-10 min-w-10 items-center justify-center gap-2 rounded-full px-2.5 text-sm font-semibold transition-colors xl:px-3',
              active ? 'bg-ink/[0.08] text-ink' : 'text-ink-soft hover:bg-ink/5 hover:text-ink',
            )}
          >
            <Icon className="size-[18px]" aria-hidden />
            <span className="sr-only xl:not-sr-only">{item.label}</span>
            {item.to === '/review' && (
              <CountBadge n={review} className="absolute -right-1 -top-0.5 xl:static" />
            )}
          </Link>
        )
      })}
    </nav>
  )
}

function SheetLink({ item, active, onClick, badge }: { item: NavItem; active: boolean; onClick: () => void; badge?: number }) {
  const { vaultUnlocked } = useAuth()
  const Icon = itemIcon(item, vaultUnlocked)
  return (
    <Link
      to={item.to}
      onClick={onClick}
      aria-current={active ? 'page' : undefined}
      className={clsx(
        'flex h-12 items-center gap-3 rounded-xl px-3 text-base font-medium',
        active ? 'bg-ink/[0.08] text-ink' : 'text-ink hover:bg-ink/5',
      )}
    >
      <Icon className="size-5 text-ink-soft" aria-hidden />
      <span className="flex-1">{item.label}</span>
      {badge ? <CountBadge n={badge} /> : null}
    </Link>
  )
}

export function BottomTabBar({ onLogout }: { onLogout: () => void }) {
  const { pathname } = useLocation()
  const { choice, setTheme } = useTheme()
  const review = useReviewCount()
  const [more, setMore] = useState(false)
  const tabs = NAV_ITEMS.filter((i) => TAB_PATHS.includes(i.to))
  const rest = NAV_ITEMS.filter((i) => !TAB_PATHS.includes(i.to))
  const moreActive = rest.some((i) => isActive(i, pathname))

  // close the sheet when the route changes by other means (back button)
  useEffect(() => setMore(false), [pathname])

  return (
    <>
      <nav
        aria-label="Main"
        className="fixed inset-x-0 bottom-0 z-30 border-t border-line bg-bg/95 pb-[env(safe-area-inset-bottom)] backdrop-blur md:hidden"
      >
        <ul className="mx-auto grid h-[var(--tabbar-h)] max-w-lg grid-cols-5">
          {tabs.map((item) => {
            const active = isActive(item, pathname)
            return (
              <li key={item.to}>
                <Link
                  to={item.to}
                  aria-current={active ? 'page' : undefined}
                  className={clsx(
                    'flex h-full flex-col items-center justify-center gap-1 text-[11px] font-semibold',
                    active ? 'text-accent' : 'text-ink-soft',
                  )}
                >
                  <item.icon className="size-5" aria-hidden />
                  {item.label}
                </Link>
              </li>
            )
          })}
          <li>
            <button
              type="button"
              onClick={() => setMore(true)}
              aria-haspopup="dialog"
              aria-expanded={more}
              className={clsx(
                'relative flex h-full w-full flex-col items-center justify-center gap-1 text-[11px] font-semibold',
                moreActive ? 'text-accent' : 'text-ink-soft',
              )}
            >
              <Menu className="size-5" aria-hidden />
              More
              {review > 0 && <span className="absolute right-[calc(50%-18px)] top-2.5 size-2 rounded-full bg-accent" aria-hidden />}
            </button>
          </li>
        </ul>
      </nav>
      <Modal open={more} onClose={() => setMore(false)} title="More">
        <div className="flex flex-col gap-1">
          {rest.map((item) => (
            <SheetLink
              key={item.to}
              item={item}
              active={isActive(item, pathname)}
              onClick={() => setMore(false)}
              badge={item.to === '/review' ? review : undefined}
            />
          ))}
        </div>
        <div className="mt-4 border-t border-line pt-4">
          <p className="mb-2 text-sm font-medium text-ink-soft" id="sheet-theme">
            Theme
          </p>
          <div role="group" aria-labelledby="sheet-theme" className="grid grid-cols-3 gap-1 rounded-full border border-line p-0.5">
            {THEME_CHOICES.map((t) => (
              <button
                key={t.value}
                type="button"
                aria-pressed={choice === t.value}
                onClick={() => setTheme(t.value)}
                className={clsx(
                  'h-10 rounded-full text-sm font-medium',
                  choice === t.value ? 'bg-ink text-bg' : 'text-ink-soft hover:bg-ink/5',
                )}
              >
                {t.label}
              </button>
            ))}
          </div>
          <button
            type="button"
            onClick={() => {
              setMore(false)
              onLogout()
            }}
            className="mt-4 flex h-12 w-full items-center gap-3 rounded-xl px-3 text-base font-medium text-danger hover:bg-danger/10"
          >
            <LogOut className="size-5" aria-hidden />
            Sign out
          </button>
        </div>
      </Modal>
    </>
  )
}
