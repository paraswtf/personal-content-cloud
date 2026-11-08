// Main navigation (docs/ARCHITECTURE.md section 14). Desktop (md and up): Library and People as links, a
// Documents menu (all documents and each document group) and a More menu (Browse: Albums, Map, Vault;
// Tidy up: Review with its count, Trash), icon only below lg; the header adds Upload and the account menu
// (Header.tsx). Phones: a bottom tab bar with the four most used sections and a More sheet with the same
// groups, theme and sign out.
import { useEffect, useState, type ComponentType } from 'react'
import { Link, useLocation } from 'react-router'
import { useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import { ClipboardCheck, Ellipsis, FileText, Images, Lock, LockOpen, LogOut, Menu, Settings, Trash2, Upload, Users, BookImage, Map } from 'lucide-react'
import { GROUP_META, GROUP_ORDER } from '../documents/docMeta'
import HeaderMenu, { type MenuEntry } from './HeaderMenu'
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

/** The More menu (desktop) and the More sheet (phones): what is not one click away, in two groups. */
export const MORE_GROUPS: { label: string; paths: string[] }[] = [
  { label: 'Browse', paths: ['/albums', '/map', '/vault'] },
  { label: 'Tidy up', paths: ['/review', '/trash'] },
]
const MORE_HINTS: Record<string, string> = {
  '/review': 'Mis-clicks, guesses and forwards',
}

function navItem(path: string): NavItem {
  return NAV_ITEMS.find((i) => i.to === path)!
}

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

const LINK =
  'relative inline-flex h-10 min-w-10 items-center justify-center gap-2 rounded-full px-2.5 text-sm font-semibold transition-colors outline-none focus-visible:ring-2 focus-visible:ring-accent lg:px-3'

export function DesktopNav() {
  const { pathname, search } = useLocation()
  const { vaultUnlocked } = useAuth()
  const review = useReviewCount()
  const group = new URLSearchParams(search).get('group')
  const onDocs = pathname === '/documents' || pathname.startsWith('/documents/')

  const docEntries: MenuEntry[] = [
    { kind: 'link', to: '/documents', label: 'All documents', icon: FileText, active: onDocs && !group },
    { kind: 'separator' },
    ...GROUP_ORDER.map(
      (g): MenuEntry => ({
        kind: 'link',
        to: `/documents?group=${g}`,
        label: GROUP_META[g].label,
        icon: GROUP_META[g].icon,
        active: onDocs && group === g,
      }),
    ),
  ]
  const moreEntries: MenuEntry[] = MORE_GROUPS.flatMap((g, k): MenuEntry[] => [
    ...(k > 0 ? [{ kind: 'separator' } as MenuEntry] : []),
    { kind: 'heading', label: g.label },
    ...g.paths.map((path): MenuEntry => {
      const item = navItem(path)
      return {
        kind: 'link',
        to: item.to,
        label: item.label,
        icon: itemIcon(item, vaultUnlocked),
        active: isActive(item, pathname),
        badge: path === '/review' ? review : undefined,
        hint: path === '/vault' ? (vaultUnlocked ? 'Open now' : 'Locked') : MORE_HINTS[path],
      }
    }),
  ])
  const moreActive = MORE_GROUPS.some((g) => g.paths.some((p) => isActive(navItem(p), pathname)))

  return (
    <nav aria-label="Main" className="hidden items-center gap-0.5 md:flex">
      {['/', '/people'].map((path) => {
        const item = navItem(path)
        const active = isActive(item, pathname)
        return (
          <Link
            key={path}
            to={item.to}
            aria-current={active ? 'page' : undefined}
            title={item.label}
            className={clsx(LINK, active ? 'bg-ink/[0.08] text-ink' : 'text-ink-soft hover:bg-ink/5 hover:text-ink')}
          >
            <item.icon className="size-[18px]" aria-hidden />
            <span className="sr-only lg:not-sr-only">{item.label}</span>
          </Link>
        )
      })}
      <HeaderMenu label="Documents" icon={FileText} entries={docEntries} active={onDocs} />
      <HeaderMenu label="More" icon={Ellipsis} entries={moreEntries} active={moreActive} dot={review > 0} />
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
        <div className="flex flex-col gap-3">
          {[...MORE_GROUPS, { label: 'Account', paths: rest.map((i) => i.to).filter((p) => !MORE_GROUPS.some((g) => g.paths.includes(p))) }].map((g) => (
            <div key={g.label} role="group" aria-label={g.label} className="flex flex-col gap-1">
              <p className="px-3 text-xs font-semibold tracking-wide text-ink-faint uppercase" aria-hidden>
                {g.label}
              </p>
              {g.paths.map((path) => {
                const item = navItem(path)
                return (
                  <SheetLink
                    key={item.to}
                    item={item}
                    active={isActive(item, pathname)}
                    onClick={() => setMore(false)}
                    badge={item.to === '/review' ? review : undefined}
                  />
                )
              })}
            </div>
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
