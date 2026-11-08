// Sticky top bar: the cloud's name, the main navigation on desktop (Library, People, the Documents menu and
// the More menu, Nav.tsx), then on the right the vault countdown while it is unlocked, Upload and the
// account menu (Settings and Theme as submenus, Sign out). On phones the navigation lives in the bottom
// tab bar instead (Nav.tsx) and the header keeps the theme toggle.
import { Link, useLocation } from 'react-router'
import clsx from 'clsx'
import { Cloud, LogOut, Monitor, Moon, Settings, Sun, Upload } from 'lucide-react'
import { useAuth } from '../../lib/auth'
import { THEME_CHOICES, useTheme } from '../../lib/theme'
import { Avatar } from '../ui'
import HeaderMenu, { type MenuEntry } from './HeaderMenu'
import { DesktopNav } from './Nav'
import { SETTINGS_SECTIONS, settingsHref } from './settings/sections'
import ThemeToggle from './ThemeToggle'
import VaultBadge from './VaultBadge'
import { DEFAULT_SITE_TITLE } from './title'

const THEME_ICONS = { system: Monitor, light: Sun, dark: Moon } as const

function AccountMenu({ onLogout, loggingOut }: { onLogout: () => void; loggingOut: boolean }) {
  const { state } = useAuth()
  const { choice, setTheme } = useTheme()
  const { pathname, search } = useLocation()
  const tab = new URLSearchParams(search).get('tab') ?? 'account'
  const name = state?.username ?? ''
  const themeLabel = THEME_CHOICES.find((t) => t.value === choice)?.label ?? 'System'
  const entries: MenuEntry[] = [
    ...(name ? [{ kind: 'heading', label: `Signed in as ${name}` } as MenuEntry] : []),
    {
      kind: 'submenu',
      label: 'Settings',
      icon: Settings,
      active: pathname === '/settings',
      entries: SETTINGS_SECTIONS.map(
        (sec): MenuEntry => ({
          kind: 'link',
          to: settingsHref(sec.id),
          label: sec.label,
          icon: sec.icon,
          active: pathname === '/settings' && tab === sec.id,
        }),
      ),
    },
    {
      kind: 'submenu',
      label: 'Theme',
      icon: THEME_ICONS[choice],
      hint: themeLabel,
      entries: THEME_CHOICES.map(
        (t): MenuEntry => ({
          kind: 'radio',
          label: t.label,
          icon: THEME_ICONS[t.value],
          checked: choice === t.value,
          onSelect: () => setTheme(t.value),
        }),
      ),
    },
    { kind: 'separator' },
    { kind: 'action', label: 'Sign out', icon: LogOut, danger: true, disabled: loggingOut, onSelect: onLogout },
  ]
  return (
    <HeaderMenu
      label="Account"
      ariaLabel={name ? `Account menu, signed in as ${name}` : 'Account menu'}
      entries={entries}
      align="right"
      active={pathname === '/settings'}
      trigger={<Avatar src={null} name={name || '?'} size={34} className={clsx(pathname === '/settings' && 'ring-2 ring-accent')} />}
    />
  )
}

export default function Header({ onLogout, loggingOut }: { onLogout: () => void; loggingOut: boolean }) {
  const { state } = useAuth()
  const { pathname } = useLocation()
  const title = state?.site_title || DEFAULT_SITE_TITLE
  const uploading = pathname === '/upload'
  return (
    <header className="sticky top-0 z-30 border-b border-line bg-bg/90 pt-[env(safe-area-inset-top)] backdrop-blur supports-[backdrop-filter]:bg-bg/75">
      <div className="mx-auto flex h-14 max-w-7xl items-center gap-2 px-4">
        <Link to="/" className="mr-1 flex min-w-0 shrink items-center gap-2.5 rounded-lg lg:mr-3" aria-label={`${title}, library`}>
          <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-ink text-bg">
            <Cloud className="size-[18px]" aria-hidden />
          </span>
          <span className="truncate text-[15px] font-bold tracking-tight md:max-lg:sr-only">{title}</span>
        </Link>
        <DesktopNav />
        <div className="ml-auto flex shrink-0 items-center gap-1.5">
          <VaultBadge />
          <Link
            to="/upload"
            aria-current={uploading ? 'page' : undefined}
            title="Upload"
            className={clsx(
              'hidden h-10 items-center gap-2 rounded-full px-3 text-sm font-semibold text-white transition-colors outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 md:inline-flex lg:px-4',
              uploading ? 'bg-accent-strong' : 'bg-accent hover:bg-accent-strong',
            )}
          >
            <Upload className="size-[18px]" aria-hidden />
            <span className="sr-only lg:not-sr-only">Upload</span>
          </Link>
          <ThemeToggle className="md:hidden" />
          <div className="hidden md:block">
            <AccountMenu onLogout={onLogout} loggingOut={loggingOut} />
          </div>
        </div>
      </div>
    </header>
  )
}
