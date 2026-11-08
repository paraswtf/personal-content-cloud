// /settings: Account, Passkeys, Sessions, System, Learning, Wallet and Appearance tabs (?tab=<id>, default
// account). The tab bodies live in components/shell/settings/.
import { useRef, type ComponentType, type KeyboardEvent } from 'react'
import { useSearchParams } from 'react-router'
import clsx from 'clsx'
import AccountTab from '../components/shell/settings/AccountTab'
import PasskeysTab from '../components/shell/settings/PasskeysTab'
import SessionsTab from '../components/shell/settings/SessionsTab'
import SystemTab from '../components/shell/settings/SystemTab'
import LearningTab from '../components/shell/settings/LearningTab'
import WalletTab from '../components/shell/settings/WalletTab'
import AppearanceTab from '../components/shell/settings/AppearanceTab'
import SharingTab from '../components/shell/settings/SharingTab'
import { usePageTitle } from '../components/shell/title'
import { SETTINGS_SECTIONS, type SettingsSectionId } from '../components/shell/settings/sections'

const BODIES: Record<SettingsSectionId, ComponentType> = {
  account: AccountTab,
  passkeys: PasskeysTab,
  sessions: SessionsTab,
  system: SystemTab,
  learning: LearningTab,
  sharing: SharingTab,
  wallet: WalletTab,
  appearance: AppearanceTab,
}
const TABS = SETTINGS_SECTIONS.map((t) => ({ ...t, body: BODIES[t.id] }))

type TabId = SettingsSectionId

export default function SettingsPage() {
  const [params, setParams] = useSearchParams()
  const raw = params.get('tab')
  const tab = TABS.find((t) => t.id === raw) ?? TABS[0]
  usePageTitle(`${tab.label} settings`)
  const list = useRef<HTMLDivElement>(null)

  const select = (id: TabId, focus = false) => {
    const next = new URLSearchParams(params)
    if (id === 'account') next.delete('tab')
    else next.set('tab', id)
    setParams(next, { replace: true })
    if (focus) window.setTimeout(() => list.current?.querySelector<HTMLButtonElement>(`#tab-${id}`)?.focus(), 0)
  }

  // arrow keys move between tabs (WAI-ARIA tabs pattern)
  const onKeyDown = (e: KeyboardEvent) => {
    const i = TABS.findIndex((t) => t.id === tab.id)
    let j = -1
    if (e.key === 'ArrowRight') j = (i + 1) % TABS.length
    else if (e.key === 'ArrowLeft') j = (i - 1 + TABS.length) % TABS.length
    else if (e.key === 'Home') j = 0
    else if (e.key === 'End') j = TABS.length - 1
    if (j < 0) return
    e.preventDefault()
    select(TABS[j].id, true)
  }

  const Body = tab.body
  return (
    <div className="mx-auto w-full max-w-5xl px-4 py-6">
      <h1 className="mb-4 text-2xl font-bold tracking-tight">Settings</h1>
      <div
        ref={list}
        role="tablist"
        aria-label="Settings sections"
        onKeyDown={onKeyDown}
        className="-mx-4 mb-5 flex gap-1 overflow-x-auto px-4 pb-1 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
      >
        {TABS.map((t) => {
          const on = t.id === tab.id
          return (
            <button
              key={t.id}
              id={`tab-${t.id}`}
              type="button"
              role="tab"
              aria-selected={on}
              aria-controls="settings-panel"
              tabIndex={on ? 0 : -1}
              onClick={() => select(t.id)}
              className={clsx(
                'inline-flex h-10 shrink-0 items-center gap-2 rounded-full px-4 text-sm font-medium transition-colors',
                on ? 'bg-ink text-bg' : 'text-ink-soft hover:bg-ink/5 hover:text-ink',
              )}
            >
              <t.icon className="size-4" aria-hidden />
              {t.label}
            </button>
          )
        })}
      </div>
      <div id="settings-panel" role="tabpanel" aria-labelledby={`tab-${tab.id}`}>
        <Body />
      </div>
    </div>
  )
}
