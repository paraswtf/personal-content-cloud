// Colour theme: 'system' follows prefers-color-scheme, 'light' and 'dark' force one. The choice is
// kept in localStorage['pcc.theme'] and applied as html[data-theme] (index.css reads it). Storage
// is a convenience only: when it is blocked the app follows the system setting.
import { useSyncExternalStore } from 'react'

export type ThemeChoice = 'system' | 'light' | 'dark'
export const THEME_KEY = 'pcc.theme'
export const THEME_CHOICES: { value: ThemeChoice; label: string }[] = [
  { value: 'system', label: 'System' },
  { value: 'light', label: 'Light' },
  { value: 'dark', label: 'Dark' },
]

const META_COLORS = { light: '#f6f6f4', dark: '#111315' } as const

function readStored(): ThemeChoice {
  try {
    const v = window.localStorage.getItem(THEME_KEY)
    return v === 'light' || v === 'dark' ? v : 'system'
  } catch {
    return 'system'
  }
}

let current: ThemeChoice = typeof window === 'undefined' ? 'system' : readStored()
const listeners = new Set<() => void>()

function systemDark(): boolean {
  return typeof window !== 'undefined' && !!window.matchMedia?.('(prefers-color-scheme: dark)').matches
}

/** The theme in effect right now ('light' or 'dark'), after resolving 'system'. */
export function resolvedTheme(choice: ThemeChoice = current): 'light' | 'dark' {
  if (choice === 'system') return systemDark() ? 'dark' : 'light'
  return choice
}

/** Write the choice to the document: data-theme on <html> and the browser chrome colour. */
export function applyTheme(choice: ThemeChoice = current): void {
  if (typeof document === 'undefined') return
  const root = document.documentElement
  if (choice === 'system') root.removeAttribute('data-theme')
  else root.setAttribute('data-theme', choice)
  const meta = document.querySelector('meta[name="theme-color"]')
  meta?.setAttribute('content', META_COLORS[resolvedTheme(choice)])
}

export function getTheme(): ThemeChoice {
  return current
}

export function setTheme(choice: ThemeChoice): void {
  current = choice
  try {
    if (choice === 'system') window.localStorage.removeItem(THEME_KEY)
    else window.localStorage.setItem(THEME_KEY, choice)
  } catch {
    // storage blocked: the choice lasts for this page load
  }
  applyTheme(choice)
  for (const l of listeners) l()
}

/** Apply the stored theme and follow system changes while the choice is 'system'. Call once at startup. */
export function initTheme(): void {
  applyTheme()
  const mq = window.matchMedia?.('(prefers-color-scheme: dark)')
  mq?.addEventListener?.('change', () => {
    if (current === 'system') {
      applyTheme()
      for (const l of listeners) l()
    }
  })
  // another tab changed the choice
  window.addEventListener('storage', (e) => {
    if (e.key !== THEME_KEY) return
    current = readStored()
    applyTheme()
    for (const l of listeners) l()
  })
}

function subscribe(l: () => void): () => void {
  listeners.add(l)
  return () => listeners.delete(l)
}

/** The stored choice and the theme in effect as one value, "choice:theme". React re-renders when the value it is
 *  given changes: with the choice on 'system' an OS switch changes only the theme in effect, and the choice alone
 *  would look unchanged. */
export function themeSnapshot(): string {
  return `${current}:${resolvedTheme(current)}`
}

/** React binding: the stored choice and the theme in effect. */
export function useTheme(): { choice: ThemeChoice; resolved: 'light' | 'dark'; setTheme: (c: ThemeChoice) => void } {
  const [choice, resolved] = useSyncExternalStore(subscribe, themeSnapshot, themeSnapshot).split(':') as [ThemeChoice, 'light' | 'dark']
  return { choice, resolved, setTheme }
}
