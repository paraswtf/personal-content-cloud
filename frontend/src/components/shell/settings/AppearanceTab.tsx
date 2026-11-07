// Settings > Appearance: the colour theme, kept in this browser only (lib/theme.ts).
import clsx from 'clsx'
import { Monitor, Moon, Sun } from 'lucide-react'
import { THEME_CHOICES, useTheme, type ThemeChoice } from '../../../lib/theme'
import { Section } from './common'

const ICONS: Record<ThemeChoice, typeof Sun> = { system: Monitor, light: Sun, dark: Moon }
const HINTS: Record<ThemeChoice, string> = {
  system: 'Follows the device setting',
  light: 'Always light',
  dark: 'Always dark',
}

export default function AppearanceTab() {
  const { choice, setTheme } = useTheme()
  return (
    <Section title="Theme" description="Saved in this browser. Other devices keep their own choice.">
      <div role="radiogroup" aria-label="Theme" className="grid gap-2 sm:grid-cols-3">
        {THEME_CHOICES.map((t) => {
          const Icon = ICONS[t.value]
          const on = choice === t.value
          return (
            <button
              key={t.value}
              type="button"
              role="radio"
              aria-checked={on}
              onClick={() => setTheme(t.value)}
              className={clsx(
                'flex min-h-16 items-center gap-3 rounded-xl border px-4 py-3 text-left transition-colors',
                on ? 'border-accent bg-accent-soft' : 'border-line hover:border-ink-faint',
              )}
            >
              <Icon className={clsx('size-5 shrink-0', on ? 'text-accent' : 'text-ink-soft')} aria-hidden />
              <span>
                <span className="block text-sm font-semibold">{t.label}</span>
                <span className="block text-xs text-ink-soft">{HINTS[t.value]}</span>
              </span>
            </button>
          )
        })}
      </div>
    </Section>
  )
}
