// Header button that cycles the colour theme: system, light, dark (lib/theme.ts).
import clsx from 'clsx'
import { Monitor, Moon, Sun } from 'lucide-react'
import { useTheme, type ThemeChoice } from '../../lib/theme'

const NEXT: Record<ThemeChoice, ThemeChoice> = { system: 'light', light: 'dark', dark: 'system' }
const LABEL: Record<ThemeChoice, string> = { system: 'System theme', light: 'Light theme', dark: 'Dark theme' }

export default function ThemeToggle({ className }: { className?: string }) {
  const { choice, setTheme } = useTheme()
  const Icon = choice === 'dark' ? Moon : choice === 'light' ? Sun : Monitor
  const next = NEXT[choice]
  return (
    <button
      type="button"
      onClick={() => setTheme(next)}
      aria-label={`${LABEL[choice]}. Switch to ${LABEL[next].toLowerCase()}`}
      title={`${LABEL[choice]} (switch to ${next})`}
      className={clsx(
        'inline-flex size-10 shrink-0 items-center justify-center rounded-full text-ink-soft transition-colors hover:bg-ink/5 hover:text-ink',
        className,
      )}
    >
      <Icon className="size-5" aria-hidden />
    </button>
  )
}
