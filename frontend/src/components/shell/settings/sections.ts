// The Settings sections (/settings?tab=<id>, account by default), shared by the Settings page's tabs and the
// header's account menu.
import { Fingerprint, GraduationCap, Link2, MonitorSmartphone, Palette, Server, UserRound, Wallet, type LucideIcon } from 'lucide-react'

export const SETTINGS_SECTIONS = [
  { id: 'account', label: 'Account', icon: UserRound },
  { id: 'passkeys', label: 'Passkeys', icon: Fingerprint },
  { id: 'sessions', label: 'Sessions', icon: MonitorSmartphone },
  { id: 'system', label: 'System', icon: Server },
  { id: 'learning', label: 'Learning', icon: GraduationCap },
  { id: 'sharing', label: 'Sharing', icon: Link2 },
  { id: 'wallet', label: 'Wallet', icon: Wallet },
  { id: 'appearance', label: 'Appearance', icon: Palette },
] as const satisfies readonly { id: string; label: string; icon: LucideIcon }[]

export type SettingsSectionId = (typeof SETTINGS_SECTIONS)[number]['id']

export function settingsHref(id: SettingsSectionId): string {
  return id === 'account' ? '/settings' : `/settings?tab=${id}`
}
