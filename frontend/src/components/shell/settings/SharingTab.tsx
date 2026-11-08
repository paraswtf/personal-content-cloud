// Settings > Sharing (docs/ARCHITECTURE.md section 26): the share links, with copy, edit and revoke.
import { Section } from './common'

export default function SharingTab() {
  return (
    <Section title="Share links" description="Links you created to show photos to people without an account.">
      <p className="text-sm text-ink-soft">No links yet.</p>
    </Section>
  )
}
