// Settings > Wallet: whether Apple Wallet passes can be made from event and boarding passes, and if
// not, exactly what to put where. Secrets never go through this page: the owner places files in the
// wallet folder and sets environment variables on the server.
import type { ReactNode } from 'react'
import { CheckCircle2, CircleAlert, Wallet } from 'lucide-react'
import { Badge, ErrorState, Spinner } from '../../ui'
import { useSystem } from './SystemTab'
import { Section } from './common'

function Code({ children }: { children: ReactNode }) {
  return <code className="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-[0.85em] text-ink">{children}</code>
}

function Block({ children }: { children: string }) {
  return (
    <pre className="mt-2 overflow-x-auto rounded-xl bg-surface-2 p-3 font-mono text-xs leading-relaxed text-ink">{children}</pre>
  )
}

const ENV_HELP: Record<string, string> = {
  WALLET_PASS_TYPE_ID: 'the Pass Type ID you registered, for example pass.com.example.cloud',
  WALLET_TEAM_ID: 'your 10-character Apple Developer Team ID (Membership details)',
  WALLET_ORG_NAME: 'the name shown on passes (optional; defaults to the site title)',
}

export default function WalletTab() {
  const q = useSystem(false)
  if (q.isPending)
    return (
      <div className="flex justify-center py-10">
        <Spinner label="Loading wallet status" />
      </div>
    )
  if (q.isError) return <ErrorState error={q.error} onRetry={() => void q.refetch()} />
  const w = q.data.wallet
  const dir = w.dir || 'DATA_DIR/wallet'

  return (
    <div className="flex flex-col gap-4">
      <Section
        title="Apple Wallet"
        description="Event tickets and boarding passes found among your documents can be added to Apple Wallet. Passes must be signed with your own Apple Developer certificate."
        aside={w.configured ? <Badge tone="ok">Ready</Badge> : <Badge tone="warn">Not set up</Badge>}
      >
        {w.configured ? (
          <p className="flex items-start gap-2 text-sm">
            <CheckCircle2 className="mt-0.5 size-4 shrink-0 text-ok" aria-hidden />
            Wallet passes are set up. Open a pass in Documents and choose Add to Apple Wallet.
          </p>
        ) : (
          <div className="flex flex-col gap-2 text-sm">
            <p className="flex items-start gap-2">
              <CircleAlert className="mt-0.5 size-4 shrink-0 text-warn" aria-hidden />
              Still missing:
            </p>
            <ul className="ml-6 list-disc text-ink-soft">
              {w.missing.map((m) => (
                <li key={m}>
                  <Code>{m}</Code>
                  {ENV_HELP[m] ? `: ${ENV_HELP[m]}` : ''}
                </li>
              ))}
            </ul>
          </div>
        )}
      </Section>

      {!w.configured && (
        <Section title="How to set it up" description="Everything happens on the server and in your Apple Developer account. This page never asks for keys or passwords.">
          <ol className="flex list-decimal flex-col gap-4 pl-5 text-sm text-ink-soft marker:font-semibold marker:text-ink">
            <li>
              In the Apple Developer portal, open Certificates, Identifiers and Profiles, then Identifiers, and register a{' '}
              <span className="font-semibold text-ink">Pass Type ID</span> (for example <Code>pass.com.example.cloud</Code>).
            </li>
            <li>
              Create a <span className="font-semibold text-ink">Pass Type ID certificate</span> for it, download it, open it in
              Keychain Access on a Mac and export it with its private key as <Code>pass.p12</Code>.
            </li>
            <li>
              Convert it to PEM files (OpenSSL 3 may need <Code>-legacy</Code> for files exported by Keychain):
              <Block>{`openssl pkcs12 -in pass.p12 -clcerts -nokeys -out cert.pem
openssl pkcs12 -in pass.p12 -nocerts -nodes -out key.pem`}</Block>
            </li>
            <li>
              Download the Apple WWDR intermediate certificate (G4) from Apple PKI and convert it:
              <Block>{`openssl x509 -inform der -in AppleWWDRCAG4.cer -out wwdr.pem`}</Block>
            </li>
            <li>
              Put <Code>cert.pem</Code>, <Code>key.pem</Code> and <Code>wwdr.pem</Code> in the wallet folder on the server and make
              the key readable only by the server user:
              <Block>{`${dir}/cert.pem
${dir}/key.pem      (chmod 600)
${dir}/wwdr.pem`}</Block>
            </li>
            <li>
              Set these environment variables for the server, then restart it:
              <Block>{`WALLET_PASS_TYPE_ID=pass.com.example.cloud
WALLET_TEAM_ID=ABCDE12345
WALLET_ORG_NAME=My Cloud            (optional)`}</Block>
            </li>
            <li>Come back to this tab. It shows Ready once every item is in place.</li>
          </ol>
          <p className="mt-4 flex items-start gap-2 text-xs text-ink-faint">
            <Wallet className="mt-0.5 size-3.5 shrink-0" aria-hidden />
            The pass is built and signed on this server; nothing is sent to Apple until you add the pass on your device.
          </p>
        </Section>
      )}
    </div>
  )
}
