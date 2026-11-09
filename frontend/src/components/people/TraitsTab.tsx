// A person's "Tattoos and nails" tab (section 34): their tattoos (long term, each with the photos it was matched in
// and when it was first and last seen), and the mehndi and nails of the last 60 days (short term). Each card has
// "Not a tattoo" (nails, mehndi) and "Not NAME's" for its crop, undoable; "Not a tattoo" also hides look-alikes of it.
// A crop opens its photo; a tattoo's photos open as one run in the viewer.
import { useMemo, useState, type ReactNode } from 'react'
import { useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import { Eye, EyeOff, Hand, MoreHorizontal, PenTool, UserX } from 'lucide-react'
import { api } from '../../lib/api'
import { formatCount, formatDate } from '../../lib/format'
import type { TraitGroupOut, TraitOut } from '../../lib/types'
import Viewer from '../Viewer'
import { useToast } from '../Toast'
import { useViewerParam } from '../public/useViewerParam'
import { Button, EmptyState, ErrorState, PageSpinner } from '../ui'
import { safeHex } from './evidence'
import { notTraitLabel, traitActionKey, useTraitActions } from './useTraitActions'

/** "March 2025". */
function monthYear(iso: string | null | undefined): string {
  if (!iso) return ''
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleDateString(undefined, { month: 'long', year: 'numeric' })
}

/** "In 14 photos, March 2025 to October 2026". */
function seenLine(g: TraitGroupOut): string {
  const from = monthYear(g.first_seen)
  const to = monthYear(g.last_seen)
  const span = from && to && from !== to ? `, ${from} to ${to}` : from || to ? `, ${from || to}` : ''
  return `In ${formatCount(g.photos, 'photo')}${span}`
}

type Acts = ReturnType<typeof useTraitActions>

function TraitCard({
  trait,
  title,
  sub,
  name,
  acts,
  onOpen,
}: {
  trait: TraitOut
  title: string
  sub: string
  /** The person's name, or null when unnamed ("Not this person's"). */
  name: string | null
  acts: Acts
  onOpen?: () => void
}) {
  const [menu, setMenu] = useState(false)
  const hex = trait.kind === 'nails' ? safeHex(trait.hex) : undefined
  const busy = acts.pending
  const crop = (
    <img
      src={trait.crop_url}
      alt={onOpen ? '' : trait.label}
      loading="lazy"
      className={clsx('size-20 rounded-xl bg-surface-2 object-cover ring-1 ring-line', trait.hidden && 'opacity-50')}
    />
  )
  return (
    <article className="flex flex-col gap-3 rounded-2xl border border-line bg-surface p-3">
      <div className="flex items-start gap-3">
        {onOpen ? (
          <button
            type="button"
            onClick={onOpen}
            aria-label={`Open the photo: ${trait.label}`}
            className="shrink-0 rounded-xl hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          >
            {crop}
          </button>
        ) : (
          <span className="shrink-0">{crop}</span>
        )}
        <div className="min-w-0 flex-1">
          <p className={clsx('flex items-center gap-1.5 text-sm font-semibold', trait.hidden && 'text-ink-faint line-through')}>
            {hex && <span aria-hidden className="size-3.5 shrink-0 rounded-full ring-1 ring-line" style={{ backgroundColor: hex }} />}
            <span className="min-w-0">{title}</span>
          </p>
          <p className="mt-0.5 text-xs text-ink-soft">{sub}</p>
          {trait.hidden && <p className="mt-0.5 text-xs text-ink-soft">Hidden: it does not count for guesses.</p>}
        </div>
        <button
          type="button"
          onClick={() => setMenu((v) => !v)}
          aria-expanded={menu}
          aria-label={`Options for ${trait.label}`}
          className="flex size-10 shrink-0 items-center justify-center rounded-full text-ink-soft hover:bg-surface-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
        >
          <MoreHorizontal className="size-5" aria-hidden />
        </button>
      </div>
      {menu && (
        <div role="group" aria-label={`What to do with ${trait.label}`} className="flex flex-wrap gap-2">
          {trait.hidden ? (
            <Button variant="secondary" disabled={busy} loading={acts.pendingKey === traitActionKey(trait.id, null)} onClick={() => acts.show(trait)}>
              <Eye className="size-4" aria-hidden />
              Show again
            </Button>
          ) : (
            <>
              <Button
                variant="secondary"
                disabled={busy}
                loading={acts.pendingKey === traitActionKey(trait.id, 'not_trait')}
                onClick={() => acts.hide(trait, 'not_trait')}
              >
                <EyeOff className="size-4" aria-hidden />
                {notTraitLabel(trait.kind)}
              </Button>
              <Button
                variant="secondary"
                disabled={busy}
                loading={acts.pendingKey === traitActionKey(trait.id, 'not_theirs')}
                onClick={() => acts.hide(trait, 'not_theirs', name)}
              >
                <UserX className="size-4" aria-hidden />
                {name ? `Not ${name}'s` : "Not this person's"}
              </Button>
            </>
          )}
        </div>
      )}
    </article>
  )
}

function Section({ id, icon, title, note, children }: { id: string; icon: ReactNode; title: string; note: string; children: ReactNode }) {
  return (
    <section aria-labelledby={id} className="mb-6">
      <h2 id={id} className="flex items-center gap-1.5 text-sm font-semibold text-ink">
        {icon}
        {title}
      </h2>
      <p className="mb-3 mt-0.5 text-xs text-ink-soft">{note}</p>
      <ul className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">{children}</ul>
    </section>
  )
}

export default function TraitsTab({ id, name: rawName }: { id: number; name: string | null }) {
  const name = rawName?.trim() ? rawName : null
  const who = name ?? 'this person'
  const toast = useToast()
  // under the 'person' key, so useRefreshLibrary asks again after any change
  const q = useQuery({ queryKey: ['person', id, 'traits'], queryFn: () => api.people.traits(id), staleTime: 30_000 })
  const acts = useTraitActions({ onDone: (m) => toast.success(m), onError: (m) => toast.error(m) })
  // the photos the viewer steps through: those of the tattoo opened, or the one photo of a mehndi or nails crop
  const [run, setRun] = useState<string[]>([])
  const runItems = useMemo(() => run.map((uid) => ({ uid })), [run])
  const viewer = useViewerParam(runItems, false, () => undefined)
  const openRun = (uids: string[], first: string) => {
    setRun(uids)
    viewer.open(first)
  }

  if (q.isPending) return <PageSpinner />
  if (q.isError) return <ErrorState error={q.error} onRetry={() => void q.refetch()} />
  const { tattoos, mehndi, nails } = q.data
  const viewerEl = viewer.uid && (
    <Viewer uid={viewer.uid} onClose={viewer.close} onPrev={viewer.onPrev} onNext={viewer.onNext} position={viewer.position} />
  )
  if (!tattoos.length && !mehndi.length && !nails.length)
    return (
      <>
        <EmptyState icon={<PenTool className="size-8" aria-hidden />} title="No tattoos or nails yet">
          Tattoos, nails and mehndi matched to {who} show here once their photos are read.
        </EmptyState>
        {viewerEl}
      </>
    )

  const shortCard = (t: TraitOut) => {
    const uid = t.item_uid
    return (
      <li key={t.id}>
        <TraitCard
          trait={t}
          title={t.label}
          sub={t.taken_at ? formatDate(t.taken_at) : 'Date not known'}
          name={name}
          acts={acts}
          onOpen={uid ? () => openRun([uid], uid) : undefined}
        />
      </li>
    )
  }

  return (
    <div>
      {tattoos.length > 0 && (
        <Section
          id={`traits-tattoos-${id}`}
          icon={<PenTool className="size-4" aria-hidden />}
          title="Tattoos"
          note={`Long term: the same tattoo helps find ${who} in photos taken months or years apart.`}
        >
          {tattoos.map((g) => {
            const first = g.best.item_uid ?? g.item_uids[0]
            return (
              <li key={g.best.id}>
                <TraitCard
                  trait={g.best}
                  title={g.label}
                  sub={seenLine(g)}
                  name={name}
                  acts={acts}
                  onOpen={first ? () => openRun(g.item_uids.length ? g.item_uids : [first], first) : undefined}
                />
              </li>
            )
          })}
        </Section>
      )}
      {mehndi.length > 0 && (
        <Section
          id={`traits-mehndi-${id}`}
          icon={<Hand className="size-4" aria-hidden />}
          title="Mehndi lately"
          note="Short term: mehndi helps only for photos taken within about three weeks."
        >
          {mehndi.map(shortCard)}
        </Section>
      )}
      {nails.length > 0 && (
        <Section
          id={`traits-nails-${id}`}
          icon={<Hand className="size-4" aria-hidden />}
          title="Nails lately"
          note="Short term: nail polish only adds weight to other signs, for photos taken within about two weeks."
        >
          {nails.map(shortCard)}
        </Section>
      )}
      {viewerEl}
    </div>
  )
}
