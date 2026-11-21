import type { ReactNode } from 'react'
import type { ApiErrorBody, ExtractResponse, Variant } from '../api'
import { seconds, VARIANT_LABELS } from '../format'
import { Checks, InvoiceView } from './InvoiceView'

export type RunState =
  | { status: 'idle' }
  | { status: 'loading'; startedAt: number }
  | { status: 'done'; data: ExtractResponse; text: string }
  | { status: 'error'; error: ApiErrorBody; text: string }

function Pill({ children, tone = 'stone' }: { children: ReactNode; tone?: 'stone' | 'amber' | 'teal' }) {
  const tones = {
    stone: 'bg-stone-100 text-stone-700 ring-stone-200 dark:bg-stone-800 dark:text-stone-300 dark:ring-stone-700',
    amber: 'bg-amber-50 text-amber-800 ring-amber-200 dark:bg-amber-500/10 dark:text-amber-300 dark:ring-amber-500/30',
    teal: 'bg-teal-50 text-teal-800 ring-teal-200 dark:bg-teal-500/10 dark:text-teal-300 dark:ring-teal-500/30',
  }
  return (
    <span className={`inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-xs font-medium ring-1 ${tones[tone]}`}>
      {children}
    </span>
  )
}

function EmptyState() {
  return (
    <div className="flex h-full min-h-[420px] flex-col items-center justify-center px-8 text-center">
      <div className="grid size-12 place-items-center rounded-2xl bg-teal-50 text-teal-700 ring-1 ring-teal-100 dark:bg-teal-500/10 dark:text-teal-300 dark:ring-teal-500/20">
        <svg viewBox="0 0 24 24" className="size-6" fill="none" stroke="currentColor" strokeWidth="1.8">
          <path d="M7 3h7l5 5v13H7z" strokeLinejoin="round" />
          <path d="M14 3v5h5M10 13h6M10 17h4" strokeLinecap="round" />
        </svg>
      </div>
      <h3 className="mt-4 font-semibold">No extraction yet</h3>
      <p className="mt-1 max-w-sm text-sm text-stone-500 dark:text-stone-400">
        Pick a sample or paste an invoice on the left, then run it through either model. Run both to see
        where they disagree.
      </p>
    </div>
  )
}

function Loading({ variant }: { variant: Variant }) {
  return (
    <div className="space-y-4" aria-busy="true">
      <p className="text-sm text-stone-500 dark:text-stone-400">
        Running {VARIANT_LABELS[variant].detail}. Greedy decoding on a small GPU takes 5 to 20 seconds.
      </p>
      <div className="grid gap-3 sm:grid-cols-2">
        {[0, 1].map((i) => (
          <div key={i} className="h-20 animate-pulse rounded-xl bg-stone-100 dark:bg-stone-800/70" />
        ))}
      </div>
      <div className="h-14 animate-pulse rounded-xl bg-stone-100 dark:bg-stone-800/70" />
      <div className="h-40 animate-pulse rounded-xl bg-stone-100 dark:bg-stone-800/70" />
      <div className="ml-auto h-28 w-64 animate-pulse rounded-xl bg-stone-100 dark:bg-stone-800/70" />
    </div>
  )
}

function RawOutput({ text }: { text: string }) {
  return (
    <details className="group rounded-xl border border-stone-200 dark:border-stone-800">
      <summary className="cursor-pointer select-none px-4 py-2.5 text-sm font-medium text-stone-600 marker:text-stone-400 dark:text-stone-300">
        Raw model output
      </summary>
      <pre className="max-h-72 overflow-auto border-t border-stone-200 bg-stone-50 px-4 py-3 font-mono text-xs leading-relaxed whitespace-pre-wrap break-all text-stone-700 dark:border-stone-800 dark:bg-stone-900 dark:text-stone-300">
        {text}
      </pre>
    </details>
  )
}

function ErrorView({ error }: { error: ApiErrorBody }) {
  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 dark:border-rose-500/30 dark:bg-rose-500/10">
        <p className="text-sm font-semibold text-rose-800 dark:text-rose-300">{error.message}</p>
        <p className="mt-0.5 font-mono text-xs text-rose-700/80 dark:text-rose-300/70">{error.code}</p>
        {error.details.length > 0 && (
          <ul className="mt-3 space-y-1 text-xs text-rose-900 dark:text-rose-200">
            {error.details.slice(0, 8).map((d, i) => (
              <li key={i} className="font-mono">
                {Array.isArray(d.loc) && d.loc.length > 0 ? `${d.loc.join('.')}: ` : ''}
                {String(d.msg ?? JSON.stringify(d))}
              </li>
            ))}
          </ul>
        )}
      </div>
      {error.raw_output && <RawOutput text={error.raw_output} />}
    </div>
  )
}

export function ResultPanel({
  variant,
  state,
  other,
  currentText,
}: {
  variant: Variant
  state: RunState
  other: RunState
  currentText: string
}) {
  if (state.status === 'idle') return <EmptyState />
  if (state.status === 'loading') return <Loading variant={variant} />
  const stale = state.text !== currentText
  if (state.status === 'error') return <ErrorView error={state.error} />

  const { data } = state
  const compare = other.status === 'done' && other.text === state.text ? other.data.invoice : undefined
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-2">
        <Pill tone="teal">{seconds(data.latency_ms)}</Pill>
        <Pill>
          {data.prompt_tokens} in / {data.completion_tokens} out tokens
        </Pill>
        {data.json_repaired && <Pill tone="amber">JSON repaired</Pill>}
        {compare && <Pill tone="amber">Highlighted fields differ from the other model</Pill>}
        {stale && <Pill tone="amber">Text changed since this run</Pill>}
      </div>
      <InvoiceView invoice={data.invoice} compareWith={compare} />
      <div>
        <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">
          Consistency checks
        </h3>
        <Checks checks={data.checks} />
      </div>
      <RawOutput text={data.raw_output} />
    </div>
  )
}
