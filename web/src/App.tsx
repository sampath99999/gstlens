import { useCallback, useEffect, useRef, useState } from 'react'
import { api, ApiError, type Health, type Sample, type Variant } from './api'
import { ResultPanel, type RunState } from './components/ResultPanel'
import { TEMPLATE_LABELS, VARIANT_LABELS } from './format'

const VARIANTS: Variant[] = ['base', 'finetuned']

function useTheme() {
  const [dark, setDark] = useState(() => {
    const forced = new URLSearchParams(window.location.search).get('theme')
    if (forced === 'dark' || forced === 'light') return forced === 'dark'
    try {
      const saved = localStorage.getItem('gstlens-theme')
      if (saved) return saved === 'dark'
    } catch {
      /* storage unavailable */
    }
    return window.matchMedia('(prefers-color-scheme: dark)').matches
  })
  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    try {
      localStorage.setItem('gstlens-theme', dark ? 'dark' : 'light')
    } catch {
      /* storage unavailable */
    }
  }, [dark])
  return [dark, () => setDark((d) => !d)] as const
}

function HealthBadge({ health, error }: { health: Health | null; error: string | null }) {
  if (error)
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-rose-600 dark:text-rose-400">
        <span className="size-1.5 rounded-full bg-rose-500" /> API offline
      </span>
    )
  if (!health)
    return <span className="text-xs text-stone-400">Connecting...</span>
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-stone-500 dark:text-stone-400">
      <span className="size-1.5 rounded-full bg-emerald-500" />
      {health.base_model} · {health.models.length === 2 ? 'base + LoRA loaded' : 'base only'}
    </span>
  )
}

export default function App() {
  const [dark, toggleTheme] = useTheme()
  const [health, setHealth] = useState<Health | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)
  const [samples, setSamples] = useState<Sample[]>([])
  const [sampleId, setSampleId] = useState<string | null>(null)
  const [text, setText] = useState('')
  const [active, setActive] = useState<Variant>('finetuned')
  const [runs, setRuns] = useState<Record<Variant, RunState>>({
    base: { status: 'idle' },
    finetuned: { status: 'idle' },
  })
  const controllers = useRef<Partial<Record<Variant, AbortController>>>({})

  useEffect(() => {
    api
      .health()
      .then(setHealth)
      .catch((e: ApiError) => setHealthError(e.body.message))
    api
      .samples()
      .then((s) => {
        setSamples(s)
        if (s.length > 0) {
          const wanted = new URLSearchParams(window.location.search).get('sample') ?? 'einvoice'
          const first = s.find((x) => x.template === wanted || x.id === wanted) ?? s[0]
          setSampleId(first.id)
          setText(first.text)
        }
      })
      .catch(() => setSamples([]))
  }, [])

  // ?run=both|base|finetuned starts an extraction once the sample is loaded (handy for demos).
  const autorun = useRef(new URLSearchParams(window.location.search).get('run'))
  useEffect(() => {
    const mode = autorun.current
    if (!mode || !text || !health) return
    autorun.current = null
    if (mode === 'both') void runBoth()
    else if (mode === 'base' || mode === 'finetuned') {
      setActive(mode)
      void run(mode)
    }
  }, [text, health])

  const run = useCallback(
    async (variant: Variant) => {
      controllers.current[variant]?.abort()
      const controller = new AbortController()
      controllers.current[variant] = controller
      const submitted = text
      setRuns((r) => ({ ...r, [variant]: { status: 'loading', startedAt: Date.now() } }))
      try {
        const data = await api.extract(submitted, variant, controller.signal)
        setRuns((r) => ({ ...r, [variant]: { status: 'done', data, text: submitted } }))
      } catch (e) {
        if (controller.signal.aborted) return
        const error =
          e instanceof ApiError ? e.body : { code: 'unknown', message: String(e), details: [] }
        setRuns((r) => ({ ...r, [variant]: { status: 'error', error, text: submitted } }))
      }
    },
    [text],
  )

  async function runBoth() {
    setActive('finetuned')
    await run('finetuned')
    await run('base')
  }

  const available = (v: Variant) => !health || health.models.includes(v)
  const busy = VARIANTS.some((v) => runs[v].status === 'loading')
  const tooShort = text.trim().length < 20

  return (
    <div className="min-h-screen">
      <header className="border-b border-stone-200 bg-white/80 backdrop-blur dark:border-stone-800 dark:bg-stone-950/80">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
          <div className="flex items-center gap-3">
            <img src="/favicon.svg" alt="" className="size-8" />
            <div>
              <h1 className="text-[15px] font-semibold leading-tight">GSTLens</h1>
              <p className="text-xs text-stone-500 dark:text-stone-400">
                Structured fields from Indian GST invoices, base model vs LoRA fine-tune
              </p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <HealthBadge health={health} error={healthError} />
            <button
              onClick={toggleTheme}
              className="grid size-8 place-items-center rounded-lg text-stone-500 ring-1 ring-stone-200 hover:bg-stone-100 dark:text-stone-400 dark:ring-stone-800 dark:hover:bg-stone-900"
              aria-label="Toggle dark mode"
            >
              {dark ? (
                <svg viewBox="0 0 24 24" className="size-4" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="4" />
                  <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
                </svg>
              ) : (
                <svg viewBox="0 0 24 24" className="size-4" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
                </svg>
              )}
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl gap-6 px-4 py-6 sm:px-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,6fr)]">
        <section className="flex flex-col rounded-2xl border border-stone-200 bg-white shadow-sm dark:border-stone-800 dark:bg-stone-900/40">
          <div className="border-b border-stone-200 px-5 pt-4 pb-3 dark:border-stone-800">
            <div className="flex items-baseline justify-between">
              <h2 className="text-sm font-semibold">Invoice text</h2>
              <span className="tabular text-xs text-stone-400">{text.length.toLocaleString()} chars</span>
            </div>
            {samples.length > 0 && (
              <div className="mt-3 flex flex-wrap gap-1.5">
                {samples
                  .filter((s, i, all) => all.findIndex((x) => x.template === s.template) === i)
                  .map((s) => (
                    <button
                      key={s.id}
                      onClick={() => {
                        setSampleId(s.id)
                        setText(s.text)
                      }}
                      className={`rounded-full px-2.5 py-1 text-xs font-medium ring-1 transition-colors ${
                        sampleId === s.id
                          ? 'bg-teal-700 text-white ring-teal-700 dark:bg-teal-500 dark:text-stone-950 dark:ring-teal-500'
                          : 'text-stone-600 ring-stone-200 hover:bg-stone-100 dark:text-stone-300 dark:ring-stone-700 dark:hover:bg-stone-800'
                      }`}
                    >
                      {TEMPLATE_LABELS[s.template] ?? s.template}
                      {s.template === 'einvoice' && (
                        <span className={sampleId === s.id ? 'opacity-80' : 'text-amber-600 dark:text-amber-400'}>
                          {' '}
                          · held out
                        </span>
                      )}
                    </button>
                  ))}
              </div>
            )}
          </div>
          <textarea
            value={text}
            onChange={(e) => {
              setText(e.target.value)
              setSampleId(null)
            }}
            spellCheck={false}
            placeholder="Paste the text of a GST invoice here (OCR output, an email, a Tally print...)"
            className="min-h-[460px] flex-1 resize-none bg-transparent px-5 py-4 font-mono text-[12px] leading-[1.55] whitespace-pre text-stone-800 outline-none placeholder:text-stone-400 dark:text-stone-200"
          />
          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-stone-200 px-5 py-3 dark:border-stone-800">
            <div className="inline-flex rounded-lg bg-stone-100 p-0.5 dark:bg-stone-800" role="tablist">
              {VARIANTS.map((v) => (
                <button
                  key={v}
                  role="tab"
                  aria-selected={active === v}
                  disabled={!available(v)}
                  onClick={() => setActive(v)}
                  className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
                    active === v
                      ? 'bg-white text-stone-900 shadow-sm dark:bg-stone-950 dark:text-stone-100'
                      : 'text-stone-500 hover:text-stone-800 dark:text-stone-400 dark:hover:text-stone-200'
                  }`}
                >
                  {VARIANT_LABELS[v].name}
                </button>
              ))}
            </div>
            <div className="flex gap-2">
              <button
                onClick={runBoth}
                disabled={busy || tooShort || !available('base') || !available('finetuned')}
                className="rounded-lg px-3 py-2 text-sm font-medium text-stone-700 ring-1 ring-stone-300 hover:bg-stone-50 disabled:opacity-40 dark:text-stone-200 dark:ring-stone-700 dark:hover:bg-stone-800"
              >
                Run both
              </button>
              <button
                onClick={() => run(active)}
                disabled={busy || tooShort || !available(active)}
                className="rounded-lg bg-teal-700 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-teal-800 disabled:opacity-40 dark:bg-teal-500 dark:text-stone-950 dark:hover:bg-teal-400"
              >
                {runs[active].status === 'loading' ? 'Extracting...' : `Extract with ${VARIANT_LABELS[active].name.toLowerCase()}`}
              </button>
            </div>
          </div>
        </section>

        <section className="rounded-2xl border border-stone-200 bg-white shadow-sm dark:border-stone-800 dark:bg-stone-900/40">
          <div className="flex items-center justify-between border-b border-stone-200 px-5 py-3 dark:border-stone-800">
            <div className="flex gap-4">
              {VARIANTS.map((v) => {
                const s = runs[v].status
                return (
                  <button
                    key={v}
                    onClick={() => setActive(v)}
                    className={`-mb-3 border-b-2 pb-3 text-sm font-medium transition-colors ${
                      active === v
                        ? 'border-teal-600 text-stone-900 dark:border-teal-400 dark:text-stone-100'
                        : 'border-transparent text-stone-500 hover:text-stone-800 dark:text-stone-400 dark:hover:text-stone-200'
                    }`}
                  >
                    {VARIANT_LABELS[v].name}
                    <span
                      className={`ml-2 inline-block size-1.5 rounded-full align-middle ${
                        s === 'done'
                          ? 'bg-emerald-500'
                          : s === 'error'
                            ? 'bg-rose-500'
                            : s === 'loading'
                              ? 'animate-pulse bg-amber-500'
                              : 'bg-stone-300 dark:bg-stone-700'
                      }`}
                    />
                  </button>
                )
              })}
            </div>
            <span className="text-xs text-stone-400">{VARIANT_LABELS[active].detail}</span>
          </div>
          <div className="p-5">
            <ResultPanel
              variant={active}
              state={runs[active]}
              other={runs[active === 'base' ? 'finetuned' : 'base']}
              currentText={text}
            />
          </div>
        </section>
      </main>
    </div>
  )
}
