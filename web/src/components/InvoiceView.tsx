import type { ReactNode } from 'react'
import type { Check, Invoice } from '../api'
import { longDate, money, quantity } from '../format'

type Compare = Invoice | undefined

function differs(a: unknown, b: unknown): boolean {
  if (typeof a === 'number' && typeof b === 'number') return Math.abs(a - b) > 0.005
  return String(a ?? '').trim().toLowerCase() !== String(b ?? '').trim().toLowerCase()
}

function Diff({ show, other, children }: { show: boolean; other: string; children: ReactNode }) {
  if (!show) return <>{children}</>
  return (
    <span
      className="rounded bg-amber-100 px-1 ring-1 ring-amber-300 dark:bg-amber-500/15 dark:ring-amber-500/40"
      title={`Other model: ${other}`}
    >
      {children}
    </span>
  )
}

function Field({
  label,
  value,
  other,
  mono,
}: {
  label: string
  value: ReactNode
  other?: { show: boolean; text: string }
  mono?: boolean
}) {
  return (
    <div className="min-w-0">
      <dt className="text-[11px] font-medium uppercase tracking-wider text-stone-500 dark:text-stone-400">
        {label}
      </dt>
      <dd className={`mt-1 truncate text-sm text-stone-900 dark:text-stone-100 ${mono ? 'font-mono' : ''}`}>
        <Diff show={other?.show ?? false} other={other?.text ?? ''}>
          {value}
        </Diff>
      </dd>
    </div>
  )
}

function PartyCard({
  role,
  name,
  gstin,
  otherName,
  otherGstin,
  compare,
}: {
  role: string
  name: string
  gstin: string | null
  otherName?: string
  otherGstin?: string | null
  compare: boolean
}) {
  return (
    <div className="rounded-xl border border-stone-200 bg-stone-50/60 p-4 dark:border-stone-800 dark:bg-stone-900/60">
      <p className="text-[11px] font-semibold uppercase tracking-wider text-teal-700 dark:text-teal-400">{role}</p>
      <p className="mt-1.5 truncate font-medium">
        <Diff show={compare && differs(name, otherName)} other={otherName ?? ''}>
          {name}
        </Diff>
      </p>
      <p className="mt-1 font-mono text-xs text-stone-600 dark:text-stone-400">
        <Diff show={compare && differs(gstin, otherGstin)} other={otherGstin ?? 'none'}>
          {gstin ?? 'Unregistered'}
        </Diff>
      </p>
    </div>
  )
}

const CHECK_LABELS: Record<string, string> = {
  seller_gstin_checksum: 'Seller GSTIN checksum',
  buyer_gstin_checksum: 'Buyer GSTIN checksum',
  items_sum_to_taxable: 'Items add up to taxable value',
  total_matches_tax: 'Taxable + tax = total',
  single_tax_regime: 'One tax regime',
}

export function Checks({ checks }: { checks: Check[] }) {
  return (
    <ul className="flex flex-wrap gap-2">
      {checks.map((c) => (
        <li
          key={c.name}
          title={c.detail}
          className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ring-1 ${
            c.passed
              ? 'bg-emerald-50 text-emerald-800 ring-emerald-200 dark:bg-emerald-500/10 dark:text-emerald-300 dark:ring-emerald-500/30'
              : 'bg-rose-50 text-rose-800 ring-rose-200 dark:bg-rose-500/10 dark:text-rose-300 dark:ring-rose-500/30'
          }`}
        >
          <span aria-hidden className={`size-1.5 rounded-full ${c.passed ? 'bg-emerald-500' : 'bg-rose-500'}`} />
          {CHECK_LABELS[c.name] ?? c.name}
        </li>
      ))}
    </ul>
  )
}

export function InvoiceView({ invoice, compareWith }: { invoice: Invoice; compareWith: Compare }) {
  const cmp = compareWith !== undefined
  const o = compareWith
  const diff = (key: keyof Invoice, fmt: (v: never) => string = String) => ({
    show: cmp && differs(invoice[key], o?.[key]),
    text: o ? fmt(o[key] as never) : '',
  })
  const totals: { key: 'taxable_value' | 'cgst' | 'sgst' | 'igst'; label: string }[] = [
    { key: 'taxable_value', label: 'Taxable value' },
    { key: 'cgst', label: 'CGST' },
    { key: 'sgst', label: 'SGST' },
    { key: 'igst', label: 'IGST' },
  ]

  return (
    <div className="space-y-5">
      <div className="grid gap-3 sm:grid-cols-2">
        <PartyCard
          role="Seller"
          name={invoice.seller_name}
          gstin={invoice.seller_gstin}
          otherName={o?.seller_name}
          otherGstin={o?.seller_gstin}
          compare={cmp}
        />
        <PartyCard
          role="Buyer"
          name={invoice.buyer_name}
          gstin={invoice.buyer_gstin}
          otherName={o?.buyer_name}
          otherGstin={o?.buyer_gstin}
          compare={cmp}
        />
      </div>

      <dl className="grid grid-cols-3 gap-4 rounded-xl border border-stone-200 px-4 py-3 dark:border-stone-800">
        <Field label="Invoice no." value={invoice.invoice_number} other={diff('invoice_number')} mono />
        <Field
          label="Date"
          value={longDate(invoice.invoice_date)}
          other={diff('invoice_date', (v: string) => longDate(v))}
        />
        <Field label="Place of supply" value={invoice.place_of_supply} other={diff('place_of_supply')} />
      </dl>

      <div className="overflow-x-auto rounded-xl border border-stone-200 dark:border-stone-800">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-stone-200 bg-stone-50 text-left text-[11px] uppercase tracking-wider text-stone-500 dark:border-stone-800 dark:bg-stone-900 dark:text-stone-400">
              <th className="px-3 py-2 font-medium">Item</th>
              <th className="px-3 py-2 font-medium">HSN</th>
              <th className="px-3 py-2 text-right font-medium">Qty</th>
              <th className="px-3 py-2 text-right font-medium">Rate</th>
              <th className="px-3 py-2 text-right font-medium">Amount</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-stone-100 dark:divide-stone-800/70">
            {invoice.line_items.map((item, i) => {
              const other = o?.line_items[i]
              const d = (a: unknown, b: unknown) => cmp && (other === undefined || differs(a, b))
              return (
                <tr key={i} className="align-top">
                  <td className="px-3 py-2">
                    <Diff show={d(item.description, other?.description)} other={other?.description ?? 'missing'}>
                      {item.description}
                    </Diff>
                  </td>
                  <td className="px-3 py-2 font-mono text-xs text-stone-600 dark:text-stone-400">
                    <Diff show={d(item.hsn, other?.hsn)} other={other?.hsn ?? 'missing'}>
                      {item.hsn}
                    </Diff>
                  </td>
                  <td className="tabular px-3 py-2 text-right">
                    <Diff show={d(item.quantity, other?.quantity)} other={String(other?.quantity ?? 'missing')}>
                      {quantity(item.quantity)}
                    </Diff>
                  </td>
                  <td className="tabular px-3 py-2 text-right">
                    <Diff show={d(item.rate, other?.rate)} other={other ? money(other.rate) : 'missing'}>
                      {money(item.rate)}
                    </Diff>
                  </td>
                  <td className="tabular px-3 py-2 text-right font-medium">
                    <Diff show={d(item.amount, other?.amount)} other={other ? money(other.amount) : 'missing'}>
                      {money(item.amount)}
                    </Diff>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
        {cmp && o && o.line_items.length !== invoice.line_items.length && (
          <p className="border-t border-stone-100 px-3 py-2 text-xs text-amber-700 dark:border-stone-800 dark:text-amber-400">
            The other model found {o.line_items.length} line item{o.line_items.length === 1 ? '' : 's'}.
          </p>
        )}
      </div>

      <div className="ml-auto w-full max-w-xs space-y-1.5 text-sm">
        {totals.map(({ key, label }) => (
          <div key={key} className="flex justify-between text-stone-600 dark:text-stone-400">
            <span>{label}</span>
            <span className="tabular">
              <Diff show={cmp && differs(invoice[key], o?.[key])} other={o ? money(o[key]) : ''}>
                {money(invoice[key])}
              </Diff>
            </span>
          </div>
        ))}
        <div className="flex justify-between border-t border-stone-200 pt-2 text-base font-semibold dark:border-stone-800">
          <span>Total</span>
          <span className="tabular">
            <Diff show={cmp && differs(invoice.total, o?.total)} other={o ? money(o.total) : ''}>
              {money(invoice.total)}
            </Diff>
          </span>
        </div>
      </div>
    </div>
  )
}
