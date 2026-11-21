const inr = new Intl.NumberFormat('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
const qty = new Intl.NumberFormat('en-IN', { maximumFractionDigits: 3 })

export const money = (value: number) => `₹${inr.format(value)}`
export const quantity = (value: number) => qty.format(value)

export function longDate(iso: string): string {
  const d = new Date(`${iso}T00:00:00`)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' })
}

export function seconds(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`
}

export const TEMPLATE_LABELS: Record<string, string> = {
  classic: 'Classic',
  tally: 'Tally print',
  key_value: 'Key-value',
  thermal: 'POS receipt',
  email: 'Email',
  csv: 'Spreadsheet',
  einvoice: 'e-Invoice',
}

export const VARIANT_LABELS = {
  base: { name: 'Base', detail: 'Qwen2.5-0.5B, 2-shot prompt' },
  finetuned: { name: 'Fine-tuned', detail: 'Qwen2.5-0.5B + LoRA' },
} as const
