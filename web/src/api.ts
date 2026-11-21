export type Variant = 'base' | 'finetuned'

export interface LineItem {
  description: string
  hsn: string
  quantity: number
  rate: number
  amount: number
}

export interface Invoice {
  seller_name: string
  seller_gstin: string
  buyer_name: string
  buyer_gstin: string | null
  invoice_number: string
  invoice_date: string
  place_of_supply: string
  line_items: LineItem[]
  taxable_value: number
  cgst: number
  sgst: number
  igst: number
  total: number
}

export interface Check {
  name: string
  passed: boolean
  detail: string
}

export interface ExtractResponse {
  model: Variant
  invoice: Invoice
  checks: Check[]
  latency_ms: number
  prompt_tokens: number
  completion_tokens: number
  json_repaired: boolean
  raw_output: string
}

export interface ApiErrorBody {
  code: string
  message: string
  details: Record<string, unknown>[]
  raw_output?: string | null
  latency_ms?: number | null
}

export interface Health {
  status: 'ok'
  backend: string
  base_model: string
  models: Variant[]
}

export interface Sample {
  id: string
  template: string
  text: string
}

export class ApiError extends Error {
  readonly body: ApiErrorBody
  readonly status: number

  constructor(status: number, body: ApiErrorBody) {
    super(body.message)
    this.status = status
    this.body = body
  }
}

const BASE = import.meta.env.VITE_API_BASE ?? '/api'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${BASE}${path}`, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...init?.headers },
    })
  } catch {
    throw new ApiError(0, {
      code: 'network_error',
      message: 'Could not reach the API. Is the backend running on port 8000?',
      details: [],
    })
  }
  const data = await res.json().catch(() => null)
  if (!res.ok) {
    const body: ApiErrorBody = data?.error ?? {
      code: `http_${res.status}`,
      message: res.statusText || 'Request failed',
      details: [],
    }
    throw new ApiError(res.status, body)
  }
  return data as T
}

export const api = {
  health: () => request<Health>('/health'),
  samples: () => request<Sample[]>('/samples'),
  extract: (text: string, model: Variant, signal?: AbortSignal) =>
    request<ExtractResponse>('/extract', {
      method: 'POST',
      body: JSON.stringify({ text, model }),
      signal,
    }),
}
