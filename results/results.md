### Held-out layout (e-invoice print, never seen in training) (n=100)

| System | Raw valid JSON | JSON after repair | Schema valid | Field F1 | Line-item F1 | All items exact | Whole invoice exact | Latency p50 | Latency p95 |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B, zero-shot | 91.0% | 92.0% | 3.0% | 0.568 | 0.235 | 9.0% | 0.0% | 12.7s | 19.3s |
| Qwen2.5-0.5B, 2-shot | 81.0% | 86.0% | 21.0% | 0.805 | 0.205 | 9.0% | 3.0% | 15.3s | 30.0s |
| Qwen2.5-0.5B + LoRA | 100.0% | 100.0% | 99.0% | 0.901 | 0.978 | 95.0% | 44.0% | 11.1s | 15.1s |
| qwen2.5:3b (Ollama Q4), 2-shot | 97.0% | 97.0% | 18.0% | 0.928 | 0.881 | 67.0% | 50.0% | 14.5s | 22.5s |

### Seen layouts (same six templates as training, new invoices) (n=100)

| System | Raw valid JSON | JSON after repair | Schema valid | Field F1 | Line-item F1 | All items exact | Whole invoice exact | Latency p50 | Latency p95 |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B, zero-shot | 92.0% | 94.0% | 37.0% | 0.766 | 0.343 | 22.0% | 0.0% | 14.0s | 20.3s |
| Qwen2.5-0.5B, 2-shot | 86.0% | 86.0% | 10.0% | 0.806 | 0.534 | 28.0% | 5.0% | 12.9s | 20.5s |
| Qwen2.5-0.5B + LoRA | 100.0% | 100.0% | 100.0% | 0.992 | 1.000 | 100.0% | 95.0% | 10.8s | 15.2s |
| qwen2.5:3b (Ollama Q4), 2-shot | 100.0% | 100.0% | 81.0% | 0.990 | 0.776 | 71.0% | 66.0% | 11.4s | 16.1s |

### Exact match per field (held-out / seen)

| Field | Qwen2.5-0.5B, zero-shot | Qwen2.5-0.5B, 2-shot | Qwen2.5-0.5B + LoRA | qwen2.5:3b (Ollama Q4), 2-shot |
|---|---|---|---|---|
| `seller_name` | 58.0% / 80.0% | 76.0% / 80.0% | 84.0% / 99.0% | 86.0% / 100.0% |
| `seller_gstin` | 59.0% / 82.0% | 75.0% / 80.0% | 84.0% / 99.0% | 86.0% / 100.0% |
| `buyer_name` | 40.0% / 81.0% | 79.0% / 80.0% | 85.0% / 99.0% | 86.0% / 99.0% |
| `buyer_gstin` | 2.0% / 18.0% | 80.0% / 80.0% | 84.0% / 99.0% | 80.0% / 96.0% |
| `invoice_number` | 85.0% / 90.0% | 70.0% / 82.0% | 100.0% / 99.0% | 97.0% / 100.0% |
| `invoice_date` | 3.0% / 56.0% | 84.0% / 83.0% | 100.0% / 100.0% | 97.0% / 100.0% |
| `place_of_supply` | 89.0% / 80.0% | 79.0% / 73.0% | 100.0% / 100.0% | 97.0% / 100.0% |
| `taxable_value` | 4.0% / 77.0% | 49.0% / 71.0% | 58.0% / 100.0% | 87.0% / 99.0% |
| `cgst` | 73.0% / 59.0% | 74.0% / 62.0% | 97.0% / 99.0% | 94.0% / 96.0% |
| `sgst` | 79.0% / 73.0% | 74.0% / 65.0% | 97.0% / 99.0% | 92.0% / 97.0% |
| `igst` | 43.0% / 67.0% | 69.0% / 58.0% | 92.0% / 98.0% | 94.0% / 97.0% |
| `total` | 76.0% / 81.0% | 81.0% / 63.0% | 100.0% / 100.0% | 94.0% / 99.0% |

### Whole invoice exact match by layout

| Layout | n | Qwen2.5-0.5B, zero-shot | Qwen2.5-0.5B, 2-shot | Qwen2.5-0.5B + LoRA | qwen2.5:3b (Ollama Q4), 2-shot |
|---|---|---|---|---|---|
| classic | 17 | 0.0% | 0.0% | 100.0% | 88.2% |
| csv | 16 | 0.0% | 6.2% | 93.8% | 56.2% |
| einvoice | 100 | 0.0% | 3.0% | 44.0% | 50.0% |
| email | 16 | 0.0% | 12.5% | 100.0% | 31.2% |
| key_value | 17 | 0.0% | 11.8% | 94.1% | 88.2% |
| tally | 17 | 0.0% | 0.0% | 82.4% | 70.6% |
| thermal | 17 | 0.0% | 0.0% | 100.0% | 58.8% |
