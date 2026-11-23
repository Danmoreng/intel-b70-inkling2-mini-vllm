## Full-matrix totals

| Release | Input tokens | Output tokens | Summed wave time | Delivered output | Card energy | Peak running / waiting |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Inkling | 2,302,177 | 126,976 | 40.41 min | 52.37 tok/s | 0.121 kWh | 1 / 3 |
| Qwen | 1,960,760 | 126,976 | 40.95 min | 51.68 tok/s | 0.122 kWh | 4 / 3 |

## One client

Each row shows the median over the indicated waves on identical input text. The last row is one observation. K denotes 1,024 tokens.

| Qwen budget | Waves | Actual input, Inkling / Qwen | Prefill tok/s, Inkling / Qwen | Decode tok/s, Inkling / Qwen | First token (s), Inkling / Qwen |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 | 5 | 625 / 496 | 2,563 / 1,122 | 81.60 / 61.16 | 0.29 / 0.45 |
| 2K | 5 | 2,337 / 2,011 | 3,335 / 2,050 | 80.78 / 64.51 | 0.75 / 0.98 |
| 4K | 5 | 5,292 / 4,083 | 3,402 / 2,180 | 81.16 / 60.64 | 1.59 / 1.89 |
| 8K | 5 | 9,526 / 8,178 | 3,265 / 2,263 | 77.89 / 59.61 | 3.24 / 3.63 |
| 16K | 5 | 20,134 / 16,348 | 3,159 / 2,176 | 73.45 / 61.87 | 6.51 / 7.56 |
| 32K | 5 | 39,087 / 32,723 | 3,037 / 2,024 | 71.96 / 59.66 | 13.07 / 16.22 |
| 64K | 3 | 77,274 / 65,520 | 2,822 / 1,705 | 74.46 / 53.26 | 27.47 / 38.52 |
| 128K | 3 | 154,584 / 131,065 | 2,455 / 1,268 | 61.82 / 44.61 | 63.12 / 103.50 |
| 195K | 1 | 230,861 / 199,673 | 2,187 / 1,007 | 51.13 / 41.36 | 105.82 / 198.55 |

## Simultaneous clients

Wave throughput includes queueing, prompt processing and all output. First-token latency is the median of per-wave client medians. The last columns measure aggregate decode while all clients are generating. A dash means there was no measurable overlap interval.

| Budget | Clients | Wave output tok/s, Inkling / Qwen | First token (s), Inkling / Qwen | Wave time (s), Inkling / Qwen | Fully overlapping decode tok/s, Inkling / Qwen |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2K | 2 | 74.59 / 72.06 | 8.18 / 11.98 | 27.46 / 28.42 | — / 111.47 |
| 2K | 3 | 74.06 / 138.66 | 14.63 / 2.58 | 41.48 / 22.16 | — / 161.02 |
| 2K | 4 | 75.13 / 160.53 | 22.27 / 3.28 | 54.52 / 25.52 | — / 190.94 |
| 4K | 2 | 73.45 / 94.98 | 8.36 / 3.48 | 27.88 / 21.56 | — / 117.68 |
| 4K | 3 | 71.10 / 121.40 | 16.70 / 5.04 | 43.21 / 25.31 | — / 156.29 |
| 4K | 4 | 74.98 / 138.63 | 22.43 / 6.19 | 54.63 / 29.55 | — / 189.20 |
| 16K | 2 | 54.84 / 59.50 | 15.87 / 12.12 | 37.35 / 34.42 | — / 107.24 |
| 16K | 3 | 53.31 / 68.88 | 25.52 / 17.60 | 57.62 / 44.60 | — / 140.72 |
| 16K | 4 | 54.38 / 76.50 | 35.25 / 20.89 | 75.32 / 53.54 | — / 173.28 |

## Prefix reuse

The first request seeds each prefix and the next two resend the same text. Earlier matrix scenarios can leave a partial prefix resident. The additional Inkling worker starts with an empty cache.

| Release | Nominal input | Actual input | Cached tokens, first / second / third | Time to first token (s), first / second / third |
| --- | --- | ---: | ---: | ---: |
| Inkling | 16K matrix | 18,187 | 0 / 18,176 / 18,176 | 5.70 / 0.13 / 0.13 |
| Inkling | 64K matrix | 74,690 | 0 / 74,688 / 0 | 27.10 / 21.62 / 27.07 |
| Qwen | 16K matrix | 16,382 | 0 / 16,128 / 16,128 | 7.40 / 0.32 / 0.33 |
| Qwen | 64K matrix | 65,469 | 14,336 / 65,280 / 65,280 | 32.37 / 0.55 / 0.57 |
| Inkling | 64K isolated | 74,690 | 0 / 74,688 / 0 | 47.08 / 18.31 / 27.16 |

| Release | Prefix | Native prefill tok/s, first → resends | Native decode tok/s, first → resends |
| --- | ---: | ---: | ---: |
| Inkling | 16K | 3,214.93 → 129.93 | 81.96 → 80.47 |
| Inkling | 64K | 2,835.36 → 1,449.21 | 69.42 → 58.85 |
| Qwen | 16K | 2,221.37 → 851.47 | 76.00 → 68.31 |
| Qwen | 64K | 1,583.61 → 399.97 | 58.03 → 51.90 |
