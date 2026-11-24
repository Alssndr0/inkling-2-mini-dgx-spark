# Confirmation run, 2026-11-24

This repository's `serve.sh`, from a clean clone, one profile at a time on one DGX Spark; stock measured the same
day by the same harness (stock P1: 2026-11-22, same requests and seeds). Raw values: `confirmation.json`.

| metric | stock | lossless | fast | fastest |
|---|---:|---:|---:|---:|
| Agentic decode, P1 (tok/s; mean of the runs) | 62.3 | 74.5 | 83.6 | 81.5 |
| Decode step, greedy agentic replay (ms) | 65.4 | 54.8 | 48.6 | 47.4 |
| Single-stream decode, short prompts (tok/s) | 40.7 | 47.7 | 49.9 | 48.6 |
| 4 streams, total (tok/s) | 75.3 | 94.2 | 91.3 | 92.9 |
| Real agent prompts, 7K-80K tokens (tok/s) | 2,590 | 4,225 | 4,519 | 4,800 |
| Documents 4K / 32K / 128K (tok/s) | 1,404 / 2,483 / 2,189 | 3,976 / 4,024 / 2,932 | 4,161 / 4,226 / 3,347 | 4,265 / 4,508 / 3,532 |
| Time to first token, short prompt (ms) | 221 | 173 | 163 | 162 |
| Startup, warm caches (s) | 360 | 79 | 78 | 76 |
| Peak memory (GiB) | 105 | 94 | 95 | 95 |

P1 per run vs stock (paired by request, 95 % CI over sessions):

- lossless: 74.50 tok/s, +19.54 % [+15.72, +24.02], tokens per step 4.131 (stock 4.084), tool-call arguments valid 167/167, cut at the cap 22 (stock 14)
- fast: 83.61 tok/s, +34.16 % [+30.13, +38.83], tokens per step 4.080 (stock 4.084), tool-call arguments valid 162/162, cut at the cap 18 (stock 14)
- fast: 83.58 tok/s, +34.10 % [+30.10, +38.74], tokens per step 4.080 (stock 4.084), tool-call arguments valid 162/162, cut at the cap 18 (stock 14)
- fastest: 80.82 tok/s, +29.68 % [+25.30, +33.84], tokens per step 3.945 (stock 4.084), tool-call arguments valid 168/168, cut at the cap 20 (stock 14)
- fastest: 82.09 tok/s, +31.70 % [+27.22, +35.92], tokens per step 3.945 (stock 4.084), tool-call arguments valid 168/168, cut at the cap 20 (stock 14)

Cold start of the default profile with an empty cache directory: ready after 224 s.
Teacher-forced KL of each profile against the configuration measured during tuning: 0 (identical).
