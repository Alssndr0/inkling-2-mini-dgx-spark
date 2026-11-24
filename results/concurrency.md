# Decode by workload and concurrency (2026-11-24)

Matched 2x2 per profile: prose / agentic coding x 1 / 4 streams, same requests and seeds everywhere.

- **Prose:** the bench panel's 4 prose prompts (TCP explainer, short story, B-tree vs LSM comparison, 1929 summary),
  T=1, max 1,024 tokens, default reasoning. 1 stream: one at a time (2 passes). 4 streams: all 4 at once (3 passes).
- **Agentic coding:** the 177 recorded OpenCode / Pi requests (25 sessions), replayed turn by turn with their recorded
  seeds and sampling; 1 worker, or 4 workers replaying 4 sessions concurrently.
- **Decode throughput** of a k-stream cell: tokens generated while exactly k streams were decoding (after each
  request's first token), divided by the length of those intervals. Prefill never falls inside such an interval.
- **Tokens per step:** 1 + accepted / drafts from vLLM's speculative-decoding counters over the cell.

| profile | prose 1 stream | prose 4 streams total / per stream | agentic 1 stream | agentic 4 streams total / per stream | tokens per step, prose / agentic |
|---|---:|---:|---:|---:|---:|
| stock vLLM | 34.6 | 81.6 / 20.4 | 60.7 | 125.7 / 31.4 | 2.15 / 3.99 |
| `lossless` | 39.3 | 90.6 / 22.6 | 70.4 | 140.1 / 35.0 | 2.06 / 3.96 |
| `fast` | 46.2 | 89.8 / 22.4 | 80.4 | 141.1 / 35.3 | 2.11 / 4.00 |
| `fastest` | 47.0 | 90.0 / 22.5 | 81.0 | 144.2 / 36.1 | 2.11 / 3.98 |

All agentic cells: 177 requests, 0 errors. The headline table's "Agentic decode" row (P1) uses the lab replay
harness (per-request decode time, no think time) on the same 177 requests, so its values are close to but not the
same as the "agentic coding, 1 stream" cell here.

Why 4 streams give about 2x, not 4x: speculative decoding verifies 8 tokens per stream per step; 4 streams send 32
tokens through each MoE layer, which together read most of the 128 experts (8 tokens already read about 40 %), so a
4-stream step reads roughly twice the expert bytes of a 1-stream step. The int8 projection and split-K decode
kernels of `fast` and `fastest` act on up to 16 rows (2 concurrent requests); larger batches use the stock kernels.
