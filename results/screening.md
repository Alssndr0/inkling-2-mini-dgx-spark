# Per-change screening results

Each change was measured against the configuration before it, with control boots of that configuration in the same
session (numbers after "vs"). Lean screening boots: an 18-request agentic replay at temperature 1 and at 0, the 13
real agent prompts of 7K-80K tokens for prompt processing, a short-prompt panel, and the teacher-forced KL probe.
Dates are 2026-11-22 / 23 (UTC).

| change | result | KL vs parent | class |
|---|---|---:|---|
| Persistent caches | startup 194 s vs 368-370 s | 0 | bit-exact |
| + eager weight loading | startup 93.4 s vs 194 s | 0 | bit-exact |
| Pruned draft vocabulary (32,768) | replay 70.80 vs 64.20 / 64.22 tok/s (+10.3 %); greedy replay 74.94 vs 68.91 / 68.88 (identical text); short prompts 42.81 vs 40.75-40.88 | 0 | draft-side exact |
| Expert GEMM tiles, prefill | real-prompt prefill 3,061 vs 2,699 / 2,752 tok/s (+11-13 %) | 0 | bit-exact |
| Expert GEMM tiles, decode | replay 64.85 vs 64.22 / 63.97 tok/s (+1.0-1.4 %), identical text | 0 | bit-exact |
| 8,192-token prefill chunks | real-prompt prefill 3,266 (+3.1-4.7 %); prompts over 30K 3,092 vs 2,829 / 2,877 (+7.5-9.3 %) | 0 (probe documents fit one chunk) | rounding-level |
| Native MXFP8 MMA, prefill tiles | real-prompt prefill 3,248 (+2.5-4.1 %) | 0.00003 (top-1 99.83 %) | rounding-level |
| Humming NVFP4 expert backend | real-prompt prefill 4,194 vs 3,660 / 3,666 (+14.5 %); replay +2.0-2.6 %; identical replies | 0 | bit-exact |
| KV pool 64 GiB, GPU utilization 0.85 | identical replies; peak memory 105 -> 94 GiB; removes the host-memory pressure that made prefill readings drift by up to 10 % between boots | 0 | bit-exact |
| 16x512 GEMV tiles | decode step 57.0 vs 57.1 / 57.1 ms; identical replies | 0 | bit-exact |
| int8 draft-only weights | decode step 54.9 vs 57.1 ms (-3.9 %); short prompts 47.58 vs 44.25 (+7.5 %); identical greedy replies | 0 | draft-side exact |
| Fused attention gate (on the lossless line) | real-prompt prefill 4,229 vs 4,160 (+1.7 %); identical replies | 0 | bit-exact |
| int8 attention projections and LM head (decode) | decode step 51.4 vs 58.2 ms (-11.7 %); greedy replay +10.2 % | 0.0075 (top-1 96.8 %) | output-changing: quality battery passed |
| Far-block attention path | real-prompt prefill 4,440 vs 4,263 / 4,259 (+4.2 %); prompts over 30K +5.3 % | 0.00071 (top-1 99.52 %) | rounding-level |
| K and V projections in one launch | decode step 48.6 vs 48.8 ms (-0.4 %); identical replies | 0 | bit-exact |
| Fused attention gate (on the fast line) | GPU time of an 8K chunk -1.7 %, at 32K depth -3.9 %; identical replies | 0 | bit-exact |
| Fused loop controller (prefill) | real-prompt prefill 4,647 vs 4,358 (+6.6 %); prompts over 30K +6.5 % | 0.0035 (top-1 97.8 %) | output-changing: quality battery and investigation passed |
| Split-K small matmuls (decode) | decode step 47.8 vs 48.6 ms (-1.6 %) | 0 on the probe (decode path only) | output-changing: quality battery and investigation passed |

Rounding-level gate: KL <= 0.002 mean, top-1 agreement >= 99.5 %, NLL change within 0.002. Output-changing changes
need the full quality battery ([quality-battery.md](quality-battery.md)).
