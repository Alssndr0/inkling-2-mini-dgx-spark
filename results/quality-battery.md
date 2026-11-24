# Quality battery

All runs on the same DGX Spark, same image, same weights. "Stock" is the image's default launch (s0).
Thinking off unless stated. Pass rules were fixed before the runs (see "How it was measured" in the README).

## Output-changing steps: full battery

| test | stock | int8 attention + head (fin-w8, the int8 step of `fast`) | `fastest` bundle (dsm-c2, same outputs as `fastest`) | rule |
|---|---:|---:|---:|---|
| Q6 HumanEval+ / MBPP+ plus tests (542 tasks, greedy) | 146 + 288 = **434** | 143 + 291 = **434** (only stock 14, only fin-w8 14, McNemar p 1.000) | 144 + 294 = **438** (only stock 16, only fastest 20, McNemar p 0.618) | not lower by more than 5, p >= 0.05 |
| Q6 base tests | 153 + 347 = 500 | 152 + 345 = 497 | 152 + 354 = 506 | reported |
| Q4 400 exact-answer problems (two runs) | 358 / 356 (mean 357) | 356 / 355 (mean 355.5) | 360 / 352 (mean 356.0) | mean within 2 of stock |
| Q2 language guard (400 sampled replies, English and Italian, thinking off and on): CJK replies / wrong language | 0 / 0 | 0 / 0 | 0 / 0 | no new CJK |
| Q3 agentic replay rows (P1, 177 requests): tool-call arguments valid | 169 / 169 | 173 / 173 | 172 / 172 | not worse |
| Q3: CJK replies / empty replies / cut at the token cap | 0 / 0 / 14 | 0 / 0 / 16 | 0 / 0 / 13 | not worse |
| Needle retrieval at ~128K and ~244K tokens | 3/3, 3/3 | 3/3, 3/3 | 3/3, 3/3 | pass |
| KL vs stock (teacher-forced, 11,214 positions) / top-1 agreement | 0 / 100 % | 0.0075 / 96.8 % | 0.0074 / 96.7 % | reported |

## Rounding-level and exact steps: KL gate

| profile / step | KL vs its parent | top-1 agreement | dNLL | class |
|---|---:|---:|---:|---|
| `lossless` (fin-g) vs stock | 0.00003 | 99.83 % | +0.0001 | rounding-level (MXFP8 MMA on prefill tiles, 8K chunks); everything else bit-exact or draft-side exact |
| `lossless` (fin-g) vs fin (without the fused gate) | 0 | 100 % | 0 | bit-exact (same replies, same fingerprints) |
| far-block attention path vs fin-w8 | 0.00071 | 99.52 % | -0.0001 | rounding-level |
| `fast` (gsf-f2) vs fin2 | 0 | 100 % | 0 | bit-exact |
| `fastest` (fin3c) vs dsm-c2 | 0 | 100 % | 0 | bit-exact |
| gate: KL <= 0.002, top-1 >= 99.5 %, abs(dNLL) <= 0.002 | | | | |

## Draft side (speculative decoding)

The drafter changes (pruned draft vocabulary, int8 draft-only weights) cannot change the output distribution:
rejection sampling accepts a draft token only with the target model's own probability. Checked with the Q0
output-distribution test (300 samples x 6 tokens on 8 prompts, against stock): minimum Bonferroni p 0.731 / 0.594
on the lossless line (threshold 0.01). T=0 replies were identical with and without each drafter change.
