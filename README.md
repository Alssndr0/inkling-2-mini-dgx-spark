# Inkling 2 Mini on DGX Spark

**Thinking Machines' Inkling 2 Mini on one NVIDIA DGX Spark: 31 % faster agentic decoding and 85 % faster
prompt processing than the stock vLLM launch, with a quality check for every step.**

Three profiles, one script. Each profile is stock vLLM plus a few kernel files mounted over the image's own, all
checked by SHA-256 before launch. Every number below was measured on the same machine with the same weights.

| | stock vLLM | `lossless` | `fast` | **`fastest`** (default) |
|---|---:|---:|---:|---:|
| Agentic decode, 177 real coding-agent requests (tok/s) | 62.3 | 74.5 | 83.6 | **81.5** |
| &nbsp;&nbsp;vs stock | | +20 % | +34 % | **+31 %** |
| Decode step, 18 agentic requests decoded greedily (ms per step) | 65.4 | 54.8 | 48.6 | 47.4 |
| Decode, agentic coding, 1 stream (tok/s) | 60.7 | 70.4 | 80.4 | **81.0** |
| Decode, agentic coding, 4 streams: total / per stream (tok/s) | 125.7 / 31.4 | 140.1 / 35.0 | 141.1 / 35.3 | **144.2 / 36.1** |
| Decode, prose, 1 stream (tok/s) | 34.6 | 39.3 | 46.2 | **47.0** |
| Decode, prose, 4 streams: total / per stream (tok/s) | 81.6 / 20.4 | 90.6 / 22.6 | 89.8 / 22.4 | **90.0 / 22.5** |
| Prompt processing, real agent prompts of 7K-80K tokens (tok/s) | 2,590 | 4,225 | 4,519 | **4,800** |
| Prompt processing, 4K / 32K / 128K-token documents (tok/s) | 1,404 / 2,483 / 2,189 | 3,976 / 4,024 / 2,932 | 4,161 / 4,226 / 3,347 | 4,265 / 4,508 / 3,532 |
| Time to first token, short prompt (ms) | 221 | 173 | 163 | 162 |
| Startup with warm caches (s) | 360 | 79 | 78 | 76 |
| Peak memory (GiB of 121) | 105 | 94 | 95 | 95 |
| Output vs stock | same model | same outputs up to rounding | int8 decode weights | int8 + two rounding changes |

Agentic decode is the main metric: it replays 177 recorded OpenCode and Pi requests at temperature 1 and counts
generated tokens per second (mean of two runs for `fast` and `fastest`). `fast` and `fastest` write different,
equally valid sampled replies, and how many draft tokens are accepted per step depends on the reply text: in these
runs `fastest`'s replies accepted 3 % fewer per step (3.95 vs 4.08) although its decode step is shorter, and in the
tuning runs the order was reversed (84.0 vs 81.6). Treat the two as equal on decode; `fastest` gains in prompt
processing. Every run with its 95 % CI: [results/headline.md](results/headline.md).

**Prose vs agentic coding, 1 vs 4 streams.** The four "Decode, ..." rows are one matched measurement
([results/concurrency.md](results/concurrency.md)): the same requests and seeds for every profile, decode speed only
(prompt processing is excluded). Agentic coding replays the same 177 recorded requests as the headline row (one
session at a time, or four sessions at once); prose is four explanation and writing prompts at temperature 1. Code
decodes about twice as fast as prose because the drafter guesses it better: about 4.0 tokens are accepted per
verify step on agentic coding against 2.1 on prose. Four streams give 1.75-2.35x the total of one stream, not 4x:
each stream verifies 8 tokens per step, so 4 streams send 32 tokens through the MoE layers and together touch
most experts (8 tokens already touch about 40 % of them), and the int8 and split-K decode kernels act only up to
2 concurrent requests. Per stream, 4 users each get about 35 tok/s on agentic coding and 22 tok/s on prose. Details in
[How it was measured](#how-it-was-measured).

## Quick start

You need a DGX Spark (GB10), Docker with the NVIDIA runtime, about 50 GB of disk (20 GB of weights plus the 27 GB vLLM image), and about 100 GiB of free unified
memory (stop other GPU work first).

```bash
git clone https://github.com/Alssndr0/inkling-2-mini-dgx-spark
cd inkling-2-mini-dgx-spark
docker pull vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e
./download.sh          # ~20 GB into ./models, every file verified against Hugging Face
./serve.sh             # the fastest profile on 127.0.0.1:8000
```

The first start compiles kernels into `./cache` (224 s here); later starts take about 76 s.

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "thinkingmachines/Inkling-2-Mini",
  "messages": [{"role": "user", "content": "Write a Python function that checks whether a string is a palindrome."}],
  "max_tokens": 2048
}' | python3 -m json.tool
```

The API is OpenAI-compatible, with reasoning (`reasoning_content` in replies), tool calling and image input
enabled. Point OpenCode, Pi or any OpenAI client at `http://127.0.0.1:8000/v1`.

```bash
./serve.sh fast        # another profile (stop the running one first)
./serve.sh stop        # waits for running requests to finish, then stops
DRY_RUN=1 ./serve.sh   # print the docker command without running it
```

Settings are environment variables: `PORT` (8000), `BIND` (127.0.0.1; use 0.0.0.0 for your network), `MODEL_DIR`,
`CACHE_DIR`, `CONTAINER` (inkling), `KV_CACHE` (64G), `GPU_UTIL` (0.85), `DRAIN_S` (120).

## Profiles

| profile | what it adds | outputs | use it when |
|---|---|---|---|
| `lossless` | faster kernels, a smaller and int8 drafter, memory caps | same as stock up to rounding (KL 0.00003, 99.83 % same top token) | you want outputs as close to stock as possible |
| `fast` | `lossless` + int8 attention and LM-head weights for decoding + a faster long-context attention path | int8 weights change outputs slightly; full quality battery passed | you want speed with the fewest output changes |
| **`fastest`** | `fast` + a fused loop controller for prompts + small-batch matmul kernels for decoding | two more rounding changes; full quality battery and a dedicated investigation passed | everyday single-user agentic coding (default) |

All three run one model on one Spark, tuned for one user at a time. They also serve several requests at once,
but the decode kernels added by `fast` and `fastest` only act on small batches (up to 2 requests).

## What we changed

Measured effects come from screening runs: each change against the configuration before it, with controls in the
same session. Classes:
**bit-exact** (same outputs bit for bit),
**draft-side exact** (changes only the drafter of speculative decoding; rejection sampling keeps the output
distribution),
**rounding-level** (passes a KL gate of 0.002),
**output-changing** (needs the full quality battery).

| change | why, in plain words | measured effect | class | profiles |
|---|---|---|---|---|
| Persistent caches and eager weight loading | compiled kernels and graphs are reused; weights load in one pass instead of on demand | startup 368 s -> 93 s (76-79 s in the final profiles) | bit-exact | all |
| Pruned draft vocabulary (32,768 tokens) | the drafter scored all 201,024 tokens 7 times per step; most are never proposed | agentic replay +10.3 %, short-prompt decode +5 % | draft-side exact | all |
| Expert GEMM launch tiles for GB10 | the stock prefill tiles re-read each block of activations for every 64 output columns; 256-wide tiles reuse it 4x more | real-prompt prefill +11-13 %, decode +1 % | bit-exact | all |
| 8,192-token prefill chunks | fewer, larger steps for long prompts | real-prompt prefill +3-5 %, prompts over 30K +8-9 % | rounding-level | all |
| Native MXFP8 MMA for prefill tiles (layers 20-23) | uses the FP8 tensor cores instead of converting to BF16 first | real-prompt prefill +2.5-4 % | rounding-level (KL 0.00003) | all |
| Humming NVFP4 expert backend (`--moe-backend humming`) | a faster, bit-identical NVFP4 expert GEMM already in the image | real-prompt prefill +14 %, decode +2 % | bit-exact | all |
| KV pool capped at 64 GiB, GPU memory utilization 0.85 | the growable KV pool on unified memory starved the host and slowed prefill | peak memory 105 -> 94 GiB, prefill stable | bit-exact | all |
| 16x512 GEMV tiles (from sxuff) | wider tiles stream the large decode projections closer to memory bandwidth | decode step -0.2 % | bit-exact | all |
| int8 draft-only weights (from sxuff) | the drafter reads half the bytes | decode step -3.9 %, short-prompt decode +7.5 % | draft-side exact | all |
| Fused attention gate | the prefill attention applies its output gate itself instead of a separate pass over an fp32 copy | 8K prompt chunk -1.7 %, at 32K depth -3.9 % (GPU time) | bit-exact | all |
| int8 attention projections and LM head, decode only | decoding is memory-bound; these weights are read at every step | decode step -11.7 %, agentic decode +8 % | output-changing, battery passed | fast, fastest |
| Far-block attention path | key blocks beyond the 1,024-token relative-bias window skip the mask work | real-prompt prefill +4.2 %, prompts over 30K +5.3 % | rounding-level (KL 0.0007) | fast, fastest |
| K and V projections in one launch | two small launches under-filled the GPU | decode step -0.4 % | bit-exact | fast, fastest |
| Fused loop controller (prefill) | the looped layers wrote and re-read ~1.8 GB of intermediates per layer visit | real-prompt prefill +6.6 % | output-changing, battery and investigation passed | fastest |
| Split-K small matmuls (decode) | the controller and conv-tap projections ran on 2-12 thread blocks | decode step -1.6 % | output-changing (rounding), battery and investigation passed | fastest |

## Quality evidence

- **`lossless`** has no output-changing step. Its teacher-forced next-token distribution matches stock with KL 0.00003
  and the same top token at 99.83 % of 11,214 positions. The drafter changes passed a distribution test against
  stock, and greedy replies were identical with and without each of them.
- **`fast`** adds int8 decode weights. Full quality battery against stock: HumanEval+ / MBPP+ 434 vs 434 (McNemar p
  1.0), 400 exact-answer problems 355.5 vs 357, no CJK or wrong-language replies in 400, every tool call valid in the
  177 agentic replies, needle retrieval 3/3 at 128K and 244K tokens.
- **`fastest`** passed the same battery (438 vs 434, 356 vs 357, all else clean). A dedicated investigation then
  compared it with `fast` on long-context coding, exact-answer problems, 38 fresh post-cutoff bug-fix tasks run
  through OpenCode, CJK risk and tool calls. No task metric showed a loss:

| `fastest` vs `fast` | `fast` | `fastest` | difference, 95 % CI |
|---|---:|---:|---|
| Fresh bug-fix tasks (Go / Python, OpenCode, isolated) | 12 / 38 | 11 / 38 | -1 task [-5, +2] |
| HumanEval+ / MBPP+ with a 2K-token context, 2 samples at T=1 | 79.5 % | 80.7 % | +1.2 pp [-0.5, +2.9] |
| 400 exact-answer problems with a 2K-token context | 365 | 369 | +1.0 pp [-2.0, +4.0] |
| Invalid tool calls in ~3,400 per arm | 0 | 0 | |

  The coding test rules out a loss larger than 0.5 pp; the other two only rule out large losses. The one measurable
  difference is a 0.2 % higher NLL on a reference text corpus, smaller than the spread of all these configurations
  around stock.

Full tables: [results/quality-battery.md](results/quality-battery.md) and
[results/fin3c-quality-investigation.md](results/fin3c-quality-investigation.md).

## How it was measured

- **Machine:** one DGX Spark (GB10, sm_121, 121 GiB unified memory, GPU clock capped at 2,200 MHz), one serve at a
  time, nothing else on the GPU.
- **Agentic decode (P1):** 177 recorded requests from 25 OpenCode and Pi coding sessions (English), replayed one at a
  time at temperature 1 with a fixed seed per request. Metric: generated tokens per second of decoding. Each profile
  is paired with stock request by request; CIs bootstrap over sessions.
- **Single-stream and 4-stream decode:** a panel of short prompts at temperature 1 (median of 6 for one stream; the
  total of 4 concurrent streams).
- **Prompt processing:** time to first token of single requests. Real agent prompts: the 13 prompts of 7K-80K tokens
  in an 18-request replay set, sent cold. Documents: one 4K, 32K and 128K-token prompt each.
- **Startup:** container start until `/v1/models` answers, with warm caches unless stated.
- **Screening vs confirmation:** the per-change effects in [What we changed](#what-we-changed) come from lean screening
  boots during the tuning. The headline table is a confirmation run of this repository's `serve.sh` from a clean
  clone (2026-11-24), with stock measured the same day by the same harness except for its agentic decode, which
  is from 2026-11-22 (same requests, same seeds). Startup and benchmark numbers come from a second pass with
  warm caches; the first pass compiled the overlay kernels while it measured.
- **Quality:** a teacher-forced KL probe (11,214 positions of code and prose), HumanEval+ / MBPP+ scored by EvalPlus
  in a sandbox, 400 problems with exactly computable answers, a language guard, the agentic replay's own replies,
  needle retrieval at 128K and 244K tokens, and an output-distribution test for drafter changes. Each change class
  had its pass rule written down before the run.
- **Clean-clone check (2026-11-24):** default profile with an empty cache: ready after 224 s, model listed,
  a chat reply with reasoning, a tool-call round trip, an image input, and a teacher-forced KL of exactly 0 against
  the configuration measured during tuning. All three profiles' launches were also compared mechanically with the
  measured ones: identical docker flags, environment, image, vLLM arguments and file contents.
- Raw numbers: [results/](results/).

## What did not work

| idea | result |
|---|---|
| FP8 KV cache | decode step -3 %, but prefill at depth -3 to -5 % and KL 0.008; not worth a new quality question |
| 4-bit attention projections or LM head (also with GPTQ) | the drafter's acceptance dropped 3-4.5 %, more than the bytes saved |
| cuBLASLt for the small decode GEMMs | decode -1.6 %, but KL 0.0022 (over the 0.002 gate) |
| More splits for decode attention | slower at every context length tested |
| Faster int8 dequant in the decode GEMV | the largest one, the LM head, already streams at 97 % of memory bandwidth |
| Relative-position bias computed inside the prefill attention | 10-54 % slower than reading it from memory |
| FP8 prefill GEMMs | needs a fused activation quantizer; 3.75 % output error |
| Fewer draft tokens, greedy drafts, block verification | all slower than 7 probabilistic drafts with standard verification |
| 16K prefill chunks | real-prompt prefill -8 % vs 8K |

## Limits

- Built and measured for one GB10 (sm_121) with this exact image. The overlays replace files inside vLLM's Inkling2
  model code and will not apply to other vLLM versions.
- Tuned for one user. The KV pool holds about 2.4 full-length (262K-token) requests; raise `KV_CACHE` for more
  concurrent long contexts, at the cost of host memory.
- Decoding is close to the memory-bandwidth limit: the expert weights stream at full bandwidth and the int8 GEMVs at
  80-97 %. More speed would need fewer bytes per token (lossier formats, which cost the drafter acceptance) or more
  accepted tokens per step.
- `reasoning_effort: "none"` shortens Inkling's reasoning but does not turn it off. On long structured tasks it can
  loop until the token limit; give such requests a generous `max_tokens`, or start the reply with `{` using
  `continue_final_message` (vLLM then cannot apply a JSON schema, so parse leniently).
- The `fastest` and `fast` decode kernels act on batches of up to 16 rows (two concurrent requests with 7 drafts each);
  larger batches use the stock kernels.

## Pins

| what | pin |
|---|---|
| Image | `vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e` (tag `inkling-2-mini`; arm64 manifest `sha256:ecdf0ccc663d86332459f5a50a38c30861a5079bc3924aa08fdc657ef10dd648`) |
| vLLM in the image | `7cb5dae`, torch 2.14, Triton 3.8, FlashInfer 0.7.4 |
| Model | [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) revision `b85c3cfa615fc413cc9af3379f491b963f3b586a` |
| Model files | [checksums/Inkling-2-Mini-b85c3cf.txt](checksums/Inkling-2-Mini-b85c3cf.txt) (Hugging Face LFS SHA-256 and Git blob SHA-1) |
| Overlay files | [patches/SHA256SUMS](patches/SHA256SUMS), checked by `serve.sh` before every start |
| Hardware | DGX Spark, GB10, 121 GiB unified memory |

## Credits

- [Thinking Machines](https://huggingface.co/thinkingmachines) for Inkling 2 Mini and its DSpark drafter.
- [vLLM](https://github.com/vllm-project/vllm) for the engine, the Inkling2 model code and the Humming expert backend
  that these overlays build on.
- [sxuff/inkling-2-mini-spark](https://github.com/sxuff/inkling-2-mini-spark) for the int8 draft-only weights, the
  16x512 GEMV tiles and the idea of reading kernel profiles by exclusive time.
- [Sybil Solutions' inference-optimisation skill](https://github.com/sybil-solutions/inference-optimisation-skill)
  (MIT) for the method: ceilings first, pre-registered quality gates, noise-controlled A/B runs, confirmation at the
  end.

## License

Apache-2.0 ([LICENSE](LICENSE)). The overlay files are modified vLLM files and carry change notices; see
[NOTICE.md](NOTICE.md). The model weights are not included.
