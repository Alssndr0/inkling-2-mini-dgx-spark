# Notices

This repository is licensed under the Apache License 2.0 (`LICENSE`).

## vLLM

The Python files in `patches/` are modified copies of files from [vLLM](https://github.com/vllm-project/vllm)
(Apache-2.0, copyright the vLLM contributors), taken from the image `vllm/vllm-openai:inkling-2-mini`
(index digest `sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e`, vLLM `7cb5dae`), or new files
built on vLLM's Triton utilities. Each file carries a change notice between the lines
`# ---- recipe change notice ----` and `# ---- end of recipe change notice ----`.

| file | upstream file | status |
|---|---|---|
| `patches/draft/speculator.py` | `vllm/models/inkling2/speculator.py` | modified: pruned draft vocabulary |
| `patches/draft/dspark.py` | `vllm/models/inkling2/nvidia/dspark.py` | modified: pruned draft vocabulary; int8 draft-only weights (from sxuff, see below) |
| `patches/draft/gemv_q8.py` | none | new file from sxuff/inkling-2-mini-spark, unchanged |
| `patches/moe/moe_sm120.py` | `vllm/models/inkling2/nvidia/ops/moe_sm120.py` | modified: expert GEMM launch tiles, native MXFP8 MMA for prefill tiles |
| `patches/gemv/gemv.py` | `vllm/models/inkling2/nvidia/ops/gemv.py` | from sxuff/inkling-2-mini-spark, unchanged: 16x512 GEMV tiles |
| `patches/int8/gemv.py` | `vllm/models/inkling2/nvidia/ops/gemv.py` | modified: sxuff's 16x512 tiles plus int8 weight dispatch |
| `patches/int8/w8.py` | none | new file: int8 decode weights and GEMV |
| `patches/int8/model.py` | `vllm/models/inkling2/nvidia/model.py` | modified: registers weights for int8 decode |
| `patches/attention/attention_sm90.py` | `vllm/models/inkling2/nvidia/ops/attention_sm90.py` | modified: far-block path, fused gate epilogue |
| `patches/attention/attention.py` | `vllm/models/inkling2/nvidia/attention.py` | modified: one launch for K / V, fused gate |
| `patches/attention-exact/attention_sm90.py` | `vllm/models/inkling2/nvidia/ops/attention_sm90.py` | modified: fused gate epilogue only |
| `patches/attention-exact/attention.py` | `vllm/models/inkling2/nvidia/attention.py` | modified: fused gate only |
| `patches/loop/ops_loop.py` | `vllm/models/inkling2/nvidia/ops/loop.py` | modified: fused loop controller, split-K small matmul |
| `patches/loop/loop.py` | `vllm/models/inkling2/nvidia/loop.py` | modified: dispatch to the above |
| `patches/loop/sconv.py` | `vllm/models/inkling2/nvidia/sconv.py` | modified: split-K conv taps |

`patches/draft/draft_vocab.json` is data made for this recipe: the 32,768 token ids the drafter scores.

The Inkling2 model implementation, the DSpark speculator and the Humming expert backend these files plug into are
vLLM's work.

## sxuff/inkling-2-mini-spark

Parts of this recipe come from [sxuff/inkling-2-mini-spark](https://github.com/sxuff/inkling-2-mini-spark)
(Apache-2.0): the int8 copies of the draft-only weights (`patches/draft/dspark.py`, `patches/draft/gemv_q8.py`), the
16x512 GEMV tiles (`patches/gemv/gemv.py`, `patches/int8/gemv.py`), and the idea of attributing kernel time by
exclusive (non-overlapped) intervals, which this work used to read its profiles.

## Inkling 2 Mini

The model is [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) by Thinking
Machines. Its weights are not redistributed here: `download.sh` fetches the pinned revision from Hugging Face and
verifies every file.

## Method

The tuning followed the [inference-optimisation skill](https://github.com/sybil-solutions/inference-optimisation-skill)
by Sybil Solutions (MIT): ceilings first, pre-registered quality gates per change class, noise-controlled A/B runs,
and a confirmation of the final configuration. No code from it is included here.
