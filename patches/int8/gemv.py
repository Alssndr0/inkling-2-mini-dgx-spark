# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ---- recipe change notice ----
# Modified copy of vllm/models/inkling2/nvidia/ops/gemv.py from the vllm/vllm-openai:inkling-2-mini image (vLLM 7cb5dae, Apache-2.0).
# Changed: the 16x512 GEMV tiles from sxuff/inkling-2-mini-spark (https://github.com/sxuff/inkling-2-mini-spark, Apache-2.0),
# plus dispatch of registered weights to the int8 GEMV in int8/w8.py (INKLING_W8).
# Change notice added for the inkling-2-mini-dgx-spark recipe, 2026-11 (Apache-2.0).
# ---- end of recipe change notice ----
# Modified for the inkling-2-mini-gb10 recipe (2026-11-21): 16x512 tiles for N >= 4096
# (bit-identical outputs), INKLING2_GEMV_TILE.
"""Low-M BF16 tensor-core projections for SM120/SM121."""

import os

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

MAX_M = 16
_BN, _BK = 32, 256
# Wide projections stream 16-column x 512-deep tiles: the same per-column
# K-ordered tensor-core chain (bit-identical outputs), ~3-7% closer to HBM
# bandwidth on GB10 for N >= 4096.
_WIDE_N = 4096 if os.environ.get("INKLING2_GEMV_TILE", "1") != "0" else 1 << 62

# Campaign option "w8" (class C, labelled): BF16 weights registered here are stored as 8-bit values (int8 or FP8
# e4m3, power-of-two scale per 32 along K; ops/w8.py) and replaced in place by their exact BF16 dequantization W', so
# cuBLAS (prefill) and this low-M GEMV (decode / verify, reading the 8-bit bytes) use the same effective weights. Row
# slices of a registered weight (full layers project q / k / v from slices of qkvr) map to the matching rows.
_FP8W = []


def register_w8(weight: torch.Tensor, fmt: str = "int8") -> None:
    from .w8 import quantize_

    assert weight.dtype == torch.bfloat16 and weight.dim() == 2 and weight.stride(1) == 1
    q, s = quantize_(weight.data, fmt)
    start = weight.data_ptr()
    _FP8W.append((start, start + weight.shape[0] * weight.stride(0) * 2, weight.stride(0), weight.shape[1], q, s))


def _fp8w_lookup(weight: torch.Tensor):
    p = weight.data_ptr()
    for start, end, stride, k, q, s in _FP8W:
        if start <= p < end and weight.dim() == 2 and weight.stride(0) == stride and weight.shape[1] == k:
            off = (p - start) // 2
            if off % stride:
                return None
            r = off // stride
            return q[r : r + weight.shape[0]], s[r : r + weight.shape[0]]
    return None
_BN, _BK = 32, 256




@triton.jit
def _gemv_kernel(
    X,
    W,
    B,
    Y,
    M,
    N,
    K,
    SX,
    SW,
    SY,
    HAS_BIAS: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    MP: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    rows = tl.arange(0, MP)
    acc = tl.zeros((MP, BN), tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        w = tl.load(
            W + cols[None, :].to(tl.int64) * SW + ks[:, None],
            (cols[None, :] < N) & (ks[:, None] < K),
            other=0,
        )
        x = tl.load(
            X + rows[:, None] * SX + ks[None, :],
            (rows[:, None] < M) & (ks[None, :] < K),
            other=0,
        )
        acc += tl.dot(x, w)
    if HAS_BIAS:
        acc += tl.load(B + cols, cols < N, other=0).to(tl.float32)[None, :]
    tl.store(
        Y + rows[:, None] * SY + cols[None, :],
        acc.to(Y.dtype.element_ty),
        (rows[:, None] < M) & (cols[None, :] < N),
    )


def sm12x_bf16_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if _FP8W:
        hit = _fp8w_lookup(weight)
        if hit is not None:
            from .w8 import gemv as w8_gemv

            return w8_gemv(x, hit[0], hit[1], bias, out_dtype, out)
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    m, k = x2.shape
    assert m <= MAX_M
    n = weight.shape[0]
    if out is None:
        out = torch.empty((m, n), device=x.device, dtype=out_dtype or x.dtype)
    if m:
        bn, bk, stages = (16, 512, 3) if n >= _WIDE_N else (_BN, _BK, 3)
        _gemv_kernel[(triton.cdiv(n, bn),)](
            x2,
            weight,
            bias,
            out,
            m,
            n,
            k,
            x2.stride(0),
            weight.stride(0),
            out.stride(0),
            HAS_BIAS=bias is not None,
            BN=bn,
            BK=bk,
            MP=16,
            num_warps=4,
            num_stages=stages,
        )
    return out.view(*lead, n)


def use_sm12x_bf16_gemv(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> bool:
    return (
        x.is_cuda
        and weight.device == x.device
        and x.dtype == weight.dtype == torch.bfloat16
        and x.ndim >= 2
        and weight.ndim == 2
        and x.shape[-1] == weight.shape[1]
        and x.stride(-1) == weight.stride(-1) == 1
        and 0 < x.numel() // x.shape[-1] <= MAX_M
        and (
            bias is None
            or (
                bias.dtype == torch.bfloat16
                and bias.device == x.device
                and bias.stride(0) == 1
            )
        )
        and current_platform.is_device_capability_family(120)
    )
