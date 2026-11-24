# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ---- recipe change notice ----
# Modified copy of vllm/models/inkling2/nvidia/ops/loop.py from the vllm/vllm-openai:inkling-2-mini image (vLLM 7cb5dae, Apache-2.0).
# Changed: the loop controller fused into two kernels for prefill chunks of >= 1,024 rows, and a split-K
# small matmul for decode batches of <= 16 rows (both change rounding).
# Change notice added for the inkling-2-mini-dgx-spark recipe, 2026-11 (Apache-2.0).
# ---- end of recipe change notice ----
"""Loop message controllers, batched over streams.

A controller blends a fp32 message into a fp32 sublayer input `dest`: from
`h = down(rms([dest, message]))` (bf16), `out = (1 - tanh(forget(h))) * dest +
tanh(gain(h)) * message` in fp32, unfused as the reference. The output feeds
the sublayer's RMSNorm, applied in the same kernel.
"""

import torch

from vllm.triton_utils import tl, tldevice, triton

from .norm import rsqrt


@triton.jit
def _pair_rms_norm_kernel(
    Dest,
    Message,
    Out,
    stride_dest_s,
    T,
    eps,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    s = tl.program_id(0).to(tl.int64)
    t = tl.program_id(1).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    dest = tl.load(Dest + s * stride_dest_s + t * D + cols, mask, other=0)
    message = tl.load(Message + (s * T + t) * D + cols, mask, other=0)
    dest_rstd = rsqrt(tl.div_rn(tl.sum(dest * dest), D) + eps)
    message_rstd = rsqrt(tl.div_rn(tl.sum(message * message), D) + eps)
    out = Out + (s * T + t) * (2 * D)
    tl.store(out + cols, (dest * dest_rstd).to(Out.dtype.element_ty), mask)
    tl.store(out + D + cols, (message * message_rstd).to(Out.dtype.element_ty), mask)


def pair_rms_norm(dest: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
    """bf16 `(S, T, 2D)` `[rms(dest), rms(message)]` of fp32 `(S, T, D)`
    inputs, weightless; `dest` may broadcast over streams (stride 0)."""
    num_streams, num_tokens, d = message.shape
    assert dest.shape == message.shape and message.is_contiguous()
    assert dest.stride(1) == d and dest.stride(2) == 1
    assert dest.dtype == message.dtype == torch.float32
    out = message.new_empty((num_streams, num_tokens, 2 * d), dtype=torch.bfloat16)
    if num_tokens:
        _pair_rms_norm_kernel[(num_streams, num_tokens)](
            dest,
            message,
            out,
            dest.stride(0),
            num_tokens,
            1e-6,
            D=d,
            BLOCK=triton.next_power_of_2(d),
            num_warps=8,
        )
    return out


@triton.jit
def _blend_rms_norm_kernel(
    Dest,
    Message,
    GainForget,
    NormW,
    Out,
    stride_dest_s,
    stride_w_s,
    T,
    norm_eps,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    s = tl.program_id(0).to(tl.int64)
    t = tl.program_id(1).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    row = s * T + t
    dest = tl.load(Dest + s * stride_dest_s + t * D + cols, mask, other=0)
    message = tl.load(Message + row * D + cols, mask, other=0)
    gain = tldevice.tanh(tl.load(GainForget + row * (2 * D) + cols, mask, other=0))
    forget = tldevice.tanh(
        tl.load(GainForget + row * (2 * D) + D + cols, mask, other=0)
    )
    x = (1.0 - forget) * dest + gain * message
    rstd = rsqrt(tl.div_rn(tl.sum(x * x), D) + norm_eps)
    y = x * rstd
    y = y * tl.load(NormW + s * stride_w_s + cols, mask, other=0).to(tl.float32)
    tl.store(Out + row * D + cols, y.to(Out.dtype.element_ty), mask)


def blend_rms_norm(
    dest: torch.Tensor,
    message: torch.Tensor,
    gain_forget: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor:
    """bf16 `(S, T, D)` RMSNorm (row `s` of `norm_weight` `(S, D)`) of
    `(1 - tanh(forget)) * dest + tanh(gain) * message`, with fp32 `(S, T, D)`
    `dest` (may broadcast over streams) and `message`, and fp32 `(S, T, 2D)`
    `gain_forget` = `[gain, forget]`. `norm_weight` may broadcast too."""
    num_streams, num_tokens, d = message.shape
    assert dest.shape == message.shape and message.is_contiguous()
    assert dest.stride(1) == d and dest.stride(2) == 1
    assert gain_forget.shape == (num_streams, num_tokens, 2 * d)
    assert gain_forget.is_contiguous() and gain_forget.dtype == torch.float32
    assert norm_weight.shape == (num_streams, d) and norm_weight.stride(1) == 1
    out = message.new_empty((num_streams, num_tokens, d), dtype=torch.bfloat16)
    if num_tokens:
        _blend_rms_norm_kernel[(num_streams, num_tokens)](
            dest,
            message,
            gain_forget,
            norm_weight,
            out,
            dest.stride(0),
            norm_weight.stride(0),
            num_tokens,
            norm_eps,
            D=d,
            BLOCK=triton.next_power_of_2(d),
            num_warps=8,
            # The blend is unfused (no FMA), as the reference.
            enable_fp_fusion=False,
        )
    return out


# ---------------------------------------------------------------------------
# Campaign patch ctlf2 (inkling2-mini-spark, 2026-11-23; ctlf + persistent row loop): the loop controller
# fused into two kernels for large row counts (prefill chunks). The reference
# path writes the bf16 pair (S, T, 2D) and the fp32 gain/forget (S, T, 2D) to
# memory and reads them back; here the pair norm feeds the down projection
# directly (kernel A) and the gain/forget projection is recomputed per column
# chunk inside the blend + RMSNorm (kernel B). Same roundings in the same
# places (bf16 pair, bf16 hidden, fp32 gain/forget, unfused blend); only the
# summation orders differ (row sums of squares chunked, GEMM K order).
# ---------------------------------------------------------------------------


@triton.jit
def _ctl_down_kernel(
    Dest,
    Message,
    Down,
    Hidden,
    stride_dest_s,
    stride_down_s,
    T,
    eps,
    D: tl.constexpr,
    R: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Persistent over row blocks (grid axis 1 = programs per stream): few
    # enough programs that the second pass over a row block hits L2.
    s = tl.program_id(0).to(tl.int64)
    ks = tl.arange(0, BLOCK_K)
    rs = tl.arange(0, R)
    w_ptr = Down + s * stride_down_s + rs[None, :] * (2 * D) + ks[:, None]
    for rb in range(tl.program_id(1), tl.cdiv(T, BLOCK_T), tl.num_programs(1)):
        rows = (rb * BLOCK_T + tl.arange(0, BLOCK_T)).to(tl.int64)
        rmask = rows < T
        dest_ptr = Dest + s * stride_dest_s + rows[:, None] * D + ks[None, :]
        msg_ptr = Message + (s * T + rows[:, None]) * D + ks[None, :]
        ss_d = tl.zeros([BLOCK_T], dtype=tl.float32)
        ss_m = tl.zeros([BLOCK_T], dtype=tl.float32)
        for k in range(0, D, BLOCK_K):
            d = tl.load(dest_ptr + k, rmask[:, None], other=0)
            m = tl.load(msg_ptr + k, rmask[:, None], other=0)
            ss_d += tl.sum(d * d, 1)
            ss_m += tl.sum(m * m, 1)
        rd = rsqrt(tl.div_rn(ss_d, D) + eps)
        rm = rsqrt(tl.div_rn(ss_m, D) + eps)
        acc = tl.zeros([BLOCK_T, R], dtype=tl.float32)
        for k in range(0, D, BLOCK_K):
            d = tl.load(dest_ptr + k, rmask[:, None], other=0)
            acc = tl.dot((d * rd[:, None]).to(tl.bfloat16), tl.load(w_ptr + k), acc)
        for k in range(0, D, BLOCK_K):
            m = tl.load(msg_ptr + k, rmask[:, None], other=0)
            acc = tl.dot((m * rm[:, None]).to(tl.bfloat16), tl.load(w_ptr + D + k), acc)
        out = Hidden + (s * T + rows[:, None]) * R + rs[None, :]
        tl.store(out, acc.to(Hidden.dtype.element_ty), rmask[:, None])


@triton.jit
def _ctl_blend_kernel(
    Dest,
    Message,
    Hidden,
    GainForget,
    NormW,
    Out,
    stride_dest_s,
    stride_gf_s,
    stride_w_s,
    T,
    norm_eps,
    D: tl.constexpr,
    R: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    s = tl.program_id(0).to(tl.int64)
    rs = tl.arange(0, R)
    cols = tl.arange(0, BLOCK_D)
    # gain rows [0, D), forget rows [D, 2D) of the (2D, R) weight, read as (R, BLOCK_D).
    g_ptr = GainForget + s * stride_gf_s + cols[None, :] * R + rs[:, None]
    for rb in range(tl.program_id(1), tl.cdiv(T, BLOCK_T), tl.num_programs(1)):
        rows = (rb * BLOCK_T + tl.arange(0, BLOCK_T)).to(tl.int64)
        rmask = rows < T
        h = tl.load(Hidden + (s * T + rows[:, None]) * R + rs[None, :], rmask[:, None], other=0)
        dest_ptr = Dest + s * stride_dest_s + rows[:, None] * D + cols[None, :]
        msg_ptr = Message + (s * T + rows[:, None]) * D + cols[None, :]
        ss = tl.zeros([BLOCK_T], dtype=tl.float32)
        for k in range(0, D, BLOCK_D):
            gain = tldevice.tanh(tl.dot(h, tl.load(g_ptr + k * R)))
            forget = tldevice.tanh(tl.dot(h, tl.load(g_ptr + (D + k) * R)))
            dest = tl.load(dest_ptr + k, rmask[:, None], other=0)
            message = tl.load(msg_ptr + k, rmask[:, None], other=0)
            x = (1.0 - forget) * dest + gain * message
            ss += tl.sum(x * x, 1)
        rstd = rsqrt(tl.div_rn(ss, D) + norm_eps)
        out_ptr = Out + (s * T + rows[:, None]) * D + cols[None, :]
        for k in range(0, D, BLOCK_D):
            gain = tldevice.tanh(tl.dot(h, tl.load(g_ptr + k * R)))
            forget = tldevice.tanh(tl.dot(h, tl.load(g_ptr + (D + k) * R)))
            dest = tl.load(dest_ptr + k, rmask[:, None], other=0)
            message = tl.load(msg_ptr + k, rmask[:, None], other=0)
            x = (1.0 - forget) * dest + gain * message
            y = x * rstd[:, None]
            y = y * tl.load(NormW + s * stride_w_s + k + cols).to(tl.float32)[None, :]
            tl.store(out_ptr + k, y.to(Out.dtype.element_ty), rmask[:, None])


# (BLOCK_T, BLOCK_K or BLOCK_D, num_warps, num_stages, programs in flight over all streams; 0 = one per row block)
CTL_FUSED_CFG = {"a": (16, 128, 8, 3, 96), "b": (16, 128, 8, 3, 96)}


def _programs(num_streams, num_tokens, bt, total):
    blocks = triton.cdiv(num_tokens, bt)
    return blocks if total <= 0 else max(1, min(blocks, triton.cdiv(total, num_streams)))


def controller_blend_fused(
    dest: torch.Tensor,
    message: torch.Tensor,
    down: torch.Tensor,
    gain_forget: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    cfg: dict | None = None,
) -> torch.Tensor:
    """`blend_rms_norm(dest, message, bmm(bmm(pair_rms_norm(dest, message),
    down^T), gain_forget^T, fp32), norm_weight, norm_eps)` in two kernels.
    `down` (S, R, 2D) and `gain_forget` (S, 2D, R) bf16, rows contiguous."""
    num_streams, num_tokens, d = message.shape
    rank = down.shape[1]
    assert dest.shape == message.shape and message.is_contiguous()
    assert dest.stride(1) == d and dest.stride(2) == 1
    assert dest.dtype == message.dtype == torch.float32
    assert down.shape == (num_streams, rank, 2 * d) and down.stride()[1:] == (2 * d, 1)
    assert gain_forget.shape == (num_streams, 2 * d, rank)
    assert gain_forget.stride()[1:] == (rank, 1)
    assert norm_weight.shape == (num_streams, d) and norm_weight.stride(1) == 1
    cfg = cfg or CTL_FUSED_CFG
    hidden = message.new_empty((num_streams, num_tokens, rank), dtype=torch.bfloat16)
    out = message.new_empty((num_streams, num_tokens, d), dtype=torch.bfloat16)
    if not num_tokens:
        return out
    bt, bk, nw, ns, total = cfg["a"]
    _ctl_down_kernel[(num_streams, _programs(num_streams, num_tokens, bt, total))](
        dest, message, down, hidden, dest.stride(0), down.stride(0), num_tokens, 1e-6,
        D=d, R=rank, BLOCK_T=bt, BLOCK_K=bk, num_warps=nw, num_stages=ns,
    )
    bt, bd, nw, ns, total = cfg["b"]
    _ctl_blend_kernel[(num_streams, _programs(num_streams, num_tokens, bt, total))](
        dest, message, hidden, gain_forget, norm_weight, out, dest.stride(0),
        gain_forget.stride(0), norm_weight.stride(0), num_tokens, norm_eps,
        D=d, R=rank, BLOCK_T=bt, BLOCK_D=bd, num_warps=nw, num_stages=ns,
        # The blend is unfused (no FMA), as the reference.
        enable_fp_fusion=False,
    )
    return out


# ---------------------------------------------------------------------------
# Campaign patch dsm (2026-11-23, on top of ctlf2): small-batch (<= 16 rows,
# decode / verify) `x @ w^T` for the loop controllers' down projection (K =
# 4,608, N = 64) and the K / V short convs' dynamic taps (K = 2,304, N = 32).
# cuBLAS runs them on 2-12 CTAs of an SM80 wmma kernel (16-37 us each, 35-55
# GB/s); here K is split over ~50 CTAs and the fp32 partials are summed in
# a fixed order by a second kernel (deterministic; rounding-level vs cuBLAS).
# ---------------------------------------------------------------------------


@triton.jit
def _smm_partial_kernel(
    X, W, P, T, N, K, sx_s, sx_t, sw_s, sw_n,
    KC: tl.constexpr, BK: tl.constexpr, TP: tl.constexpr, NP: tl.constexpr,
):
    s = tl.program_id(0).to(tl.int64)
    sp = tl.program_id(1)
    nsplit = tl.num_programs(1)
    rows = tl.arange(0, TP)
    ns = tl.arange(0, NP)
    acc = tl.zeros([TP, NP], dtype=tl.float32)
    for k0 in range(sp * KC, sp * KC + KC, BK):
        ks = k0 + tl.arange(0, BK)
        km = ks < K
        x = tl.load(X + s * sx_s + rows[:, None] * sx_t + ks[None, :],
                    (rows[:, None] < T) & km[None, :], other=0)
        w = tl.load(W + s * sw_s + ns[None, :] * sw_n + ks[:, None],
                    (ns[None, :] < N) & km[:, None], other=0)
        acc = tl.dot(x, w, acc)
    tl.store(P + ((s * nsplit + sp) * TP + rows[:, None]) * NP + ns[None, :], acc)


@triton.jit
def _smm_reduce_kernel(
    P, Y, T, N, sy_s, sy_t,
    NSPLIT: tl.constexpr, TP: tl.constexpr, NP: tl.constexpr,
):
    s = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, TP)
    ns = tl.arange(0, NP)
    acc = tl.zeros([TP, NP], dtype=tl.float32)
    for sp in tl.static_range(NSPLIT):
        acc += tl.load(P + ((s * NSPLIT + sp) * TP + rows[:, None]) * NP + ns[None, :])
    tl.store(Y + s * sy_s + rows[:, None] * sy_t + ns[None, :], acc.to(Y.dtype.element_ty),
             (rows[:, None] < T) & (ns[None, :] < N))


SMALL_MM_MAX_ROWS = 16


def small_mm_nt(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """bf16 (S, T, K) @ (S, N, K)^T -> bf16 (S, T, N), fp32 accumulation, T <= 16, N <= 64; split-K, fixed-order
    reduction."""
    num_streams, num_tokens, k = x.shape
    n = w.shape[1]
    assert w.shape == (num_streams, n, k) and n <= 64 and num_tokens <= SMALL_MM_MAX_ROWS
    assert x.stride(2) == 1 and w.stride(2) == 1
    y = x.new_empty((num_streams, num_tokens, n))
    if not num_tokens:
        return y
    bk = 64
    target = triton.cdiv(64, num_streams)
    kc = max(bk, triton.next_power_of_2(triton.cdiv(k, target)))
    nsplit = triton.cdiv(k, kc)
    np_ = max(16, triton.next_power_of_2(n))
    part = torch.empty((num_streams, nsplit, 16, np_), dtype=torch.float32, device=x.device)
    _smm_partial_kernel[(num_streams, nsplit)](
        x, w, part, num_tokens, n, k, x.stride(0), x.stride(1), w.stride(0), w.stride(1),
        KC=kc, BK=bk, TP=16, NP=np_, num_warps=4, num_stages=3,
    )
    _smm_reduce_kernel[(num_streams,)](
        part, y, num_tokens, n, y.stride(0), y.stride(1), NSPLIT=nsplit, TP=16, NP=np_, num_warps=4,
    )
    return y
