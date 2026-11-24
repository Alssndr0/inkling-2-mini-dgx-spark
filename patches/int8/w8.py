# SPDX-License-Identifier: Apache-2.0
# ---- recipe change notice ----
# Not part of the image; overlaid next to its files at serve time.
# New file: int8 storage (power-of-two scale per 32 weights) of BF16 linears for the decode path, its GEMV,
# and one launch for the K and V projections (bit-identical to two launches).
# Change notice added for the inkling-2-mini-dgx-spark recipe, 2026-11 (Apache-2.0).
# ---- end of recipe change notice ----
"""8-bit weight storage for BF16 linears + a low-M 8-bit-weight GEMV (campaign options "w8": int8 or fp8, class C).

A BF16 weight W [N, K] is quantized per row in blocks of 32 along K with one power-of-two scale per block:
  fmt "int8": q = round(W / 2^e) in [-127, 127], 2^e the smallest power of two with absmax / 2^e <= 127
  fmt "fp8":  q = e4m3(W / 2^e), 2^e the smallest power of two with absmax / 2^e <= 448 (MXFP8-like)
Because the scale is a power of two, q x 2^e is exactly representable in BF16 (int8: 7 significant bits; e4m3: 4), so
  - `quantize_(W, fmt)` overwrites W in place with W' = q x 2^e (BF16, exact) and returns (q, e + 127 as uint8);
  - prefill keeps cuBLAS BF16 on W' (unchanged speed); decode / verify (<= 16 rows) runs `gemv`, which reads only
    the 8-bit values + scales (about half the bytes) and dequantizes in registers to the same W';
  - so the teacher-forced KL probe (a prefill) measures exactly the quality effect of the decode path.
int8 keeps ~4x less rounding error than e4m3 at the same bytes (bench_w8.py measures both).

Usage (inside the engine image): from w8 import quantize_, gemv; q, s = quantize_(W, "int8"); y = gemv(x, q, s)
Generalise: any bandwidth-bound low-M BF16 projection; the in-place exact W' keeps the model's large-M path as is.
"""
import torch

from vllm.triton_utils import tl, triton

MAX_M = 16
_QMAX = {"int8": 127.0, "fp8": 448.0}


@torch.no_grad()
def quantize_(w: torch.Tensor, fmt: str = "int8", chunk_rows: int = 8192):
    """In place: w <- q x 2^e (bf16, exact); returns (q [N, K] int8 or fp8 e4m3, s uint8 [N, K // 32] = e + 127)."""
    n, k = w.shape
    assert k % 32 == 0 and w.dtype == torch.bfloat16 and fmt in _QMAX
    qdtype = torch.int8 if fmt == "int8" else torch.float8_e4m3fn
    q = torch.empty((n, k), dtype=qdtype, device=w.device)
    s = torch.empty((n, k // 32), dtype=torch.uint8, device=w.device)
    for r0 in range(0, n, chunk_rows):
        blk = w[r0:r0 + chunk_rows].float().view(-1, k // 32, 32)
        amax = blk.abs().amax(-1)
        e = torch.ceil(torch.log2(amax / _QMAX[fmt])).clamp(-126, 127)
        e = torch.where(amax > 0, e, torch.full_like(e, -126))
        scale = torch.exp2(e)[..., None]
        if fmt == "int8":
            qq = torch.round(blk / scale).clamp(-127, 127)
            qf = qq
            qq = qq.to(torch.int8)
        else:
            qq = (blk / scale).to(torch.float8_e4m3fn)
            qf = qq.float()
        q[r0:r0 + chunk_rows] = qq.view(-1, k)
        s[r0:r0 + chunk_rows] = (e + 127).to(torch.uint8)
        w[r0:r0 + chunk_rows] = (qf * scale).view(-1, k).to(torch.bfloat16)
    return q, s


@triton.jit
def _gemv_w8_kernel(X, Q, S, B, Y, M, N, K, SX, SQ, SS, SY, FP8: tl.constexpr,
                    HAS_BIAS: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, MP: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    rows = tl.arange(0, MP)
    acc = tl.zeros((MP, BN), tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        kmask = ks < K
        qb = tl.load(Q + cols[None, :].to(tl.int64) * SQ + ks[:, None],
                     (cols[None, :] < N) & kmask[:, None], other=0)          # raw bytes (uint8 view)
        if FP8:
            qv = qb.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        else:
            qv = qb.to(tl.int8, bitcast=True).to(tl.float32)
        e = tl.load(S + cols[None, :].to(tl.int64) * SS + ks[:, None] // 32,
                    (cols[None, :] < N) & kmask[:, None], other=127).to(tl.uint32)
        scale = (e << 23).to(tl.float32, bitcast=True)   # 2^(e - 127); e >= 1 by construction
        w = (qv * scale).to(tl.bfloat16)                 # exact (power-of-two scale)
        x = tl.load(X + rows[:, None] * SX + ks[None, :], (rows[:, None] < M) & kmask[None, :], other=0)
        acc += tl.dot(x, w)
    if HAS_BIAS:
        acc += tl.load(B + cols, cols < N, other=0).to(tl.float32)[None, :]
    tl.store(Y + rows[:, None] * SY + cols[None, :], acc.to(Y.dtype.element_ty),
             (rows[:, None] < M) & (cols[None, :] < N))


def _config(n):
    """(BN, BK, warps, stages) per output width, from bench_fp8w.py / bench_w8.py --grid on GB10 (8 rows)."""
    if n <= 2304:
        return 16, 256, 2, 3
    return 32, 256, 4, 2


def gemv(x, q, s, bias=None, out_dtype=None, out=None, BN=None, BK=None, num_warps=None, num_stages=None):
    cfg = _config(q.shape[0])
    BN, BK = BN or cfg[0], BK or cfg[1]
    num_warps, num_stages = num_warps or cfg[2], num_stages or cfg[3]
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    m, k = x2.shape
    assert m <= MAX_M and q.shape[1] == k and q.stride(1) == 1 and s.stride(1) == 1
    n = q.shape[0]
    if out is None:
        out = torch.empty((m, n), device=x.device, dtype=out_dtype or x.dtype)
    if m:
        _gemv_w8_kernel[(triton.cdiv(n, BN),)](
            x2, q.view(torch.uint8), s, bias, out, m, n, k, x2.stride(0), q.stride(0), s.stride(0), out.stride(0),
            FP8=q.dtype == torch.float8_e4m3fn, HAS_BIAS=bias is not None, BN=BN, BK=BK, MP=16,
            num_warps=num_warps, num_stages=num_stages)
    return out.view(*lead, n)


# Campaign patch kvm (2026-11-23): the K and V projections of a layer with separate K / V inputs (full-attention
# layers, loop passes 2-3) as ONE launch over their adjacent rows of qkvr: the first n1 output columns read X1, the
# rest X2. Each column runs the same instruction sequence as `gemv` with the same config, so outputs are bit-identical
# to two `gemv` calls; one launch fills the GPU (2 x 64 programs of 16 columns) where each alone ran 64 programs on
# 48 SMs at ~120 GB/s.
@triton.jit
def _gemv_w8_dual_kernel(X1, X2, Q, S, Y1, Y2, M, N1, N, K, SX1, SX2, SQ, SS, SY1, SY2, FP8: tl.constexpr,
                         BN: tl.constexpr, BK: tl.constexpr, MP: tl.constexpr):
    pid = tl.program_id(0)
    nb1 = tl.cdiv(N1, BN)
    second = pid >= nb1
    cols = tl.where(second, N1 + (pid - nb1) * BN, pid * BN) + tl.arange(0, BN)
    lim = tl.where(second, N, N1)
    rows = tl.arange(0, MP)
    if second:
        xb = X2
        sx = SX2
    else:
        xb = X1
        sx = SX1
    acc = tl.zeros((MP, BN), tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        kmask = ks < K
        qb = tl.load(Q + cols[None, :].to(tl.int64) * SQ + ks[:, None],
                     (cols[None, :] < lim) & kmask[:, None], other=0)
        if FP8:
            qv = qb.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        else:
            qv = qb.to(tl.int8, bitcast=True).to(tl.float32)
        e = tl.load(S + cols[None, :].to(tl.int64) * SS + ks[:, None] // 32,
                    (cols[None, :] < lim) & kmask[:, None], other=127).to(tl.uint32)
        scale = (e << 23).to(tl.float32, bitcast=True)
        w = (qv * scale).to(tl.bfloat16)
        x = tl.load(xb + rows[:, None] * sx + ks[None, :], (rows[:, None] < M) & kmask[None, :], other=0)
        acc += tl.dot(x, w)
    if second:
        tl.store(Y2 + rows[:, None] * SY2 + (cols - N1)[None, :], acc.to(Y2.dtype.element_ty),
                 (rows[:, None] < M) & (cols[None, :] < lim))
    else:
        tl.store(Y1 + rows[:, None] * SY1 + cols[None, :], acc.to(Y1.dtype.element_ty),
                 (rows[:, None] < M) & (cols[None, :] < lim))


def gemv_dual(x1, x2, q, s, n1):
    """(x1 @ W[:n1].T, x2 @ W[n1:].T) for the 8-bit weight (q, s) of W [N, K]; bit-identical to two `gemv` calls
    whose widths get the same launch config (asserted)."""
    n, k = q.shape
    cfg = _config(n1)
    assert _config(n - n1) == cfg and n1 % cfg[0] == 0
    BN, BK, num_warps, num_stages = cfg
    a = x1.reshape(-1, k)
    b = x2.reshape(-1, k)
    m = a.shape[0]
    assert b.shape[0] == m <= MAX_M and q.stride(1) == 1 and s.stride(1) == 1
    assert a.stride(1) == 1 and b.stride(1) == 1
    y1 = torch.empty((m, n1), device=a.device, dtype=a.dtype)
    y2 = torch.empty((m, n - n1), device=a.device, dtype=a.dtype)
    if m:
        grid = (triton.cdiv(n1, BN) + triton.cdiv(n - n1, BN),)
        _gemv_w8_dual_kernel[grid](
            a, b, q.view(torch.uint8), s, y1, y2, m, n1, n, k, a.stride(0), b.stride(0), q.stride(0), s.stride(0),
            y1.stride(0), y2.stride(0), FP8=q.dtype == torch.float8_e4m3fn, BN=BN, BK=BK, MP=16,
            num_warps=num_warps, num_stages=num_stages)
    return y1.view(*x1.shape[:-1], n1), y2.view(*x2.shape[:-1], n - n1)
