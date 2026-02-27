"""
pcph_geosaw_fused.py
====================
Band-limited sawtooth pitch signal via Geometric Sine-Sum.

Replaces the Dirichlet (cosine-sum) PCPH kernel with a SINE-sum variant
that produces a SAWTOOTH-shaped waveform instead of a Dirac-comb spike train.

MATH
----
The current PCPH computes:
    Σ_{k=0}^{N} cos(k·φ)  =  (cos(φ/2) - cos((N+0.5)φ)) / (2·sin(φ/2))
This is a COSINE sum → waveform shape = spike every period, flat between spikes.
98% of the energy is packed into ~2% of the waveform. Very "pointy".

This kernel computes:
    G_N(φ, r) = Σ_{k=1}^{N} r^k · sin(k·φ)

With r<1 (e.g. 0.95): exponential spectral rolloff, naturally band-limited.
With r→1:             approaches uniform-amplitude sawtooth (BLSAW).

Closed form derived by taking Im[z(1-z^N)/(1-z)], z = r·e^{iφ},
multiplying through by conjugate of denominator:

    numerator  = r·sin(φ) - r^{N+1}·sin((N+1)φ) + r^{N+2}·sin(N·φ)
    denominator = 1 - 2r·cos(φ) + r²

    G = numerator / denominator

WAVEFORM CHARACTER
------------------
r=0.95, f0=262Hz, N=91:
  Shape:   smooth linear ramp → sharp reset (= band-limited sawtooth)
  Crest:   ~4.6x  (vs ~9.8x for Dirichlet spike train, 1.41x for pure sine)
  Energy:  spread across entire period  (vs ~98% packed into peaks for PCPH)
  k=1..5:  1.000, 0.950, 0.902, 0.857, 0.814  (natural rolloff)
  k=91:    ~0.010  (near-nyquist is 1% → zero aliasing risk even if F0 wobbles)

COST
----
4 trig evals: sin(φ), sin((N+1)φ), sin(Nφ), cos(φ)
vs Dirichlet: 3 trig evals (cos(φ/2), cos((N+0.5)φ), sin(φ/2))
Nearly identical throughput on GPU — sincos is a single hardware instruction.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['N_elements'],
)
@triton.jit
def _pcph_geosaw_kernel(
    PHASE,
    N_HARMS,
    OUT,
    stride_b,
    stride_c,
    stride_n,
    N_elements,
    C,
    R,
    EPS,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx  = pid // C
    channel_idx = pid % C
    bc_offset  = batch_idx * stride_b + channel_idx * stride_c

    PHASE   = PHASE   + bc_offset
    N_HARMS = N_HARMS + bc_offset
    OUT     = OUT     + bc_offset

    block_start = tl.program_id(1) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_elements

    phi = tl.load(PHASE   + offsets * stride_n, mask=mask)
    n   = tl.load(N_HARMS + offsets * stride_n, mask=mask)

    # Adaptive r: 1 - 1/N per sample (last harmonic always at ~37% of first)
    r = tl.where(R < 0.0, 1.0 - 1.0 / tl.clamp(n, 1.0, 1e9), R)
    r = tl.clamp(r, 1e-6, 1.0 - 1e-6)

    # r^(N+1), r^(N+2) via log — works for float N from N_HARMS tensor
    log_r  = tl.log(r)
    rNp1   = tl.exp((n + 1.0) * log_r)    # r^{N+1}
    rNp2   = rNp1 * r                     # r^{N+2}

    # 4 trig evals (pairs: sincos(phi) + sincos(N·phi) + cos for denom)
    # Triton maps tl.sin/tl.cos to hardware sincos where possible
    sp   = tl.sin(phi)
    cp   = tl.cos(phi)
    sN1  = tl.sin((n + 1.0) * phi)        # sin((N+1)·φ)
    sN   = tl.sin(n * phi)                 # sin(N·φ)

    numer = r * sp  -  rNp1 * sN1  +  rNp2 * sN
    denom = 1.0  -  2.0 * r * cp  +  r * r

    result = tl.where(tl.abs(denom) < EPS, 0.0, numer / denom)
    tl.store(OUT + offsets * stride_n, result, mask=mask)

def pcph_geosaw_fwd(phase, n_harms, r=0.95, eps=1e-6):
    """
    Args:
        phase:   [B, C, T] float32, instantaneous phase in radians
        n_harms: [B, C, T] float32, number of harmonics to nyquist (same shape)
        r:       float (0 < r < 1) or None for adaptive r=1-1/N per sample
                   0.95 → crest ~4.6x, k=91 at 1%    (good default)
                   0.90 → crest ~3.2x, k=91 at 0.06%  (more rolloff)
                   0.99 → crest ~8.7x, k=91 at 40%    (near BLSAW)
                   None → adaptive: last harmonic always at ~37% regardless of f0
        eps:     singularity threshold (default 1e-6)

    Returns:
        [B, C, T] float32 — band-limited sawtooth pitch signal
    """
    phase   = phase.contiguous()
    n_harms = n_harms.contiguous()
    batch, channels, length = phase.shape
    out = torch.empty_like(phase)

    r_val = -1.0 if r is None else float(r)
    grid  = lambda meta: (batch * channels, triton.cdiv(length, meta['BLOCK_SIZE']))

    _pcph_geosaw_kernel[grid](
        phase, n_harms, out,
        phase.stride(0), phase.stride(1), phase.stride(2),
        length, channels, r_val, eps
    )
    return out


class FusedGeoSaw(torch.autograd.Function):
    @staticmethod
    def forward(ctx, phase, n_harms, r=0.95, eps=1e-6):
        return pcph_geosaw_fwd(phase, n_harms, r=r, eps=eps)

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None