from typing import Union, Tuple
import torch
import triton
import triton.language as tl
# ---------------------------------------------------------------------------
# "Short" Mode: Standard Sequential IIR ( For short sequences / training )
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
    ],
    key=['N_elements', 'T'],
)
@triton.jit
def _iir_fwd_kernel(
    X, ALPHA, Y,
    H_IN, H_OUT,
    stride_bc, stride_t,
    N_elements, T,
    HAS_H_IN:  tl.constexpr,
    HAS_H_OUT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_elements

    tl.assume(stride_t == 1)
    tl.assume(T > 0)
    
    if HAS_H_IN:
        h = tl.load(H_IN + offsets, mask=mask, other=0.0)
    else:
        h = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for t in range(T):
        ptrs_x = X + offsets * stride_bc + t * stride_t
        ptrs_a = ALPHA + offsets * stride_bc + t * stride_t
        ptrs_y = Y + offsets * stride_bc + t * stride_t

        x_t = tl.load(ptrs_x, mask=mask, other=0.0, cache_modifier=".cg")
        a_t = tl.load(ptrs_a, mask=mask, other=0.0, cache_modifier=".cg")

        h = a_t * h + x_t

        tl.store(ptrs_y, h, mask=mask)

    if HAS_H_OUT:
        tl.store(H_OUT + offsets, h, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
    ],
    key=['N_elements', 'T'],
)
@triton.jit
def _iir_bwd_kernel(
    GRAD_Y, ALPHA, Y,
    GRAD_X, GRAD_ALPHA,
    stride_bc, stride_t,
    N_elements, T,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_elements

    tl.assume(stride_t == 1)
    tl.assume(T > 0)

    dx_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for t_rev in range(T):
        t = T - 1 - t_rev

        ptrs_g = GRAD_Y + offsets * stride_bc + t * stride_t
        g_t = tl.load(ptrs_g, mask=mask, other=0.0, cache_modifier=".cg")

        a_next_mask = mask & (t < T - 1)
        ptrs_a_next = ALPHA + offsets * stride_bc + (t + 1) * stride_t
        a_next = tl.load(ptrs_a_next, mask=a_next_mask, other=0.0, cache_modifier=".cg")

        dx_acc = g_t + a_next * dx_acc

        ptrs_dx = GRAD_X + offsets * stride_bc + t * stride_t
        tl.store(ptrs_dx, dx_acc, mask=mask)

        y_prev_mask = mask & (t > 0)
        ptrs_y_prev = Y + offsets * stride_bc + (t - 1) * stride_t
        y_prev = tl.load(ptrs_y_prev, mask=y_prev_mask, other=0.0, cache_modifier=".cg")

        da_t = y_prev * dx_acc
        ptrs_da = GRAD_ALPHA + offsets * stride_bc + t * stride_t
        tl.store(ptrs_da, da_t, mask=mask)


class FusedFastIIR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        assert x.dim() == 3 and alpha.dim() == 3
        B, C, T = x.shape

        x = x.contiguous()
        alpha = alpha.contiguous()
        y = torch.empty_like(x)

        N_elements = B * C
        grid = lambda meta: (triton.cdiv(N_elements, meta['BLOCK_SIZE']),)

        _iir_fwd_kernel[grid](
            x, alpha, y,
            alpha, alpha,
            T, 1,
            N_elements, T,
            False, False
        )

        ctx.save_for_backward(alpha, y)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        alpha, y = ctx.saved_tensors
        B, C, T = y.shape

        grad_y = grad_y.contiguous()
        grad_x = torch.empty_like(y)
        grad_alpha = torch.empty_like(alpha)

        N_elements = B * C
        grid = lambda meta: (triton.cdiv(N_elements, meta['BLOCK_SIZE']),)

        _iir_bwd_kernel[grid](
            grad_y, alpha, y,
            grad_x, grad_alpha,
            T, 1,
            N_elements, T
        )

        return grad_x, grad_alpha


# ---------------------------------------------------------------------------
# "Long" Mode: Associative Scan IIR ( Forward Only, Fast Inference on Long T )
# ---------------------------------------------------------------------------

@triton.jit
def combine_fn(a1, x1, a2, x2):
    """
    Tuple combinator for the parallel prefix sum. 
    Applies the recurrent relation statematically.
    """
    return a2 * a1, a2 * x1 + x2


@triton.jit
def _iir_fwd_long_kernel(
    X, ALPHA, Y,
    stride_bc, stride_t,
    N_elements, T,
    BLOCK_BC: tl.constexpr = 32,
    BLOCK_T: tl.constexpr = 1024
):
    # Process a 2D block: [BLOCK_BC, BLOCK_T]
    pid = tl.program_id(0)
    bc_offsets = pid * BLOCK_BC + tl.arange(0, BLOCK_BC)
    mask_bc = bc_offsets < N_elements

    # Carry state across time blocks
    h_prev = tl.zeros([BLOCK_BC], dtype=tl.float32)
    tl.assume(stride_t == 1)

    for t_start in range(0, T, BLOCK_T):
        t_offsets = t_start + tl.arange(0, BLOCK_T)
        mask = mask_bc[:, None] & (t_offsets[None, :] < T)

        # Compute pointers
        offs = bc_offsets[:, None] * stride_bc + t_offsets[None, :] * stride_t
        x = tl.load(X + offs, mask=mask, other=0.0)
        a = tl.load(ALPHA + offs, mask=mask, other=0.0)

        # Fast parallel scan
        a_scan, x_scan = tl.associative_scan((a, x), axis=1, combine_fn=combine_fn)

        # Apply carry from previous block
        y_chunk = x_scan + a_scan * h_prev[:, None]

        # Update carry for next block
        last_idx = tl.minimum(BLOCK_T, T - t_start) - 1
        col_mask = tl.arange(0, BLOCK_T)[None, :] == last_idx
        h_prev   = tl.sum(tl.where(col_mask, y_chunk, 0.0), axis=1)

        tl.store(Y + offs, y_chunk, mask=mask)

def fast_iir_filter_long(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    B, C, T = x.shape
    x, alpha = x.contiguous(), alpha.contiguous()
    y = torch.empty_like(x)
    N_elements = B * C

    BLOCK_BC = 32
    grid = (triton.cdiv(N_elements, BLOCK_BC),)

    _iir_fwd_long_kernel[grid](
        x, alpha, y,
        T, 1,
        N_elements, T
    )
    return y


# ---------------------------------------------------------------------------
# Unified Interface
# ---------------------------------------------------------------------------

def fast_iir_filter_triton(
    x: torch.Tensor,
    alpha: torch.Tensor,
    mode: str = "short_train",
    initial_state: torch.Tensor = None
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

    B, C, T = x.shape
    N_elements = B * C

    if mode == "long_infer":
        return fast_iir_filter_long(x, alpha), None

    elif mode == "short_infer":
        x, alpha = x.contiguous(), alpha.contiguous()
        y = torch.empty_like(x)

        h_out = torch.empty((B, C), device=x.device, dtype=x.dtype)

        has_h_in = initial_state is not None
        h_in = initial_state.contiguous() if has_h_in else alpha

        grid = lambda meta: (triton.cdiv(N_elements, meta['BLOCK_SIZE']),)

        _iir_fwd_kernel[grid](
            x, alpha, y,
            h_in, h_out,
            T, 1,
            N_elements, T,
            has_h_in, True
        )

        return y, h_out

    elif mode == "short_train":
        return FusedFastIIR.apply(x, alpha), None