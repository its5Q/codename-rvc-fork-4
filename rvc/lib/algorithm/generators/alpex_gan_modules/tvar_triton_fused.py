import torch
import triton
import triton.language as tl
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Forward Kernel
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_BT': 64,  'BLOCK_K': 64},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_BT': 64,  'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_BT': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_BT': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_BT': 256, 'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_BT': 512, 'BLOCK_K': 64},  num_warps=8, num_stages=2),
    ],
    key=['BT_total', 'K'],
)
@triton.jit
def _tvfir_fwd_kernel(
    SIGNAL, COEF, OUT,
    B, T, K,
    stride_sig_b, stride_sig_t,
    stride_coef_b, stride_coef_t, stride_coef_k,
    stride_out_b, stride_out_t,
    BT_total,
    BLOCK_BT: tl.constexpr,
    BLOCK_K:  tl.constexpr,
):
    pid = tl.program_id(0)
    bt_offsets = pid * BLOCK_BT + tl.arange(0, BLOCK_BT)
    bt_mask = bt_offsets < BT_total

    b_idx = bt_offsets // T
    t_idx = bt_offsets % T

    acc = tl.zeros([BLOCK_BT], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # CORRECTED mapping: matches PyTorch padding and as_strided offset natively
        sig_t_idx = t_idx[:, None] + k_offsets[None, :] 
        
        sig_ptr = SIGNAL + b_idx[:, None] * stride_sig_b + sig_t_idx * stride_sig_t
        sig_vals = tl.load(sig_ptr, mask=bt_mask[:, None] & k_mask[None, :], other=0.0, cache_modifier=".cg")

        coef_ptr = COEF + b_idx[:, None] * stride_coef_b + t_idx[:, None] * stride_coef_t + k_offsets[None, :] * stride_coef_k
        coef_vals = tl.load(coef_ptr, mask=bt_mask[:, None] & k_mask[None, :], other=0.0, cache_modifier=".cg")

        acc += tl.sum(sig_vals * coef_vals, axis=1)

    out_ptr = OUT + b_idx * stride_out_b + t_idx * stride_out_t
    tl.store(out_ptr, acc, mask=bt_mask)


# ---------------------------------------------------------------------------
# Backward Kernel: Gradient w.r.t Signal (dx)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_BT': 64,  'BLOCK_K': 64},  num_warps=4),
        triton.Config({'BLOCK_BT': 128, 'BLOCK_K': 64},  num_warps=4),
        triton.Config({'BLOCK_BT': 256, 'BLOCK_K': 64},  num_warps=8),
    ],
    key=['BT_total', 'K'],
)
@triton.jit
def _tvfir_bwd_dx_kernel(
    GRAD_Y, COEF, GRAD_X,
    B, T, K,
    stride_gy_b, stride_gy_t,
    stride_c_b, stride_c_t, stride_c_k,
    stride_gx_b, stride_gx_t,
    BT_total,
    BLOCK_BT: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    bt_offsets = pid * BLOCK_BT + tl.arange(0, BLOCK_BT)
    bt_mask = bt_offsets < BT_total

    b_idx = bt_offsets // T
    t_idx = bt_offsets % T

    acc = tl.zeros([BLOCK_BT], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Cross-correlation index mapping for transposed filter
        tau = t_idx[:, None] + (K - 1) - k_offsets[None, :]
        tau_mask = (tau >= 0) & (tau < T)
        valid_mask = bt_mask[:, None] & k_mask[None, :] & tau_mask

        gy_ptr = GRAD_Y + b_idx[:, None] * stride_gy_b + tau * stride_gy_t
        gy_vals = tl.load(gy_ptr, mask=valid_mask, other=0.0)

        c_ptr = COEF + b_idx[:, None] * stride_c_b + tau * stride_c_t + k_offsets[None, :] * stride_c_k
        c_vals = tl.load(c_ptr, mask=valid_mask, other=0.0)

        acc += tl.sum(gy_vals * c_vals, axis=1)

    gx_ptr = GRAD_X + b_idx * stride_gx_b + t_idx * stride_gx_t
    tl.store(gx_ptr, acc, mask=bt_mask)


# ---------------------------------------------------------------------------
# Backward Kernel: Gradient w.r.t Coefficients (dw)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_BT': 64,  'BLOCK_K': 64},  num_warps=4),
        triton.Config({'BLOCK_BT': 128, 'BLOCK_K': 64},  num_warps=4),
        triton.Config({'BLOCK_BT': 256, 'BLOCK_K': 64},  num_warps=8),
    ],
    key=['BT_total', 'K'],
)
@triton.jit
def _tvfir_bwd_dw_kernel(
    GRAD_Y, SIGNAL, GRAD_W,
    B, T, K,
    stride_gy_b, stride_gy_t,
    stride_sig_b, stride_sig_t,
    stride_gw_b, stride_gw_t, stride_gw_k,
    BT_total,
    BLOCK_BT: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_bt = tl.program_id(0)
    pid_k = tl.program_id(1)

    bt_offsets = pid_bt * BLOCK_BT + tl.arange(0, BLOCK_BT)
    bt_mask = bt_offsets < BT_total

    b_idx = bt_offsets // T
    t_idx = bt_offsets % T

    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K

    gy_ptr = GRAD_Y + b_idx * stride_gy_b + t_idx * stride_gy_t
    gy_vals = tl.load(gy_ptr, mask=bt_mask, other=0.0)

    sig_t_idx = t_idx[:, None] + k_offsets[None, :]
    sig_ptr = SIGNAL + b_idx[:, None] * stride_sig_b + sig_t_idx * stride_sig_t
    sig_vals = tl.load(sig_ptr, mask=bt_mask[:, None] & k_mask[None, :], other=0.0)

    gw_vals = gy_vals[:, None] * sig_vals

    gw_ptr = GRAD_W + b_idx[:, None] * stride_gw_b + t_idx[:, None] * stride_gw_t + k_offsets[None, :] * stride_gw_k
    tl.store(gw_ptr, gw_vals, mask=bt_mask[:, None] & k_mask[None, :])


# ---------------------------------------------------------------------------
# Autograd Function Wrapper
# ---------------------------------------------------------------------------
class TVFIRFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, signal, f_coef):
        B, T = signal.shape
        K = f_coef.shape[-1]

        padded = F.pad(signal, (K - 1, 0)).contiguous()
        f_coef = f_coef.contiguous()

        out = torch.empty((B, T), device=signal.device, dtype=signal.dtype)
        BT_total = B * T

        grid = lambda meta: (triton.cdiv(BT_total, meta['BLOCK_BT']),)
        
        _tvfir_fwd_kernel[grid](
            padded, f_coef, out,
            B, T, K,
            padded.stride(0), padded.stride(1),
            f_coef.stride(0), f_coef.stride(1), f_coef.stride(2),
            out.stride(0), out.stride(1),
            BT_total,
        )

        # Context required for computing transposed gradients
        ctx.save_for_backward(padded, f_coef)
        ctx.K = K
        
        return out

    @staticmethod
    def backward(ctx, grad_out):
        padded, f_coef = ctx.saved_tensors
        K = ctx.K
        B, T = f_coef.shape[0], f_coef.shape[1]

        grad_out = grad_out.contiguous()
        BT_total = B * T

        grad_signal = None
        if ctx.needs_input_grad[0]:
            grad_signal = torch.empty((B, T), device=grad_out.device, dtype=grad_out.dtype)
            grid_dx = lambda meta: (triton.cdiv(BT_total, meta['BLOCK_BT']),)
            
            _tvfir_bwd_dx_kernel[grid_dx](
                grad_out, f_coef, grad_signal,
                B, T, K,
                grad_out.stride(0), grad_out.stride(1),
                f_coef.stride(0), f_coef.stride(1), f_coef.stride(2),
                grad_signal.stride(0), grad_signal.stride(1),
                BT_total,
            )

        grad_f_coef = None
        if ctx.needs_input_grad[1]:
            grad_f_coef = torch.empty((B, T, K), device=grad_out.device, dtype=grad_out.dtype)
            grid_dw = lambda meta: (triton.cdiv(BT_total, meta['BLOCK_BT']), triton.cdiv(K, meta['BLOCK_K']))
            
            _tvfir_bwd_dw_kernel[grid_dw](
                grad_out, padded, grad_f_coef,
                B, T, K,
                grad_out.stride(0), grad_out.stride(1),
                padded.stride(0), padded.stride(1),
                grad_f_coef.stride(0), grad_f_coef.stride(1), grad_f_coef.stride(2),
                BT_total,
            )

        return grad_signal, grad_f_coef

# ---------------------------------------------------------------------------
# Module Definition
# ---------------------------------------------------------------------------
class TimeVarFIRFilter(torch.nn.Module):
    def __init__(self):
        super(TimeVarFIRFilter, self).__init__()

    def forward(self, signal: torch.Tensor, f_coef: torch.Tensor) -> torch.Tensor:
        if signal.dim() == 3:
            signal = signal.squeeze(1)

        if f_coef.shape[1] != signal.shape[1]: 
            f_coef = f_coef.transpose(1, 2) 

        # We call the autograd-enabled object instead of native Triton Kernel here
        out = TVFIRFunction.apply(signal, f_coef)

        return out.unsqueeze(1)