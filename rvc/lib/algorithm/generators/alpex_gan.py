import sys, os
import pathlib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
import random, time
import soundfile as sf


import math
from typing import Optional, Tuple, List
from itertools import chain

import torch
from torch import Tensor
import numpy as np

import torch.nn as nn
import torch.nn.init as init

from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

import torch.nn.functional as F

from torch.amp import autocast # guard
from torch.utils.checkpoint import checkpoint

from rvc.lib.algorithm.generators.alpex_gan_modules import PchipF0UpsamplerTorch, FusedGeoSaw, Snake


def apply_mask(tensor: torch.Tensor, mask: Optional[torch.Tensor]):
    return tensor * mask if mask else tensor

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

def remove_weight_norm_legacy_safe(module):
    if is_parametrized(module, "weight"):
        remove_parametrizations(module, "weight", leave_parametrized=True)
    else:
        remove_weight_norm(module)

def create_ups_convtranspose1d_layer(in_channels, out_channels, kernel_size, stride):
    m = torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - stride) // 2)
    return weight_norm(m)

def create_conv1d_layer(channels, kernel_size, dilation):
    return weight_norm(torch.nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation, padding=get_padding(kernel_size, dilation)))


class CyclicNoiseGenerator(nn.Module):
    """
    Cyclic noise excitation ( Wang 2020, cyc-noise-NSF ) - Inspired module
    Focused on achieving a similar thing but in a way more optimized manner.
    """
    def __init__(self, samp_rate, noise_std=0.003, unvoiced_noise_std=0.003, voiced_threshold=0):
        super().__init__()
        self.samp_rate = samp_rate
        self.noise_std = noise_std
        self.unvoiced_noise_std = unvoiced_noise_std
        self.voiced_threshold = voiced_threshold

    def _pulse_train(self, f0s, pulse_locs):
        """
        f0s: [B, T, 1] - audio-rate F0, 0 for unvoiced
        pulse_locs: [B, 1, T] bool - exact FGSS peak locations
        Returns: pulse_train, uv, pulse_noise - all [B, T, 1]
        """
        with torch.no_grad():
            uv = (f0s > self.voiced_threshold).float()

            # [B, 1, T] → [B, T, 1] to match f0s layout
            loc = pulse_locs.permute(0, 2, 1)

            noise_amp = uv * self.noise_std + (1.0 - uv) * self.unvoiced_noise_std
            pulse_noise = torch.randn_like(f0s) * noise_amp
            pulse_train = loc.float() + pulse_noise * loc.float() + pulse_noise * (1.0 - uv)

        return pulse_train, uv, pulse_noise

    def _decay_kernel(self, beta, f0mean):
        """
        Exponentially decaying noise kernel, truncated at -40 dB.
        Returns [L, 1].
        """
        with torch.no_grad():
            L = int(4.6 * self.samp_rate / f0mean)
            t = torch.arange(L, device=beta.device, dtype=torch.float32)
            decay = torch.exp(-t * float(f0mean) / beta.item() / self.samp_rate)
        noise = torch.randn(L, 1, device=beta.device, dtype=torch.float32)
        return noise * self.noise_std * decay.unsqueeze(-1)

    def _causal_conv_fft(self, signal, kernel):
        """
        FFT causal convolution. Uses overlap-add for long signals (same threshold as GaussianDecimator: 2^21 points).
        signal: [B, T, 1], kernel: [L, 1] --> [B, T, 1]
        """
        B, T, _ = signal.shape
        L = kernel.shape[0]
        conv_len = T + L - 1
        N_direct = 1 << (conv_len - 1).bit_length()

        sig_f = signal[:, :, 0].float()
        ker_f = kernel[:, 0].float()

        if N_direct <= (1 << 21):
            # Direct FFT conv - for shorter signals
            S = torch.fft.rfft(sig_f, n=N_direct)
            K = torch.fft.rfft(ker_f, n=N_direct)
            out = torch.fft.irfft(S * K, n=N_direct)[:, :T]
        else:
            # Overlap-add - for longer signals
            # Block size: next pow2 >= 4*L keeps each block FFT cheap
            N_ola = 1 << (max(2 * L, 1) - 1).bit_length()
            step = N_ola - (L - 1)
            K = torch.fft.rfft(ker_f, n=N_ola)
            out_full = torch.zeros(B, conv_len, device=signal.device, dtype=torch.float32)
            pos = 0
            while pos < T:
                chunk = sig_f[:, pos:pos + step]
                if chunk.shape[-1] < N_ola:
                    chunk = F.pad(chunk, (0, N_ola - chunk.shape[-1]))
                y_block = torch.fft.irfft(torch.fft.rfft(chunk, n=N_ola) * K, n=N_ola)
                write_end = min(pos + N_ola, conv_len)
                out_full[:, pos:write_end] += y_block[:, :write_end - pos]
                pos += step
            out = out_full[:, :T]

        return out.unsqueeze(-1).to(signal.dtype)

    def forward(self, f0s, beta, pulse_locs):
        pulse_train, uv, noise = self._pulse_train(f0s, pulse_locs)
        pure_pulse = pulse_train - noise * (1.0 - uv)

        if (uv < 1).all():
            cyc_noise = torch.zeros_like(f0s)
        else:
            f0mean = f0s[uv > 0].mean()
            kernel = self._decay_kernel(beta, f0mean)
            cyc_noise = self._causal_conv_fft(pure_pulse, kernel)

        cyc_noise = cyc_noise + noise * (1.0 - uv)
        return cyc_noise


class GaussianDecimator(nn.Module):
    """
    Single-stage decimation:
        Kaiser-windowed sinc lowpass via FFT convolution, followed by integer striding.
        No learnable parameters - purely a deterministic signal processing module.
    """
    _AA_ROLLOFF = 0.95
    _AA_STOPBAND_DB = 120.0

    def __init__(self, downsample_factor: int):
        super().__init__()
        self.factor = downsample_factor

        cutoff = self._AA_ROLLOFF / (2.0 * downsample_factor)
        delta_f = (1.0 - self._AA_ROLLOFF) / (2.0 * downsample_factor)
        beta = 0.1102 * (self._AA_STOPBAND_DB - 8.7)
        n_taps = int(math.ceil((self._AA_STOPBAND_DB - 8.0) / (2.285 * 2.0 * math.pi * delta_f)))

        if n_taps % 2 == 0:
            n_taps += 1

        half_k = (n_taps - 1) // 2
        n_arr = torch.arange(-half_k, half_k + 1, dtype=torch.float64)
        h = torch.sinc(2.0 * cutoff * n_arr)
        win = torch.kaiser_window(n_taps, periodic=False, beta=beta, dtype=torch.float64)
        h = (h * win).float()
        h = h / h.sum()

        self.register_buffer('aa_filter', h.view(1, 1, n_taps))
        self._half_k = half_k
        self._n_taps = n_taps
        self._fft_cache = {}

    def _apply(self, fn):
        self._fft_cache.clear()
        return super()._apply(fn)

    def _get_cached_filter(self, n_fft, device, dtype):
        """Retrieves or computes the FFT of the filter for a given size."""
        key = (n_fft, device, dtype)
        if key not in self._fft_cache:
            w_f32 = self.aa_filter.to(device, dtype=dtype)
            self._fft_cache[key] = torch.fft.rfft(w_f32, n=n_fft)
        return self._fft_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]  -->  [B, C, T // factor]
        b, c, t = x.shape
        taps = self._n_taps
        half_k = self._half_k

        N_fft_single = 1 << math.ceil(math.log2(t + taps - 1))

        if N_fft_single <= (1 << 21):
            W = self._get_cached_filter(N_fft_single, x.device, torch.float32)
            y = torch.fft.irfft(torch.fft.rfft(x.float(), n=N_fft_single) * W, n=N_fft_single)
            x = y[..., half_k : half_k + t].to(x.dtype)
        else:
            # Overlap-Add
            N_fft_ola = 1 << 19
            step = N_fft_ola - (taps - 1)
            W = self._get_cached_filter(N_fft_ola, x.device, torch.float32)
            x_fp32 = x.float()
            conv_len = t + taps - 1
            out_full = torch.zeros(b, c, conv_len, device=x.device, dtype=torch.float32)
            pos = 0
            while pos < t:
                chunk = x_fp32[..., pos : pos + step]
                if chunk.shape[-1] < N_fft_ola:
                    chunk = F.pad(chunk, (0, N_fft_ola - chunk.shape[-1]))
                y_block = torch.fft.irfft(torch.fft.rfft(chunk, n=N_fft_ola) * W, n=N_fft_ola)
                write_end = min(pos + N_fft_ola, conv_len)
                out_full[..., pos : write_end] += y_block[..., : write_end - pos]
                pos += step
            x = out_full[..., half_k : half_k + t].to(x.dtype)

        # Stride: integer decimation
        return x[:, :, ::self.factor].contiguous()


class SincUpsamplerFFT(nn.Module):
    """
    Ideal sinc upsampling via FFT spectral zero-padding.

    For a signal that is bandlimited to f < sr / (2 * factor) — which is true
    of any GaussianDecimator output by construction
    this is mathematically exact (Whittaker-Shannon reconstruction).
    For periodic signals (pitched speech) the periodic-sinc kernel used by DFT
    zero-padding is identical to the ideal sinc kernel, so there is no approximation error.

    Nyquist bin halving ( even-length inputs ):
        In rfft, the Nyquist bin encodes both positive and negative Nyquist
        frequency energy combined. When zero-padding expands the spectrum, this
        energy must be split so that it maps to a single interior frequency bin
        rather than remaining at the boundary.
        Without halving, there is a spectral artifact exactly at the coarse stage's Nyquist
        which is the precise frequency where the Laplacian residual boundary sits.
        The halving is applied in-place on a clone to avoid in-graph mutation.
    """

    def __init__(self, upsample_factor: int):
        super().__init__()
        self.factor = upsample_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T_coarse] --> [B, C, T_coarse * factor]
        t_coarse = x.shape[-1]
        t_fine = t_coarse * self.factor

        x_f32 = x.float()
        X = torch.fft.rfft(x_f32, n=t_coarse) # [B, C, t_coarse//2 + 1] complex

        # Nyquist bin halving for even-length input
        if t_coarse % 2 == 0:
            X = X.clone()
            X[..., -1] = X[..., -1] * 0.5

        # Zero-pad spectrum from coarse to fine length
        n_coarse_bins = t_coarse // 2 + 1
        n_fine_bins = t_fine // 2 + 1
        X_padded = F.pad(X, (0, n_fine_bins - n_coarse_bins))

        x_up = torch.fft.irfft(X_padded, n=t_fine) * self.factor # energy normalization

        return x_up.to(x.dtype)


class LaplacianExcitationPyramid(nn.Module):
    """
    Decomposes the full-rate excitation waveform into a Laplacian pyramid.

    A Laplacian pyramid is a frequency-band decomposition:
        G[i] =  Gaussian level i (lowpassed and decimated excitation)
        L[i] =  G[i] - sinc_upsample(G[i+1])  for i < coarsest
        L[-1] = G[-1] (coarsest base)

    Each L[i] contains only the frequency band addable at generator stage i:
        Stage 0 gets L[0]: 0 Hz to stage-0-Nyquist  (the base pitch structure)
        Stage 1 gets L[1]: stage-0-Nyquist to stage-1-Nyquist  (lower harmonics)
        Stage 2 gets L[2]: stage-1-Nyquist to stage-2-Nyquist  (mid harmonics)
        Stage 3 gets L[3]: stage-2-Nyquist to full Nyquist     (fine detail)

    Cascade decimation (fine-to-coarse, using upsample_rates[1:][::-1]):
        48k  [12,10,2,2]  -->  cascade factors [2, 2, 10]:  ~1,875 total taps
        40k  [10,10,2,2]  -->  cascade factors [2, 2, 10]:  ~1,875 total taps
        32k  [10,8,2,2]   -->  cascade factors [2, 2, 8]:   ~1,567 total taps
        24k  [10,6,2,2]   -->  cascade factors [2, 2, 6]:   ~1,259 total taps

    """
    def __init__(self, upsample_rates: List[int]):
        super().__init__()

        # Cascade decimation factors: reversed(upsample_rates[1:])
        # Builds Gaussian pyramid from finest to coarsest, one step at a time
        cascade_factors = list(reversed(upsample_rates[1:]))
        self.decimators = nn.ModuleList([GaussianDecimator(f) for f in cascade_factors])

        # Inter-level upsampling factors for Laplacian residuals: upsample_rates[1:]
        # Used to upsample each coarser Gaussian level back to the finer level's
        # resolution before computing the band residual
        inter_factors = upsample_rates[1:]
        self.upsamplers = nn.ModuleList([SincUpsamplerFFT(f) for f in inter_factors])

    def forward(self, exc: torch.Tensor) -> List[torch.Tensor]:
        """
        exc: [B, 1, T_audio]
        Returns: list of [B, 1, T_stage_i], length = num_upsamples
            index 0 = coarsest (stage 0), index -1 = finest (last stage)
        """
        # Build Gaussian pyramid via cascade decimation (fine --> coarse)
        # gaussian[0] = exc (full rate)
        # gaussian[k] = decimate(gaussian[k-1], cascade_factors[k-1])
        gaussian: List[torch.Tensor] = [exc]
        for dec in self.decimators:
            gaussian.append(dec(gaussian[-1]))
        # gaussian is now: [G_finest, G_mid..., G_coarsest]

        # Reverse to coarse-first for consistent indexing with generator stages
        gaussian = list(reversed(gaussian))
        # gaussian[0] = G_coarsest, gaussian[-1] = exc

        # Build Laplacian pyramid
        # L[0] = G_coarsest  (base, contains 0..stage0_nyquist)
        # L[i] = G[i] - sinc_upsample(G[i-1], inter_factors[i-1])
        laplacian: List[torch.Tensor] = [gaussian[0]]
        for i in range(1, len(gaussian)):
            g_up = self.upsamplers[i - 1](gaussian[i - 1])

            # Length guard: ±1 sample from integer rounding through cascade
            t_target = gaussian[i].shape[-1]
            if g_up.shape[-1] != t_target:
                g_up = F.pad(g_up, (0, t_target - g_up.shape[-1])) if g_up.shape[-1] < t_target else g_up[..., :t_target]

            laplacian.append(gaussian[i] - g_up)

        return laplacian


class ResBlock(torch.nn.Module):
    """
    A residual block module that applies a series of 1D convolutional layers
    with residual connections.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: Tuple[int] = (1, 3, 5),
    ):
        super().__init__()
        self.convs1 = self._create_convs(channels, kernel_size, dilations)
        self.convs2 = self._create_convs(channels, kernel_size, [1] * len(dilations))

        self.snake1 = Snake(channels, init='periodic', correction=None)
        self.snake2 = Snake(channels, init='periodic', correction=None)

    @staticmethod
    def _create_convs(channels: int, kernel_size: int, dilations: Tuple[int]):
        layers = torch.nn.ModuleList(
            [create_conv1d_layer(channels, kernel_size, d) for d in dilations]
        )
        return layers

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        for conv1, conv2 in zip(self.convs1, self.convs2):

            x_residual = x

            xt = self.snake1(x)
            xt = apply_mask(xt, x_mask)
            xt = conv1(xt)

            xt = self.snake2(xt)
            xt = apply_mask(xt, x_mask)
            xt = conv2(xt)

            x = xt + x_residual
            x = apply_mask(x, x_mask)

        return x

    def remove_weight_norm(self):
        for conv in chain(self.convs1, self.convs2):
            remove_weight_norm_legacy_safe(conv)


def _find_fgss_pulse_locs(signal: torch.Tensor, phase_cycles_f64: torch.Tensor, voiced_mask: torch.Tensor) -> torch.Tensor:
    """
    Function to find exact FGSS signal peaks within each pitch period.

    Period boundaries are defined by integer crossings of "phase_cycles_f64".
    Within each voiced period the argmax of the FGSS waveform is the exact peak location.
    """
    B, _, T = signal.shape
    device = signal.device

    # Period label per sample
    period_id = torch.floor(phase_cycles_f64).long()
    pid_offset = period_id[:, :, :1]
    pid_norm = (period_id - pid_offset).clamp(min=0)
    P = int(pid_norm.max().item()) + 1

    # Mask unvoiced to -inf
    NEG_INF = float('-inf')
    sig_masked = torch.where(voiced_mask > 0.5, signal, torch.full_like(signal, NEG_INF)) # [B, 1, T]

    # Per-period maximum value
    period_max = sig_masked.new_full((B, 1, P), NEG_INF)
    period_max.scatter_reduce_(2, pid_norm, sig_masked, reduce='amax', include_self=True)

    # Candidate mask: voiced sample whose value == its period's max
    sample_max = period_max.gather(2, pid_norm) # [B, 1, T]
    is_candidate = (sig_masked == sample_max) & (voiced_mask > 0.5)

    # Keep only the first candidate per period ( handles ties )
    # global_cumsum[t] counts candidates in [0..t].
    cand_f = is_candidate.float()
    global_cs = torch.cumsum(cand_f, dim=2) # [B, 1, T]
    cs_before = global_cs - cand_f # cumsum before t

    # period-start positions: where floor(phase_cycles_f64) increments
    floor_curr = period_id.float()
    floor_prev = F.pad(floor_curr[:, :, :-1], (1, 0), value=float((pid_offset - 1).min().item()))
    is_start = (floor_curr > floor_prev) # [B, 1, T]

    # For each period, gather cs_before at its start sample via scatter amin.
    # Non-start positions contribute +inf so they never win.
    cs_at_start = torch.where(is_start, cs_before, torch.full_like(cs_before, float('inf')))
    period_offset = sig_masked.new_full((B, 1, P), float('inf'))
    period_offset.scatter_reduce_(2, pid_norm, cs_at_start, reduce='amin', include_self=True)
    period_offset = period_offset.clamp(min=0.0)

    # Local cumsum within each period
    local_cs = global_cs - period_offset.gather(2, pid_norm) # [B, 1, T]

    # Pulse = first candidate in its period ( local_cs transitions 0→1 here )
    pulse_locs = is_candidate & (local_cs == 1.0)

    return pulse_locs


def fgss_generator(
    f0: torch.Tensor,
    hop_length: int,
    sample_rate: int,
    r: float = 0.92,
    random_init_phase: bool = False,
    power_factor: float = 0.1,
    max_frequency: Optional[float] = None,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Finite Geometric Sine Series (FGSS) excitation signal generator.
    """
    batch, _, _ = f0.size()
    device = f0.device

    upsampler = PchipF0UpsamplerTorch(scale_factor=hop_length).to(device)
    f0_upsampled = upsampler(f0)

    if torch.all(f0_upsampled < 1.0):
        _, _, total_length = f0_upsampled.size()
        zeros = torch.zeros((batch, 1, total_length), device=device, dtype=f0_upsampled.dtype)
        return zeros, zeros, torch.zeros((batch, 1, total_length), dtype=torch.bool, device=device)

    voiced_mask = (f0_upsampled > 1.0).float()

    phase_increment_f64 = f0_upsampled.double() / sample_rate
    if random_init_phase:
        init_phase = torch.rand((batch, 1, 1), device=device, dtype=torch.float64)
        phase_increment_f64[:, :, :1] += init_phase
    phase_cycles_f64 = torch.cumsum(phase_increment_f64, dim=2)

    # Phase for FusedGeoSaw: frac of cycle in [0, 2π), exactly 0 at period start
    frac = phase_cycles_f64 % 1.0
    floor_curr = torch.floor(phase_cycles_f64)
    floor_prev = F.pad(floor_curr[:, :, :-1], (1, 0), value=0.0)
    reset = (floor_curr > floor_prev) & (f0_upsampled > 1.0)
    frac_snapped = torch.where(reset, torch.zeros_like(frac), frac)
    phase = (frac_snapped * (2.0 * torch.pi)).float()

    nyquist = sample_rate / 2.0
    limit_freq = max_frequency if max_frequency is not None else nyquist
    safe_f0 = torch.clamp(f0_upsampled, min=1e-5)
    N = torch.floor(limit_freq / safe_f0)

    # Fused kernel for FGSS
    harmonics = FusedGeoSaw.apply(phase, N, r, epsilon)

    # Normalization
    r2  = r * r
    r2N = torch.pow(torch.tensor(r, device=device, dtype=phase.dtype), 2.0 * N)
    amp_scale = power_factor * torch.sqrt(2.0 * (1.0 - r2) / (r2 * torch.clamp(1.0 - r2N, min=epsilon)))

    signal = harmonics * amp_scale * voiced_mask

    # Exact pulse locations: argmax of signal within each period window.
    # Uses phase_cycles_f64 integer crossings as period boundaries
    # same source as frac_snapped, so windows are coherent with FusedGeoSaw.
    pulse_locs = _find_fgss_pulse_locs(signal, phase_cycles_f64, voiced_mask)

    return signal, f0_upsampled, pulse_locs


class ExcitationSynthesizer(torch.nn.Module):
    """
    Synthesizes the excitation signal from F0.
    """
    def __init__(
        self,
        sample_rate: int,
        hop_length: int = 480,
        random_init_phase: bool = False,
        power_factor: float = 0.1,
        add_noise_std: float = 0.003,
        beta: float = 0.870
    ):
        super(ExcitationSynthesizer, self).__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.random_init_phase = random_init_phase
        self.power_factor = power_factor
        self.noise_std = add_noise_std
        self.beta = beta

        self.l_cyc_noise = CyclicNoiseGenerator(
            samp_rate=sample_rate,
            noise_std=add_noise_std,
            unvoiced_noise_std=power_factor / 3.0,
            voiced_threshold=0,
        )

        self.l_linear = torch.nn.Linear(1, 1, bias=False)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, f0: torch.Tensor, g: Optional[torch.Tensor] = None, upsample_factor: int = None):
        if f0.dim() == 2:
            f0 = f0.unsqueeze(1)

        hop = upsample_factor if upsample_factor is not None else self.hop_length

        with autocast('cuda', enabled=False):
            f0 = f0.float()

            with torch.no_grad():
                fgss_harmonic_signal, f0_upsampled, pulse_locs = fgss_generator(
                    f0,
                    hop_length=hop,
                    sample_rate=self.sample_rate,
                    random_init_phase=self.random_init_phase,
                    power_factor=self.power_factor,
                )

                f0_for_cyc = f0_upsampled.permute(0, 2, 1)  # [B, 1, T] -> [B, T, 1]
                f0_for_cyc = f0_for_cyc * (f0_for_cyc > 1.0).float()
                beta = torch.ones(1, 1, 1, device=f0.device) * self.beta

                cyc_noise = self.l_cyc_noise(f0_for_cyc, beta, pulse_locs)
                cyc_noise = cyc_noise.permute(0, 2, 1)  # [B, T, 1] -> [B, 1, T]
                excitation_signal = fgss_harmonic_signal + cyc_noise

        excitation_signal = excitation_signal.to(dtype=self.l_linear.weight.dtype)
        excitation_signal = excitation_signal.transpose(1, 2)
        excitation = self.l_tanh(self.l_linear(excitation_signal))
        excitation = excitation.transpose(1, 2)

        return excitation


class ALPEX_GAN_Generator(nn.Module):
    """
    Experimental neural vocoder for GAN-based voice synthesis.

    ALPEX stands for:
        A  — Adaptive harmonics  ( N scales with F0 to stay below Nyquist )
        L  — Laplacian pyramid   ( excitation decomposed into per-stage frequency bands )
        P  — Pyramid injection   ( each generator stage receives its own band )
        EX — Complex excitation  ( band-limited sawtooth + cyclic noise source )
    """

    def __init__(
        self,
        initial_channel, # 192
        resblock_kernel_sizes, # [3, 7, 11]
        resblock_dilation_sizes, # variable
        upsample_rates, # variable
        upsample_initial_channel, # 512
        upsample_kernel_sizes, # variable
        gin_channels, # 256
        sr,
        use_inplace: bool = True,
    ):
        super(ALPEX_GAN_Generator, self).__init__()
        self.use_inplace = use_inplace
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        total_ups_factor = math.prod(upsample_rates)

        # Excitation source
        self.excitation_synthesizer = ExcitationSynthesizer(
            sample_rate=sr,
            hop_length=total_ups_factor,
            random_init_phase=False,
            power_factor=0.1,
            add_noise_std=0.003,
        )

        # Laplacian pyramid: decomposes full-rate excitation into per-stage bands
        self.laplacian_pyramid = LaplacianExcitationPyramid(upsample_rates=upsample_rates)

        # Pre convolution
        self.conv_pre = weight_norm(Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3))

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        self.exc_proj = nn.ModuleList()

        ch = upsample_initial_channel  # 512
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            ch //= 2  # 256 --> 128 --> 64 --> 32

            # Upsamplers
            self.ups.append(create_ups_convtranspose1d_layer(2 * ch, ch, k, u))

            # Residual blocks with snake activation
            for j, (kk, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, kk, d))

            # Excitation signal projection
            self.exc_proj.append(
                nn.Conv1d(1, ch, kernel_size=7, padding=3, bias=False))

        # Post convolution 
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3, bias=False))

        # Speaker embedding conditioning
        if gin_channels != 0:
            self.cond = Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x: torch.Tensor, f0: torch.Tensor, g: Optional[torch.Tensor] = None):
        # x:  [B, 192, T]
        # f0: [B, T] or [B, 1, T]

        # Generate full-rate excitation waveform [B, 1, T]
        excitation = self.excitation_synthesizer(f0, g=g)

        # Decompose into Laplacian bands - one per stage, coarse to fine
        laplacian: List[torch.Tensor] = self.laplacian_pyramid(excitation)

        # Feature pre-conv ( 192 -> 512 )
        x = self.conv_pre(x)

        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.silu(x, inplace=self.use_inplace)
            x = self.ups[i](x)

            # Project Laplacian band [B,1,T] --> [B,ch,T] and inject
            exc_i = self.exc_proj[i](laplacian[i])

            # Length guard: ConvTranspose1d ±1 with odd input lengths
            if exc_i.shape[-1] != x.shape[-1]:
                diff = x.shape[-1] - exc_i.shape[-1]
                exc_i = F.pad(exc_i, (0, diff)) if diff > 0 else exc_i[..., :x.shape[-1]]

            if self.use_inplace:
                x.add_(exc_i)
            else:
                x = x + exc_i

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.silu(x, inplace=self.use_inplace)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        # pre convolution
        remove_weight_norm_legacy_safe(self.conv_pre)
        # upsamplers
        for l in self.ups:
            remove_weight_norm_legacy_safe(l)
        # ResBlocks
        for l in self.resblocks:
            l.remove_weight_norm()
        # post convolution
        remove_weight_norm_legacy_safe(self.conv_post)

    def __prepare_scriptable__(self):
        # pre convolution
        for hook in self.conv_pre._forward_pre_hooks.values():
            if (
                hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                and hook.__class__.__name__ == "WeightNorm"
            ):
                remove_weight_norm_legacy_safe(self.conv_pre)
        # upsamplers
        for l in self.ups:
            for hook in l._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    remove_weight_norm_legacy_safe(l)
        # ResBlocks
        for l in self.resblocks:
            for hook in l._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    remove_weight_norm_legacy_safe(l)
        # post convolution
        for hook in self.conv_post._forward_pre_hooks.values():
            if (
                hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                and hook.__class__.__name__ == "WeightNorm"
            ):
                remove_weight_norm_legacy_safe(self.conv_post)

        return self
