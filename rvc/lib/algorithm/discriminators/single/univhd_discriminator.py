"""
PyTorch implementation of UnivHD:

,, A Universal Harmonic Discriminator for High-quality GAN-based Vocoders ''
Nan Xu, Zhaolong Huang, Xiao Zeng  (arXiv:2512.03486v1, Dec 2025)

"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


def _next_pow2(x: float) -> int:
    return 2 ** math.ceil(math.log2(x))


def _derive_n_fft(sample_rate: int) -> int:
    """
    Scale paper's n_fft=1024@24kHz to nearest power-of-2.
    Ensures Hz/bin < ERB(fmin) ~ 28 Hz (filter resolution constraint).

        24 kHz -> 1024  (23.4 Hz/bin)
        32 kHz -> 2048  (15.6 Hz/bin)
        40 kHz -> 2048  (19.5 Hz/bin)
        48 kHz -> 2048  (23.4 Hz/bin)
    """
    return _next_pow2(1024 * sample_rate / 24_000)


def _derive_hop(sample_rate: int) -> int:
    """
    Scale hop=256@24kHz linearly -> ~10.7 ms/frame at any sample rate.
    T_frames is then nearly constant across rates for the same audio duration.
    """
    return round(256 * sample_rate / 24_000)


def _compute_n_bins(sample_rate: int, n_harmonics: int, bins_per_octave: int, fmin: float) -> int:
    """
    Number of harmonic frequency bins F.  (§ IV-B: 124 at 24 kHz)

    fmax_first = fs / (2*H)   [Nyquist bound for first harmonic, § III-A]
    F = floor(B * log2(fmax_first / fmin))   [CQT convention, Eq. 5]

        24 kHz -> 124
        32 kHz -> 134
        40 kHz -> 142
        48 kHz -> 148
    """
    fmax_first = sample_rate / (2.0 * n_harmonics)
    return int(math.floor(bins_per_octave * math.log2(fmax_first / fmin)))


def _freq_after_mdc(f: int, n_mdc: int = 3, k: int = 5, stride: int = 2, pad: int = 2) -> int:
    """Frequency dimension after n_mdc MDC blocks (each has stride-2 final conv)."""
    for _ in range(n_mdc):
        f = math.floor((f + 2 * pad - k) / stride) + 1
    return f


class HarmonicFilter(nn.Module):
    """
    Triangular band-pass harmonic filter bank with learnable bandwidths.

    Input:  STFT magnitude [B, N_stft, T]
    Output: harmonic tensor [B, n_total, F, T]

    All buffers (fc, stft_freqs, harmonic_orders) are derived from sample_rate.
    """

    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        n_harmonics: int = 10,
        bins_per_octave: int = 24,
        fmin: float = 32.7,
        add_half_harmonic: bool = True,
    ) -> None:
        super().__init__()

        self.n_fft = n_fft
        self.n_bins = _compute_n_bins(sample_rate, n_harmonics, bins_per_octave, fmin)

        # fc_k = fmin * 2^(k/B),  k=0..F-1
        k  = torch.arange(self.n_bins, dtype=torch.float32)
        fc = fmin * torch.pow(2.0, k / bins_per_octave)
        self.register_buffer("fc", fc)

        # STFT frequency axis
        stft_freqs = torch.arange(n_fft // 2 + 1, dtype=torch.float32) * (sample_rate / n_fft)
        self.register_buffer("stft_freqs", stft_freqs)

        # Harmonic orders: [0.5, 1, 2, ..., H]
        orders = ([0.5] if add_half_harmonic else []) + [float(h) for h in range(1, n_harmonics + 1)]
        self.n_total = len(orders)
        self.register_buffer("harmonic_orders", torch.tensor(orders, dtype=torch.float32))

        # Single scalar controls overall bandwidth scaling for all harmonics.
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, stft_mag: torch.Tensor) -> torch.Tensor:
        """[B, N_stft, T] -> [B, n_total, F, T]"""
        gamma = torch.clamp(self.gamma, min=1.0)

        h_fc = self.harmonic_orders.unsqueeze(1) * self.fc.unsqueeze(0)
        h_bw = (0.1079 * h_fc + 24.7) / gamma.unsqueeze(1)

        diff = (self.stft_freqs.unsqueeze(0).unsqueeze(0) - h_fc.unsqueeze(2)).abs()

        filter_bank = F.relu(1.0 - 2.0 * diff / h_bw.unsqueeze(2))

        return torch.einsum("hfn,bnt->bhft", filter_bank, stft_mag)


class HybridConvBlock(nn.Module):
    """
    HCB: depthwise-separable branch (intra-harmonic) + normal conv branch
    """

    def __init__(self, in_channels: int, out_channels: int = 32, kernel_size: Tuple[int, int] = (7, 7)) -> None:
        super().__init__()
        pad = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.ds_conv     = weight_norm(nn.Conv2d(in_channels, in_channels, kernel_size, padding=pad, groups=in_channels))
        self.p_conv      = weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.normal_conv = weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, in_ch, F, T] -> [B, out_ch, F, T]"""
        return self.p_conv(self.ds_conv(x)) + self.normal_conv(x)


class MultiScaleDilatedConv(nn.Module):
    """
    MDC: dilated convs (d=1,2,4) -> LeakyReLU -> stride-(2,1) conv.
    """

    def __init__(
        self,
        in_channels:    int,
        out_channels:   int             = 32,
        kernel_size:    int             = 5,
        dilation_rates: Tuple[int, ...] = (1, 2, 4),
        lrelu_slope:    float           = 0.1,
    ) -> None:
        super().__init__()
        self.lrelu_slope = lrelu_slope
        k = kernel_size

        layers, ch = [], in_channels
        for d in dilation_rates:
            layers.append(weight_norm(nn.Conv2d(ch, out_channels, (k, k), stride=(1, 1), dilation=(d, 1), padding=(d*(k-1)//2, (k-1)//2))))
            ch = out_channels
        self.dilated_convs = nn.ModuleList(layers)

        self.final_conv = weight_norm(nn.Conv2d(out_channels, out_channels, (k, k), stride=(2, 1), padding=((k-1)//2, (k-1)//2)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, in_ch, F, T] -> [B, out_ch, ceil(F/2), T]"""
        for conv in self.dilated_convs:
            x = conv(x)
        return self.final_conv(F.leaky_relu(x, self.lrelu_slope))


class UnivHD(nn.Module):
    """
    Universal Harmonic Discriminator.

    All parameters are derived from sample_rate in __init__:
        n_fft, hop_length -> _derive_n_fft / _derive_hop
        n_bins (F)        -> _compute_n_bins (from fmax_first = sr/(2*H))
        final_conv kernel -> _freq_after_mdc(n_bins)   <- the key sr-varying piece

    Usage:
        disc = UnivHD(sample_rate=48_000)
        logits, feat_maps = disc(waveform)

    Returns:
        logits    [B, T_frames]   raw scores (no sigmoid)
        feat_maps List[Tensor]x4  [HCB, MDC1, MDC2, MDC3] for feature-matching loss
    """

    _N_MDC:   int = 3
    _HCB_OUT: int = 32
    _MDC_OUT: int = 32

    def __init__(
        self,
        sample_rate: int,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        n_harmonics: int = 10,
        bins_per_octave: int = 24,
        fmin: float = 32.7,
        add_half_harmonic: bool = True,
        lrelu_slope: float = 0.1,
    ) -> None:
        super().__init__()

        n_fft = n_fft if n_fft is not None else _derive_n_fft(sample_rate)
        hop_length = hop_length if hop_length is not None else _derive_hop(sample_rate)
        win_length = win_length if win_length is not None else n_fft

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.register_buffer("window", torch.hann_window(win_length))

        # Harmonic filter
        self.harmonic_filter = HarmonicFilter(
            sample_rate=sample_rate, n_fft=n_fft,
            n_harmonics=n_harmonics, bins_per_octave=bins_per_octave,
            fmin=fmin, add_half_harmonic=add_half_harmonic,
        )
        n_total = self.harmonic_filter.n_total   # H + 1 = 11

        # HCB
        self.hcb = HybridConvBlock(in_channels=n_total, out_channels=self._HCB_OUT)

        # MDC x 3
        self.mdc_blocks = nn.ModuleList([
            MultiScaleDilatedConv(
                in_channels  = self._HCB_OUT if i == 0 else self._MDC_OUT,
                out_channels = self._MDC_OUT,
                lrelu_slope  = lrelu_slope,
            )
            for i in range(self._N_MDC)
        ])

        freq_kernel = _freq_after_mdc(self.harmonic_filter.n_bins)
        self.final_conv = weight_norm(nn.Conv2d(self._MDC_OUT, 1, kernel_size=(freq_kernel, 1)))

    def _stft_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, T] -> squeeze to [B, T] for torch.stft
        return torch.stft(x.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length,
                          win_length=self.win_length, window=self.window,
                          center=True, return_complex=True).abs()

    def forward(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # waveform: [B, 1, T]
        x = self.harmonic_filter(self._stft_magnitude(waveform))

        feat_maps: List[torch.Tensor] = []
        x = self.hcb(x)

        for mdc in self.mdc_blocks:
            x = mdc(x)
            feat_maps.append(x)

        return self.final_conv(x).squeeze(1).squeeze(1), feat_maps


class UniversalHarmonicDiscriminator(nn.Module):
    """
    Multi-scale wrapper for UnivHD.

    Takes (y, y_hat) and returns (y_d_rs, y_d_gs, fmap_rs, fmap_gs).

    UnivHD is a single-STFT-resolution discriminator by design
    (multi-scale comes from its internal MDC blocks, not from running multiple STFTs).
    This wrapper holds a single UnivHD instance for the given sample_rate.
    """

    def __init__(self, sample_rate: int, **kwargs) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList([UnivHD(sample_rate=sample_rate, **kwargs)])

    def forward(
        self,
        y:     torch.Tensor,
        y_hat: torch.Tensor,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []

        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
