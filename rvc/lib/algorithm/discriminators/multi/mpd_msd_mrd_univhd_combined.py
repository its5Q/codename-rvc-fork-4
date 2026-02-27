import math
from typing import Optional, List, Union, Dict, Tuple


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.parametrizations import weight_norm, spectral_norm
from torch.nn import Conv2d

from torchaudio.transforms import Spectrogram, Resample

from torch.utils.checkpoint import checkpoint
from rvc.train.utils import AttrDict

from rvc.lib.algorithm.commons import get_padding
from rvc.lib.algorithm.residuals import LRELU_SLOPE


class MPD_MSD_MRD_UnivHD_Combined(torch.nn.Module):
    """
    MPD + MSD + MRD + UnivHD — four complementary discriminators.

      MSD    (DiscriminatorS) - multi-scale waveform consistency
      MPD x5 (DiscriminatorP) - periodic structure (periods 2,3,5,7,11)
      MRD x3 (DiscriminatorR) - fine-grained flat spectral detail (sibilants)
      UnivHD                  - harmonic-aware dynamic spectral resolution
    """

    def __init__(
        self,
        sample_rate:       int,
        use_spectral_norm: bool = False,
        use_checkpointing: bool = False,
        **multi_resolution_cfg,
    ):
        super().__init__()
        self.use_checkpointing = use_checkpointing

        periods     = [2, 3, 5, 7, 11]
        resolutions = multi_resolution_cfg["resolutions"]
        assert len(resolutions) == 3, \
            f"MRD requires exactly 3 resolutions, got {resolutions}"

        self.discriminators = torch.nn.ModuleList(
            [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
            + [DiscriminatorP(p, use_spectral_norm=use_spectral_norm) for p in periods]
            + [DiscriminatorR(multi_resolution_cfg, res) for res in resolutions]
            + [UnivHD(sample_rate=sample_rate)]
        )

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []

        for d in self.discriminators:
            if self.training and self.use_checkpointing:
                y_d_r, fmap_r = checkpoint(d, y, use_reentrant=False)
                y_d_g, fmap_g = checkpoint(d, y_hat, use_reentrant=False)
            else:
                y_d_r, fmap_r = d(y)
                y_d_g, fmap_g = d(y_hat)

            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    """
    Discriminator for the short-term component.

    This class implements a discriminator for the short-term component
    of the audio signal. The discriminator is composed of a series of
    convolutional layers that are applied to the input signal.
    """

    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()

        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.convs = torch.nn.ModuleList(
            [
                norm_f(torch.nn.Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(torch.nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(torch.nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(torch.nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(torch.nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(torch.nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(torch.nn.Conv1d(1024, 1, 3, 1, padding=1))
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE, inplace=True)

    def forward(self, x):
        fmap = []
        for conv in self.convs:
            x = self.lrelu(conv(x))
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class DiscriminatorP(torch.nn.Module):
    """
    Discriminator for the long-term component.

    This class implements a discriminator for the long-term component
    of the audio signal. The discriminator is composed of a series of
    convolutional layers that are applied to the input signal at a given
    period.

    Args:
        period (int): Period of the discriminator.
        kernel_size (int): Kernel size of the convolutional layers. Defaults to 5.
        stride (int): Stride of the convolutional layers. Defaults to 3.
        use_spectral_norm (bool): Whether to use spectral normalization. Defaults to False.
    """

    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.period = period
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        in_channels = [1, 32, 128, 512, 1024]
        out_channels = [32, 128, 512, 1024, 1024]
        strides = [3, 3, 3, 3, 1]

        self.convs = torch.nn.ModuleList(
            [
                norm_f(
                    torch.nn.Conv2d(
                        in_ch,
                        out_ch,
                        (kernel_size, 1),
                        (s, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                )
                for in_ch, out_ch, s in zip(in_channels, out_channels, strides)
            ]
        )

        self.conv_post = norm_f(torch.nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE, inplace=True)

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = torch.nn.functional.pad(x, (0, n_pad), "reflect")
        x = x.view(b, c, -1, self.period)

        for conv in self.convs:
            x = self.lrelu(conv(x))
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class DiscriminatorR(nn.Module):
    def __init__(self, cfg: AttrDict, resolution: List[List[int]]):
        super().__init__()
        self.cfg = cfg

        self.resolution = resolution
        assert len(self.resolution) == 3, f"MRD layer requires list with len=3, got {self.resolution}"

        self.lrelu_slope = 0.1
        self.d_mult = 1
        n_fft, hop_length, win_length = self.resolution
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(1, int(32 * self.d_mult), (3, 9), padding=(1, 4))),
                weight_norm(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 3),
                        padding=(1, 1),
                    )
                ),
            ]
        )
        self.conv_post = weight_norm(
            nn.Conv2d(int(32 * self.d_mult), 1, (3, 3), padding=(1, 1))
        )

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        n_fft, hop_length, win_length = self.resolution

        p = (n_fft - hop_length) // 2
        x = F.pad(x, (p, p), mode="reflect").squeeze(1)

        x = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=self.window, 
            center=False,
            return_complex=True,
        )

        return torch.abs(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        x = self.spectrogram(x).unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.lrelu_slope, inplace=True)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


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
        gamma = self.gamma
        h_fc = self.harmonic_orders.unsqueeze(1) * self.fc.unsqueeze(0)
        h_bw = (0.1079 * h_fc + 24.7) / gamma.unsqueeze(1)

        diff = (self.stft_freqs.unsqueeze(0).unsqueeze(0) - h_fc.unsqueeze(2)).abs()

        filter_bank = F.relu(1.0 - 2.0 * diff / h_bw.unsqueeze(2))

        return torch.einsum("hfn,bnt->bhft", filter_bank, stft_mag)


def _next_pow2(x: float) -> int:
    return 2 ** math.ceil(math.log2(x))


def _derive_n_fft(sample_rate: int) -> int:
    return _next_pow2(1024 * sample_rate / 24_000)


def _derive_hop(sample_rate: int) -> int:
    return round(256 * sample_rate / 24_000)


def _compute_n_bins(sample_rate: int, n_harmonics: int, bins_per_octave: int, fmin: float) -> int:
    fmax_first = sample_rate / (2.0 * n_harmonics)
    return int(math.floor(bins_per_octave * math.log2(fmax_first / fmin)))


def _freq_after_mdc(f: int, n_mdc: int = 3, k: int = 5, stride: int = 2, pad: int = 2) -> int:
    for _ in range(n_mdc):
        f = math.floor((f + 2 * pad - k) / stride) + 1
    return f
