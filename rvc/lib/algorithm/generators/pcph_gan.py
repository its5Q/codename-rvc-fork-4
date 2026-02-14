import math
from typing import Optional, Tuple
from itertools import chain

import torch
from torch import Tensor
import numpy as np

import torch.nn as nn
import torch.nn.init as init

from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm

import torch.nn.functional as F

from torch.amp import autocast # guard
from torch.utils.checkpoint import checkpoint

from rvc.lib.algorithm.generators.pcph_gan_modules.pcph_dirichlet_fused import FusedDirichlet
from rvc.lib.algorithm.generators.pcph_gan_modules.spectral_tilt_triton_fused import fast_iir_filter_triton
from rvc.lib.algorithm.generators.pcph_gan_modules.snake_beta_fused_triton import SnakeBeta as SnakeBetaFused, snake_kaiming_normal_
from rvc.lib.algorithm.generators.pcph_gan_modules.PchipF0UpsamplerTorch import PchipF0UpsamplerTorch

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def create_resblock_conv1d_layer(channels, kernel_size, dilation):
    m = torch.nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation, padding=get_padding(kernel_size, dilation))
    # SnakeBeta adapted initialization
    snake_kaiming_normal_(m.weight, alpha=2.9, beta=2.9, kind='approx')
    if m.bias is not None:
        nn.init.constant_(m.bias, 0)
    return weight_norm(m)


def create_ups_convtranspose1d_layer(in_channels, out_channels, kernel_size, stride):
    m = torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - stride) // 2)
    return weight_norm(m)


class PhaseDispersion(nn.Module):
    def __init__(self, sample_rate, duration_ms=2.66, dispersion_factor=50.0):
        super().__init__()
        kernel_size = int((duration_ms / 1000.0) * sample_rate)
        if kernel_size % 2 == 0:
            kernel_size += 1

        self.padding = (kernel_size - 1) // 2

        t = torch.linspace(-0.5, 0.5, kernel_size)

        chirp = torch.sin(dispersion_factor * t**2) 

        window = torch.hann_window(kernel_size)
        impulse_response = (chirp * window).view(1, 1, -1)

        impulse_response = impulse_response / (torch.norm(impulse_response) + 1e-8)

        self.register_buffer("dispersion_kernel", impulse_response)

    def forward(self, x):
        b, c, t = x.shape
        kernel = self.dispersion_kernel.expand(c, -1, -1)

        return F.conv1d(
            x, 
            kernel, 
            padding=self.padding, 
            groups=c
        )


class DynamicSpectralTilt_IIR(nn.Module):
    def __init__(self, sample_rate, hidden_dim=64):
        super().__init__()
        self.sample_rate = sample_rate
        
        self.f0_to_alpha = nn.Sequential(
            nn.Conv1d(1, hidden_dim, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_dim, 1, 1),
            nn.Sigmoid() 
        )

        nn.init.zeros_(self.f0_to_alpha[2].weight)
        nn.init.constant_(self.f0_to_alpha[2].bias, -4.0)

    def forward(self, x, f0_upsampled, voiced_mask, initial_state=None):
        nyquist = self.sample_rate / 2.0
        f0_norm = f0_upsampled / nyquist
        f0_input = torch.log2(f0_norm * 10.0 + 1.0)

        alpha = self.f0_to_alpha(f0_input) * 0.98
        alpha = alpha * voiced_mask

        input_signal = (1.0 - alpha) * x

        if not self.training:
            iir_mode = "long_infer" # "short_infer" is also available, but its purpose is for Streaming-Inference, not offline.
        else:
            iir_mode = "short_train"

        return fast_iir_filter_triton(input_signal, alpha, mode=iir_mode, initial_state=initial_state)


class TimeVarFIRFilter(torch.nn.Module):
    def __init__(self):
        super(TimeVarFIRFilter, self).__init__()

    def forward(self, signal, f_coef):
        if signal.dim() == 3:
            signal = signal.squeeze(1)

        if f_coef.shape[1] != signal.shape[1]: 
            f_coef = f_coef.transpose(1, 2) 

        b, t = signal.shape
        k = f_coef.shape[-1]

        padded_signal = F.pad(signal, (k - 1, 0))
        stride_b, stride_t = padded_signal.stride()

        windows = padded_signal.as_strided(size=(b, t, k), stride=(stride_b, stride_t, stride_t))
        y = torch.sum(windows * f_coef, dim=-1, keepdim=True)

        return y.transpose(1, 2)


class SincFilter(torch.nn.Module):
    def __init__(self, sample_rate, min_duration_ms=0.646):
        super(SincFilter, self).__init__()
        filter_order = int((min_duration_ms / 1000.0) * sample_rate)

        if filter_order % 2 == 0:
            filter_order += 1

        self.half_k = (filter_order - 1) // 2
        self.order = self.half_k * 2 + 1

        n_index = torch.arange(-self.half_k, self.half_k + 1, dtype=torch.float32)
        self.register_buffer("n_index", n_index.view(1, 1, -1))

        window = 0.54 + 0.46 * torch.cos(2 * np.pi * n_index / self.order)
        self.register_buffer("window", window.view(1, 1, -1))

        flip = torch.pow(-1, self.n_index)
        self.register_buffer("flip", flip)

        impulse = torch.zeros_like(self.window)
        center_idx = self.half_k
        impulse[:, :, center_idx] = self.window[:, :, center_idx]
        self.register_buffer("impulse", impulse)

    def forward(self, cut_f):
        if cut_f.dim() == 2:
            cut_f = cut_f.unsqueeze(-1)
        if cut_f.shape[1] == 1 and cut_f.shape[2] != 1:
             cut_f = cut_f.transpose(1, 2)

        sinc_term = torch.sinc(cut_f * self.n_index)
        lp_c = cut_f * sinc_term * self.window

        hp_c = self.impulse - lp_c

        lp_coef_norm = torch.sum(lp_c, dim=2, keepdim=True)
        hp_coef_norm = torch.sum(hp_c * self.flip, dim=2, keepdim=True)

        lp_c = lp_c / (lp_coef_norm + 1e-8)
        hp_c = hp_c / (hp_coef_norm + 1e-8)

        return lp_c, hp_c


class pu_downsampler(nn.Module):
    '''
    Space-to-Depth (Pixel Unshuffle) style downsampler for dense PCPH harmonic signal
    '''
    def __init__(self, out_channels, downsample_factor):
        super().__init__()
        self.factor = downsample_factor

        self.mixer = nn.Conv1d(
            in_channels=downsample_factor,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

    def forward(self, x):
        # x: [B, 1, T]
        b, c, t = x.size()

        if t % self.factor != 0:
            pad_amt = self.factor - (t % self.factor)
            x = F.pad(x, (0, pad_amt))
            t = x.shape[-1]

        x = x.view(b, c, t // self.factor, self.factor)
        x = x.permute(0, 3, 2, 1) 
        x = x.reshape(b, self.factor, t // self.factor)

        return self.mixer(x)

 
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
        """
        Initializes the ResBlock.

        Args:
            channels (int): Number of input and output channels for the convolution layers.
            kernel_size (int): Size of the convolution kernel. Defaults to 3.
            dilations (Tuple[int]): Tuple of dilation rates for the convolution layers in the first set.
        """
        super().__init__()
        self.convs1 = self._create_convs(channels, kernel_size, dilations)
        self.convs2 = self._create_convs(channels, kernel_size, [1] * len(dilations))

        self.acts1 = nn.ModuleList([
            SnakeBetaFused(num_channels=channels, init=2.9, beta_init=2.9, log_scale=True) for _ in dilations])
        self.acts2 = nn.ModuleList([
            SnakeBetaFused(num_channels=channels, init=2.9, beta_init=2.9, log_scale=True) for _ in dilations])

    @staticmethod
    def _create_convs(channels: int, kernel_size: int, dilations: Tuple[int]):
        """
        Creates a list of 1D convolutional layers with specified dilations.

        Args:
            channels (int): Number of input and output channels for the convolution layers.
            kernel_size (int): Size of the convolution kernel.
            dilations (Tuple[int]): Tuple of dilation rates for each convolution layer.
        """
        layers = torch.nn.ModuleList(
            [create_resblock_conv1d_layer(channels, kernel_size, d) for d in dilations]
        )
        return layers

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None):
        for i, (conv1, conv2) in enumerate(zip(self.convs1, self.convs2)):

            x_residual = x # Residual store

            xt = self.acts1[i](x) # Activation 1

            if x_mask is not None:
                xt = xt * x_mask # Masking 1

            xt = conv1(xt) # Conv 1

            xt = self.acts2[i](xt) # Activation 2

            if x_mask is not None:
                xt = xt * x_mask # Masking 2

            xt = conv2(xt) # Conv 2

            x = xt + x_residual # Residual connection

            if x_mask is not None: # Final mask
                x = x * x_mask

        return x

    def remove_weight_norm(self):
        for conv in chain(self.convs1, self.convs2):
            remove_weight_norm(conv)


def pcph_generator_v3(
    f0: torch.Tensor,
    hop_length: int,
    sample_rate: int,
    random_init_phase: Optional[bool] = True,
    power_factor: Optional[float] = 0.1,
    max_frequency: Optional[float] = None,
    epsilon: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    An optimized O(1) generator for Pseudo-Constant-Power Harmonic waveforms.
    """
    batch, _, _ = f0.size()
    device = f0.device

    # F0 upsampling
    pchip_f0_upsampler = PchipF0UpsamplerTorch(scale_factor=hop_length)
    f0_upsampled = pchip_f0_upsampler(f0)

    # Early return for mute / silent / all unvoiced
    if torch.all(f0_upsampled < 1.0):
        _, _, total_length = f0_upsampled.size()
        zeros = torch.zeros((batch, 1, total_length), device=device, dtype=f0_upsampled.dtype)
        return zeros, zeros, zeros, zeros

    # Preparation
    voiced_mask = (f0_upsampled > 1.0).float()

    # Calculate Phase (Theta)
    phase_increment = f0_upsampled / sample_rate # phase = 2 * pi * integral(f0 / sr)

    # Randomize initial phase
    if random_init_phase:
        init_phase = torch.rand((batch, 1, 1), device=device)
        phase_increment[:, :, :1] += init_phase

    # Cumsum
    # Multiplying by 2pi at the end to save ops during the cumsum
    phase = torch.cumsum(phase_increment.double(), dim=2) * 2.0 * torch.pi
    phase = torch.fmod(phase, 2.0 * torch.pi)
    phase = phase.float()

    # Dynamic harmonic count (N)
    # N is the max harmonic index before aliasing (Nyquist)
    # N(t) = floor( MaxFreq / f0(t) )
    nyquist = sample_rate / 2.0
    limit_freq = max_frequency if max_frequency is not None else nyquist

    # Zero-Division safety for unvoiced segments
    safe_f0 = torch.clamp(f0_upsampled, min=1e-5)
    N = torch.floor(limit_freq / safe_f0)

    # Uses fused Triton dirichlet summation to calculate the raw harmonics
    harmonics = FusedDirichlet.apply(phase, N, epsilon)

    # Amplitude Normalization (Pseudo-Constant-Power)
    # Power Factor Normalization: amp = P * sqrt(2/N)
    amp_scale = power_factor * torch.sqrt(2.0 / torch.clamp(N, min=1.0))

    # Apply masks and scale
    pcph_harmonic_signal = harmonics * amp_scale * voiced_mask

    return pcph_harmonic_signal, voiced_mask, phase, f0_upsampled


class SourceModulePCPH(torch.nn.Module):
    """
    Source Module using PCPH harmonics + Cyclic Noise with Sinc-based mixing.
    """
    def __init__(
        self,
        sample_rate: int,
        hop_length: int = 480,
        random_init_phase: bool = True,
        power_factor: float = 0.1,
        add_noise_std: float = 0.003,
        beta: float = 0.870,
    ):
        super(SourceModulePCPH, self).__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.random_init_phase = random_init_phase
        self.power_factor = power_factor
        self.noise_std = add_noise_std
        self.beta = beta

        self.sinc_filter = SincFilter(sample_rate, min_duration_ms=0.646) # Maintains ~31 taps @ 48k
        self.dispersion = PhaseDispersion(sample_rate, duration_ms=2.66, dispersion_factor=50.0) # Maintains ~128 taps @ 48k

        self.tv_filter = TimeVarFIRFilter()
        self.dynamic_tilt_filter = DynamicSpectralTilt_IIR(sample_rate=sample_rate)
        self.f0_to_cut = nn.Conv1d(1, 1, kernel_size=1)
        self.l_linear = torch.nn.Linear(1, 1, bias=False)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, f0: torch.Tensor, upsample_factor: int = None, initial_state=None):
        """
        f0: (Batch, Frames) or (Batch, 1, Frames)
        """
        if f0.dim() == 2:
            f0 = f0.unsqueeze(1)

        hop = upsample_factor if upsample_factor is not None else self.hop_length

        with autocast('cuda', enabled=False):
            f0 = f0.float()

            # Generate pcph harmonics, mask, and phase
            with torch.no_grad():
                pcph_harmonic_signal, voiced_mask, phase, f0_upsampled = pcph_generator_v3(
                    f0,
                    hop_length=hop,
                    sample_rate=self.sample_rate,
                    random_init_phase=self.random_init_phase,
                    power_factor=self.power_factor
                )

            # Phase dispersion
            pcph_harmonic_signal = self.dispersion(pcph_harmonic_signal)

            # Spectral tilting
            res = self.dynamic_tilt_filter(pcph_harmonic_signal, f0_upsampled, voiced_mask, initial_state=initial_state)
            if isinstance(res, tuple):
                pcph_harmonic_signal, new_state = res
            else:
                pcph_harmonic_signal, new_state = res, None


            # Noise Generation
            noise = torch.randn_like(pcph_harmonic_signal)          #  Base noise
            decay = torch.exp(-phase / (2 * torch.pi * self.beta))  #  Cyclic Decay Envelope
            cyclic_noise = noise * decay * self.noise_std           #  Apply decay only to voiced regions
            unvoiced_gain = self.power_factor / 3.0

            flat_noise = torch.randn_like(pcph_harmonic_signal) * unvoiced_gain
            final_noise = torch.where(voiced_mask > 0.5, cyclic_noise, flat_noise)  # Combine: Cyclic in voiced, Flat in unvoiced

            # Learn a shift/scale on F0 for optimal cutoff
            nyquist = self.sample_rate / 2
            cut_f = torch.sigmoid(self.f0_to_cut(f0_upsampled / nyquist))
            lp_coef, hp_coef = self.sinc_filter(cut_f)  # Get Sinc coefs

            # Apply Filters: Harmonics -> LP, Noise -> HP
            filtered_harm = self.tv_filter(pcph_harmonic_signal, lp_coef)
            filtered_noise = self.tv_filter(final_noise, hp_coef)
            excitation_signal = filtered_harm + filtered_noise

        excitation_signal = excitation_signal.to(dtype=self.l_linear.weight.dtype)

        # Trainable projection ( linear -> tanh )
        excitation_signal = excitation_signal.transpose(1, 2)
        excitation = self.l_tanh(self.l_linear(excitation_signal))
        excitation = excitation.transpose(1, 2)

        return excitation, new_state

class PCPH_GAN_Generator(nn.Module):
    def __init__(
        self,
        initial_channel, # 192
        resblock_kernel_sizes, # [3,7,11]
        resblock_dilation_sizes, # [[1,3,5], [1,3,5], [1,3,5]]
        upsample_rates, # [12, 10, 2, 2]
        upsample_initial_channel, # 512
        upsample_kernel_sizes, # [24, 20, 4, 4]
        gin_channels, # 256
        sr, # 48000,
        checkpointing: bool = False, # not implemented yet.
        use_inplace: bool = True,
    ):
        super(PCPH_GAN_Generator, self).__init__()
        self.checkpointing = checkpointing
        self.use_inplace = use_inplace
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        total_ups_factor = math.prod(upsample_rates)

        self.use_tanh = False

        # PCPH handler
        self.m_source = SourceModulePCPH(
            sample_rate=sr,
            hop_length=total_ups_factor,
            random_init_phase=True,
            power_factor=0.1,
            add_noise_std=0.003,
            beta=0.870
        )
        
        # Initial feats conv, projection: 192 -> 512
        self.conv_pre = weight_norm(Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3))

        # Module containers init
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        self.har_convs = nn.ModuleList()

        ch = upsample_initial_channel # 512

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            # 512:  256 --> 128 --> 64 --> 32
            ch //= 2

            # Features upsampling convs
            self.ups.append(create_ups_convtranspose1d_layer(2 * ch, ch, k, u))

            # Resblocks
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d))

            # Harmonic prior downsampling convs
            if i + 1 < len(upsample_rates):
                s_c = int(math.prod(upsample_rates[i + 1:]))
                # Space-to-depth downsampling
                self.har_convs.append(pu_downsampler(out_channels=ch, downsample_factor=s_c))
            else:
                # Projecting 1 channel -> ch channels
                self.har_convs.append(Conv1d(1, ch, kernel_size=1, bias=False))

        # Post convolution
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3, bias=False))

        # embedding / spk conditioning layer
        if gin_channels != 0:
            self.cond = Conv1d(gin_channels, upsample_initial_channel, 1)


    def forward(self, x: torch.Tensor, f0: torch.Tensor, g: Optional[torch.Tensor] = None):
        # x: [B, 192, Frames]
        # f0: [B, Frames]

        # Generate the prior and retrieve the state
        har_prior, new_state = self.m_source(f0) # Output: B, 1, F

        # Pre-convolution ( Channel expansion: 192 -> 512 )
        x = self.conv_pre(x)

        # Apply spk emb conditioning
        if g is not None:
            x = x + self.cond(g)

        # Main loop
        for i in range(self.num_upsamples):

            # pre-upsampling activation
            x = F.silu(x, inplace=self.use_inplace)
            # Upsample features
            x = self.ups[i](x)

            # pcph harmonic injection
            if self.use_inplace:
                x.add_(self.har_convs[i](har_prior))
            else:
                har_prior_injection = self.har_convs[i](har_prior)
                x = x + har_prior_injection

            # Resblocks processing
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # Final act
        x = F.silu(x, inplace=self.use_inplace)
        # Post conv
        x = self.conv_post(x)
        
        # Tanh / Clamp
        if self.use_tanh:
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1.0, max=1.0)

        return x, new_state

    def remove_weight_norm(self):
        # Upsamplers
        for l in self.ups:
            remove_weight_norm(l)
        # ResBlocks
        for l in self.resblocks:
            l.remove_weight_norm()
        # pre convolution
        remove_weight_norm(self.conv_pre)
        # post convolution
        remove_weight_norm(self.conv_post)

    def __prepare_scriptable__(self):
        # Pre convolution
        for hook in self.conv_pre._forward_pre_hooks.values():
            if (
                hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                and hook.__class__.__name__ == "WeightNorm"
            ):
                remove_weight_norm(self.conv_pre)
        # Upsamplers
        for l in self.ups:
            for hook in l._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    remove_weight_norm(l)
        # ResBlocks
        for l in self.resblocks:
            for hook in l._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    remove_weight_norm(l)
        # Post convolution
        for hook in self.conv_post._forward_pre_hooks.values():
            if (
                hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                and hook.__class__.__name__ == "WeightNorm"
            ):
                remove_weight_norm(self.conv_post)

        return self
