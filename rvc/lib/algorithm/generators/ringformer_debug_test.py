import math
from typing import Optional

import sys, os
import pathlib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm

from torch.nn import Conv1d, ConvTranspose1d


import einops
import numpy as np

from rvc.lib.algorithm.residuals import ResBlock
from rvc.lib.algorithm.conformer.conformer import Conformer

from rvc.lib.algorithm.commons import init_weights
from rvc.lib.algorithm.conformer.stft import TorchSTFT

import random
import time


def padDiff(x):
    return F.pad(F.pad(x, (0,0,-1,1), 'constant', 0) - x, (0,0,0,-1), 'constant', 0)


class SineGen(torch.nn.Module):
    """ Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """

    def __init__(self, samp_rate, upsample_scale, harmonic_num=0,
                 sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0,
                 flag_for_pulse=False):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
        self.upsample_scale = upsample_scale

    def _f02uv(self, f0):
        # generate uv signal
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv

    def _f02sine(self, f0_values):
        """ f0_values: (batchsize, length, dim)
            where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The interger part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1
        print(f"[SineGen] rad_values: mean={rad_values.mean().item()} std={rad_values.std().item()}")


        # initial phase noise (no noise for fundamental component)
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], \
                              device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        if not self.flag_for_pulse:
            rad_values = torch.nn.functional.interpolate(
                rad_values.transpose(1, 2), 
                scale_factor=1/self.upsample_scale, 
                mode="linear"
            ).transpose(1, 2)

            phase = torch.cumsum(rad_values, dim=1) * 2 * np.pi
            phase = torch.nn.functional.interpolate(
                phase.transpose(1, 2) * self.upsample_scale, 
                scale_factor=self.upsample_scale,
                mode="linear"
            ).transpose(1, 2)

            sines = torch.sin(phase)
            
        else:
            # If necessary, make sure that the first time step of every
            # voiced segments is sin(pi) or cos(0)
            # This is used for pulse-train generation

            # identify the last time step in unvoiced segments
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)

            # get the instantanouse phase
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            # different batch needs to be processed differently
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                # stores the accumulation of i.phase within
                # each voiced segments
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum

            # rad_values - tmp_cumsum: remove the accumulation of i.phase
            # within the previous voiced segment.
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)

            # get the sines
            sines = torch.cos(i_phase * 2 * np.pi)
        return sines

    def forward(self, f0):
        """ sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim,
                             device=f0.device)
        # fundamental component
        fn = torch.multiply(f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device))

        # generate sine waveforms
        sine_waves = self._f02sine(fn) * self.sine_amp
        print(f"[SineGen] sine_waves: min={sine_waves.min().item()} max={sine_waves.max().item()} mean={sine_waves.mean().item()}")

        # generate uv signal
        uv = self._f02uv(f0)

        # noise: for unvoiced should be similar to sine_amp
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)

        # first: set the unvoiced part to 0 by uv
        # then: additive noise
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise



class SourceModuleHnNSF(torch.nn.Module):
    """
    Source Module for generating harmonic and noise components for audio synthesis.

    This module generates a harmonic source signal using sine waves and adds
    optional noise. It's often used in neural vocoders as a source of excitation.

    Args:
        sample_rate (int): Sampling rate of the audio in Hz.
        harmonic_num (int, optional): Number of harmonic overtones to generate above the fundamental frequency (F0). Defaults to 0.
        sine_amp (float, optional): Amplitude of the sine wave components. Defaults to 0.1.
        add_noise_std (float, optional): Standard deviation of the additive white Gaussian noise. Defaults to 0.003.
        voiced_threshod (float, optional): Threshold for the fundamental frequency (F0) to determine if a frame is voiced. If F0 is below this threshold, it's considered unvoiced. Defaults to 0.
    """

    def __init__(
        self,
        sample_rate: int,
        upsample_scale: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshod: float = 0,
    ):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        self.l_sin_gen = SineGen(
            sample_rate, upsample_scale, harmonic_num, sine_amp, add_noise_std, voiced_threshod
        )

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x):
        """
        Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        noise_source (batchsize, length 1)
        """
        print(f"[SRC MODULE] F0 in: min={x.min().item()} max={x.max().item()} mean={x.mean().item()}")
        # source for harmonic branch
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))

        # source for noise branch, in the same shape as uv
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv



class RingFormerGenerator(torch.nn.Module):
    def __init__(
        self,
        initial_channel, # 192
        resblock_kernel_sizes, # [3,7,11]
        resblock_dilation_sizes, # [[1,3,5], [1,3,5], [1,3,5]]
        upsample_rates, # [12, 10]
        upsample_initial_channel, # 512
        upsample_kernel_sizes, # [24, 20]
        gen_istft_n_fft, # 24
        gen_istft_hop_size, # 4
        gin_channels, # 256
        sr, # 48000
        harmonic_num = 8,
        block_size_custom = 512,
    ):
        super(RingFormerGenerator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.gen_istft_n_fft = gen_istft_n_fft
        self.gen_istft_hop_size = gen_istft_hop_size
        self.conv_pre = weight_norm(Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3))


        self.m_source = SourceModuleHnNSF(
            sample_rate=sr,
            upsample_scale=math.prod(upsample_rates) * gen_istft_hop_size,
            harmonic_num=harmonic_num,
            voiced_threshod=0,
        )

        self.f0_upsamp = torch.nn.Upsample(
            scale_factor=math.prod(upsample_rates) * self.gen_istft_hop_size
        )

        self.noise_convs = nn.ModuleList()

        self.noise_res = nn.ModuleList()

        self.ups = nn.ModuleList()

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2 ** i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2
                    )
                )
            )

            c_cur = upsample_initial_channel // (2 ** (i + 1))

            if i + 1 < len(upsample_rates):
                stride_f0 = math.prod(upsample_rates[i + 1:])
                kernel = stride_f0 * 2 - stride_f0 % 2
                padding = 0 if stride_f0 == 1 else (kernel - stride_f0) // 2


                self.noise_convs.append(Conv1d(
                    self.gen_istft_n_fft + 2, c_cur, kernel_size=kernel, stride=stride_f0, padding=padding))
                self.noise_res.append(ResBlock(c_cur, 7, [1, 3, 5]))

            else:
                self.noise_convs.append(Conv1d(self.gen_istft_n_fft + 2, c_cur, kernel_size=1))
                self.noise_res.append(ResBlock(c_cur, 11, [1, 3, 5]))



        self.alphas = nn.ParameterList()
        self.alphas.append(nn.Parameter(torch.ones(1, upsample_initial_channel, 1)))
        self.resblocks = nn.ModuleList()



        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            self.alphas.append(nn.Parameter(torch.ones(1, ch, 1)))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d))



        self.conformers = nn.ModuleList()
        self.post_n_fft = self.gen_istft_n_fft
        self.conv_post = weight_norm(Conv1d(128, self.post_n_fft + 2, 7, 1, padding=3))



        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** i)
            self.conformers.append(
                Conformer(
                    dim=ch,
                    depth=2,
                    dim_head=64,
                    heads=8,
                    ff_mult=4,
                    conv_expansion_factor = 2,
                    conv_kernel_size=31,
                    attn_dropout=0.1,
                    ff_dropout=0.1,
                    conv_dropout=0.1,
                    block_size=block_size_custom,
                )
            )



        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))

        self.stft = TorchSTFT(
            "cuda",
            filter_length=self.gen_istft_n_fft,
            hop_length=self.gen_istft_hop_size,
            win_length=self.gen_istft_n_fft
        )



        if gin_channels != 0:
            self.cond = Conv1d(gin_channels, upsample_initial_channel, 1)



    def forward(self, x: torch.Tensor, f0: torch.Tensor, g: Optional[torch.Tensor] = None):
        print(f"[DEC forward] Input x shape (latent): {x.shape}")  # [B, C, T]
        print(f"[DEC forward] Input f0 shape: {f0.shape}")

        print(f"[DEBUG] Raw F0 min/max: {f0.min().item()} {f0.max().item()}")
        print(f"[DEBUG] Raw F0 mean: {f0.mean().item()}")


        if len(f0.shape) == 1:
            f0 = f0.unsqueeze(0)

        f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # [B, T, 1] -> upsampled
        print(f"[DEC forward] Input f0 shape after f0_upsamp: {f0.shape}")
        print(f"[DEBUG] Upsampled F0 min/max: {f0.min().item()} {f0.max().item()}")
        print(f"[DEBUG] Upsampled F0 mean: {f0.mean().item()}")



        har_source, _, _ = self.m_source(f0)
        print(f"[DEC forward] Harmonic source shape (before squeeze): {har_source.shape}")
        har_source = har_source.transpose(1, 2).squeeze(1)
        print(f"[DEC forward] Harmonic source shape (after squeeze): {har_source.shape}")

        har_spec, har_phase = self.stft.transform(har_source)
        print(f"[DEC forward] STFT har_spec shape: {har_spec.shape}")
        print(f"[DEC forward] STFT har_phase shape: {har_phase.shape}")
        har = torch.cat([har_spec, har_phase], dim=1)
        print(f"[DEC forward] Concatenated har shape: {har.shape}")

        x = self.conv_pre(x)
        print(f"[DEC forward] After conv_pre: {x.shape}")

        if g is not None and self.cond is not None:
            #print(f"[DEC forward] Conditioning enabled. g shape: {g.shape}")
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            print(f"\n[DEC forward] === Upsample stage {i} ===")
            print(f"[stage {i}] x before sine transform: {x.shape}")
            x = x + (1 / self.alphas[i]) * (torch.sin(self.alphas[i] * x) ** 2)
            x = einops.rearrange(x, 'b f t -> b t f')
            print(f"[stage {i}] x before conformer: {x.shape}")

            x = self.conformers[i](x)
            x = einops.rearrange(x, 'b t f -> b f t')
            print(f"[stage {i}] x after conformer: {x.shape}")

            x_source = self.noise_convs[i](har)
            print(f"[stage {i}] x_source after noise_conv: {x_source.shape}")
            x_source = self.noise_res[i](x_source)
            print(f"[stage {i}] x_source after noise_res: {x_source.shape}")

            x = self.ups[i](x)
            print(f"[stage {i}] x after upsample: {x.shape}")
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
                #print(f"[stage {i}] x after reflection pad: {x.shape}")

            x = x + x_source
            print(f"[stage {i}] x after adding source: {x.shape}")

            xs = None
            for j in range(self.num_kernels):
                res_out = self.resblocks[i * self.num_kernels + j](x)
                print(f"[stage {i}] resblock {j} output shape: {res_out.shape}")
                xs = res_out if xs is None else xs + res_out
            x = xs / self.num_kernels
            print(f"[stage {i}] x after averaging resblocks: {x.shape}")

        x = x + (1 / self.alphas[i + 1]) * (torch.sin(self.alphas[i + 1] * x) ** 2)

        x = self.conv_post(x)
        print(f"[DEC forward] x after conv_post: {x.shape}")

        spec = torch.exp(x[:, :self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :])
        print(f"[DEC forward] spec shape: {spec.shape}")
        print(f"[DEC forward] phase shape: {phase.shape}")

        out = self.stft.inverse(spec, phase).to(x.device)
        print(f"[DEC forward] Output waveform shape: {out.shape}")

        return out, spec, phase


    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

    def __prepare_scriptable__(self):
        for l in self.ups:
            for hook in l._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    remove_weight_norm(l)
        for l in self.resblocks:
            for hook in l._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    remove_weight_norm(l)
        # conv_pre, conv_post
        for hook in self.conv_pre._forward_pre_hooks.values():
            if (
                hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                and hook.__class__.__name__ == "WeightNorm"
            ):
                remove_weight_norm(self.conv_pre)
        for hook in self.conv_post._forward_pre_hooks.values():
            if (
                hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                and hook.__class__.__name__ == "WeightNorm"
            ):
                remove_weight_norm(self.conv_post)

        return self




if __name__ == "__main__":
    import soundfile as sf
    import random, time

    # === CONFIGURATION ===
    loops = 5                 # How many times to run the vocoder
    spacing_ms = 50           # Silence between outputs, in milliseconds
    uniform_length = 36       # Trim spectrogram and f0 to this length (set None to disable)
    sr = 48000                # Sampling rate
    # ======================

    script_dir = pathlib.Path(__file__).parent.resolve()
    spec_path = script_dir / "spec.pt"
    f0_path = script_dir / "f0.npy"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === Load inputs ===
    print(f"Loading: {spec_path}")
    x_raw = torch.load(spec_path, map_location=device)  # [1, 1025, T]
    print(f"Loading: {f0_path}")
    f0_orig = torch.from_numpy(np.load(f0_path)).float().to(device)

    # Clamp if needed
    if uniform_length is not None:
        if x_raw.shape[-1] > uniform_length:
            x_raw = x_raw[..., :uniform_length]
            print(f"[debug] x_raw clamped to length {x_raw.shape[-1]}")
        if f0_orig.ndim == 1 and f0_orig.shape[0] > uniform_length:
            f0_orig = f0_orig[:uniform_length]
        elif f0_orig.ndim == 2 and f0_orig.shape[1] > uniform_length:
            f0_orig = f0_orig[:, :uniform_length]
        print(f"[debug] f0 clamped to length {f0_orig.shape}")

    output_segments = []
    spacing_samples = int(sr * spacing_ms / 1000)
    silence = torch.zeros(1, spacing_samples).to(device)

    with torch.no_grad():
        for i in range(loops):
            time.sleep(0.01)  # ensures time advances enough for seed

            # Reseed randomness (simulates a new process)
            seed = int(time.time() * 1000) % (2**32 - 1)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            model = RingFormerGenerator(
                initial_channel=192,
                resblock_kernel_sizes=[3, 7, 11],
                resblock_dilation_sizes=[[1, 3, 5]] * 3,
                upsample_rates=[12, 10],
                upsample_initial_channel=512,
                upsample_kernel_sizes=[24, 20],
                gen_istft_n_fft=24,
                gen_istft_hop_size=4,
                gin_channels=256,
                sr=sr,
                harmonic_num=8,
                block_size_custom=512,
            ).to(device)

            model.eval()

            projection = torch.nn.Conv1d(1025, 192, kernel_size=1).to(device)
            x = projection(x_raw.clone().to(device))

            print(f"[main] Loop {i+1}/{loops} | seed: {seed}")
            out, _, _ = model(x, f0_orig.clone())
            out = out.squeeze(0)
            output_segments.append(out)
            if i < loops - 1:
                output_segments.append(silence.clone())

    # Concatenate all outputs + spacers
    full_output = torch.cat(output_segments, dim=-1)
    print(f"[main] Final output shape: {full_output.shape}")

    out_path = script_dir / f"output_loops{loops}.wav"
    sf.write(out_path, full_output.squeeze(0).cpu().numpy(), samplerate=sr)
    print(f"[main] Output audio saved to {out_path}")

