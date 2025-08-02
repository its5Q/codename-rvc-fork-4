import sys, os
sys.path.insert(0, os.getcwd())

import torch
import numpy as np
import soundfile as sf
from rvc.lib.algorithm.synthesizers import Synthesizer



def _log_tensor(name, x):
    if x is None:
        print(f"[DEBUG] {name} is None")
        return
    try:
        print(f"[DEBUG] {name}: shape={x.shape}, dtype={x.dtype}, min={x.min().item():.3f}, max={x.max().item():.3f}, mean={x.float().mean().item():.3f}")
    except Exception as e:
        print(f"[DEBUG] {name}: shape={x.shape}, dtype={x.dtype}, (min/max/mean unavailable): {e}")


# ====================  Synth init  ====================

checkpoint = torch.load("logs/BENCHMARK_RING/BENCHMARK_RING_2000e_2000s.pth", map_location="cpu")

config_tuple = checkpoint["config"]  # this is the list of args for Synthesizer
use_f0 = checkpoint.get("f0", 1)
vocoder = checkpoint.get("vocoder", "RingFormer")
hidden_dim = 768 if checkpoint.get("version","v1")== "v2" else 256

synth = Synthesizer(
    *config_tuple,
    use_f0=use_f0,
    text_enc_hidden_dim=hidden_dim,
    vocoder=vocoder,
    gen_istft_n_fft=32,
    gen_istft_hop_size=4,
)

#del synth.enc_q # not used rn


# load weights, move to GPU if available, eval, strip weight‑norm
synth.load_state_dict(checkpoint["weight"], strict=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
synth = synth.to(device).eval()
synth.dec = synth.dec.__prepare_scriptable__()


# ====================  Data loading  ====================

reference_path = os.path.join("debug")

# Phone / contentvec features
phone = np.load(os.path.join(reference_path, "ref_phone.npy"))
phone = np.repeat(phone, 2, axis=0)  # Match pitch frame rate
# Pitch
pitch = np.load(os.path.join(reference_path, "ref_pitch.npy"))
pitchf = np.load(os.path.join(reference_path, "ref_pitchf.npy"))


# Find minimum length
min_len = min(len(phone), len(pitch), len(pitchf))

# Trim all to same length
phone = phone[:min_len]
pitch = pitch[:min_len]
pitchf = pitchf[:min_len]


# Convert to tensors
phone = torch.FloatTensor(phone).unsqueeze(0).to(device)
phone_lengths = torch.LongTensor([phone.shape[1]]).to(device)
pitch = torch.LongTensor(pitch).unsqueeze(0).to(device)
pitchf = torch.FloatTensor(pitchf).unsqueeze(0).to(device)

# Default speaker
sid = torch.LongTensor([0]).to(device)


_log_tensor("phone", phone)
_log_tensor("pitch", pitch)
_log_tensor("pitchf", pitchf)
print("[DEBUG] phone_lengths:", phone_lengths.item())



with torch.no_grad():
    o, *_ = synth.infer(
        phone,
        phone_lengths,
        pitch,
        pitchf,
        sid=sid
    )


wav = o.clamp(-1,1).cpu().numpy()
if wav.ndim == 3:  # (batch, channels, time)
    wav = wav[0].transpose(1,0)  # (time, channels)
elif wav.ndim == 2:  # (batch, time)
    wav = wav[0]  # (time,)
# else leave as is if 1D

sf.write("recon_debug_direct.wav", wav, samplerate=checkpoint["config"][-1])