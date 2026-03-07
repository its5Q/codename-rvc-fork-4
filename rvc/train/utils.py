import os
import glob
import json
import signal
import sys

import torch
import torch.distributed as dist
from torch.nn import functional as F

import numpy as np
import soundfile as sf

from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm


MATPLOTLIB_FLAG = False

debug_save_load = False

from itertools import chain
from utils_cdnm import check_optimizer_coverage, verify_optimizer_has_all_params
from mel_processing import mel_spectrogram_torch
from rvc.train.process.extract_model import extract_model

def replace_keys_in_dict(d, old_key_part, new_key_part):
    """
    Recursively replace parts of the keys in a dictionary.

    Args:
        d (dict or OrderedDict): The dictionary to update.
        old_key_part (str): The part of the key to replace.
        new_key_part (str): The new part of the key.
    """
    updated_dict = OrderedDict() if isinstance(d, OrderedDict) else {}
    for key, value in d.items():
        new_key = (
            key.replace(old_key_part, new_key_part) if isinstance(key, str) else key
        )
        updated_dict[new_key] = (
            replace_keys_in_dict(value, old_key_part, new_key_part)
            if isinstance(value, dict)
            else value
        )
    return updated_dict


def load_checkpoint(checkpoint_path, model, optimizer=None, load_opt=1):
    assert os.path.isfile(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    model_state = model.module if hasattr(model, "module") else model
    model_state.load_state_dict(checkpoint_dict["model"], strict=True)

    if optimizer and load_opt == 1:
        opt_state = checkpoint_dict.get("optimizer")
        if opt_state:
            optimizer.load_state_dict(opt_state)
            print("Loaded optimizer state.")
        else:
            raise ValueError(f"[ERROR] Missing optimizer state...")

    print(f"Loaded checkpoint '{checkpoint_path}' (iteration {checkpoint_dict['iteration']})")
    return (
        model,
        optimizer,
        checkpoint_dict.get("learning_rate", 0),
        checkpoint_dict["iteration"],
        checkpoint_dict.get("gradscaler", {})
    )

def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path, gradscaler=None):
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    checkpoint_data = {
        "model": state_dict,
        "iteration": iteration,
        "optimizer": optimizer.state_dict(),
        "learning_rate": learning_rate,
    }

    if gradscaler is not None:
        checkpoint_data["gradscaler"] = gradscaler.state_dict()

    torch.save(checkpoint_data, checkpoint_path)
    print(f"Saved model to {checkpoint_path}")

def summarize(
    writer,
    global_step,
    scalars={},
    histograms={},
    images={},
    audios={},
    audio_sample_rate=22050,
):
    """
    Log various summaries to a TensorBoard writer.

    Args:
        writer (SummaryWriter): The TensorBoard writer.
        global_step (int): The current global step.
        scalars (dict, optional): Dictionary of scalar values to log.
        histograms (dict, optional): Dictionary of histogram values to log.
        images (dict, optional): Dictionary of image values to log.
        audios (dict, optional): Dictionary of audio values to log.
        audio_sample_rate (int, optional): Sampling rate of the audio data.
    """
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sample_rate)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    """
    Get the latest checkpoint file in a directory.

    Args:
        dir_path (str): The directory to search for checkpoints.
        regex (str, optional): The regular expression to match checkpoint files.
    """
    checkpoints = sorted(
        glob.glob(os.path.join(dir_path, regex)),
        key=lambda f: int("".join(filter(str.isdigit, f))),
    )
    return checkpoints[-1] if checkpoints else None


def plot_spectrogram_to_numpy(spectrogram):
    """
    Convert a spectrogram to a NumPy array for visualization.

    Args:
        spectrogram (numpy.ndarray): The spectrogram to plot.
    """
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        plt.switch_backend("Agg")
        MATPLOTLIB_FLAG = True

    fig, ax = plt.subplots(figsize=(10, 2.5))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    data = data[:, :, :3]
    plt.close(fig)
    return data


def load_wav_to_torch(full_path):
    """
    Load a WAV file into a PyTorch tensor.

    Args:
        full_path (str): The path to the WAV file.
    """
    data, sample_rate = sf.read(full_path, dtype="float32")
    return torch.FloatTensor(data), sample_rate


def load_filepaths_and_text(filename, split="|"):
    """
    Load filepaths and associated text from a file.

    Args:
        filename (str): The path to the file.
        split (str, optional): The delimiter used to split the lines.
    """
    with open(filename, encoding="utf-8") as f:
        return [line.strip().split(split) for line in f]


class HParams:
    """
    A class for storing and accessing hyperparameters.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = HParams(**v) if isinstance(v, dict) else v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return repr(self.__dict__)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_config_from_json(config_save_path):
    try:
        with open(config_save_path, "r") as f:
            config = json.load(f)
        config = HParams(**config)
        return config
    except FileNotFoundError:
        print(
            f"Model config file not found at {config_save_path}. Did you run preprocessing and feature extraction steps?"
        )
        sys.exit(1)


def mel_spec_similarity(y_hat_mel, y_mel):
    device = y_hat_mel.device
    y_mel = y_mel.to(device)

    if y_hat_mel.shape != y_mel.shape:
        trimmed_shape = tuple(min(dim_a, dim_b) for dim_a, dim_b in zip(y_hat_mel.shape, y_mel.shape))
        y_hat_mel = y_hat_mel[..., :trimmed_shape[-1]]
        y_mel = y_mel[..., :trimmed_shape[-1]]
    
    loss_mel = F.l1_loss(y_hat_mel, y_mel)

    mel_spec_similarity = 100.0 - (loss_mel * 100.0)
    mel_spec_similarity = mel_spec_similarity.clamp(0.0, 100.0)

    return mel_spec_similarity


def flush_writer(writer, rank):
    if rank == 0 and writer is not None:
        writer.flush()


# Currently has no use, kept for future ig.
def flush_writer_grad(writer, rank, global_step):
    if rank == 0 and writer is not None and global_step % 10 == 0:
        writer.flush()


def block_tensorboard_flush_on_exit(writer):
    def handler(signum, frame):
        print("[Warning] Training interrupted. Skipping flush to avoid partial logs.")
        try:
            writer.close()
        except:
            pass
        os._exit(1)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def si_sdr(preds, target, eps=1e-8):
    """Scale-Invariant SDR"""
    preds = preds - preds.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    target_energy = (target ** 2).sum(dim=-1, keepdim=True)
    scaling_factor = (preds * target).sum(dim=-1, keepdim=True) / (target_energy + eps)
    projection = scaling_factor * target

    noise = preds - projection

    si_sdr_value = 10 * torch.log10((projection ** 2).sum(dim=-1) / (noise ** 2).sum(dim=-1) + eps)

    return si_sdr_value.mean()


def wave_to_mel(config, waveform, half):
    mel_spec = mel_spectrogram_torch(
        waveform.float().squeeze(1),
        config.data.filter_length,
        config.data.n_mel_channels,
        config.data.sample_rate,
        config.data.hop_length,
        config.data.win_length,
        config.data.mel_fmin,
        config.data.mel_fmax,
    )
    if half == torch.float16:
        mel_spec = mel_spec.half()
    elif half == torch.float32 or half == torch.bfloat16:
        pass

    return mel_spec


def small_model_naming(model_name, epoch, global_step):
    return f"{model_name}_{epoch}e_{global_step}s.pth"


def old_session_cleanup(now_dir, model_name):
    for root, dirs, files in os.walk(os.path.join(now_dir, "logs", model_name), topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            file_name, file_extension = os.path.splitext(name)
            if (
                file_extension == ".0"
                or (file_name.startswith("D_") and file_extension == ".pth")
                or (file_name.startswith("G_") and file_extension == ".pth")
                or (file_name.startswith("added") and file_extension == ".index")
            ):
                os.remove(file_path)
        for name in dirs:
            if name == "eval":
                folder_path = os.path.join(root, name)
                for item in os.listdir(folder_path):
                    item_path = os.path.join(folder_path, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                os.rmdir(folder_path)

    print("    ██████  Cleanup done!")


def print_init_setup(
    warmup_duration,
    rank,
    use_warmup,
    config,
    optimizer_choice,
    lr_scheduler,
    exp_decay_gamma,
    use_kl_annealing,
    kl_annealing_cycle_duration,
    spectral_loss,
    adversarial_loss,
    vits2_mode,
    use_tstp
):
    # Warmup init msg:
    if rank == 0:

        # Precision init msg:
        if not config.train.fp16_run:
            if torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32:
                print("    ██████  PRECISION: TF32")
            else:
                print("    ██████  PRECISION: FP32")
        elif config.train.fp16_run:
            if torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32:
                print("    ██████  PRECISION: TF32 / FP16 - AMP")
            else:
                print("    ██████  PRECISION: FP32 / FP16 - AMP")
        # cudnn backend checks:
        if torch.backends.cudnn.benchmark:
            print("    ██████  cudnn.benchmark: True")
        else:
            print("    ██████  cudnn.benchmark: False")

        if torch.backends.cudnn.deterministic:
            print("    ██████  cudnn.deterministic: True")
        else:
            print("    ██████  cudnn.deterministic: False")

        # Optimizer check:
        print(f"    ██████  Optimizer used: {optimizer_choice}")


        # Spectral loss check:
        if spectral_loss == "L1 Mel Loss":
            print("    ██████  Spectral loss: Single-Scale (L1) Mel loss function")
        elif spectral_loss == "Multi-Scale Mel Loss":
            print("    ██████  Spectral loss: Multi-Scale Mel loss function")
        elif spectral_loss == "Multi-Res STFT Loss":
            print("    ██████  Spectral loss: Multi-Resolution STFT loss function")

        # Adversarial loss check:
        if adversarial_loss == "tprls":
            print("    ██████  Adversarial loss: TPRLS")
        elif adversarial_loss == "hinge":
            print("    ██████  Adversarial loss: HINGE")
        elif adversarial_loss == "lsgan":
            print("    ██████  Adversarial loss: LSGAN")

        # Vits maode checkup:
        if vits2_mode:
            print("    ██████  Vits mode: vits-based + few vits2 tweaks (Custom)")
        else:
            print("    ██████  Vits mode: vits-based (Default RVC)")

        # LR scheduler check:
        if lr_scheduler == "exp decay":
            print(f"    ██████  lr scheduler: exponential lr decay with gamma of: {exp_decay_gamma}")
        elif lr_scheduler == "cosine annealing":
            print(f"    ██████  lr scheduler: cosine annealing")

        # Warmup
        if use_warmup:
            print(f"    ██████  Warmup Enabled for: {warmup_duration} epochs.")

        # Kl annealing
        if use_kl_annealing:
            print(f"    ██████  KL loss annealing enabled with cycle duration of: {kl_annealing_cycle_duration} epochs.")

        # Tstp
        if use_tstp:
            print(f"    ██████  Two-Stage Training Protocol: Enabled")

def train_loader_safety(train_loader):
    if len(train_loader) < 3:
        print("Not enough data present in the training set. Perhaps you didn't slice the audio files? ( Preprocessing step )")
        os._exit(2333333)


def verify_spk_dim(
    config,
    model_info_path,
    experiment_dir,
    latest_checkpoint_path,
    rank,
    pretrainG
):
    embedder_name = "contentvec" # Default embedder
    spk_dim = config.model.spk_embed_dim  # 109 default speakers

    if rank == 0:
        try:
            with open(model_info_path, "r") as f:
                model_info = json.load(f)
                embedder_name = model_info["embedder_model"]
                spk_dim = model_info["speakers_id"]
        except Exception as e:
            print(f"Could not load model info file: {e}. Using defaults.")

        # Try to load speaker dim from latest checkpoint or pretrainG
        try:
            last_g = latest_checkpoint_path(experiment_dir, "G_*.pth")
            chk_path = (last_g if last_g else (pretrainG if pretrainG not in ("", "None") else None))
            if chk_path:
                ckpt = torch.load(chk_path, map_location="cpu", weights_only=True)
                spk_dim = ckpt["model"]["emb_g.weight"].shape[0]
                del ckpt

        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Using default number of speakers.")

        # update config before the model init
        print(f"    ██████  Initializing the generator with: {spk_dim} speakers.")

    return spk_dim


def early_stopper(
    stopper, 
    rank, 
    global_step, 
    epoch, 
    architecture, 
    nets, 
    optims, 
    config, 
    experiment_dir, 
    gradscaler, 
    save_weight_models,
    model_name,
    vocoder,
    vits2_mode,
    n_gpus
):
    if stopper is not None and stopper.stop_triggered:
        net_g, net_d = nets
        optim_g, optim_d = optims

        if rank == 0:
            print(f"[TRAINING] Saving the models at steps: '{global_step}' in progress ...")

            g_path = os.path.join(experiment_dir, f"G_{global_step}.pth")
            d_path = os.path.join(experiment_dir, f"D_{global_step}.pth")

            # Save Generator checkpoint
            save_checkpoint(net_g, optim_g, config.train.learning_rate, epoch, g_path, gradscaler)
            # Save Discriminator checkpoint
            save_checkpoint(net_d, optim_d, config.train.learning_rate, epoch, d_path, gradscaler)

            # Save small weight model
            if save_weight_models:
                weight_model_name = small_model_naming(model_name, epoch, global_step)
                model_path = os.path.join(experiment_dir, weight_model_name)

                ckpt = net_g.module.state_dict() if hasattr(net_g, "module") else net_g.state_dict()
                extract_model(
                    ckpt=ckpt, 
                    sr=config.data.sample_rate, 
                    name=model_name, 
                    model_path=model_path, 
                    epoch=epoch, 
                    step=global_step, 
                    hps=config, 
                    vocoder=vocoder, 
                    architecture=architecture, 
                    vits2_mode=vits2_mode
                )
                print(f"[TRAINING] All finished .. You can ignore anything past this msg.")
        if n_gpus > 1:
            dist.barrier()
        return True
    return False


class WeightTrajectoryVisualizer:
    def __init__(self, history_limit=50):
        self.history_limit = history_limit
        self.downsample_size = 8000

        self.history = {
            "vocoder": {"weights": [], "epoch": []},
            "context": {"weights": [], "epoch": []}
        }

    def update(self, model, epoch):
        if hasattr(model, 'module'):
            model_ref = model.module
        else:
            model_ref = model

        vocoder_weights = []
        context_weights = []

        for name, param in model_ref.named_parameters():
            if not param.requires_grad:
                continue

            data = param.detach().cpu().flatten()
            if name.startswith("dec."):
                vocoder_weights.append(data)
            else:
                context_weights.append(data)

        self._process_group("vocoder", vocoder_weights, epoch)
        self._process_group("context", context_weights, epoch)

    def _process_group(self, key, weights_list, epoch):
        if not weights_list:
            return

        flat = torch.cat(weights_list)

        if flat.numel() > self.downsample_size:
            step = flat.numel() // self.downsample_size
            flat = flat[::step].clone()

        group = self.history[key]
        group["weights"].append(flat.numpy())
        group["epoch"].append(epoch)

        if len(group["weights"]) > self.history_limit:
            group["weights"].pop(0)
            group["epoch"].pop(0)

    def get_plot(self):
        """
        2-Row Dashboard:
        Top; Vocoder ( Decoder )
        Bottom; Context ( Encoders / Flow )
        """
        if len(self.history["vocoder"]["weights"]) < 3 and len(self.history["context"]["weights"]) < 3:
            return None

        fig = plt.figure(figsize=(15, 9), dpi=100)
        gs = gridspec.GridSpec(
            2, 3,
            width_ratios=[2.4, 1.1, 1.1],
            height_ratios=[1, 1]
        )
        fig.patch.set_facecolor('#f2f2f2')

        self._plot_row(fig, gs, 0, "vocoder", "Vocoder")
        self._plot_row(fig, gs, 1, "context", "Context (Enc/Flow)")

        plt.tight_layout()
        fig.canvas.draw()

        data_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data_np = data_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return data_np

    def _plot_row(self, fig, gs, row_idx, key, title_prefix):
        group = self.history[key]
        if len(group["weights"]) < 3:
            return

        data = np.stack(group["weights"])
        epochs = group["epoch"]

        mean = np.mean(data, axis=0)
        centered = data - mean
        u, s, vh = np.linalg.svd(centered, full_matrices=False)
        coords = u[:, :2] * s[:2]

        expl_var = (s[:2] ** 2) / np.sum(s ** 2)
        total_var = np.sum(expl_var) * 100

        velocity = np.linalg.norm(np.diff(data, axis=0), axis=1)

        drift = np.linalg.norm(data - data[0], axis=1)

        ax_traj = fig.add_subplot(gs[row_idx, 0])
        colors = cm.cividis(np.linspace(0.15, 0.85, len(coords)))

        for i in range(len(coords) - 1):
            ax_traj.annotate('', xy=coords[i+1], xytext=coords[i], arrowprops=dict(arrowstyle="->", color=colors[i], lw=1.4))

        ax_traj.scatter(coords[:, 0], coords[:, 1], c=colors, s=30)
        ax_traj.text(coords[0, 0], coords[0, 1], 'Start', fontsize=8, fontweight='bold')
        ax_traj.text(coords[-1, 0], coords[-1, 1], f'Ep {epochs[-1]}', fontsize=8, fontweight='bold', color='red')

        ax_traj.set_title(f"{title_prefix} Trajectory (Var: {total_var:.1f}%)", fontsize=11, fontweight='bold')
        ax_traj.grid(True, alpha=0.3)
        ax_traj.set_xticks([])
        ax_traj.set_yticks([])

        ax_vel = fig.add_subplot(gs[row_idx, 1])
        ax_vel.plot(epochs[1:], velocity, color='#2e7d32', lw=1.6)
        ax_vel.set_title("Update Speed", fontsize=10)
        ax_vel.grid(True, alpha=0.3)
        ax_vel.tick_params(axis='x', labelsize=8)

        ax_drift = fig.add_subplot(gs[row_idx, 2])
        ax_drift.plot(epochs, drift, color='teal', lw=1.5)
        ax_drift.fill_between(epochs, drift, color='teal', alpha=0.1)
        ax_drift.set_title("Total Drift", fontsize=10)
        ax_drift.grid(True, alpha=0.3)
        ax_drift.tick_params(axis='x', labelsize=8)

        for ax in [ax_traj, ax_vel, ax_drift]:
            ax.set_facecolor('#f7f7f7')
