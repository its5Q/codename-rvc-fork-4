import os
import sys
import time
from scipy import signal
from scipy.io import wavfile
import numpy as np
import concurrent.futures
from tqdm import tqdm
import json
from distutils.util import strtobool
import librosa
import multiprocessing
import noisereduce as nr
import shutil

import soxr
from fractions import Fraction

now_directory = os.getcwd()
sys.path.append(now_directory)

from rvc.lib.utils import load_audio, load_audio_ffmpeg, loudness_normalize_audio
from rvc.train.preprocess.slicer import Slicer

import logging
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logging.getLogger("numba.core.byteflow").setLevel(logging.WARNING)
logging.getLogger("numba.core.ssa").setLevel(logging.WARNING)
logging.getLogger("numba.core.interpreter").setLevel(logging.WARNING)



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

OVERLAP = 0.3
PERCENTAGE = 3.0
MAX_AMPLITUDE = 0.9
ALPHA = 0.75
HIGH_PASS_CUTOFF = 48
SAMPLE_RATE_16K = 16000
RES_TYPE = "soxr_vhq"


def secs_to_samples(secs, sr):
    """Return an *exact* integer number of samples for `secs` seconds at `sr` Hz.
       Raises if the result is not an integer (prevents float drift)."""
    frac = Fraction(str(secs)) * sr
    if frac.denominator != 1:
        raise ValueError(f"{secs}s × {sr}Hz is not an integer sample count")
    return frac.numerator


class PreProcess:
    def __init__(self, sr: int, exp_dir: str):
        self.slicer = Slicer(
            sr=sr,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.sr = sr
        self.b_high, self.a_high = signal.butter(N=5, Wn=HIGH_PASS_CUTOFF, btype="high", fs=self.sr)
        self.exp_dir = exp_dir
        self.device = "cpu"
        self.gt_wavs_dir = os.path.join(exp_dir, "sliced_audios")
        self.wavs16k_dir = os.path.join(exp_dir, "sliced_audios_16k")
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

    def _normalize_audio_blend(self, audio: np.ndarray):
        tmp_max = np.abs(audio).max()
        if tmp_max > 2.5:
            return None
        return (audio / tmp_max * (MAX_AMPLITUDE * ALPHA)) + (1 - ALPHA) * audio

    def _normalize_audio_rms(self, audio: np.ndarray):
        rms = np.sqrt(np.mean(audio**2))
        if rms < 1e-8:  # Avoid division by zero for silent audio
            return None
        target_rms = 0.13 # or 0.16
        return audio * (target_rms / rms)

    def process_audio_segment(
        self,
        normalized_audio: np.ndarray,
        sid: int,
        idx0: int,
        idx1: int,
        normalization_mode: str,
        loading_resampling: str,
        target_lufs: float,
    ):
        # Warning for discarded / rejected slices
        if normalized_audio is None:
            logger.warning(f"{sid}-{idx0}-{idx1}-filtered (audio discarded)")
            return
        # Post-normalization on slices
        if normalization_mode == "post":
            normalized_audio = loudness_normalize_audio(normalized_audio, self.sr, target_lufs)
            # Safety
            if not isinstance(normalized_audio, np.ndarray):
                logger.warning(f"Audio segment {sid}-{idx0}-{idx1} discarded due to clipping. LUFS target you picked isn't adequate for your set.")
                return
        # Saving slices for GroundTruth ( 'sliced_audios' dir )
        wavfile.write(
            os.path.join(self.gt_wavs_dir, f"{sid}_{idx0}_{idx1}.wav"),
            self.sr,
            normalized_audio.astype(np.float32),
        )
        # Resampling of slices for wavs16k ( 'sliced_audios_16k' dir )
        if loading_resampling == "librosa":
            chunk_16k = librosa.resample(
                normalized_audio, orig_sr=self.sr, target_sr=SAMPLE_RATE_16K, res_type=RES_TYPE
            )
        else: # ffmpeg
            chunk_16k = load_audio_ffmpeg(
                normalized_audio, sample_rate=SAMPLE_RATE_16K, source_sr=self.sr,
            )
        # Saving slices for 16khz ( 'sliced_audios_16k' dir )
        wavfile.write(
            os.path.join(self.wavs16k_dir, f"{sid}_{idx0}_{idx1}.wav"),
            SAMPLE_RATE_16K,
            chunk_16k.astype(np.float32),
        )
    def simple_cut(
        self,
        audio: np.ndarray,
        sid: int,
        idx0: int,
        chunk_len: float,
        overlap_len: float,
        normalization_mode: str,
        loading_resampling: str,
        target_lufs: float,
    ):
        chunk_len_smpl = secs_to_samples(chunk_len, self.sr)
        stride = chunk_len_smpl - secs_to_samples(overlap_len, self.sr)

        slice_idx = 0
        i = 0
        while i < len(audio):
            chunk = audio[i : i + chunk_len_smpl]
            # Post-normalization on slices
            if normalization_mode == "post":
                chunk = loudness_normalize_audio(chunk, self.sr, target_lufs)
                # Safety
                if not isinstance(chunk, np.ndarray):
                    raise ValueError(f"Loudness post-normalization failed for audio segment ({sid}-{idx0}-{slice_idx}) due to clipping. LUFS target you picked isn't adequate for your set.")
            # If the last slice's below 3 seconds, discard it.
            if len(chunk) < chunk_len_smpl:
                logger.warning(f"The last resulting slice ({sid}-{idx0}-{slice_idx}) is too short: {len(chunk)} < {chunk_len_smpl} samples - Discarding!")
                break
            # Saving slices for GroundTruth ( 'sliced_audios' dir )
            wavfile.write(
                os.path.join(self.gt_wavs_dir, f"{sid}_{idx0}_{slice_idx}.wav"),
                self.sr, chunk.astype(np.float32))
            # Resampling of slices for wavs16k ( 'sliced_audios_16k' dir )
            if loading_resampling == "librosa":
                chunk_16k = librosa.resample(
                    chunk, orig_sr=self.sr, target_sr=SAMPLE_RATE_16K, res_type=RES_TYPE
                )
            else: # ffmpeg
                chunk_16k = load_audio_ffmpeg(
                    chunk, sample_rate=SAMPLE_RATE_16K, source_sr=self.sr,
                )
            # Saving slices for 16khz ( 'sliced_audios_16k' dir )
            wavfile.write(
                os.path.join(self.wavs16k_dir, f"{sid}_{idx0}_{slice_idx}.wav"),
                SAMPLE_RATE_16K, chunk_16k.astype(np.float32))
            slice_idx += 1
            i += stride

    def process_audio(
        self,
        path: str,
        idx0: int,
        sid: int,
        cut_preprocess: str,
        process_effects: bool,
        noise_reduction: bool,
        reduction_strength: float,
        chunk_len: float,
        overlap_len: float,
        normalization_mode: str,
        loading_resampling: str,
        target_lufs: float,
    ):
        audio_length = 0
        try:
            # Loading the audio
            if loading_resampling == "librosa":
                audio = load_audio(path, self.sr) # Librosa's using SoXr
            else:
                audio = load_audio_ffmpeg(path, self.sr) # FFmpeg's using Windowed Sinc filter with Blackman-Nuttall window.

            # Getting the length
            audio_length = librosa.get_duration(y=audio, sr=self.sr)

            # Processing, Filtering, Noise reduction
            if process_effects:
                audio = signal.lfilter(self.b_high, self.a_high, audio)
            if normalization_mode == "pre":
                audio = loudness_normalize_audio(audio, self.sr, target_lufs)
                # Safety
                if not isinstance(audio, np.ndarray):
                    raise ValueError(f"Loudness pre-normalization failed for file {path} due to clipping. ( LUFS target you picked isn't adequate for your set.)")
            if noise_reduction:
                audio = nr.reduce_noise(y=audio, sr=self.sr, prop_decrease=reduction_strength)

            # Slicing approach
            if cut_preprocess == "Skip":
                self.process_audio_segment(audio, sid, idx0, 0, normalization_mode, loading_resampling, target_lufs)
            elif cut_preprocess == "Simple":
                self.simple_cut(audio, sid, idx0, chunk_len, overlap_len, normalization_mode, loading_resampling, target_lufs)
            elif cut_preprocess == "Automatic":
                idx1 = 0
                for audio_segment in self.slicer.slice(audio):
                    i = 0
                    while True:
                        start = int(self.sr * (PERCENTAGE - OVERLAP) * i)
                        i += 1
                        if len(audio_segment[start:]) > (PERCENTAGE + OVERLAP) * self.sr:
                            tmp_audio = audio_segment[start : start + int(PERCENTAGE * self.sr)]
                            self.process_audio_segment(tmp_audio, sid, idx0, idx1, normalization_mode, loading_resampling, target_lufs)
                            idx1 += 1
                        else:
                            tmp_audio = audio_segment[start:]
                            self.process_audio_segment(tmp_audio, sid, idx0, idx1, normalization_mode, loading_resampling, target_lufs)
                            idx1 += 1
                            break
        except Exception as e:
            logger.error(f"Error processing {path}: {e}")
            raise e
        return audio_length

def format_duration(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def save_dataset_duration(file_path, dataset_duration):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}

    formatted_duration = format_duration(dataset_duration)
    new_data = {
        "total_dataset_duration": formatted_duration,
        "total_seconds": dataset_duration,
    }
    data.update(new_data)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def cleanup_dirs(exp_dir):
    gt_wavs_dir = os.path.join(exp_dir, "sliced_audios")
    wavs16k_dir = os.path.join(exp_dir, "sliced_audios_16k")
    logger.info("Cleaning up partially processed audio directories...")
    if os.path.exists(gt_wavs_dir):
        shutil.rmtree(gt_wavs_dir)
        logger.info(f"Deleted directory: {gt_wavs_dir}")
    if os.path.exists(wavs16k_dir):
        shutil.rmtree(wavs16k_dir)
        logger.info(f"Deleted directory: {wavs16k_dir}")


def process_audio_worker(args):
    # Each worker creates its own PreProcess instance
    (
        path,
        idx0,
        sid,
        sr,
        exp_dir,
        cut_preprocess,
        process_effects,
        noise_reduction,
        reduction_strength,
        chunk_len,
        overlap_len,
        normalization_mode,
        loading_resampling,
        target_lufs,
    ) = args
    pp = PreProcess(sr, exp_dir)
    return pp.process_audio(
        path,
        idx0,
        sid,
        cut_preprocess,
        process_effects,
        noise_reduction,
        reduction_strength,
        chunk_len,
        overlap_len,
        normalization_mode,
        loading_resampling,
        target_lufs,
    )

def preprocess_training_set(
    input_root: str,
    sr: int,
    num_processes: int,
    exp_dir: str,
    cut_preprocess: str,
    process_effects: bool,
    noise_reduction: bool,
    reduction_strength: float,
    chunk_len: float,
    overlap_len: float,
    normalization_mode: str,
    loading_resampling: str,
    target_lufs: float,
):
    start_time = time.time()
    logger.info(f"Starting preprocess with {num_processes} processes...")

    files = []
    idx = 0

    for root, _, filenames in os.walk(input_root):
        try:
            sid = 0 if root == input_root else int(os.path.basename(root))
            for f in filenames:
                if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                    files.append((os.path.join(root, f), idx, sid))
                    idx += 1
        except ValueError:
            logger.warning(
                f'Speaker ID folder is expected to be integer, got "{os.path.basename(root)}" instead.'
            )

    while True:
        try:
            audio_lengths = []

            gt_wavs_dir = os.path.join(exp_dir, "sliced_audios")
            wavs16k_dir = os.path.join(exp_dir, "sliced_audios_16k")
            os.makedirs(gt_wavs_dir, exist_ok=True)
            os.makedirs(wavs16k_dir, exist_ok=True)

            arg_list = [
                (
                    file_path,
                    idx0,
                    sid,
                    sr,
                    exp_dir,
                    cut_preprocess,
                    process_effects,
                    noise_reduction,
                    reduction_strength,
                    chunk_len,
                    overlap_len,
                    normalization_mode,
                    loading_resampling,
                    target_lufs,
                )
                for (file_path, idx0, sid) in files
            ]

            with tqdm(total=len(arg_list)) as pbar:
                with multiprocessing.Pool(processes=num_processes) as pool:
                    for result in pool.imap_unordered(process_audio_worker, arg_list):
                        audio_lengths.append(result)
                        pbar.update(1)

            total_audio_length = sum(audio_lengths)
            save_dataset_duration(os.path.join(exp_dir, "model_info.json"), total_audio_length)

            elapsed_time = time.time() - start_time
            logger.info(f"Preprocess completed in {elapsed_time:.2f} seconds "
                        f"on {format_duration(total_audio_length)} of audio.")

            break

        except ValueError as e:
            logger.error(f"Preprocessing aborted: {e}")
            cleanup_dirs(exp_dir)
            target_lufs -= 1.0
            logger.info(f"Retrying with a lower LUFS target: {target_lufs}")
            if target_lufs < -40.0:
                logger.error("LUFS target too low. Aborting preprocessing.. uhh.. Guess your dataset is a no-go haha.")
                break


if __name__ == "__main__":
    if len(sys.argv) < 14:
        print("Usage: python preprocess.py <experiment_directory> <input_root> <sample_rate> <num_processes or 'none'> <cut_preprocess> <process_effects> <noise_reduction> <reduction_strength> <chunk_len> <overlap_len> <normalization_mode> <loading_resampling> <target_lufs>")
        sys.exit(1)
    experiment_directory = str(sys.argv[1])
    input_root = str(sys.argv[2])
    sample_rate = int(sys.argv[3])
    num_processes = sys.argv[4]

    if num_processes.lower() == "none":
        num_processes = multiprocessing.cpu_count()
    else:
        num_processes = int(num_processes)

    cut_preprocess = str(sys.argv[5])
    process_effects = bool(strtobool(sys.argv[6]))
    noise_reduction = bool(strtobool(sys.argv[7]))
    reduction_strength = float(sys.argv[8])
    chunk_len = float(sys.argv[9])
    overlap_len = float(sys.argv[10])
    normalization_mode = str(sys.argv[11])
    loading_resampling = str(sys.argv[12])
    target_lufs = float(sys.argv[13])

    preprocess_training_set(
        input_root,
        sample_rate,
        num_processes,
        experiment_directory,
        cut_preprocess,
        process_effects,
        noise_reduction,
        reduction_strength,
        chunk_len,
        overlap_len,
        normalization_mode,
        loading_resampling,
        target_lufs,
    )
