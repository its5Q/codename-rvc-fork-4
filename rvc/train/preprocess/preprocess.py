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

import soxr
from fractions import Fraction

now_directory = os.getcwd()
sys.path.append(now_directory)

from rvc.lib.utils import load_audio
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
        self.b_high, self.a_high = signal.butter(
            N=5, Wn=HIGH_PASS_CUTOFF, btype="high", fs=self.sr
        )
        self.exp_dir = exp_dir
        self.device = "cpu"
        self.gt_wavs_dir = os.path.join(exp_dir, "sliced_audios")
        self.wavs16k_dir = os.path.join(exp_dir, "sliced_audios_16k")
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

    def _normalize_audio(self, audio: np.ndarray):
        tmp_max = np.abs(audio).max()
        if tmp_max > 2.5:
            return None
        return (audio / tmp_max * (MAX_AMPLITUDE * ALPHA)) + (1 - ALPHA) * audio

    def process_audio_segment(
        self,
        normalized_audio: np.ndarray,
        sid: int,
        idx0: int,
        idx1: int,
        normalization_mode: str,
    ):
        if normalized_audio is None:
            logger.warning(f"{sid}-{idx0}-{idx1}-filtered (audio discarded)")
            return
        if normalization_mode == "post":
            normalized_audio = self._normalize_audio(normalized_audio)
        wavfile.write(
            os.path.join(self.gt_wavs_dir, f"{sid}_{idx0}_{idx1}.wav"),
            self.sr,
            normalized_audio.astype(np.float32),
        )
        audio_16k = librosa.resample(
            normalized_audio,
            orig_sr=self.sr,
            target_sr=SAMPLE_RATE_16K,
            res_type=RES_TYPE,
        )
        wavfile.write(
            os.path.join(self.wavs16k_dir, f"{sid}_{idx0}_{idx1}.wav"),
            SAMPLE_RATE_16K,
            audio_16k.astype(np.float32),
        )

    def simple_cut(
        self,
        audio: np.ndarray,
        sid: int,
        idx0: int,
        chunk_len: float,
        overlap_len: float,
        normalization_mode: str,
    ):
        chunk_len_smpl = secs_to_samples(chunk_len, self.sr)
        stride = chunk_len_smpl - secs_to_samples(overlap_len, self.sr)

        slice_idx = 0
        i = 0
        while i < len(audio):
            chunk = audio[i : i + chunk_len_smpl]
            if normalization_mode == "post":
                chunk = self._normalize_audio(chunk)

            if len(chunk) < chunk_len_smpl:
                logger.warning(f"[SKIP] SimpleCut tail: {sid}-{idx0}-{slice_idx} ({len(chunk)} < {chunk_len_smpl} samples)")
                break

            # save full-rate
            wavfile.write(
                os.path.join(self.gt_wavs_dir, f"{sid}_{idx0}_{slice_idx}.wav"),
                self.sr, chunk.astype(np.float32))
            # save 16 kHz
            chunk_16k = librosa.resample(
                chunk, orig_sr=self.sr, target_sr=SAMPLE_RATE_16K, res_type=RES_TYPE
            )
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
        normalization_mode: str
    ):
        audio_length = 0
        try:
            audio = load_audio(path, self.sr)
            audio_length = librosa.get_duration(y=audio, sr=self.sr)

            if process_effects:
                audio = signal.lfilter(self.b_high, self.a_high, audio)
            if normalization_mode == "pre":
                audio = self._normalize_audio(audio)
            if noise_reduction:
                audio = nr.reduce_noise(
                    y=audio, sr=self.sr, prop_decrease=reduction_strength
                )
            if cut_preprocess == "Skip":
                self.process_audio_segment(audio, sid, idx0, 0, normalization_mode)
            elif cut_preprocess == "Simple":
                self.simple_cut(audio, sid, idx0, chunk_len, overlap_len, normalization_mode)
            elif cut_preprocess == "Automatic":
                idx1 = 0
                for audio_segment in self.slicer.slice(audio):
                    i = 0
                    while True:
                        start = int(self.sr * (PERCENTAGE - OVERLAP) * i)
                        i += 1
                        if len(audio_segment[start:]) > (PERCENTAGE + OVERLAP) * self.sr:
                            tmp_audio = audio_segment[start : start + int(PERCENTAGE * self.sr)]
                            self.process_audio_segment(tmp_audio, sid, idx0, idx1, normalization_mode)
                            idx1 += 1
                        else:
                            tmp_audio = audio_segment[start:]
                            self.process_audio_segment(tmp_audio, sid, idx0, idx1, normalization_mode)
                            idx1 += 1
                            break
        except Exception as e:
            logger.error(f"Error processing {path}: {e}")
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

    audio_lengths = []

    # Prepare argument list for multiprocessing
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
            normalization_mode
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

if __name__ == "__main__":
    if len(sys.argv) < 12:
        print("Usage: python preprocess.py <experiment_directory> <input_root> <sample_rate> <num_processes or 'none'> <cut_preprocess> <process_effects> <noise_reduction> <reduction_strength> <chunk_len> <overlap_len>")
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
        normalization_mode
    )
