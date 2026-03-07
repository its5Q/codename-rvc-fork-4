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
import shutil
import soundfile as sf
import io
import soxr
from fractions import Fraction

now_directory = os.getcwd()
sys.path.append(now_directory)

from rvc.lib.utils import load_audio, load_audio_ffmpeg
from rvc.train.preprocess.slicer import Slicer
from rvc.train.preprocess.smartcutter.inference import SmartCutterInterface

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

OVERLAP = 0.3
PERCENTAGE = 3.0
MAX_AMPLITUDE = 0.9
ALPHA = 0.75
HIGH_PASS_CUTOFF = 48
SAMPLE_RATE_16K = 16000
RES_TYPE = "soxr_vhq"
FLAC_COMPRESSION_LEVEL = 5 / 8 # FLAC level 5, libsndfile uses 0.0 - 1.0 range for compression level


def secs_to_samples(secs, sr):
    """Return an *exact* integer number of samples for `secs` seconds at `sr` Hz.
       Raises if the result is not an integer (prevents float drift)."""
    frac = Fraction(str(secs)) * sr
    if frac.denominator != 1:
        raise ValueError(f"{secs}s × {sr}Hz is not an integer sample count")
    return frac.numerator

def save_audio(path: str, name: str, sample_rate: int, format: str, audio: np.ndarray):
    """
    Save audio to file.
    Args:
        path: Path to a directory where the audio file will be saved.
        name: Name of the audio file without the extension.
        sample_rate: Sample rate of the audio file.
        format: Format of the audio file (WAV or FLAC).
        audio: Audio data array.
    """
    if format.lower() == "flac":
        memory_file = io.BytesIO()
        sf.write(
            memory_file,
            audio,
            sample_rate,
            format="FLAC",
            subtype="PCM_24",
            compression_level=FLAC_COMPRESSION_LEVEL
        )
        memory_file.seek(0)
        with open(os.path.join(path, f"{name}.flac"), "wb") as f:
            f.write(memory_file.read())
    else:
        wavfile.write(
            os.path.join(path, f"{name}.wav"),
            sample_rate,
            audio.astype(np.float32),
        )

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

        self.gt_wavs_dir = os.path.join(exp_dir, "sliced_audios")
        self.wavs16k_dir = os.path.join(exp_dir, "sliced_audios_16k")
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)


    def process_audio_segment(
        self,
        audio: np.ndarray,
        sid: int,
        idx0: int,
        idx1: int,
        loading_resampling: str,
        dataset_format: str
    ):
        # Saving slices for GroundTruth ( 'sliced_audios' dir )
        save_audio(self.gt_wavs_dir, f"{sid}_{idx0}_{idx1}", self.sr, dataset_format, audio)

        # Resampling of slices for wavs16k ( 'sliced_audios_16k' dir )
        if loading_resampling == "librosa":
            chunk_16k = librosa.resample(
                audio, orig_sr=self.sr, target_sr=SAMPLE_RATE_16K, res_type=RES_TYPE
            )
        else: # ffmpeg
            chunk_16k = load_audio_ffmpeg(
                audio, sample_rate=SAMPLE_RATE_16K, source_sr=self.sr,
            )

        # Saving slices for 16khz ( 'sliced_audios_16k' dir )
        save_audio(self.wavs16k_dir, f"{sid}_{idx0}_{idx1}", SAMPLE_RATE_16K, dataset_format, chunk_16k)


    def simple_cut(
        self,
        audio: np.ndarray,
        sid: int,
        idx0: int,
        chunk_len: float,
        overlap_len: float,
        loading_resampling: str,
        dataset_format: str
    ):
        chunk_len_smpl = secs_to_samples(chunk_len, self.sr)
        stride = chunk_len_smpl - secs_to_samples(overlap_len, self.sr)

        slice_idx = 0
        i = 0
        while i < len(audio):
            chunk = audio[i : i + chunk_len_smpl]

            # If the last slice's below 3 seconds, we're padding it to 3 secs.
            if len(chunk) < chunk_len_smpl:
                padding_needed = chunk_len_smpl - len(chunk)
                if len(chunk) > self.sr * 1.0: 
                    padding = np.zeros(padding_needed, dtype=np.float32)
                    chunk = np.concatenate((chunk, padding))
                    logger.info(f"Padded final slice {sid}_{idx0}_{slice_idx} with {padding_needed} samples.")
                else:
                    break

            # Saving slices
            save_audio(self.gt_wavs_dir, f"{sid}_{idx0}_{slice_idx}", self.sr, dataset_format, chunk)

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
            save_audio(self.wavs16k_dir, f"{sid}_{idx0}_{slice_idx}", SAMPLE_RATE_16K, dataset_format, chunk_16k)

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
        loading_resampling: str,
        dataset_format: str
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
            if noise_reduction:
                import noisereduce as nr
                audio = nr.reduce_noise(y=audio, sr=self.sr, prop_decrease=reduction_strength)

            # Slicing approach
            if cut_preprocess == "Skip":
                self.process_audio_segment(audio, sid, idx0, 0, loading_resampling, dataset_format)
            elif cut_preprocess == "Simple":
                self.simple_cut(audio, sid, idx0, chunk_len, overlap_len, loading_resampling, dataset_format)
            elif cut_preprocess == "Automatic":
                idx1 = 0
                for audio_segment in self.slicer.slice(audio):
                    i = 0
                    while True:
                        start = int(self.sr * (PERCENTAGE - OVERLAP) * i)
                        i += 1
                        if len(audio_segment[start:]) > (PERCENTAGE + OVERLAP) * self.sr:
                            tmp_audio = audio_segment[start : start + int(PERCENTAGE * self.sr)]
                            self.process_audio_segment(tmp_audio, sid, idx0, idx1, loading_resampling, dataset_format)
                            idx1 += 1
                        else:
                            tmp_audio = audio_segment[start:]
                            self.process_audio_segment(tmp_audio, sid, idx0, idx1, loading_resampling, dataset_format)
                            idx1 += 1
                            break
        except Exception as e:
            logger.error(f"Error processing {path}: {e}")
            raise e
        return audio_length

def _process_audio_worker(args):
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
        loading_resampling,
        dataset_format
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
        loading_resampling,
        dataset_format
    )

def _process_and_save_worker(args):
    file_name, gt_wavs_dir, wavs16k_dir = args
    try:
        # Process ground truth audio
        gt_path = os.path.join(gt_wavs_dir, file_name)
        gt_audio, gt_sr = sf.read(gt_path)

        # Simple peak normalization to 0.95
        peak_amp = np.max(np.abs(gt_audio))
        if peak_amp > 0:
            gt_normalized = (gt_audio / peak_amp) * 0.95
        else:
            gt_normalized = gt_audio

        save_audio(gt_wavs_dir, file_name.split(".")[0], gt_sr, file_name.split(".")[1], gt_normalized)

        # Process 16k audio
        k16_path = os.path.join(wavs16k_dir, file_name)
        k16_audio, k16_sr = sf.read(k16_path)

        # Peak norm to 0.95
        peak_amp_16k = np.max(np.abs(k16_audio))
        if peak_amp_16k > 0:
            k16_normalized = (k16_audio / peak_amp_16k) * 0.95
        else:
            k16_normalized = k16_audio

        save_audio(wavs16k_dir, file_name.split(".")[0], k16_sr, file_name.split(".")[1], k16_normalized)
    except Exception as e:
        logger.error(f"Error classic normalizing {file_name}: {e}")
        raise e

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
    logger.info("Cleaning up partially processed audio directories if they exist...")
    if os.path.exists(gt_wavs_dir):
        shutil.rmtree(gt_wavs_dir)
        logger.info(f"Deleted directory: {gt_wavs_dir}")
    if os.path.exists(wavs16k_dir):
        shutil.rmtree(wavs16k_dir)
        logger.info(f"Deleted directory: {wavs16k_dir}")


def run_smart_cutter_stage(input_root, exp_dir, sr):
    """
    Stage 1: Sequential GPU processing.
    Reads from input_root, writes to 'smart_cut_temp' inside exp_dir.
    Returns the path to the NEW input root (the temp folder).
    """
    ckpt_dir = os.path.join(now_directory, r"rvc/models/smartcutter")
    output_root = os.path.join(exp_dir, "smart_cut_temp")
    os.makedirs(output_root, exist_ok=True)

    print("[SmartCutter] Starting .. this may take a bit ... ")
    print(f"[SmartCutter] Original Input: {input_root}")
    print(f"[SmartCutter] Temp Output: {output_root}")

    # Initialize Interface
    engine = SmartCutterInterface(sr, ckpt_dir)
    engine.load_model()

    files_to_process = []
    # Scan files
    for root, _, filenames in os.walk(input_root):
        for f in filenames:
            if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                full_path = os.path.join(root, f)

                # Replicate folder structure in temp dir
                rel_path = os.path.relpath(full_path, input_root)
                out_path = os.path.join(output_root, rel_path)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                files_to_process.append((full_path, out_path))

    # Process Loop
    with tqdm(total=len(files_to_process), desc="SmartCutting") as pbar:
        for in_p, out_p in files_to_process:
            engine.process_file(in_p, out_p)
            pbar.update(1)

    # Cleanup GPU
    engine.unload()
    print("SmartCutter Stage Complete. Proceeding to Slicing...")

    return output_root


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
    use_smart_cutter: bool,
    dataset_format: str
):
    start_time = time.time()

    sc_engine = None
    if use_smart_cutter:
        try:
            ckpt_dir = os.path.join(now_directory, r"rvc/models/smartcutter")
            sc_engine = SmartCutterInterface(sr, ckpt_dir)
            sc_engine.load_model()
            logger.info("SmartCutter model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load SmartCutter: {e}")
            return

    speaker_map = {}

    root_files = [f for f in os.listdir(input_root) if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".opus", ".aac"))]
    if root_files:
        speaker_map[input_root] = [os.path.join(input_root, f) for f in root_files]

    for root, dirs, filenames in os.walk(input_root):
        if root == input_root:
            continue

        audio_files = [os.path.join(root, f) for f in filenames if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".opus", ".aac"))]
        if audio_files:
            speaker_map[root] = audio_files

    speaker_count = len(speaker_map)

    if speaker_count > 1:
        detected_sids = set()
        for folder_path in speaker_map.keys():
            if folder_path == input_root:
                detected_sids.add(0)
            else:
                try:
                    folder_name = os.path.basename(folder_path)
                    sid = int(folder_name.split('_')[0])
                    detected_sids.add(sid)
                except (ValueError, IndexError):
                    logger.error(f"FATAL: Folder '{folder_name}' is invalid for multi-speaker. "
                                 f"Folders must start with an integer (e.g., '0_name').")
                    sys.exit(1)

        expected_sids = set(range(speaker_count))
        if detected_sids != expected_sids:
            missing = sorted(list(expected_sids - detected_sids))
            logger.error(f"FATAL: Speaker IDs are not contiguous or missing 0. "
                         f"Detected: {sorted(list(detected_sids))}. Missing: {missing}")
            sys.exit(1)
        else:
            logger.info("Contiguity check passed.")

    logger.info(f"Found {speaker_count} speakers to process.")
    cleanup_dirs(exp_dir)


    total_audio_length = 0

    with multiprocessing.Pool(processes=num_processes) as pool:
        for speaker_dir, audio_paths in tqdm(speaker_map.items(), desc="Processing Speakers"):

            temp_speaker_dir = None
            current_batch_paths = []
            try:
                if speaker_dir == input_root:
                    sid = 0
                else:
                    folder_name = os.path.basename(speaker_dir)
                    sid_str = folder_name.split('_')[0] 
                    sid = int(sid_str)
            except (ValueError, IndexError):
                logger.warning(f"Folder '{os.path.basename(speaker_dir)}' does not start with a valid integer ID. Using SID 0.")
                sid = 0

            if use_smart_cutter and sc_engine:
                temp_speaker_dir = os.path.join(exp_dir, "smart_cut_temp", str(sid))
                os.makedirs(temp_speaker_dir, exist_ok=True)

                for file_path in audio_paths:
                    filename = os.path.basename(file_path)
                    out_path = os.path.join(temp_speaker_dir, filename)

                    sc_engine.process_file(file_path, out_path)
                    current_batch_paths.append(out_path)
            else:
                current_batch_paths = audio_paths

            arg_list = [
                (
                    f_path,
                    idx,
                    sid,
                    sr,
                    exp_dir,
                    cut_preprocess,
                    process_effects,
                    noise_reduction,
                    reduction_strength,
                    chunk_len,
                    overlap_len,
                    loading_resampling,
                    dataset_format
                )
                for idx, f_path in enumerate(current_batch_paths)
            ]

            for result in pool.imap_unordered(_process_audio_worker, arg_list):
                if result:
                    total_audio_length += result

            if temp_speaker_dir and os.path.exists(temp_speaker_dir):
                shutil.rmtree(temp_speaker_dir)

    main_temp_dir = os.path.join(exp_dir, "smart_cut_temp")
    if os.path.exists(main_temp_dir):
        shutil.rmtree(main_temp_dir)
        
    if use_smart_cutter and sc_engine:
        sc_engine.unload()

    save_dataset_duration(os.path.join(exp_dir, "model_info.json"), total_audio_length)

    if normalization_mode == "post_peak":
        logger.info("Classic Peak Normalization enabled. Initiating...")
        gt_wavs_dir = os.path.join(exp_dir, "sliced_audios")
        wavs16k_dir = os.path.join(exp_dir, "sliced_audios_16k")

        audio_files = [f for f in os.listdir(gt_wavs_dir) if f.endswith((".wav", ".flac"))]
        audio_files.sort()

        arg_list = [(file_name, gt_wavs_dir, wavs16k_dir) for file_name in audio_files]

        with multiprocessing.Pool(processes=num_processes) as pool:
             list(tqdm(pool.imap_unordered(_process_and_save_worker, arg_list), total=len(audio_files), desc="Peak Normalization"))

    elapsed_time = time.time() - start_time
    logger.info(f"Preprocessing completed in {elapsed_time:.2f} seconds "
                f"on {format_duration(total_audio_length)} of audio.")

if __name__ == "__main__":
    if len(sys.argv) < 15:
        print("Usage: python preprocess.py <experiment_directory> <input_root> <sample_rate> <num_processes or 'none'> <cut_preprocess> <process_effects> <noise_reduction> <reduction_strength> <chunk_len> <overlap_len> <normalization_mode> <loading_resampling> <use_smart_cutter> <dataset_format>")
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
    use_smart_cutter = bool(strtobool(sys.argv[13]))
    dataset_format = str(sys.argv[14])

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
        use_smart_cutter,
        dataset_format
    )