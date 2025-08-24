import os
import sys
import soxr
import librosa
import ffmpeg
import soundfile as sf
import numpy as np
import re
import unicodedata
import wget
from torch import nn
import pyloudnorm as pyln

import logging
from transformers import HubertModel
import warnings

# Remove this to see warnings about transformers models
warnings.filterwarnings("ignore")

logging.getLogger("fairseq").setLevel(logging.ERROR)
logging.getLogger("faiss.loader").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

now_dir = os.getcwd()
sys.path.append(now_dir)

base_path = os.path.join(now_dir, "rvc", "models", "formant", "stftpitchshift")
stft = base_path + ".exe" if sys.platform == "win32" else base_path


class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

# Not used anymore. All logic contained in the ' preprocess.py '
def get_loudness_and_peak(audio: np.ndarray, sample_rate: int):
    """
    Measures the integrated loudness and true peak of an audio signal in a single pass.

    Args:
        audio (np.ndarray): The input audio signal as a NumPy array (np.float32).
        sample_rate (int): The sample rate of the audio.

    Returns:
        tuple: A tuple containing (integrated_loudness, true_peak).
    """
    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(audio)
    true_peak = 20 * np.log10(np.max(np.abs(audio))) # Calculate true peak manually
    return loudness, true_peak

# Not used anymore. All logic contained in the ' preprocess.py '
def loudness_normalize_audio(audio: np.ndarray, sample_rate: int, target_lufs: float = -23.0) -> np.ndarray:
    """
    Normalizes the perceived loudness of an audio signal to a target LUFS value.

    Args:
        audio (np.ndarray): The input audio signal as a NumPy array (np.float32).
        sample_rate (int): The sample rate of the audio.
        target_lufs (float): The desired loudness level in LUFS.

    Returns:
        np.ndarray: The loudness-normalized audio signal.
    """
    try:
        meter = pyln.Meter(sample_rate, block_size=0.200) # True Peak Meter with 200ms blocks
        loudness = meter.integrated_loudness(audio)
        normalized_audio = pyln.normalize.loudness(audio, loudness, target_lufs)

        # Safety for wrong LUFS
        if np.abs(normalized_audio).max() > 1.0:
            return None

        return normalized_audio.astype(np.float32)
    except Exception as e:
        print(f"Loudness normalization failed: {e}")
        return audio



def load_audio_16k(file):
    # this is used by f0 and feature extractions that load preprocessed 16k files, so there's no need to resample - Noobies
    try:
        audio, sr = librosa.load(file, sr=16000)
    except Exception as error:
        raise RuntimeError(f"An error occurred loading the audio: {error}")

    return audio.flatten()


def load_audio(file, sample_rate):
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        audio, sr = sf.read(file)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.T)
        if sr != sample_rate:
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=sample_rate, res_type="soxr_vhq"
            )
    except Exception as error:
        raise RuntimeError(f"An error occurred loading the audio: {error}")

    return audio.flatten()


def load_audio_ffmpeg(
    source: [str, np.ndarray],
    sample_rate: int = 48000,
    source_sr: int = None,
) -> np.ndarray:
    """
    Args:
        source (str | np.ndarray): The path to the audio file or an in-memory audio chunk.
        sample_rate (int): The target sample rate to resample the audio to.
        source_sr (int): The sample rate of the input source. Required for in-memory audio.

    Returns:
        np.ndarray: A NumPy array containing the audio waveform as 32-bit floats.
    """
    if isinstance(source, str):
        # Handle file path
        source = source.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        if not os.path.exists(source):
            raise FileNotFoundError(f"The audio file was not found at the provided path: {source}")

        try:
            out, err = (
                ffmpeg.input(source, threads=0)
                .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sample_rate)
                .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            print(e.stderr.decode())  # Print FFmpeg's error output for debugging
            raise RuntimeError(f"Failed to load audio file '{source}':\n{e.stderr.decode()}") from e
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while loading audio: {e}") from e
    elif isinstance(source, np.ndarray):
        # Handle in-memory audio chunk
        if source_sr is None:
            raise ValueError("source_sr must be provided when passing a NumPy array.")
        
        # Ensure the array is a 32-bit float and mono
        if source.dtype != np.float32:
            source = source.astype(np.float32)

        if source.ndim > 1:
            # If stereo, convert to mono
            source = np.mean(source, axis=1)

        try:
            process = (
                ffmpeg
                .input('pipe:0', format='f32le', acodec='pcm_f32le', ar=source_sr, ac=1)
                .output('pipe:1', format='f32le', acodec='pcm_f32le', ar=sample_rate)
                .run_async(pipe_stdin=True, pipe_stdout=True, quiet=True)
            )
            out, err = process.communicate(input=source.tobytes())
        except ffmpeg.Error as e:
            print(e.stderr.decode()) # Print FFmpeg's error output for debugging
            raise RuntimeError(f"Failed to resample audio chunk:\n{e.stderr.decode()}") from e
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while processing audio chunk: {e}") from e
    else:
        raise ValueError("Invalid source type. Must be a file path (str) or a NumPy array (np.ndarray).")

    return np.frombuffer(out, np.float32).flatten()


def load_audio_infer(
    file,
    sample_rate,
    **kwargs,
):
    formant_shifting = kwargs.get("formant_shifting", False)
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        if not os.path.isfile(file):
            raise FileNotFoundError(f"File not found: {file}")
        audio, sr = sf.read(file)

        print(f"[INFER] loaded audio: {file}")

        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.T)
            print("[WARNING] Provided input audio is in stereo. Converting to mono. - For future, please use mono only.")
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate, res_type="soxr_vhq")

        if formant_shifting:
            formant_qfrency = kwargs.get("formant_qfrency", 0.8)
            formant_timbre = kwargs.get("formant_timbre", 0.8)

            from stftpitchshift import StftPitchShift

            pitchshifter = StftPitchShift(1024, 32, sample_rate)
            audio = pitchshifter.shiftpitch(
                audio,
                factors=1,
                quefrency=formant_qfrency * 1e-3,
                distortion=formant_timbre,
            )
    except Exception as error:
        raise RuntimeError(f"An error occurred loading the audio: {error}")
    return np.array(audio).flatten()


def format_title(title):
    formatted_title = unicodedata.normalize("NFC", title)
    formatted_title = re.sub(r"[\u2500-\u257F]+", "", formatted_title)
    formatted_title = re.sub(r"[^\w\s.-]", "", formatted_title, flags=re.UNICODE)
    formatted_title = re.sub(r"\s+", "_", formatted_title)
    return formatted_title


def load_embedding(embedder_model, custom_embedder=None):
    embedder_root = os.path.join(now_dir, "rvc", "models", "embedders")
    embedding_list = {
        "contentvec": os.path.join(embedder_root, "contentvec"),
        "spin": os.path.join(embedder_root, "spin"),
        "chinese-hubert-base": os.path.join(embedder_root, "chinese_hubert_base"),
        "japanese-hubert-base": os.path.join(embedder_root, "japanese_hubert_base"),
        "korean-hubert-base": os.path.join(embedder_root, "korean_hubert_base"),
    }

    online_embedders = {
        "contentvec": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/contentvec/pytorch_model.bin",
        "chinese-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/chinese_hubert_base/pytorch_model.bin",
        "japanese-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/japanese_hubert_base/pytorch_model.bin",
        "korean-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/korean_hubert_base/pytorch_model.bin",
    }

    config_files = {
        "contentvec": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/contentvec/config.json",
        "chinese-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/chinese_hubert_base/config.json",
        "japanese-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/japanese_hubert_base/config.json",
        "korean-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/korean_hubert_base/config.json",
    }

    if embedder_model == "custom":
        if os.path.exists(custom_embedder):
            model_path = custom_embedder
        else:
            print(f"Custom embedder not found: {custom_embedder}, using contentvec")
            model_path = embedding_list["contentvec"]
    elif embedder_model == "spin":
        model_path = embedding_list[embedder_model]
        bin_file = os.path.join(model_path, "pytorch_model.bin")
        json_file = os.path.join(model_path, "config.json")
    else:
        model_path = embedding_list[embedder_model]
        bin_file = os.path.join(model_path, "pytorch_model.bin")
        json_file = os.path.join(model_path, "config.json")
        os.makedirs(model_path, exist_ok=True)
        if not os.path.exists(bin_file):
            url = online_embedders[embedder_model]
            print(f"Downloading {url} to {model_path}...")
            wget.download(url, out=bin_file)
        if not os.path.exists(json_file):
            url = config_files[embedder_model]
            print(f"Downloading {url} to {model_path}...")
            wget.download(url, out=json_file)

    models = HubertModelWithFinalProj.from_pretrained(model_path)
    return models
