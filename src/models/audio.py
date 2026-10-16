"""Audio decoding for Whisper: everything becomes 16 kHz mono float32."""

from __future__ import annotations

import io
from typing import Optional

import numpy as np

WHISPER_SAMPLE_RATE = 16000


def _to_mono_16k(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    samples = samples.astype(np.float32, copy=False)
    if sample_rate != WHISPER_SAMPLE_RATE:
        import soxr

        samples = soxr.resample(samples, sample_rate, WHISPER_SAMPLE_RATE).astype(np.float32, copy=False)
    return samples


def decode_file_bytes(data: bytes) -> np.ndarray:
    """Decode an audio file (wav, flac, ogg, mp3, ... anything libsndfile reads)."""
    import soundfile as sf

    try:
        samples, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"could not decode audio ({e}); send wav, flac, ogg or mp3, or raw pcm16 with sample_rate") from e
    return _to_mono_16k(samples, int(sr))


def decode_file(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        return decode_file_bytes(f.read())


def decode_pcm16(data: bytes, sample_rate: int, num_channels: int = 1) -> np.ndarray:
    """Raw little-endian 16-bit PCM, as an audio_in component emits with codec "pcm16"."""
    if sample_rate <= 0 or num_channels <= 0:
        raise ValueError("sample_rate and num_channels must be positive for pcm16 audio")
    usable = len(data) - len(data) % (2 * num_channels)
    samples = np.frombuffer(data[:usable], dtype="<i2").astype(np.float32) / 32768.0
    if num_channels > 1:
        samples = samples.reshape(-1, num_channels)
    return _to_mono_16k(samples, sample_rate)


def duration_seconds(samples: Optional[np.ndarray]) -> float:
    return 0.0 if samples is None else len(samples) / WHISPER_SAMPLE_RATE
