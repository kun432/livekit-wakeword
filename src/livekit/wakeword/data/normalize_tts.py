"""Normalize generated TTS clips before augmentation."""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import TtsBackend, WakeWordConfig

logger = logging.getLogger(__name__)

_ORIGINAL_CLIP_RE = re.compile(r"^clip_\d{6}\.wav$")
_TTS_SPLITS = [
    "positive_train",
    "positive_test",
    "negative_train",
    "negative_test",
]


@dataclass
class NormalizeSummary:
    processed: int = 0
    unchanged: int = 0
    stretched: int = 0
    dropped: int = 0


def _stretch_to_target(audio: np.ndarray, target_length: int) -> np.ndarray:
    import librosa

    if len(audio) == target_length:
        return audio.astype(np.float32, copy=False)
    rate = len(audio) / float(target_length)
    stretched = librosa.effects.time_stretch(audio.astype(np.float32, copy=False), rate=rate)
    if len(stretched) > target_length:
        stretched = stretched[:target_length]
    elif len(stretched) < target_length:
        stretched = np.pad(stretched, (0, target_length - len(stretched)))
    return stretched.astype(np.float32, copy=False)


def _normalize_directory(
    clip_dir: Path,
    rejected_dir: Path,
    *,
    target_duration_s: float,
    max_duration_s: float,
    sample_rate: int = 16000,
) -> NormalizeSummary:
    import soundfile as sf
    from tqdm import tqdm

    summary = NormalizeSummary()

    wav_files = sorted(p for p in clip_dir.glob("*.wav") if _ORIGINAL_CLIP_RE.match(p.name))
    rejected_dir.mkdir(parents=True, exist_ok=True)

    for wav_path in tqdm(wav_files, desc=f"Normalize {clip_dir.name}", unit="clip"):
        audio, sr = sf.read(str(wav_path))
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio = audio.astype(np.float32, copy=False)
        duration_s = len(audio) / float(sr)
        summary.processed += 1

        if duration_s <= target_duration_s:
            summary.unchanged += 1
            continue

        if duration_s <= max_duration_s:
            normalized = _stretch_to_target(audio, int(target_duration_s * sr))
            sf.write(str(wav_path), normalized, sr)
            summary.stretched += 1
            continue

        rejected_path = rejected_dir / wav_path.name
        if rejected_path.exists():
            rejected_path.unlink()
        shutil.move(str(wav_path), str(rejected_path))
        summary.dropped += 1

    return summary


def run_normalize_tts(config: WakeWordConfig) -> None:
    """Normalize generated TTS clips to fit the model clip duration."""
    if config.tts_backend not in {TtsBackend.voxcpm, TtsBackend.voxcpm_nanovllm}:
        logger.info(
            "Skipping TTS normalization for tts_backend=%s "
            "(only supported for voxcpm / voxcpm_nanovllm)",
            config.tts_backend.value,
        )
        return

    target_duration_s = config.augmentation.clip_duration
    max_duration_s = config.tts_normalization.max_duration_s
    if max_duration_s <= target_duration_s:
        raise ValueError(
            "tts_normalization.max_duration_s must be greater than augmentation.clip_duration"
        )

    model_dir = config.model_output_dir
    rejected_root = model_dir / "rejected_tts"

    total = NormalizeSummary()
    for split in _TTS_SPLITS:
        clip_dir = model_dir / split
        if not clip_dir.exists():
            logger.warning("Skipping %s: directory not found", split)
            continue
        summary = _normalize_directory(
            clip_dir,
            rejected_root / split,
            target_duration_s=target_duration_s,
            max_duration_s=max_duration_s,
        )
        total.processed += summary.processed
        total.unchanged += summary.unchanged
        total.stretched += summary.stretched
        total.dropped += summary.dropped
        logger.info(
            "Normalized %s: processed=%d unchanged=%d stretched=%d dropped=%d",
            split,
            summary.processed,
            summary.unchanged,
            summary.stretched,
            summary.dropped,
        )

    logger.info(
        "TTS normalization complete: processed=%d unchanged=%d stretched=%d dropped=%d",
        total.processed,
        total.unchanged,
        total.stretched,
        total.dropped,
    )
