"""Tests for TTS normalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from livekit.wakeword.config import WakeWordConfig
from livekit.wakeword.data.normalize_tts import run_normalize_tts


def _write_tone(path: Path, duration_s: float, sample_rate: int = 16000) -> None:
    samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, samples, endpoint=False)
    audio = 0.1 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    sf.write(path, audio, sample_rate)


def test_normalize_tts_stretches_medium_long_clip(tmp_path: Path) -> None:
    cfg = WakeWordConfig(
        model_name="demo",
        target_phrases=["hey"],
        output_dir=str(tmp_path),
        tts_backend="voxcpm_nanovllm",
        tts_normalization={"enabled": True, "max_duration_s": 2.5},
    )
    split_dir = cfg.model_output_dir / "positive_train"
    split_dir.mkdir(parents=True, exist_ok=True)
    wav_path = split_dir / "clip_000000.wav"
    _write_tone(wav_path, 2.25)

    run_normalize_tts(cfg)

    audio, sr = sf.read(wav_path)
    assert sr == 16000
    assert len(audio) == int(cfg.augmentation.clip_duration * sr)


def test_normalize_tts_moves_overlong_clip_to_rejected(tmp_path: Path) -> None:
    cfg = WakeWordConfig(
        model_name="demo",
        target_phrases=["hey"],
        output_dir=str(tmp_path),
        tts_backend="voxcpm_nanovllm",
        tts_normalization={"enabled": True, "max_duration_s": 2.5},
    )
    split_dir = cfg.model_output_dir / "negative_train"
    split_dir.mkdir(parents=True, exist_ok=True)
    wav_path = split_dir / "clip_000001.wav"
    _write_tone(wav_path, 2.6)

    run_normalize_tts(cfg)

    assert not wav_path.exists()
    rejected = cfg.model_output_dir / "rejected_tts" / "negative_train" / "clip_000001.wav"
    assert rejected.exists()


def test_normalize_tts_skips_non_voxcpm_backend(tmp_path: Path) -> None:
    cfg = WakeWordConfig(
        model_name="demo",
        target_phrases=["hey"],
        output_dir=str(tmp_path),
        tts_backend="piper_vits",
        tts_normalization={"enabled": True, "max_duration_s": 2.5},
    )
    split_dir = cfg.model_output_dir / "positive_train"
    split_dir.mkdir(parents=True, exist_ok=True)
    wav_path = split_dir / "clip_000000.wav"
    _write_tone(wav_path, 2.25)

    run_normalize_tts(cfg)

    audio, sr = sf.read(wav_path)
    assert sr == 16000
    assert len(audio) == int(2.25 * sr)
