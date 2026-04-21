"""Tests for the Nano-vLLM VoxCPM backend integration points."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any, cast

import pytest

from livekit.wakeword.config import TtsBackend, WakeWordConfig
from livekit.wakeword.data.tts.backends import get_tts_backend
from livekit.wakeword.data.tts.voxcpm_nanovllm_backend import VoxCpmNanoVllmBackend


def test_get_tts_backend_returns_nanovllm_backend(tmp_path: Path) -> None:
    cfg = WakeWordConfig(
        model_name="t",
        target_phrases=["hey"],
        data_dir=str(tmp_path / "data"),
        tts_backend=TtsBackend.voxcpm_nanovllm,
    )
    backend = get_tts_backend(cfg)
    assert isinstance(backend, VoxCpmNanoVllmBackend)


def test_validate_artifacts_requires_model_files(tmp_path: Path) -> None:
    model_dir = tmp_path / "data" / "voxcpm" / "VoxCPM2"
    model_dir.mkdir(parents=True)
    backend = VoxCpmNanoVllmBackend(
        model_dir=model_dir,
        voice_design_prompts=["prompt"],
        cfg_values=[2.0],
        temperature_values=[1.0],
        inference_timesteps=10,
        devices=[0],
        max_num_seqs=8,
        concurrency=4,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.9,
        enforce_eager=False,
    )
    with pytest.raises(FileNotFoundError, match="missing required files"):
        backend.validate_artifacts()


def test_validate_artifacts_requires_nanovllm_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_dir = tmp_path / "data" / "voxcpm" / "VoxCPM2"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "audiovae.pth").write_bytes(b"ok")
    (model_dir / "model.safetensors").write_bytes(b"ok")
    backend = VoxCpmNanoVllmBackend(
        model_dir=model_dir,
        voice_design_prompts=["prompt"],
        cfg_values=[2.0],
        temperature_values=[1.0],
        inference_timesteps=10,
        devices=[0],
        max_num_seqs=8,
        concurrency=4,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.9,
        enforce_eager=False,
    )
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "nanovllm_voxcpm" else object())
    with pytest.raises(ImportError, match="nano-vllm-voxcpm is not installed"):
        backend.validate_artifacts()


@pytest.mark.asyncio
async def test_generate_one_writes_clip(tmp_path: Path) -> None:
    backend = VoxCpmNanoVllmBackend(
        model_dir=tmp_path,
        voice_design_prompts=["prompt"],
        cfg_values=[2.0],
        temperature_values=[1.0],
        inference_timesteps=10,
        devices=[0],
        max_num_seqs=8,
        concurrency=4,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.9,
        enforce_eager=False,
    )

    class FakeServer:
        async def generate(self, **kwargs: Any):
            yield [0.1, -0.1, 0.2]

    out = await backend._generate_one(
        FakeServer(),
        sem=__import__("asyncio").Semaphore(1),
        phrases=["hello"],
        output_dir=tmp_path,
        sample_idx=3,
        src_sr=16000,
    )
    assert out == tmp_path / "clip_000003.wav"
    assert cast(Path, out).exists()
