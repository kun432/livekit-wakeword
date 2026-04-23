"""Nano-vLLM-backed VoxCPM2 TTS backend with concurrent generation."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Any

import numpy as np

from ...config import WakeWordConfig

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16_000


def diversification_triple_at_index(
    prompts: list[str],
    cfg_values: list[float],
    temperature_values: list[float],
    index: int,
) -> tuple[str, float, float]:
    """Return the (prompt, cfg, temperature) triple for global clip *index*."""
    np_ = len(prompts)
    nc = len(cfg_values)
    nt = len(temperature_values)
    n = np_ * nc * nt
    if n == 0:
        raise ValueError("voxcpm_nanovllm diversification lists must be non-empty")
    flat = index % n
    ti = flat % nt
    flat //= nt
    ci = flat % nc
    pi = flat // nc
    return prompts[pi], cfg_values[ci], temperature_values[ti]


class VoxCpmNanoVllmBackend:
    """Nano-vLLM-backed VoxCPM2 generator with bounded async concurrency."""

    def __init__(
        self,
        *,
        model_dir: Path,
        voice_design_prompts: list[str],
        cfg_values: list[float],
        temperature_values: list[float],
        inference_timesteps: int,
        devices: list[int],
        max_num_seqs: int,
        concurrency: int,
        max_num_batched_tokens: int,
        gpu_memory_utilization: float,
        enforce_eager: bool,
    ) -> None:
        self._model_dir = model_dir
        self._prompts = voice_design_prompts
        self._cfg_values = cfg_values
        self._temperature_values = temperature_values
        self._inference_timesteps = inference_timesteps
        self._devices = devices
        self._max_num_seqs = max_num_seqs
        self._concurrency = concurrency
        self._max_num_batched_tokens = max_num_batched_tokens
        self._gpu_memory_utilization = gpu_memory_utilization
        self._enforce_eager = enforce_eager

    @classmethod
    def from_config(cls, config: WakeWordConfig) -> VoxCpmNanoVllmBackend:
        vt = config.voxcpm_tts
        nt = config.voxcpm_nanovllm_tts
        return cls(
            model_dir=config.voxcpm_local_model_path,
            voice_design_prompts=list(vt.voice_design_prompts),
            cfg_values=list(vt.cfg_values),
            temperature_values=list(nt.temperature_values),
            inference_timesteps=nt.inference_timesteps,
            devices=list(nt.devices),
            max_num_seqs=nt.max_num_seqs,
            concurrency=nt.concurrency,
            max_num_batched_tokens=nt.max_num_batched_tokens,
            gpu_memory_utilization=nt.gpu_memory_utilization,
            enforce_eager=nt.enforce_eager,
        )

    def validate_artifacts(self) -> None:
        if not self._model_dir.is_dir():
            raise FileNotFoundError(
                f"VoxCPM model directory not found: {self._model_dir}. "
                "Run: livekit-wakeword setup --config <your.yaml>"
            )
        required = [
            self._model_dir / "config.json",
            self._model_dir / "audiovae.pth",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "VoxCPM model directory is missing required files: " + ", ".join(missing)
            )
        if not list(self._model_dir.glob("*.safetensors")):
            raise FileNotFoundError(
                f"No *.safetensors weights found under {self._model_dir}. "
                "Nano-vLLM requires safetensors-based VoxCPM checkpoints."
            )
        if not self._prompts or not self._cfg_values or not self._temperature_values:
            raise ValueError(
                "voxcpm_tts.voice_design_prompts, voxcpm_tts.cfg_values, and "
                "voxcpm_nanovllm_tts.temperature_values must be non-empty"
            )
        if importlib.util.find_spec("nanovllm_voxcpm") is None:
            raise ImportError(
                "nano-vllm-voxcpm is not installed. "
                "Install it in this environment before using tts_backend=voxcpm_nanovllm."
            )

    def synthesize_clips(
        self,
        phrases: list[str],
        output_dir: Path,
        n_samples: int,
        *,
        start_index: int = 0,
        batch_size: int = 50,
    ) -> list[Path]:
        del batch_size
        if not phrases:
            raise ValueError("phrases must be non-empty")
        output_dir.mkdir(parents=True, exist_ok=True)
        return asyncio.run(self._synthesize_async(phrases, output_dir, n_samples, start_index))

    async def _synthesize_async(
        self,
        phrases: list[str],
        output_dir: Path,
        n_samples: int,
        start_index: int,
    ) -> list[Path]:
        nanovllm_mod = importlib.import_module("nanovllm_voxcpm")
        VoxCPM = getattr(nanovllm_mod, "VoxCPM")
        logger.info("Loading Nano-vLLM VoxCPM from %s", self._model_dir)
        server = VoxCPM.from_pretrained(
            model=str(self._model_dir),
            inference_timesteps=self._inference_timesteps,
            max_num_batched_tokens=self._max_num_batched_tokens,
            max_num_seqs=self._max_num_seqs,
            gpu_memory_utilization=self._gpu_memory_utilization,
            enforce_eager=self._enforce_eager,
            devices=self._devices,
        )
        await server.wait_for_ready()

        try:
            model_info = await server.get_model_info()
            src_sr = int(model_info["output_sample_rate"])
            sem = asyncio.Semaphore(self._concurrency)
            from tqdm import tqdm  # type: ignore[import-untyped]

            tasks = {
                asyncio.create_task(
                    self._generate_one(server, sem, phrases, output_dir, sample_idx, src_sr)
                ): sample_idx
                for sample_idx in range(start_index, n_samples)
            }
            results: list[Path | None | Exception] = []
            with tqdm(
                total=max(0, n_samples - start_index),
                desc="Nano-vLLM VoxCPM clips",
                unit="clip",
                initial=0,
            ) as pbar:
                for task in asyncio.as_completed(tasks):
                    try:
                        results.append(await task)
                    except Exception as exc:
                        results.append(exc)
                    pbar.update(1)
        finally:
            await server.stop()

        generated: list[Path] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning("Nano-vLLM VoxCPM generate failed: %s", result)
                continue
            if result is not None:
                generated.append(result)

        logger.info("Generated %d clips in %s", len(generated), output_dir)
        return generated

    async def _generate_one(
        self,
        server: Any,
        sem: asyncio.Semaphore,
        phrases: list[str],
        output_dir: Path,
        sample_idx: int,
        src_sr: int,
    ) -> Path | None:
        import librosa
        import soundfile as sf  # type: ignore[import-untyped]

        async with sem:
            phrase = phrases[sample_idx % len(phrases)]
            prompt, cfg_v, temperature = diversification_triple_at_index(
                self._prompts,
                self._cfg_values,
                self._temperature_values,
                sample_idx,
            )
            text = f"({prompt}){phrase}"
            chunks: list[np.ndarray] = []
            async for chunk in server.generate(
                target_text=text,
                cfg_value=cfg_v,
                temperature=temperature,
            ):
                chunks.append(np.asarray(chunk, dtype=np.float32).flatten())

            if not chunks:
                logger.warning("Nano-vLLM VoxCPM returned empty audio at clip %d", sample_idx)
                return None

            audio = np.concatenate(chunks, axis=0)
            if audio.size == 0:
                logger.warning("Nano-vLLM VoxCPM returned empty audio at clip %d", sample_idx)
                return None

            if src_sr != TARGET_SAMPLE_RATE:
                audio = librosa.resample(
                    audio,
                    orig_sr=src_sr,
                    target_sr=TARGET_SAMPLE_RATE,
                )
            peak = float(np.max(np.abs(audio))) or 1.0
            audio_i16 = (audio * (32767.0 / peak)).astype(np.int16)

            out_path = output_dir / f"clip_{sample_idx:06d}.wav"
            sf.write(str(out_path), audio_i16, TARGET_SAMPLE_RATE)
            return out_path
