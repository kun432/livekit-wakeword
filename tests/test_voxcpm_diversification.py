"""VoxCPM diversification index must match ``itertools.product`` (resume-safe)."""

from __future__ import annotations

import itertools as it

from livekit.wakeword.data.tts.voxcpm_backend import diversification_triple_at_index
from livekit.wakeword.data.tts.voxcpm_nanovllm_backend import (
    diversification_triple_at_index as nanovllm_diversification_triple_at_index,
)


def test_diversification_triple_matches_product_order() -> None:
    prompts = ["P0", "P1", "P2"]
    cfg_values = [1.5, 2.0]
    timesteps = [8, 10, 12]
    flat = list(it.product(prompts, cfg_values, timesteps))
    n = len(prompts) * len(cfg_values) * len(timesteps)
    for i in range(n * 3 + 7):
        assert diversification_triple_at_index(prompts, cfg_values, timesteps, i) == flat[i % n]


def test_nanovllm_diversification_triple_matches_product_order() -> None:
    prompts = ["P0", "P1", "P2"]
    cfg_values = [1.5, 2.0]
    temperatures = [0.8, 1.0, 1.2]
    flat = list(it.product(prompts, cfg_values, temperatures))
    n = len(prompts) * len(cfg_values) * len(temperatures)
    for i in range(n * 3 + 7):
        assert nanovllm_diversification_triple_at_index(
            prompts, cfg_values, temperatures, i
        ) == flat[i % n]
