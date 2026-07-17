"""Moshi (Kyutai) TTS engine — full-duplex, codec-token dialogue synthesis.

Moshi is Kyutai's full-duplex speech-text foundation model: a Mimi neural
audio codec paired with an RQ-Transformer language model that autoregressively
predicts audio codec tokens conditioned on text. Kyutai ships two checkpoints
that differ only by conditioning voice — "moshika" (female) and "moshiko"
(male) — with no user-supplied voice cloning, so this engine is preset-only
and honors ``VoiceReference.preset`` as the checkpoint name (design §4.4).

The ``moshi`` package, its ``torch``/``huggingface_hub`` dependencies, and the
multi-gigabyte Mimi codec + LM weights are a heavy, GPU-hungry optional
dependency not installed in this environment, so every third-party import is
kept strictly inside the render path — module import (and therefore registry
discovery) never touches them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from satasr.core.audio import SAMPLE_RATE, AudioBuffer
from satasr.core.interfaces import VoiceReference
from satasr.tts.base import BaseTTSEngine
from satasr.tts.registry import TTS_ENGINES

#: Mimi's native codec sample rate, per the Kyutai Moshi model card.
_MOSHI_NATIVE_SAMPLE_RATE = 24_000
#: Fallback preset used only if a caller passes an empty string (guard
#: clause; BaseTTSEngine already rejects ``preset is None`` before we get
#: here). Names one of Kyutai's two shipped Moshi checkpoints.
_DEFAULT_PRESET = "moshika"
#: Hugging Face repos for Kyutai's two preset voice checkpoints.
_PRESET_REPOS = {
    "moshika": "kyutai/moshika-pytorch-bf16",
    "moshiko": "kyutai/moshiko-pytorch-bf16",
}


@TTS_ENGINES.register("moshi")
class MoshiTTSEngine(BaseTTSEngine):
    """Kyutai Moshi: preset-voice, full-duplex codec-token TTS."""

    name = "moshi"
    supports_cloning = False
    #: Both of Kyutai's preset voice checkpoints (see ``_PRESET_REPOS``
    #: above); either may be fetched depending on ``voice.preset``.
    model_ids: tuple[str, ...] = tuple(_PRESET_REPOS.values())

    def _render(self, text: str, voice: VoiceReference) -> AudioBuffer:
        """Synthesize ``text`` with Moshi's preset ``voice.preset`` checkpoint.

        Loads the named Mimi codec + Moshi LM checkpoint, autoregressively
        generates audio codec tokens conditioned on the text, decodes them to
        a waveform at Mimi's native 24 kHz, and resamples down to the
        project's 16 kHz mono float32 convention.
        """
        import torch  # type: ignore
        from huggingface_hub import hf_hub_download  # type: ignore
        from moshi.models import loaders  # type: ignore

        device = "cuda" if torch.cuda.is_available() else "cpu"
        repo = _PRESET_REPOS.get(voice.preset or "", _PRESET_REPOS[_DEFAULT_PRESET])
        mimi_path = hf_hub_download(repo, loaders.MIMI_NAME)
        moshi_path = hf_hub_download(repo, loaders.MOSHI_NAME)
        mimi = loaders.get_mimi(mimi_path, device=device)
        lm = loaders.get_moshi_lm(moshi_path, device=device)
        raw = self._generate_waveform(mimi, lm, text, device)
        return self._to_16k(raw)

    @staticmethod
    def _generate_waveform(mimi: Any, lm: Any, text: str, device: str) -> np.ndarray:
        """Autoregressively generate codec tokens for ``text`` and decode them.

        Kept as a thin, isolated seam so the heavy generation loop is easy to
        stub in integration tests once real weights are available.
        """
        import torch  # already type: ignore'd on its first import above
        from moshi.models import LMGen  # already type: ignore'd above (loaders)

        text_tokens = lm.text_tokenizer.encode(text)
        prefix = torch.tensor([text_tokens], device=device)
        lm_gen = LMGen(lm, temp=0.8, temp_text=0.7)
        with torch.no_grad():
            codes = lm_gen.generate(prefix)
            waveform = mimi.decode(codes)
        result: np.ndarray = waveform.squeeze().cpu().numpy().astype(np.float32)
        return result

    @staticmethod
    def _to_16k(raw: np.ndarray) -> AudioBuffer:
        """Downmix/resample Moshi's raw waveform to the project's 16 kHz mono.

        Uses plain numpy linear interpolation rather than scipy so this
        helper (and its fast test) needs no extra dependency beyond the
        project's existing numpy requirement.
        """
        samples = np.asarray(raw, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=0).astype(np.float32)
        resampled = _linear_resample(samples, _MOSHI_NATIVE_SAMPLE_RATE, SAMPLE_RATE)
        clipped = np.clip(resampled, -1.0, 1.0).astype(np.float32)
        return AudioBuffer(clipped, SAMPLE_RATE)


def _linear_resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample a 1-D float32 array from ``src_rate`` to ``dst_rate`` Hz."""
    if samples.size == 0:
        return samples
    duration_s = samples.shape[0] / src_rate
    dst_count = max(1, round(duration_s * dst_rate))
    src_times = np.arange(samples.shape[0], dtype=np.float64) / src_rate
    dst_times = np.arange(dst_count, dtype=np.float64) / dst_rate
    interpolated: np.ndarray = np.interp(dst_times, src_times, samples)
    return interpolated.astype(np.float32)
