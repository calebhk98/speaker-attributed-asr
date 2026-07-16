"""Dia (Nari Labs) TTS engine — dialogue-native codec-token synthesis.

Dia is a 1.6B-parameter text-to-dialogue model from Nari Labs
(``nari-labs/dia`` on GitHub, checkpoint ``nari-labs/Dia-1.6B-0626`` on the
Hugging Face Hub). Unlike single-speaker engines it is trained to generate a
whole multi-turn conversation directly from a transcript in which each turn is
prefixed with a speaker tag such as ``"[S1]"`` or ``"[S2]"``.

This pipeline calls TTS engines one sentence and one speaker at a time (see
``BaseTTSEngine.synthesize``), so this engine renders a single-turn "dialogue"
consisting of exactly one tagged utterance per call. Dia also supports
zero-shot voice cloning by conditioning on a reference ``audio_prompt``, but
per issue #16 this engine is registered preset-only (``supports_cloning =
False``): ``VoiceReference.preset`` supplies the speaker tag (e.g. ``"S1"``),
and no reference audio is used. A future ``DialogueSource`` (design §4.6)
could instead call Dia with a full multi-turn script to get its native
turn-taking; that is out of scope here.

ASSUMPTIONS (unverified in this environment — no GPU/weights available; see
the accompanying issue comment):
- The optional dependency installs as the ``dia`` package (``pip install
  git+https://github.com/nari-labs/dia.git``), exposing ``dia.model.Dia``.
- ``Dia.from_pretrained(checkpoint, compute_dtype=...)`` loads the model and
  ``model.generate(text)`` returns a 1-D numpy waveform, per the project's
  README usage example (``sf.write("simple.mp3", output, 44100)``).
- Dia's native output sample rate is 44.1 kHz (the Descript Audio Codec's
  rate), so this engine downsamples to the project's 16 kHz standard.

The ``dia`` package and its torch weights are a heavy, GPU-hungry optional
dependency that is not installed in this environment, so every import lives
strictly inside :meth:`DiaTTSEngine._render` — module import (and therefore
registry discovery) never touches it.
"""

from __future__ import annotations

import numpy as np

from satasr.core.audio import SAMPLE_RATE, AudioBuffer
from satasr.core.interfaces import VoiceReference
from satasr.tts.base import BaseTTSEngine
from satasr.tts.registry import TTS_ENGINES

#: Dia's native codec sample rate (Descript Audio Codec), per the model README.
_DIA_NATIVE_SAMPLE_RATE = 44_100
#: Published checkpoint used for inference (Hugging Face Hub model card).
_MODEL_ID = "nari-labs/Dia-1.6B-0626"
#: Speaker tag used when a caller supplies no distinguishing preset content.
_DEFAULT_SPEAKER_TAG = "S1"


@TTS_ENGINES.register("dia")
class DiaTTSEngine(BaseTTSEngine):
    """Nari Labs Dia: preset-tagged, dialogue-native codec-token TTS."""

    name = "dia"
    supports_cloning = False

    def _render(self, text: str, voice: VoiceReference) -> AudioBuffer:
        """Synthesize one speaker-tagged utterance with Dia.

        Returns 16 kHz mono float32 audio, resampled down from Dia's native
        44.1 kHz codec output.
        """
        from dia.model import Dia  # type: ignore

        model = Dia.from_pretrained(_MODEL_ID, compute_dtype="float16")
        prompt = self._build_prompt(text, voice)
        raw = model.generate(prompt, use_torch_compile=False, verbose=False)
        return self._to_16k(raw)

    @staticmethod
    def _build_prompt(text: str, voice: VoiceReference) -> str:
        """Prefix ``text`` with the speaker tag Dia expects for one turn.

        ``voice.preset`` names the tag (e.g. ``"S1"``); bare names are wrapped
        in brackets, and an already-bracketed preset is used as-is.
        """
        tag = voice.preset or _DEFAULT_SPEAKER_TAG
        bracketed = tag if tag.startswith("[") else f"[{tag}]"
        return f"{bracketed} {text}"

    @staticmethod
    def _to_16k(raw: object) -> AudioBuffer:
        """Downmix/resample Dia's raw waveform to the project's 16 kHz mono.

        Uses plain numpy linear interpolation rather than scipy so this helper
        (and its fast test) needs no extra dependency beyond the project's
        existing numpy requirement.
        """
        samples = np.asarray(raw, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=-1).astype(np.float32)
        resampled = _linear_resample(samples, _DIA_NATIVE_SAMPLE_RATE, SAMPLE_RATE)
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
