"""Voxtral TTS (Mistral) engine — vLLM-Omni-served preset-voice synthesis.

Voxtral-4B-TTS-2603 is Mistral's text-to-speech model (released March 2026,
weights on Hugging Face under CC BY-NC 4.0 — verify commercial terms before
production use). It ships 20 built-in preset voices (e.g. "casual_male") and
its model card's recommended local-inference path is *serving* it through
vLLM-Omni (>= 0.18.0), which exposes an OpenAI-compatible
``/v1/audio/speech`` endpoint — there is no first-party Python model class to
import directly, unlike most other engines in this package. Voxtral also
supports voice cloning from a reference clip, but that is only documented
through Mistral's hosted AI Studio console, not the local vLLM-Omni server,
so — per the issue's guidance — this engine defaults to
``supports_cloning = False`` and honors ``VoiceReference.preset`` only
(design §4.4). See the accompanying issue comment for the full list of
assumptions and unverified claims (no weights/GPU available here to confirm
any of this end to end).

The ``openai`` package (the client used to talk to a local vLLM-Omni server)
is a heavy optional dependency not installed in this environment, so the
import stays strictly inside :meth:`VoxtralTTSEngine._render` — module import
(and therefore registry discovery) never touches it.
"""

from __future__ import annotations

import io
import wave

import numpy as np

from satasr.core.audio import SAMPLE_RATE, AudioBuffer
from satasr.core.interfaces import VoiceReference
from satasr.tts.base import BaseTTSEngine
from satasr.tts.registry import TTS_ENGINES

#: Published checkpoint id served by vLLM-Omni (model card default).
_MODEL_ID = "mistralai/Voxtral-4B-TTS-2603"
#: Fallback preset used only if a caller passes an empty string (guard
#: clause; BaseTTSEngine already rejects ``preset is None`` before we get
#: here). Names one of the 20 shipped preset voices per the model card.
_DEFAULT_VOICE = "casual_male"
#: Local vLLM-Omni OpenAI-compatible server address (model card example).
#: ASSUMPTION: unverified — a real deployment likely configures this via env.
_DEFAULT_BASE_URL = "http://localhost:8000/v1"
#: 16-bit PCM full-scale divisor for decoding the WAV response into floats.
_INT16_FULL_SCALE = 32768.0


@TTS_ENGINES.register("voxtral")
class VoxtralTTSEngine(BaseTTSEngine):
    """Mistral Voxtral: preset-voice TTS served through vLLM-Omni."""

    name = "voxtral"
    supports_cloning = False

    def _render(self, text: str, voice: VoiceReference) -> AudioBuffer:
        """Synthesize ``text`` with Voxtral's preset ``voice.preset`` voice.

        Calls a locally-running vLLM-Omni server's OpenAI-compatible
        ``audio.speech`` endpoint, requests WAV output, and decodes +
        resamples the result down to the project's 16 kHz mono float32
        convention.
        """
        from openai import OpenAI  # type: ignore

        client = OpenAI(base_url=_DEFAULT_BASE_URL, api_key="not-needed")
        preset = voice.preset or _DEFAULT_VOICE
        response = client.audio.speech.create(
            model=_MODEL_ID, voice=preset, input=text, response_format="wav"
        )
        return self._to_16k(response.read())

    @staticmethod
    def _to_16k(wav_bytes: bytes) -> AudioBuffer:
        """Decode a WAV response and resample/downmix it to 16 kHz mono.

        Uses the standard library's ``wave`` module rather than soundfile,
        and plain numpy linear interpolation rather than scipy, so this
        helper (and its fast test) needs no extra dependency beyond the
        project's existing numpy requirement.
        """
        samples, native_rate = _decode_wav(wav_bytes)
        resampled = _linear_resample(samples, native_rate, SAMPLE_RATE)
        clipped = np.clip(resampled, -1.0, 1.0).astype(np.float32)
        return AudioBuffer(clipped, SAMPLE_RATE)


def _decode_wav(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Parse a 16-bit PCM WAV container into float32 samples + its rate."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        native_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        raw = wav_file.readframes(wav_file.getnframes())
    pcm16 = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        pcm16 = pcm16.reshape(-1, channels).mean(axis=1)
    samples = pcm16.astype(np.float32) / _INT16_FULL_SCALE
    return samples, native_rate


def _linear_resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample a 1-D float32 array from ``src_rate`` to ``dst_rate`` Hz."""
    if samples.size == 0 or src_rate == dst_rate:
        return samples
    duration_s = samples.shape[0] / src_rate
    dst_count = max(1, round(duration_s * dst_rate))
    src_times = np.arange(samples.shape[0], dtype=np.float64) / src_rate
    dst_times = np.arange(dst_count, dtype=np.float64) / dst_rate
    interpolated: np.ndarray = np.interp(dst_times, src_times, samples)
    return interpolated.astype(np.float32)
