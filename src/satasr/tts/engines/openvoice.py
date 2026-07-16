"""OpenVoice (MyShell) TTS engine — voice conversion via tone-color transfer.

OpenVoice is not a text-to-speech model on its own: it is a *voice
conversion* model that reshapes the timbre of an existing utterance to match
a reference speaker's "tone color" embedding (design §4.3, family:
voice-conversion). ``_render`` therefore composes two stages internally,
kept private to this plugin so the rest of the pipeline only ever sees a
single ``TTSEngine``:

1. A base TTS pass (MeloTTS, the base speaker OpenVoice v2 ships with)
   produces the *content* — the words, spoken in a flat, un-cloned voice.
2. OpenVoice's ``ToneColorConverter`` reshapes that audio's timbre to match
   ``voice.reference_audio`` (§4.4: clone real recorded speakers).

OpenVoice is MIT-licensed (verify current terms on the MyShell repo before
production use). The ``melo`` and ``openvoice`` packages, plus their GPU
checkpoints, are a heavy optional dependency not installed in this
environment, so every third-party import stays strictly inside the render
path — module import (and therefore registry discovery) never touches them.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager

from satasr.core.audio import SAMPLE_RATE, AudioBuffer
from satasr.core.interfaces import VoiceReference
from satasr.tts.base import BaseTTSEngine
from satasr.tts.registry import TTS_ENGINES

#: Converter checkpoint directory, per the OpenVoice v2 release layout.
_CONVERTER_DIR = "checkpoints_v2/converter"
#: Precomputed source tone-color embedding for the base speaker below.
_BASE_SPEAKER_SE_PATH = "checkpoints_v2/base_speakers/ses/en-newest.pth"
#: MeloTTS speaker key / language used to generate the base content audio.
_BASE_SPEAKER_KEY = "en-newest"
_MELO_LANGUAGE = "EN"
#: Native output sample rate of MeloTTS / OpenVoice's converter.
_NATIVE_SAMPLE_RATE = 24_000


@TTS_ENGINES.register("openvoice")
class OpenVoiceTTSEngine(BaseTTSEngine):
    """MyShell OpenVoice: base-TTS content + tone-color voice conversion."""

    name = "openvoice"
    supports_cloning = True

    def _render(self, text: str, voice: VoiceReference) -> AudioBuffer:
        """Speak ``text`` with MeloTTS, then clone ``voice`` onto it.

        ``voice.reference_audio`` is guaranteed non-None here —
        ``BaseTTSEngine._check_voice`` enforces the cloning contract before
        ``_render`` is ever called.
        """
        import torch  # type: ignore

        device = "cuda" if torch.cuda.is_available() else "cpu"
        with (
            self._base_speech_path(text, device) as base_path,
            self._reference_wav_path(voice) as ref_path,
        ):
            wav = self._convert_tone_color(base_path, ref_path, device)
        return self._to_16k(wav)

    @staticmethod
    @contextmanager
    def _base_speech_path(text: str, device: str) -> Iterator[str]:
        """Render flat content audio for ``text`` with MeloTTS's base voice."""
        from melo.api import TTS as MeloTTS  # type: ignore

        melo = MeloTTS(language=_MELO_LANGUAGE, device=device)
        speaker_id = melo.hps.data.spk2id[_BASE_SPEAKER_KEY]
        with tempfile.NamedTemporaryFile(suffix=".wav") as base_file:
            melo.tts_to_file(text, speaker_id, base_file.name, quiet=True)
            yield base_file.name

    @staticmethod
    @contextmanager
    def _reference_wav_path(voice: VoiceReference) -> Iterator[str]:
        """Write the in-memory reference clip to a temp WAV OpenVoice can read."""
        import soundfile as sf  # type: ignore

        assert voice.reference_audio is not None  # enforced by base class
        with tempfile.NamedTemporaryFile(suffix=".wav") as ref_file:
            sf.write(
                ref_file.name,
                voice.reference_audio.samples,
                voice.reference_audio.sample_rate,
            )
            yield ref_file.name

    @staticmethod
    def _convert_tone_color(base_path: str, ref_path: str, device: str) -> object:
        """Transfer the reference speaker's tone color onto the base audio."""
        # soundfile/torch already imported (with ignore) elsewhere in this
        # module, so mypy only reports the missing-stub error once per file;
        # a second "# type: ignore" here would itself be flagged as unused.
        import soundfile as sf
        import torch
        from openvoice import se_extractor  # type: ignore
        from openvoice.api import ToneColorConverter  # type: ignore

        converter = ToneColorConverter(f"{_CONVERTER_DIR}/config.json", device=device)
        converter.load_ckpt(f"{_CONVERTER_DIR}/checkpoint.pth")
        target_se, _ = se_extractor.get_se(ref_path, converter, vad=True)
        source_se = torch.load(_BASE_SPEAKER_SE_PATH, map_location=device)

        with tempfile.NamedTemporaryFile(suffix=".wav") as out_file:
            converter.convert(
                audio_src_path=base_path,
                src_se=source_se,
                tgt_se=target_se,
                output_path=out_file.name,
            )
            wav, _ = sf.read(out_file.name, dtype="float32")
            return wav

    @staticmethod
    def _to_16k(wav: object) -> AudioBuffer:
        """Downmix/resample the converted waveform to the project's 16 kHz mono."""
        import numpy as np
        from scipy.signal import resample_poly  # type: ignore

        samples = np.asarray(wav, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=-1).astype(np.float32)
        resampled = resample_poly(samples, SAMPLE_RATE, _NATIVE_SAMPLE_RATE)
        clipped = np.clip(resampled, -1.0, 1.0).astype(np.float32)
        return AudioBuffer(clipped, SAMPLE_RATE)
