"""The one canonical serialized-output (SOT) transcript format (design §4.7).

A :class:`~satasr.core.models.MixedClip` carries its ground truth as a tuple of
:class:`~satasr.core.models.PlacedUtterance`. This module is the single place
that turns that tuple into the speaker-tagged, timestamped text string the
model is trained to produce, and back. Every other module — the dataset
writer, model decoding, eval — imports these functions rather than growing its
own copy (CLAUDE.md rule 4).

Line grammar, one utterance per line, joined by ``"\\n"``::

    <SPEAKER_ID> START->END: TEXT

``START``/``END`` are ``str(float)`` seconds — Python's float formatting is the
shortest decimal that reads back to the exact same float, so a round trip
never loses precision (no fixed decimal-place rounding). ``TEXT`` is the
utterance text verbatim, including a trailing mid-word truncation marker
(``MixingConfig.truncation_marker``, §4.7 step 5) when the mixer cut the
speaker off inside a word.

Truncation contract (open Q4): ``PlacedUtterance.truncated`` is *not* written
to the string separately. On parse it is reconstructed from whether ``TEXT``
ends with the marker as its own token. A whole-word truncation (the mixer cut
exactly on a word boundary, so no marker was appended) is indistinguishable in
the trained-on text from an utterance that simply ended there — by design,
since the model only ever sees the text and timestamps, never the mixer's
internal bookkeeping. Callers that need the original bookkeeping value must
keep the :class:`~satasr.core.models.MixedClip` itself; the string format only
promises to preserve what the model is trained to reproduce.

``PlacedUtterance.words`` (per-word timestamps) is likewise not part of the
string: the SOT target is utterance-level, per the design's example output
(§1). Parsing always yields ``words=()``.
"""

from __future__ import annotations

from collections.abc import Iterable

from satasr.core.config import MixingConfig
from satasr.core.models import MixedClip, PlacedUtterance

# Single source of truth for the default marker: mirrors MixingConfig so this
# module never hardcodes its own copy of "-" (CLAUDE.md rule 4).
DEFAULT_TRUNCATION_MARKER: str = MixingConfig().truncation_marker

_HEADER_SEP = "> "
_TIME_SEP = "->"
_TEXT_SEP = ": "


def serialize_clip(clip: MixedClip) -> str:
    """Serialize a :class:`MixedClip`'s ground truth to its SOT string."""
    return serialize_utterances(clip.utterances)


def serialize_utterances(utterances: Iterable[PlacedUtterance]) -> str:
    """Serialize placed utterances, ordered deterministically (start, end,
    speaker_id) so overlapping speakers always come out in the same order."""
    ordered = sorted(utterances, key=_sort_key)
    return "\n".join(_format_line(u) for u in ordered)


def parse_utterances(
    sot: str, truncation_marker: str = DEFAULT_TRUNCATION_MARKER
) -> tuple[PlacedUtterance, ...]:
    """Parse a SOT string back into :class:`PlacedUtterance` objects."""
    return tuple(_parse_line(line, truncation_marker) for line in sot.splitlines())


def _sort_key(u: PlacedUtterance) -> tuple[float, float, str]:
    return (u.start_s, u.end_s, u.speaker_id)


def _format_line(u: PlacedUtterance) -> str:
    _validate_field(u.speaker_id, "speaker_id")
    _validate_field(u.text, "text")
    if u.start_s < 0 or u.end_s < 0:
        raise ValueError(f"timestamps must be non-negative, got {u.start_s}-{u.end_s}")
    times = f"{u.start_s}{_TIME_SEP}{u.end_s}"
    return f"<{u.speaker_id}{_HEADER_SEP}{times}{_TEXT_SEP}{u.text}"


def _validate_field(value: str, name: str) -> None:
    """Guard the characters the line grammar relies on being unambiguous."""
    if "\n" in value:
        raise ValueError(f"{name} must not contain a newline: {value!r}")
    if name == "speaker_id" and ">" in value:
        raise ValueError(f"speaker_id must not contain '>': {value!r}")


def _parse_line(line: str, marker: str) -> PlacedUtterance:
    if not line.startswith("<"):
        raise ValueError(f"malformed SOT line (missing speaker tag): {line!r}")
    speaker_id, found, rest = line[1:].partition(_HEADER_SEP)
    if not found:
        raise ValueError(f"malformed SOT line (missing '{_HEADER_SEP}'): {line!r}")

    start_str, end_str, text = _split_timestamps(rest, line)
    truncated = _ends_with_marker_token(text, marker)
    return PlacedUtterance(
        speaker_id, float(start_str), float(end_str), text, (), truncated
    )


def _split_timestamps(rest: str, line: str) -> tuple[str, str, str]:
    start_str, found, rest2 = rest.partition(_TIME_SEP)
    if not found:
        raise ValueError(f"malformed SOT line (missing '{_TIME_SEP}'): {line!r}")
    end_str, found, text = rest2.partition(_TEXT_SEP)
    if not found:
        raise ValueError(f"malformed SOT line (missing '{_TEXT_SEP}'): {line!r}")
    return start_str, end_str, text


def _ends_with_marker_token(text: str, marker: str) -> bool:
    """Was the trailing token of ``text`` the mid-word truncation marker?"""
    if text == marker:
        return True
    return text.endswith(" " + marker)
