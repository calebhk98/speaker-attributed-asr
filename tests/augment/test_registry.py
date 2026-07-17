"""All built-in augmenters register themselves under their design names."""

from __future__ import annotations

from satasr.augment import AUGMENTERS


def test_expected_names_are_registered() -> None:
    available = AUGMENTERS.available()
    for name in ("gain", "white_noise", "musan", "reverb"):
        assert name in available
