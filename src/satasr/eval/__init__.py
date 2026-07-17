"""Evaluation metrics for multi-talker, speaker-attributed ASR (design §6).

Each metric parses serialized output via ``satasr.format`` (single source of
truth) and scores predicted vs. reference utterances. Concrete metrics register
their modules here as they are added.
"""
