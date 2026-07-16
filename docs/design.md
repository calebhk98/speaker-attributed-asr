# Multi-Talker Overlapping Speech Recognition + Diarization
## Project Design Document

**Purpose of this document:** hand-off/reference doc for a project that may be
picked up in 2 days or 2 months. Assumes no prior context. Read top to bottom
once, then use as reference. This is the single source of truth for *what* the
project is and *why*; the code implements it and cites section numbers (e.g.
`§4.7`) in comments.

---

## 1. The Goal

Build a system that takes in audio with **1-4 people talking, including
overlapping speech**, and outputs a per-speaker, timestamped transcript. Example
target output:

```
Speaker A: (0:00-0:15) "So, when we go to Africa we need to get some protection from lions"
Speaker B: (0:05-0:12) "Wait, we are going to Africa?"
Speaker C: (0:00-0:15) "While those stupid heads are talking about their trip, Bethany, how was that Salon?"
```

This is **not** standard "speech-to-text" and **not** standard "speaker
diarization" run separately. It's a combined, harder problem with an actual name
in research: **multi-talker ASR with serialized output** (speaker tags +
timestamps produced jointly, handling full overlap).

### Explicit scope decisions (read this before changing anything)

- **Simultaneous overlapping speakers: a configurable parameter, not a fixed
  number.** There's a physical ceiling somewhere, but the exact cutoff is a
  tunable design choice, not hardcoded logic. Default target: **up to 6
  simultaneous speakers**, but the pipeline should make this trivial to change.
- **Total speakers present in a clip vs. how many are simultaneously speaking
  are two separate counts, and both need to be tracked.**
- **Total distinct speaker identities across a full session: up to ~20 is fine
  and realistic.** This is a tracking problem (who's who over time), not an
  acoustic separation problem. This is the correct way to interpret "handle 20
  speakers."
- **No fixed target recording environment.** Cast a wide net with augmentation
  (varied noise, varied reverb/room simulation) rather than optimizing for one
  setting.
- 20 *simultaneous* speakers at once is understood to be unachievable and is
  explicitly out of scope.

---

## 2. Why This Is Hard (background/rationale)

Two failure modes if you treat this as diarization + transcription bolted
together:

1. Diarization tools (pyannote, NeMo Sortformer) tell you *who spoke when* but
   don't transcribe.
2. ASR tools (Whisper) transcribe but fall apart on overlapping speech and don't
   track speaker identity.

The real research area combining both is **target-speaker / speaker-attributed
ASR with Serialized Output Training (SOT)**.

### DiCoW / SA-DiCoW (Diarization-Conditioned Whisper)

- Built by BUT Speech@FIT (Brno University of Technology) + collaborators.
- **Base architecture:** Whisper-large-v3-turbo as the backbone.
- **What's added:** Frame-Level Diarization-Dependent Transformations (FDDT) —
  lightweight per-layer affine transforms conditioning the Whisper encoder on
  diarization info (silence / target / non-target / overlap). Initialized as
  identity transforms, so the added parameter count is small.
- **SA-DiCoW variant** adds serialized output training so the model directly
  outputs speaker-tagged, timestamped transcripts for overlapping speech.
- Repo: `github.com/BUTSpeechFIT/DiCoW` (HF: `BUT-FIT/DiCoW_v3_MLC`).
- **~918M trainable parameters** — the realistic size floor for this task.

### Why not train from scratch?

Not necessary or advisable. Fine-tune from the existing DiCoW/SA-DiCoW
checkpoint. We're only teaching it the narrow additional skill of staying locked
onto individual speakers under overlap.

---

## 3. Hardware

- **Dual RTX 3090** (24GB VRAM each, 48GB total)
- **Ryzen 9** CPU

Good for: fine-tuning a ~918M model (full or LoRA) split across both GPUs;
running multiple TTS generation jobs. NOT for pretraining a foundation model
from scratch.

**TTS throughput gotcha:** vendor RTF numbers are batch-size-1, latency-oriented.
This project is offline batch generation — real throughput is much higher once
batched. **Benchmark your actual batched setup on ~10 hours before committing to
a full-scale timeline.** CPU (Ryzen 9) is usable for lightweight non-AR TTS
(e.g. Piper) as a bulk worker, not for AR/cloning models (XTTS, F5-TTS).

---

## 4. Data Strategy

### 4.1 Overall two-phase training plan

*(Training Phase 1/2 ≠ Data Stage 1/2. Two different pipelines.)*

1. **Training Phase 1 (full fine-tune):** train + test on synthetic TTS audio.
2. **Training Phase 2 (LoRA fine-tune):** adapt the Phase 1 model on **real
   human speech**, tested separately.

Order validated by "Mind the Gap" (§7): synthetic pretrain → real fine-tune beat
both joint training and real-only.

**Train/val/test discipline (both phases, separately):** train = learn from;
validation = tune/pick checkpoints (repeatable); test = touched exactly **once**
at the very end.

### 4.2 Target volume

**10,000 hours** of synthetic audio. ~1,000 hours per engine as a base
allocation, then scale engines up/down once results come in. Build the pipeline
to add/remove/reweight engines dynamically — do not hardcode a fixed list.

### 4.3 Data Stage 1: TTS generation — models to use

Grouped by **architecture family** (matters more than company — shared
architecture ⇒ shared artifacts). Prioritize covering families over stacking one.

- **Autoregressive codec-token:** Bark, XTTS v2 (CPML, non-commercial), Fish
  Speech / S1-S2, Chatterbox (MIT), Orpheus, Qwen3-TTS, GPT-SoVITS, VoxCPM2,
  MetaVoice-1B.
- **Flow-matching / diffusion, non-AR:** F5-TTS (CC-BY-NC 4.0), StyleTTS2.
- **Original flow-based end-to-end:** VITS, Piper (MIT weights / GPL-3.0 fork).
- **Lightweight decoder-only:** Kokoro (Apache 2.0, 82M, fast, **no cloning** —
  54 preset voices).
- **Dialogue-native (multi-speaker conversation directly):** Dia / Dia2, Sesame
  CSM, VibeVoice, Moshi.
- **Big-lab backed:** Voxtral (Mistral), NVIDIA NeMo TTS (FastPitch, RAD-TTS,
  Mixer-TTS).
- **Voice conversion framing:** OpenVoice, MeloTTS.

**Allocation strategy (recommend, not finalized):** heavier (2,000-3,000h) to
best-match engines (a cloned-voice codec-token model + a dialogue-native model),
lighter (a few hundred h) to the rest.

### 4.4 Voice diversity trick

For cloning-capable engines (XTTS v2, F5-TTS, Chatterbox, …) **clone voices from
real recorded speakers** (LibriVox / Common Voice) rather than each engine's
built-in bank — text diversity from TTS, acoustics closer to real voices.

### 4.5 Source text

Wikipedia / LLM text / social posts: clean, good diversity. Movie scripts:
**copyright flag** even as TTS input — decide before scaling. **Known gap:**
synthetic LLM text won't reliably contain natural disfluency — assume it won't,
don't rely on it.

### 4.6 Data Stage 2: Mixing script (splicing / overlap)

- Per-sentence granularity: each TTS clip is one sentence, one speaker.
- Overlap: mix 2+ speakers' clips starting at a random point.
- **Truncation logic** (mimic interruption): leave **some sentences uncut**;
  **randomly truncate others** — cut the interrupted speaker ~0.01-3s after the
  interrupter starts.
- **Simultaneous count: configurable, default up to 6.** An easily adjustable
  parameter, not hardcoded.
- **Track two separate counts per clip:** (1) total distinct speakers present,
  (2) max ever simultaneous. Balance both distributions deliberately (reference:
  Sortformer ~1/3/6/10 across 1-4 simultaneous).

### 4.7 Ground-truth transcript for truncated clips

**Primary: forced alignment BEFORE cutting, on the clean uncut audio.**

1. Generate full sentence audio (exact text is known).
2. Align the **uncut** clip → word-level start/end timestamps.
3. **Use CTC-segmentation (wav2vec2-CTC), not classic HMM forced alignment
   (MFA)** for the synthetic stage — more robust to TTS pacing.
4. Cut point = deterministic lookup: keep words before the cut, drop the rest.
5. **Mid-word cut:** drop the partial word, optionally mark with `-`.

**Fallbacks if alignment fails on an engine:** ASR-on-spliced-audio comparison;
word-by-word checker with binary search. Default to CTC-segmentation.

### 4.8 Real-audio datasets (Phase 2 LoRA + realistic turn-taking)

- **LibriSpeech / LibriSpeechMix / LibriCSS** — public domain (LibriVox).
  Overlap is artificially constructed but acoustics/conditions are real.
- **AMI Meeting Corpus** — genuine spontaneous overlap; **verify license before
  committing.**
- **Common Voice** (CC0) — extra speaker diversity for cloning sources.

### 4.9 Noise / acoustic augmentation

- **MUSAN** for background noise (pyannote uses it too).
- **Room impulse response (reverb)** — matters **substantially more for
  diarization than ASR** (~11-pt DER improvement far-field in one study). Don't
  skip it.
- No fixed target environment ⇒ broad/varied coverage (noise types, SNR, room
  sizes/mic distances).

---

## 5. Training Plan

1. **Phase 1 — full fine-tune** from a DiCoW/SA-DiCoW checkpoint on the full
   synthetic dataset; test on held-out synthetic.
2. **Phase 2 — LoRA** on real speech (LibriSpeechMix/LibriCSS + AMI); test on
   held-out real; re-check against the Phase 1 synthetic test set for regression.
3. Training compute for ~918M over this much data is multi-day-to-multi-week on
   dual 3090s — size it explicitly.

---

## 6. Evaluation

- **Diarization:** DER (lower better). Sortformer ~14.76% on DIHARD3; heavy
  overlap pushes even the best to 27%+.
- **ASR:** multi-speaker WER — **tcpWER / cpWER** (time-constrained /
  concatenated minimum-permutation WER), the DiCoW/Sortformer standard.

---

## 7. Key Research Reference

**"Mind the Gap: Impact of Synthetic Conversational Data on Multi-Talker ASR and
Speaker Diarization"** (Polok, Medennikov, Černocký, Watanabe, Burget, Cornell —
BUT / CMU / NVIDIA, 2026). arXiv: 2605.15442.

- Two-stage (synthetic → real) beat joint and real-only, for both tasks.
- Boosting overlap intensity **helps ASR but hurts diarization** — don't
  maximize overlap blindly.
- Diverse source domains beat a single source.
- Noise+reverb critical for diarization (~4-pt DER), marginal for ASR.
- Scale reference: DiCoW 500-2,500h simulated; Sortformer ~2,000h; real sets
  ~300h. Our 10k target is generous.
- **FastMSS** simulator: 1,000h annotated in <5 min on 32 CPU workers — mixing
  is not the bottleneck, TTS generation is.

---

## 8. Tools/Models Quick Reference

See §4.3 for the full grouped TTS list. Core architecture: DiCoW / SA-DiCoW
(Whisper-large-v3-turbo + FDDT, ~918M). Diarization alts: NeMo Sortformer,
pyannote.audio. Simulators: FastMSS, NeMo. Alignment: CTC-segmentation
(synthetic), MFA (real). Noise: MUSAN. Real data: LibriSpeech/Mix/CSS, AMI,
Common Voice.

---

## 9. Open Questions

1. Final per-engine TTS hour allocation — flat vs. weighted (§4.3)?
2. Movie-script source text in or out given copyright (§4.5)?
3. AMI license checked (§4.8)?
4. Mid-word truncation marker (`-`) implemented and handled by the tokenizer
   (§4.7)?
5. A hand-verified real test set set aside, separate from AMI/LibriSpeech splits?
