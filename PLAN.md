# Audio Diarization Tool — Planning Document

## Goal

A standalone Python CLI tool that takes a long audio file (up to ~3 hours), transcribes it with WhisperX, groups speech by speaker, lets the user listen to example clips per speaker, name each one, and optionally merge speaker labels before producing a final transcript.

---

## How It Works — High Level

```
Audio file
    │
    ▼
WhisperX transcription + diarization
  (saved to <filename>.diarize_cache.json)
    │
    ▼
Segments grouped by speaker (SPEAKER_00, SPEAKER_01, …)
    │
    ▼
Interactive review loop
  • Play 5 random clips per speaker
  • Show transcript of each clip
  • User types a name (or skips / merges / replays)
    │
    ▼
Final output
  • <filename>_transcript.json  (timestamped, speaker-labelled segments)
```

---

## Confirmed Decisions

| Concern | Decision |
|---|---|
| Target hardware | Apple M4 Pro (Apple Silicon) |
| Torch device | `mps` (Metal Performance Shaders) — auto-detected, falls back to `cpu` |
| Model | `large-v3` (most accurate Whisper model, good for noisy/low-quality audio) |
| Diarization | `pyannote.audio` via whisperx |
| Audio I/O | `pydub` + `ffmpeg` (clip extraction) |
| Clip playback | `afplay` via `subprocess` (macOS built-in, zero dependencies, reliable) |
| Output | JSON only |
| Resume | Always — cache intermediate results after transcription |

---

## Key Technical Choices

### Apple Silicon (M4 Pro) Notes

- WhisperX detects MPS automatically; pass `--device mps` or let it auto-detect.
- `large-v3` on M4 Pro processes roughly **4–8× faster than real-time**, so a 3-hour file should finish in ~25–45 minutes.
- pyannote diarization also runs on MPS.
- No CUDA involved — avoid any CUDA-specific installs.

### Clip Playback — `afplay`

macOS ships with `afplay`, a command-line audio player that handles WAV, AIFF, MP3, M4A natively. We call it via `subprocess.run(["afplay", clip_path])`. No pip install needed, no cross-platform abstraction layer. If the tool ever needs to run on Windows, swap in `winsound` (WAV only) or `pygame.mixer`.

### Resume / Cache Strategy

After transcription + diarization completes (the slow step), results are written to:
```
<source_audio_dir>/<filename>.diarize_cache.json
```
On next run, if this file exists and `--force` is not passed, the pipeline step is skipped entirely and the review loop reloads from cache. The cache also stores any naming/merge decisions made so far, so an interrupted review session can resume mid-way.

### Diarization Accuracy for Noisy Audio

- `large-v3` + forced alignment helps with difficult audio.
- pyannote's diarization pipeline has a `min_speakers` / `max_speakers` option — we'll expose `--min-speakers` and `--max-speakers` flags so the user can constrain the count if they know how many people were in the meeting.
- Segments shorter than ~1 second are merged into adjacent segments to reduce noise in the diarization output.

---

## Phases

### Phase 1 — Transcription & Diarization

1. Load audio with `whisperx.load_audio()`.
2. Transcribe with `large-v3` on `mps`.
3. Align transcription for word-level timestamps.
4. Run pyannote diarization → assign speaker label to each segment.
5. Write full result to `<filename>.diarize_cache.json`.

Cache JSON shape:
```json
{
  "audio_path": "/path/to/file.m4a",
  "model": "large-v3",
  "segments": [
    {
      "start": 12.4,
      "end": 15.8,
      "speaker": "SPEAKER_00",
      "text": "Good morning everyone."
    }
  ],
  "speaker_names": {},
  "merges": []
}
```

### Phase 2 — Speaker Review (Interactive CLI)

For each detected speaker, in order of first appearance:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Speaker 1 of 4  •  SPEAKER_00  •  47 segments
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[1/5]  00:00:12 – 00:00:18
  "Good morning everyone, thanks for joining today."
  ▶ playing…

[2/5]  00:14:03 – 00:14:11
  "…the budget numbers don't account for the Q3 overspend…"
  ▶ playing…

… (3 more clips) …

  [n] Name this speaker
  [s] Skip  (label stays "Speaker 1")
  [m] Merge into another speaker
  [r] Replay all clips
  [q] Save & quit
> _
```

- Clips extracted to a temp dir as WAV files, played one at a time via `afplay`.
- After naming/merging, the cache is updated immediately.
- Merging prompts: "Merge into which speaker? (1–4)" — all segments are re-labelled in the cache.

### Phase 3 — Output

Writes `<filename>_transcript.json` next to the source audio:

```json
{
  "generated_at": "2026-05-17T10:30:00",
  "audio_file": "meeting.m4a",
  "duration_seconds": 10847,
  "speakers": ["John", "Sarah", "Speaker 3"],
  "segments": [
    {
      "start": 12.4,
      "end": 15.8,
      "speaker": "John",
      "text": "Good morning everyone."
    }
  ]
}
```

---

## File Structure

```
ppc-minutes/
├── PLAN.md
├── diarize.py               ← entry point + argparse
├── requirements.txt
└── src/
    ├── pipeline.py          ← transcription, alignment, diarization, cache R/W
    ├── segments.py          ← speaker grouping, merge logic, data model
    ├── clips.py             ← extract WAV clips from source audio via pydub
    ├── playback.py          ← afplay wrapper (+ Windows fallback stub)
    ├── review.py            ← interactive CLI review loop (rich)
    └── output.py            ← write final JSON transcript
```

Flat `src/` rather than nested packages — keeps imports simple for a single-script-style tool.

---

## Dependencies

**pip:**
```
whisperx          # transcription + alignment (pulls in torch, faster-whisper, etc.)
pyannote.audio    # speaker diarization
pydub             # audio clip extraction
rich              # styled terminal output + progress bars
```

**System (brew):**
```
ffmpeg            # required by pydub
```
`afplay` is pre-installed on macOS — no action needed.

**HuggingFace:**
- Accept terms for `pyannote/speaker-diarization-3.1` on huggingface.co
- Set `HF_TOKEN` env var, or pass `--hf-token hf_xxx` at runtime

---

## CLI

```bash
# Standard run
python diarize.py meeting.m4a --hf-token hf_xxx

# Resume after interruption (skips transcription if cache exists)
python diarize.py meeting.m4a

# Force re-transcription even if cache exists
python diarize.py meeting.m4a --force

# Constrain speaker count (improves diarization accuracy)
python diarize.py meeting.m4a --min-speakers 2 --max-speakers 5

# Override model (e.g. for a quick test run)
python diarize.py meeting.m4a --model medium
```

---

## Setup Instructions (for dad's Mac)

```bash
# 1. Install Homebrew if not present, then ffmpeg
brew install ffmpeg

# 2. Create a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Set HuggingFace token (one-time)
export HF_TOKEN=hf_your_token_here
# Or add to ~/.zshrc for persistence

# 5. Run
python diarize.py /path/to/meeting.m4a
```

---

## Build Order

1. `src/pipeline.py` — transcription + diarization + cache write/read
2. `src/segments.py` — data model, grouping, merge logic
3. `src/clips.py` — extract N random clips per speaker as WAV
4. `src/playback.py` — `afplay` wrapper
5. `src/review.py` — interactive loop (rich panels, keypress handling)
6. `src/output.py` — final JSON writer
7. `diarize.py` — wire it all together with argparse
8. End-to-end test on a short clip (~5 min), then a full-length file
