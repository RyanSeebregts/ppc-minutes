import json
from pathlib import Path

CACHE_SUFFIX = ".diarize_cache.json"


def get_cache_path(audio_path: str) -> Path:
    p = Path(audio_path)
    return p.parent / (p.stem + CACHE_SUFFIX)


def load_cache(audio_path: str) -> dict | None:
    cache_path = get_cache_path(audio_path)
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    return None


def save_cache(cache: dict, audio_path: str) -> None:
    cache_path = get_cache_path(audio_path)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def _get_devices(override: str | None) -> tuple[str, str]:
    """Return (transcription_device, torch_device).

    faster-whisper (ctranslate2) doesn't support MPS, so we always use CPU
    for the transcription step on Apple Silicon. PyTorch alignment and
    diarization can use MPS.
    """
    import torch

    if override:
        torch_dev = override
        trans_dev = "cpu" if override == "mps" else override
        return trans_dev, torch_dev

    if torch.backends.mps.is_available():
        return "cpu", "mps"
    if torch.cuda.is_available():
        return "cuda", "cuda"
    return "cpu", "cpu"


_SAMPLE_RATE = 16_000
_MAX_CHUNK_S = 20 * 60   # 20-minute chunks keep STFT intermediate under ~400 MB
_OVERLAP_S = 30          # overlap so sentences straddling a boundary are heard fully


def _transcribe_chunk(model, audio_chunk, offset_s: float, language: str | None = None) -> tuple[list, str]:
    from rich.progress import (
        BarColumn,
        Progress,
        TaskProgressColumn,
        TextColumn,
        TimeRemainingColumn,
    )

    segments_gen, info = model.model.transcribe(
        audio_chunk,
        task="transcribe",
        language=language,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    segments: list[dict] = []
    with Progress(
        TextColumn("[bold cyan]Transcribing"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task_id = progress.add_task("", total=info.duration)
        for seg in segments_gen:
            segments.append({
                "start": seg.start + offset_s,
                "end": seg.end + offset_s,
                "text": seg.text,
                "words": [
                    {
                        "word": w.word,
                        "start": w.start + offset_s,
                        "end": w.end + offset_s,
                        "score": w.probability,
                    }
                    for w in (seg.words or [])
                ],
            })
            progress.update(task_id, completed=min(seg.end, info.duration))

    return segments, info.language


def _transcribe_with_progress(model, audio, trans_device: str, language: str | None = None) -> tuple[list, str]:
    total_samples = len(audio)
    max_samples = _MAX_CHUNK_S * _SAMPLE_RATE
    overlap_samples = _OVERLAP_S * _SAMPLE_RATE

    if total_samples <= max_samples:
        return _transcribe_chunk(model, audio, 0.0, language)

    n_chunks = (total_samples + max_samples - 1) // max_samples
    all_segments: list[dict] = []
    detected_language: str = language or "en"
    pos = 0

    for i in range(n_chunks):
        boundary = min(pos + max_samples, total_samples)
        chunk_end = min(boundary + overlap_samples, total_samples)
        offset_s = pos / _SAMPLE_RATE

        print(f"  chunk {i + 1}/{n_chunks} ({pos // _SAMPLE_RATE}s – {boundary // _SAMPLE_RATE}s)")
        segments, lang = _transcribe_chunk(model, audio[pos:chunk_end], offset_s, language)

        if boundary < total_samples:
            cutoff_s = boundary / _SAMPLE_RATE
            segments = [s for s in segments if s["start"] < cutoff_s]

        if i == 0 and language is None:
            detected_language = lang
        all_segments.extend(segments)
        pos = boundary

    return all_segments, detected_language


def transcribe(
    audio_path: str,
    model_name: str = "large-v3",
    hf_token: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    device_override: str | None = None,
    language: str | None = None,
) -> dict:
    import whisperx
    from rich.console import Console
    from rich.status import Status

    console = Console()
    trans_device, torch_device = _get_devices(device_override)
    if trans_device == "cuda":
        compute_type = "float16"
    elif trans_device == "cpu":
        compute_type = "int8_float32"
    else:
        compute_type = "int8"

    console.print(f"[dim]Transcription device : {trans_device} ({compute_type})[/dim]")
    console.print(f"[dim]Alignment/diarization: {torch_device}[/dim]")

    # --- Transcription ---
    console.print(f"\n[bold]Loading Whisper model[/bold] [cyan]{model_name}[/cyan]…")
    model = whisperx.load_model(model_name, trans_device, compute_type=compute_type)

    audio = whisperx.load_audio(audio_path)
    if language:
        console.print(f"[dim]Language: {language} (specified)[/dim]")
    segments, language = _transcribe_with_progress(model, audio, trans_device, language)
    console.print(f"[green]✓[/green] Language: [cyan]{language}[/cyan]")

    # --- Alignment ---
    with Status("[bold]Aligning transcription…[/bold]", console=console):
        model_a, metadata = whisperx.load_align_model(
            language_code=language, device=torch_device
        )
        result = whisperx.align(
            segments, model_a, metadata, audio, torch_device,
            return_char_alignments=False,
        )
    console.print("[green]✓[/green] Alignment done.")

    # --- Diarization ---
    diarize_kwargs: dict = {}
    if min_speakers is not None:
        diarize_kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        diarize_kwargs["max_speakers"] = max_speakers

    with Status("[bold]Diarizing speakers…[/bold]", console=console):
        diarize_model = whisperx.diarize.DiarizationPipeline(
            token=hf_token, device=torch_device
        )
        diarize_segments = diarize_model(audio_path, **diarize_kwargs)
        result = whisperx.assign_word_speakers(diarize_segments, result)
    console.print("[green]✓[/green] Diarization done.")

    # --- Build flat segment list ---
    segments = []
    for seg in result["segments"]:
        text = seg.get("text", "").strip()
        if not text:
            continue
        segments.append({
            "start": round(float(seg["start"]), 3),
            "end": round(float(seg["end"]), 3),
            "speaker": seg.get("speaker") or "SPEAKER_00",
            "text": text,
        })

    cache: dict = {
        "audio_path": str(Path(audio_path).resolve()),
        "model": model_name,
        "segments": segments,
        "speaker_names": {},
        "merges": [],
    }
    save_cache(cache, audio_path)
    console.print(f"[green]✓[/green] {len(segments)} segments saved to cache.")
    return cache
