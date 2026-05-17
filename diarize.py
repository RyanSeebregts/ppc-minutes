import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console

from src import pipeline
from src.clips import cleanup_clips
from src.output import write_transcript
from src.review import review

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe and diarise a long audio file using WhisperX."
    )
    parser.add_argument("audio", help="Path to the audio file (m4a, mp3, wav, …)")
    parser.add_argument(
        "--model", default="large-v3",
        help="Whisper model size (default: large-v3)"
    )
    parser.add_argument(
        "--hf-token",
        help="HuggingFace token for pyannote (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--min-speakers", type=int, metavar="N",
        help="Minimum number of speakers (helps diarization accuracy)"
    )
    parser.add_argument(
        "--max-speakers", type=int, metavar="N",
        help="Maximum number of speakers (helps diarization accuracy)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-transcribe even if a cache file already exists"
    )
    parser.add_argument(
        "--language", default=None,
        help="Audio language code, e.g. 'en' (skips per-chunk detection, improves accuracy)"
    )
    parser.add_argument(
        "--device",
        help="Override compute device (mps / cuda / cpu)"
    )
    parser.add_argument(
        "--skip-review", action="store_true",
        help="Skip the interactive review and write output immediately"
    )
    args = parser.parse_args()

    audio_path = str(Path(args.audio).resolve())
    if not os.path.exists(audio_path):
        console.print(f"[red]Error:[/red] file not found: {audio_path}")
        sys.exit(1)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # --- Load or create cache ---
    cache = None
    if not args.force:
        cache = pipeline.load_cache(audio_path)
        if cache:
            seg_count = len(cache.get("segments", []))
            console.print(
                f"[green]Loaded cache[/green] — {seg_count} segments, "
                f"skipping transcription. (Use --force to re-transcribe.)"
            )

    if cache is None:
        if not hf_token:
            console.print(
                "[red]Error:[/red] HuggingFace token required for diarization.\n"
                "Pass --hf-token hf_xxx or set the HF_TOKEN environment variable."
            )
            sys.exit(1)
        console.print(
            f"[bold]Starting transcription[/bold] of [cyan]{Path(audio_path).name}[/cyan]"
        )
        cache = pipeline.transcribe(
            audio_path,
            model_name=args.model,
            hf_token=hf_token,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            device_override=args.device,
            language=args.language,
        )

    # --- Interactive review ---
    if not args.skip_review:
        try:
            cache = review(cache, audio_path)
        finally:
            cleanup_clips()
    else:
        console.print("[dim]Skipping review (--skip-review)[/dim]")

    # --- Write output ---
    out_path = write_transcript(cache, audio_path)
    console.print(f"\n[bold green]Done![/bold green] Transcript written to:\n  {out_path}")


if __name__ == "__main__":
    main()
