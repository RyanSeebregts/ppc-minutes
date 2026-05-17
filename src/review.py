from rich.console import Console
from rich.prompt import Prompt
from rich.rule import Rule

from src import pipeline
from src.clips import extract_clip
from src.playback import play
from src.segments import (
    display_name,
    group_by_speaker,
    load_segments,
    merge_speakers,
    sample_clips,
    set_speaker_name,
)

console = Console()


def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _play_clips(clips, audio_path: str, speaker_id: str) -> None:
    for i, seg in enumerate(clips):
        console.print(
            f"\n  [dim][{i + 1}/{len(clips)}]  "
            f"{_fmt_time(seg.start)} → {_fmt_time(seg.end)}[/dim]"
        )
        console.print(f'  [italic]"{seg.text}"[/italic]')
        clip_path = extract_clip(audio_path, seg.start, seg.end, f"{speaker_id}_{i}")
        console.print("  [dim]▶ playing…[/dim]", end="")
        play(clip_path)
        console.print(" [dim]done[/dim]")


def review(cache: dict, audio_path: str) -> dict:
    """Interactive speaker-labelling loop. Returns updated cache."""

    while True:
        segments = load_segments(cache)
        groups = group_by_speaker(segments)
        speaker_ids = list(groups.keys())

        if not speaker_ids:
            console.print("[yellow]No speakers found in cache.[/yellow]")
            break

        all_labelled = all(
            sp in cache["speaker_names"] for sp in speaker_ids
        )
        if all_labelled:
            console.print("\n[bold green]All speakers are already labelled.[/bold green]")
            if Prompt.ask("Re-review anyway?", choices=["y", "n"], default="n") == "n":
                break

        for idx, speaker_id in enumerate(speaker_ids):
            # Speaker may have been merged away during this pass
            if speaker_id not in group_by_speaker(load_segments(cache)):
                continue

            sp_segments = groups[speaker_id]
            sp_label = display_name(cache, speaker_id, idx)
            already_named = speaker_id in cache["speaker_names"]

            console.print()
            console.print(Rule(
                f"[bold cyan]Speaker {idx + 1} of {len(speaker_ids)}[/bold cyan]"
                f"  •  {speaker_id}  •  {len(sp_segments)} segments"
            ))

            if already_named:
                console.print(f"  [green]Currently named: {sp_label}[/green]")

            clips = sample_clips(sp_segments, n=5)
            _play_clips(clips, audio_path, speaker_id)

            while True:
                console.print(
                    "\n  [bold cyan]n[/bold cyan] Name  "
                    "[bold cyan]s[/bold cyan] Skip  "
                    "[bold cyan]m[/bold cyan] Merge into another  "
                    "[bold cyan]r[/bold cyan] Replay clips  "
                    "[bold cyan]q[/bold cyan] Save & quit"
                )
                choice = Prompt.ask(
                    "  >", choices=["n", "s", "m", "r", "q"], show_choices=False
                )

                if choice == "n":
                    name = Prompt.ask("  Enter name").strip()
                    if name:
                        cache = set_speaker_name(cache, speaker_id, name)
                        pipeline.save_cache(cache, audio_path)
                        console.print(f"  [green]Saved as '{name}'[/green]")
                    break

                elif choice == "s":
                    console.print(f"  [dim]Skipped — keeping as '{sp_label}'[/dim]")
                    break

                elif choice == "m":
                    # Refresh groups in case a prior merge changed things
                    current_groups = group_by_speaker(load_segments(cache))
                    others = [
                        (i, sp) for i, sp in enumerate(speaker_ids)
                        if sp != speaker_id and sp in current_groups
                    ]
                    if not others:
                        console.print("  [red]No other speakers to merge into.[/red]")
                        continue

                    console.print("  Merge into which speaker?")
                    for i, sp in others:
                        console.print(
                            f"    [{i + 1}] {display_name(cache, sp, i)} ({sp})"
                        )

                    raw = Prompt.ask("  Speaker number").strip()
                    try:
                        target_num = int(raw) - 1
                        target_id = speaker_ids[target_num]
                        if target_id == speaker_id or target_id not in current_groups:
                            raise ValueError
                        cache = merge_speakers(cache, speaker_id, target_id)
                        pipeline.save_cache(cache, audio_path)
                        merged_into = display_name(cache, target_id, target_num)
                        console.print(f"  [green]Merged into '{merged_into}'[/green]")
                        break
                    except (ValueError, IndexError):
                        console.print("  [red]Invalid choice — try again.[/red]")

                elif choice == "r":
                    _play_clips(clips, audio_path, speaker_id)

                elif choice == "q":
                    pipeline.save_cache(cache, audio_path)
                    console.print(
                        "\n[yellow]Progress saved. Run again to continue.[/yellow]"
                    )
                    return cache

        console.print("\n[bold green]Review complete![/bold green]")
        break

    return cache
