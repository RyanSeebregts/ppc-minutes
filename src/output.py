import json
from datetime import datetime
from pathlib import Path


def write_transcript(cache: dict, audio_path: str) -> str:
    p = Path(audio_path)
    out_path = p.parent / (p.stem + "_transcript.json")

    names = cache.get("speaker_names", {})

    def resolve(speaker_id: str) -> str:
        return names.get(speaker_id, speaker_id)

    segments = cache.get("segments", [])

    # Collect unique speakers in order of first appearance
    seen: list[str] = []
    for seg in segments:
        name = resolve(seg["speaker"])
        if name not in seen:
            seen.append(name)

    duration = round(segments[-1]["end"]) if segments else 0

    output = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "audio_file": p.name,
        "duration_seconds": duration,
        "speakers": seen,
        "segments": [
            {
                "start": seg["start"],
                "end": seg["end"],
                "speaker": resolve(seg["speaker"]),
                "text": seg["text"],
            }
            for seg in segments
        ],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return str(out_path)
