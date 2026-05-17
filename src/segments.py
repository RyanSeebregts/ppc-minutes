import random
from dataclasses import dataclass


@dataclass
class Segment:
    start: float
    end: float
    speaker: str
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def load_segments(cache: dict) -> list[Segment]:
    return [Segment(**s) for s in cache["segments"]]


def group_by_speaker(segments: list[Segment]) -> dict[str, list[Segment]]:
    """Return speakers in order of first appearance."""
    groups: dict[str, list[Segment]] = {}
    for seg in segments:
        if seg.speaker not in groups:
            groups[seg.speaker] = []
        groups[seg.speaker].append(seg)
    return groups


def sample_clips(segments: list[Segment], n: int = 5) -> list[Segment]:
    """Pick n clips spread evenly across the speaker's segments.

    Prefers segments that are at least 2 seconds long with meaningful text.
    Falls back to all segments if none meet the criteria.
    """
    good = [
        s for s in segments
        if s.duration >= 2.0 and len(s.text.split()) >= 4
    ]
    pool = good if good else segments

    if len(pool) <= n:
        return pool

    bucket_size = len(pool) // n
    clips = []
    for i in range(n):
        start = i * bucket_size
        end = min((i + 1) * bucket_size, len(pool))
        clips.append(random.choice(pool[start:end]))
    return clips


def merge_speakers(cache: dict, from_id: str, into_id: str) -> dict:
    for seg in cache["segments"]:
        if seg["speaker"] == from_id:
            seg["speaker"] = into_id
    cache["merges"].append({"from": from_id, "into": into_id})
    cache["speaker_names"].pop(from_id, None)
    return cache


def set_speaker_name(cache: dict, speaker_id: str, name: str) -> dict:
    cache["speaker_names"][speaker_id] = name
    return cache


def display_name(cache: dict, speaker_id: str, fallback_index: int) -> str:
    return cache["speaker_names"].get(speaker_id) or f"Speaker {fallback_index + 1}"
