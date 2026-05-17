import os
import shutil
import subprocess
import tempfile

_clip_dir: str | None = None


def _get_clip_dir() -> str:
    global _clip_dir
    if _clip_dir is None:
        _clip_dir = tempfile.mkdtemp(prefix="diarize_clips_")
    return _clip_dir


def extract_clip(audio_path: str, start: float, end: float, clip_id: str) -> str:
    """Extract a segment from audio_path as a WAV file using ffmpeg.

    Uses seek-before-input (-ss before -i) so ffmpeg jumps directly to the
    timestamp without decoding the whole file — fast even on 3-hour files.
    """
    out_path = os.path.join(_get_clip_dir(), f"{clip_id}.wav")
    if os.path.exists(out_path):
        return out_path

    # Pad slightly so the clip doesn't feel clipped at the edges
    padded_start = max(0.0, start - 0.25)
    padded_end = end + 0.25

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(padded_start),
            "-to", str(padded_end),
            "-i", audio_path,
            "-ar", "16000",
            "-ac", "1",
            "-f", "wav",
            out_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return out_path


def cleanup_clips() -> None:
    global _clip_dir
    if _clip_dir and os.path.exists(_clip_dir):
        shutil.rmtree(_clip_dir, ignore_errors=True)
        _clip_dir = None
