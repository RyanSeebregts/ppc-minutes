import subprocess
import sys


def play(clip_path: str) -> None:
    """Play a WAV file. Blocks until playback finishes."""
    if sys.platform == "darwin":
        subprocess.run(["afplay", clip_path], check=True)
    elif sys.platform == "win32":
        import winsound
        winsound.PlaySound(clip_path, winsound.SND_FILENAME)
    else:
        # Linux fallback — requires alsa-utils
        subprocess.run(["aplay", "-q", clip_path], check=True)
