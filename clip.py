"""Encode a list of BGR frames into an MP4 clip.

Prefers H.264 via the ffmpeg CLI (yuv420p + faststart, so it streams and
previews in Telegram). Falls back to OpenCV's mp4v writer if ffmpeg is not
available — playable, but a less universal codec.
"""

import os
import shutil
import subprocess
import tempfile

import cv2


def _has_ffmpeg():
    return shutil.which("ffmpeg") is not None


def encode_clip(frames, fps):
    """Return (mp4_bytes, "mp4") for the given BGR frames, or (None, None)."""
    if not frames:
        return None, None
    h, w = frames[0].shape[:2]
    fps = max(1.0, min(30.0, float(fps)))
    if _has_ffmpeg():
        data = _encode_ffmpeg(frames, w, h, fps)
        if data:
            return data, "mp4"
    data = _encode_cv2(frames, w, h, fps)
    return (data, "mp4") if data else (None, None)


def _encode_ffmpeg(frames, w, h, fps):
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "clip.mp4")
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", f"{fps:.3f}", "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "veryfast",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", out,
        ]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            for f in frames:
                p.stdin.write(f.tobytes())
            p.stdin.close()
            err = p.stderr.read()
            p.wait(timeout=60)
        except Exception:
            p.kill()
            return None
        if p.returncode != 0 or not os.path.exists(out):
            print(f"[clip] ffmpeg failed: {err.decode(errors='replace')[:200]}",
                  flush=True)
            return None
        with open(out, "rb") as fh:
            return fh.read()


def _encode_cv2(frames, w, h, fps):
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "clip.mp4")
        vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not vw.isOpened():
            return None
        for f in frames:
            vw.write(f)
        vw.release()
        with open(out, "rb") as fh:
            return fh.read()
