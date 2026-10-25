"""Cast the annotated live feed to a Chromecast.

Chromecasts can't play MJPEG, so the JPEG frames are piped into an ffmpeg
process that writes a rolling H.264 HLS playlist to a temp dir. The app serves
that dir under an unguessable token (the Chromecast can't hold a login
session) and pychromecast tells the device to load the playlist with the
Default Media Receiver.
"""
import os
import secrets
import shutil
import socket
import subprocess
import tempfile
import threading
import time

import pychromecast

HLS_MIME = "application/vnd.apple.mpegurl"


def lan_ip():
    """Best-guess LAN address of this host (what the Chromecast can reach)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))  # no packet is sent for UDP connect
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def discover(timeout=5):
    """Friendly names of Chromecasts found on the LAN, sorted."""
    casts, browser = pychromecast.get_chromecasts(timeout=timeout)
    try:
        return sorted(c.name for c in casts)
    finally:
        browser.stop_discovery()


class Caster:
    def __init__(self):
        self._lock = threading.Lock()
        self._ffmpeg = None
        self._cast = None
        self._browser = None
        self._dir = None
        self.token = None
        self.device = None

    @property
    def active(self):
        return self._ffmpeg is not None

    def hls_dir(self, token):
        """The playlist dir if `token` matches the live session, else None."""
        if self._dir and self.token and secrets.compare_digest(token.encode(), self.token.encode()):
            return self._dir
        return None

    def start(self, device, frames, base_url):
        """Begin encoding `frames` (iterator of JPEG bytes) and cast it."""
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg is not installed")
        with self._lock:
            self._stop_locked()
            self._dir = tempfile.mkdtemp(prefix="vipervision-cast-")
            self.token = secrets.token_urlsafe(16)
            self.device = device
            try:
                self._start_ffmpeg(frames)
                self._wait_for_playlist()
                self._play(device, f"{base_url}/cast/hls/{self.token}/live.m3u8")
            except Exception:
                self._stop_locked()
                raise

    def stop(self):
        with self._lock:
            self._stop_locked()

    def _start_ffmpeg(self, frames):
        # Wall-clock timestamps because the camera's frame rate is irregular.
        # scale rounds to even dims (yuv420p needs them); main profile keeps
        # older Chromecasts happy.
        cmd = [
            "ffmpeg", "-loglevel", "error",
            "-use_wallclock_as_timestamps", "1",
            "-probesize", "100000", "-analyzeduration", "0",  # don't buffer input to probe
            "-f", "image2pipe", "-c:v", "mjpeg", "-i", "-",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=15",
            "-an", "-c:v", "libx264", "-preset", "veryfast",
            "-tune", "zerolatency", "-profile:v", "main", "-pix_fmt", "yuv420p",
            "-g", "15", "-f", "hls", "-hls_time", "1", "-hls_list_size", "10",
            "-hls_flags", "delete_segments+omit_endlist+independent_segments",
            os.path.join(self._dir, "live.m3u8"),
        ]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                             stdout=subprocess.DEVNULL)  # stderr -> app log
        self._ffmpeg = p

        def feed():
            try:
                for jpeg in frames:
                    if self._ffmpeg is not p:
                        break
                    p.stdin.write(jpeg)
                    p.stdin.flush()
            except (BrokenPipeError, ValueError, OSError):
                pass

        threading.Thread(target=feed, daemon=True).start()

    def _wait_for_playlist(self, min_segments=3, timeout=20):
        # The receiver wants a few segments buffered at the live edge; casting
        # as soon as the first one lands makes it stall and reload repeatedly.
        playlist = os.path.join(self._dir, "live.m3u8")
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._ffmpeg.poll() is not None:
                raise RuntimeError("ffmpeg exited before producing video")
            try:
                with open(playlist) as fh:
                    if fh.read().count("#EXTINF") >= min_segments:
                        return
            except OSError:
                pass
            time.sleep(0.25)
        raise RuntimeError("no video produced (is the camera connected?)")

    def _play(self, device, url):
        casts, browser = pychromecast.get_listed_chromecasts(
            friendly_names=[device], discovery_timeout=8)
        if not casts:
            browser.stop_discovery()
            raise RuntimeError(f"Chromecast '{device}' not found")
        cast = casts[0]
        self._cast, self._browser = cast, browser
        cast.wait(timeout=10)
        cast.media_controller.play_media(url, HLS_MIME, stream_type="LIVE",
                                         title="ViperVision")
        cast.media_controller.block_until_active(timeout=10)

    def _stop_locked(self):
        if self._cast is not None:
            try:
                self._cast.quit_app()
                self._cast.disconnect(timeout=5)
            except Exception:
                pass
        if self._browser is not None:
            try:
                self._browser.stop_discovery()
            except Exception:
                pass
        if self._ffmpeg is not None:
            try:
                self._ffmpeg.stdin.close()
            except Exception:
                pass
            self._ffmpeg.kill()
            self._ffmpeg.wait()
        if self._dir:
            shutil.rmtree(self._dir, ignore_errors=True)
        self._cast = self._browser = self._ffmpeg = self._dir = None
        self.token = self.device = None
