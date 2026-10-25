"""ViperVision — RTSP stream viewer.

Connects to an RTSP camera feed with OpenCV and serves it to the browser
as an MJPEG stream via Flask.

Usage:
    RTSP_URL=rtsp://user:pass@host:554/stream python app.py
    # or set the URL from the web UI at http://localhost:5000
"""

import hmac
import os
import re
import secrets
import socket
import threading
import time
import urllib.request
from collections import deque
from urllib.parse import urlparse

# Must be set before the first VideoCapture is created. Many cameras (this
# one included) only accept RTP over UDP and reject interleaved TCP.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;" + os.environ.get("RTSP_TRANSPORT", "udp"),
)

import cv2
import numpy as np
from flask import (
    Flask, Response, jsonify, redirect, render_template, request, send_from_directory,
    session
)
from werkzeug.middleware.proxy_fix import ProxyFix

from cast import Caster, discover, lan_ip
from clip import encode_clip
from detector import (
    PersonDetector, MotionDetector, COCO_ANIMALS, ANIMAL_IDS,
)
from storage import DetectionSaver
from notify import TelegramNotifier
from telegram_bot import TelegramCommandBot
from metrics import metrics

app = Flask(__name__)

# Behind SWAG/nginx: trust one hop of X-Forwarded-* so Flask sees the real
# scheme (https) and client IP instead of the proxy's http/172.18.0.1.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# --- Authentication -------------------------------------------------------
# Session cookies are signed with this key; a fresh random key each start
# means restarting the server logs everyone out (fine for a single-user app).
# Set SECRET_KEY to keep sessions valid across restarts.
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")

# Mark the session cookie Secure so it is only sent over HTTPS. Off by default
# because the app is also reachable over plain http on the LAN
# (http://<host>:5001); set SESSION_COOKIE_SECURE=1 if you only ever reach it
# through the https reverse proxy, otherwise direct-http logins won't stick.
if os.environ.get("SESSION_COOKIE_SECURE", "").lower() in ("1", "true", "yes", "on"):
    app.config.update(SESSION_COOKIE_SECURE=True)

AUTH_USER = os.environ.get("VIPERVISION_USER", "admin")
# Use the configured password, or generate one and print it at startup so the
# app is never unintentionally left with a well-known default credential.
AUTH_PASSWORD = os.environ.get("VIPERVISION_PASSWORD")
AUTH_PASSWORD_GENERATED = AUTH_PASSWORD is None
if AUTH_PASSWORD_GENERATED:
    AUTH_PASSWORD = secrets.token_urlsafe(9)

# Endpoints reachable without a session. /metrics is scraped by Prometheus
# (which can't hold a login session); it carries only aggregate counts, and an
# optional METRICS_TOKEN can gate it (see the metrics route).
PUBLIC_ENDPOINTS = {"login", "static", "metrics", "cast_hls"}

# Optional bearer token for /metrics. If set, scrapers must send
# `Authorization: Bearer <token>`; if unset, /metrics is open (fine on a
# trusted LAN — set the token if the app is exposed publicly via the proxy).
METRICS_TOKEN = os.environ.get("METRICS_TOKEN", "").strip()

# Optional camera illuminator (an ESP32-CAM's white LED via ESPHome's REST
# API). Point LIGHT_URL at the ESPHome light endpoint, e.g.
#   http://192.168.64.65:8080/light/cam2_illuminator
# (requires a `light:` component in the ESPHome config — a bare `output:` is
# not exposed over REST). Unset = no light, and the UI button stays hidden.
LIGHT_URL = os.environ.get("LIGHT_URL", "").strip().rstrip("/")
# Last commanded state; the ESP32 isn't polled (it's fragile and this keeps
# /status cheap), so this reflects what we last told it.
light_state = {"on": False}


def _bool_env(name, default):
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v.lower() in ("1", "true", "yes", "on")


# Which detectors are enabled at startup (each still needs its model/feature
# available). Default: motion ON, human OFF — override per camera in compose.
DETECT_DEFAULT = _bool_env("DETECT_DEFAULT", False)
MOTION_DEFAULT = _bool_env("MOTION_DEFAULT", True)
ANIMAL_DEFAULT = _bool_env("ANIMAL_DEFAULT", False)


def check_credentials(username, password):
    # Constant-time compare on both fields to avoid leaking either via timing.
    user_ok = hmac.compare_digest(username, AUTH_USER)
    pass_ok = hmac.compare_digest(password, AUTH_PASSWORD)
    return user_ok and pass_ok


@app.before_request
def require_login():
    if request.endpoint in PUBLIC_ENDPOINTS:
        return None
    if session.get("authed"):
        return None
    # For API/stream calls, answer with 401 rather than an HTML redirect.
    if request.path.startswith(("/status", "/ptz", "/detect", "/motion", "/animal",
                                "/light", "/record", "/keep", "/categorize", "/video_feed",
                                "/cast")):
        return jsonify({"error": "authentication required"}), 401
    # Relative redirect so it stays under any proxy path prefix.
    return redirect("login")

START_TIME = time.time()  # process start, for the uptime gauge
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "80"))
RECONNECT_DELAY = 3.0  # seconds between reconnect attempts
DETECT_INTERVAL = float(os.environ.get("DETECT_INTERVAL", "0.5"))  # seconds between inferences
# Record a clip once a person has been present continuously for this long
# (a passer-by who leaves sooner is ignored). One clip per event.
SAVE_AFTER = float(os.environ.get("SAVE_AFTER_SECONDS", "5"))
# Tolerate brief detection gaps before ending an event. On low-fps or dim
# feeds (e.g. an ESP32-CAM at 1fps) detection flickers; without this grace a
# single missed frame would reset the linger timer and no event ever completes.
EVENT_GRACE = float(os.environ.get("EVENT_GRACE_SECONDS", "2.0"))
# Clip spans CLIP_PRE seconds before the trigger (from a rolling buffer) to
# CLIP_POST seconds after, so you see the person arrive and what they do next.
CLIP_PRE = float(os.environ.get("CLIP_PRE_SECONDS", "3"))
CLIP_POST = float(os.environ.get("CLIP_POST_SECONDS", "4"))
# Rolling buffer must hold at least the whole clip window, plus slack.
BUFFER_SECONDS = CLIP_PRE + CLIP_POST + 2.0
# Animal capture: which COCO species trigger a saved snapshot + alert, and how
# long they must linger continuously first. Default: cats only, same linger as
# people. Set ALERT_ANIMALS="cat,dog" to widen, or ANIMAL_SAVE_AFTER_SECONDS.
ALERT_ANIMALS = {s.strip().lower() for s in
                 os.environ.get("ALERT_ANIMALS", "cat").split(",") if s.strip()}
ANIMAL_SAVE_AFTER = float(os.environ.get("ANIMAL_SAVE_AFTER_SECONDS", SAVE_AFTER))
# Motion-triggered clips: linger before recording. Motion is noisier than the
# object detectors, so this defaults a touch longer to avoid clip spam.
MOTION_SAVE_AFTER = float(os.environ.get("MOTION_SAVE_AFTER_SECONDS", SAVE_AFTER))
# Extend-on-motion: instead of a fixed-length clip, keep recording for as long
# as motion continues and only finalize after MOTION_END_GRACE of stillness, so
# a whole episode (e.g. sleepwalking) becomes ONE continuous clip. Capped at
# MOTION_MAX_SECONDS to bound memory. Opt-in per camera.
MOTION_EXTEND = _bool_env("MOTION_EXTEND", False)
MOTION_END_GRACE = float(os.environ.get("MOTION_END_GRACE_SECONDS", "6"))
MOTION_MAX_SECONDS = float(os.environ.get("MOTION_MAX_SECONDS", "300"))


def draw_person_boxes(frame, boxes):
    """Draw green person boxes + labels and a total onto a BGR frame."""
    for x, y, w, h, conf in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)
        label = f"person {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x, y - th - 6), (x + tw + 4, y), (0, 220, 0), -1)
        cv2.putText(frame, label, (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    if boxes:
        cv2.putText(frame, f"Humans: {len(boxes)}", (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2, cv2.LINE_AA)


def draw_motion_boxes(frame, boxes):
    """Draw amber boxes around moving regions (drawn under person boxes)."""
    for x, y, w, h in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 180, 255), 2)


def draw_animal_boxes(frame, boxes):
    """Draw cyan boxes + species labels for detected animals (cat, dog, ...)."""
    for x, y, w, h, conf, cls in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 220, 0), 2)
        label = f"{COCO_ANIMALS.get(cls, 'animal')} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x, y - th - 6), (x + tw + 4, y), (255, 220, 0), -1)
        cv2.putText(frame, label, (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


class RTSPStream:
    """Reads frames from an RTSP source on a background thread.

    Keeps only the latest frame so viewers never lag behind the live feed,
    and reconnects automatically if the stream drops.
    """

    def __init__(self):
        self._url = None
        self._frame = None
        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self._thread = None
        self._generation = 0
        self.status = "disconnected"

        # Human detection (populated by a background detector thread).
        self._raw = None            # latest decoded BGR frame, for the detector
        self._boxes = []            # latest person boxes: (x, y, w, h, conf)
        self.detector = PersonDetector()
        self.detect_enabled = self.detector.available and DETECT_DEFAULT
        self.person_count = 0

        # Motion detection — independent of person detection, toggled
        # separately. Off by default (person detection is the primary signal).
        self.motion = MotionDetector()
        self.motion_enabled = self.motion.available and MOTION_DEFAULT
        self._motion_boxes = []     # latest motion boxes: (x, y, w, h)
        self.motion_count = 0
        self._motion_event_start = None
        self._motion_event_captured = False
        self._motion_last_seen = 0.0
        # Extend-on-motion recording state (MOTION_EXTEND): while active, the
        # reader appends every frame to _rec_frames until the detect loop ends
        # the episode (stillness > MOTION_END_GRACE, or MOTION_MAX_SECONDS).
        self._rec_active = False
        self._rec_frames = []       # list of (timestamp, jpeg) during an episode
        self._rec_start = 0.0

        # Animal detection — reuses the person model (COCO), toggled separately.
        self.animal_enabled = self.detector.available and ANIMAL_DEFAULT
        self._animal_boxes = []     # latest animal boxes: (x, y, w, h, conf, cls)
        self.animal_count = 0
        self._animal_event_start = None    # time an alert-species first appeared
        self._animal_event_captured = False

        # Clip-on-detection: track how long a person has been continuously
        # present so we only record sustained detections (see SAVE_AFTER).
        self.saver = DetectionSaver()
        self.notifier = TelegramNotifier()
        self._event_start = None    # time persons first appeared, else None
        self._event_captured = False  # recorded a clip for this event yet
        self._last_person_seen = 0.0  # for gap-tolerant linger (EVENT_GRACE)
        # Rolling buffer of recent (timestamp, annotated-jpeg) for clip pre-roll.
        self._buffer = deque()
        threading.Thread(target=self._detect_loop, daemon=True).start()

    @property
    def url(self):
        return self._url

    def start(self, url):
        with self._lock:
            self._url = url
            self._generation += 1
            generation = self._generation
            self._frame = None
            self._buffer.clear()
            self.status = "connecting"
        thread = threading.Thread(target=self._reader, args=(url, generation), daemon=True)
        thread.start()
        self._thread = thread

    def stop(self):
        with self._lock:
            self._generation += 1
            self._url = None
            self._frame = None
            self._raw = None
            self._boxes = []
            self.person_count = 0
            self._motion_boxes = []
            self.motion_count = 0
            self._motion_event_start = None
            self._motion_event_captured = False
            self._rec_active = False
            self._rec_frames = []
            self._animal_boxes = []
            self.animal_count = 0
            self._event_start = None
            self._event_captured = False
            self._buffer.clear()
            self.status = "disconnected"
            self._new_frame.notify_all()

    def _current(self, generation):
        with self._lock:
            return generation == self._generation

    def _reader(self, url, generation):
        while self._current(generation):
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            # Small buffer keeps the feed close to real time.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

            if not cap.isOpened():
                cap.release()
                metrics.inc("vipervision_reconnects_total")
                with self._lock:
                    if generation == self._generation:
                        self.status = f"failed to connect, retrying in {RECONNECT_DELAY:.0f}s"
                time.sleep(RECONNECT_DELAY)
                continue

            with self._lock:
                if generation == self._generation:
                    self.status = "connected"

            while self._current(generation):
                ok, frame = cap.read()
                if not ok:
                    break
                with self._lock:
                    self._raw = frame  # hand the raw frame to the detector
                if self.detect_enabled or self.motion_enabled or self.animal_enabled:
                    self._draw_overlays(frame)
                ok, jpeg = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                )
                if not ok:
                    continue
                frame_bytes = jpeg.tobytes()
                now = time.time()
                metrics.inc("vipervision_frames_total")
                with self._lock:
                    if generation != self._generation:
                        break
                    self._frame = frame_bytes
                    # Keep the last BUFFER_SECONDS of annotated frames for clips.
                    self._buffer.append((now, frame_bytes))
                    cutoff = now - BUFFER_SECONDS
                    while self._buffer and self._buffer[0][0] < cutoff:
                        self._buffer.popleft()
                    # Extend-on-motion: collect every frame during an episode.
                    if self._rec_active:
                        self._rec_frames.append((now, frame_bytes))
                        # Safety cap in case the detect loop is delayed finalizing.
                        if now - self._rec_frames[0][0] > MOTION_MAX_SECONDS + 15:
                            drop = now - MOTION_MAX_SECONDS
                            self._rec_frames = [f for f in self._rec_frames
                                                if f[0] >= drop]
                    self._new_frame.notify_all()

            cap.release()
            metrics.inc("vipervision_reconnects_total")
            with self._lock:
                if generation == self._generation:
                    self.status = f"stream dropped, reconnecting in {RECONNECT_DELAY:.0f}s"
            time.sleep(RECONNECT_DELAY)

    def _draw_overlays(self, frame):
        with self._lock:
            person = list(self._boxes)
            motion = list(self._motion_boxes)
            animals = list(self._animal_boxes)
            person_on = self.detect_enabled
            motion_on = self.motion_enabled
            animal_on = self.animal_enabled
        # Motion first (amber), then animals (cyan), then person (green) on top.
        if motion_on:
            draw_motion_boxes(frame, motion)
        if animal_on:
            draw_animal_boxes(frame, animals)
        if person_on:
            draw_person_boxes(frame, person)

    def _detect_loop(self):
        """Run person detection on the latest frame at a throttled rate.

        Runs independently of the reader so inference latency never lowers
        the streamed frame rate; the reader just overlays the last result.
        """
        motion_was_on = False
        while True:
            person_on = self.detect_enabled and self.detector.available
            animal_on = self.animal_enabled and self.detector.available
            motion_on = self.motion_enabled and self.motion.available
            if not (person_on or animal_on or motion_on):
                self._event_start = None  # reset so re-enabling starts fresh
                self._event_captured = False
                if motion_was_on:
                    self.motion.reset()
                    motion_was_on = False
                time.sleep(0.2)
                continue
            with self._lock:
                frame = None if self._raw is None else self._raw.copy()
            if frame is None:
                time.sleep(0.1)
                continue

            # Person + animal share the one COCO model, so run a single forward
            # pass over the union of wanted classes and split the results.
            if person_on or animal_on:
                wanted = set()
                if person_on:
                    wanted.add(0)
                if animal_on:
                    wanted |= ANIMAL_IDS
                dets = self.detector.detect_objects(frame, wanted)
                metrics.inc("vipervision_inferences_total")
                if person_on:
                    persons = [d[:5] for d in dets if d[5] == 0]
                    with self._lock:
                        self._boxes = persons
                        self.person_count = len(persons)
                    self._maybe_save(frame, persons)
                else:
                    with self._lock:
                        self._boxes = []
                        self.person_count = 0
                    self._event_start = None
                    self._event_captured = False
                if animal_on:
                    animals = [d for d in dets if d[5] in ANIMAL_IDS]
                    with self._lock:
                        self._animal_boxes = animals
                        self.animal_count = len(animals)
                    self._maybe_save_animals(animals)
                else:
                    with self._lock:
                        self._animal_boxes = []
                        self.animal_count = 0
                    self._animal_event_start = None
                    self._animal_event_captured = False
            else:
                # Neither person nor animal on: clear both, stop clip events.
                with self._lock:
                    self._boxes = []
                    self.person_count = 0
                    self._animal_boxes = []
                    self.animal_count = 0
                self._event_start = None
                self._event_captured = False
                self._animal_event_start = None
                self._animal_event_captured = False

            if motion_on:
                # Fresh background model on the first frame after enabling.
                if not motion_was_on:
                    self.motion.reset()
                    motion_was_on = True
                mboxes = self.motion.detect(frame)
                with self._lock:
                    self._motion_boxes = mboxes
                    self.motion_count = len(mboxes)
                self._maybe_save_motion(mboxes)
            elif motion_was_on:
                self.motion.reset()
                motion_was_on = False
                self._flush_extended()   # don't lose an in-progress episode
                with self._lock:
                    self._motion_boxes = []
                    self.motion_count = 0
                self._motion_event_start = None
                self._motion_event_captured = False

            time.sleep(DETECT_INTERVAL)

    def _maybe_save(self, frame, boxes):
        """Once a person has lingered > SAVE_AFTER, record one clip per event
        (pre-roll from the buffer + post-roll) and dispatch it."""
        if not (self.saver.available or self.notifier.available):
            return
        now = time.time()
        if not boxes:
            # End the event only after a grace gap of no detections, so a
            # flickering low-fps feed still accumulates a continuous event.
            if self._event_start is not None and now - self._last_person_seen > EVENT_GRACE:
                self._event_start = None
                self._event_captured = False
            return
        self._last_person_seen = now
        if self._event_start is None:
            self._event_start = now
            self._event_captured = False
        if now - self._event_start >= SAVE_AFTER and not self._event_captured:
            self._event_captured = True
            # Capture off-thread: it waits CLIP_POST for post-roll, so it must
            # not block detection.
            threading.Thread(target=self._capture_clip,
                             args=(now, len(boxes)), daemon=True).start()

    def _encode_frames(self, items):
        """Encode a list of (timestamp, jpeg) into an MP4 at the frames' real
        rate (so playback is life-speed). Returns (bytes, ext) or (None, None)."""
        if len(items) < 2:
            return None, None
        span = items[-1][0] - items[0][0]
        fps = len(items) / span if span > 0 else 15.0
        frames = []
        for _, j in items:
            img = cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                frames.append(img)
        if len(frames) < 2:
            return None, None
        return encode_clip(frames, fps)

    def _encode_window(self, trigger_time):
        """Encode the buffered frames spanning [trigger-CLIP_PRE, trigger+
        CLIP_POST] into an MP4. Assumes post-roll has already been waited for."""
        lo, hi = trigger_time - CLIP_PRE, trigger_time + CLIP_POST
        with self._lock:
            items = [(t, j) for (t, j) in self._buffer if lo <= t <= hi]
        return self._encode_frames(items)

    def _capture_clip(self, trigger_time, person_count):
        """Wait for post-roll, then encode the buffered window into an MP4 and
        hand it to storage + notifier."""
        time.sleep(CLIP_POST)
        data, ext = self._encode_window(trigger_time)
        if not data:
            return
        metrics.inc("vipervision_clips_captured_total")
        self.saver.save(data, person_count, trigger_time,
                        ext=ext, content_type="video/mp4")
        self.notifier.notify(data, person_count, trigger_time, media="video")

    def _maybe_save_animals(self, animals):
        """Once an alert species (default: cat) has lingered > ANIMAL_SAVE_AFTER,
        save one annotated snapshot per event and dispatch a Telegram alert."""
        if not (self.saver.available or self.notifier.available):
            return
        wanted = [a for a in animals if COCO_ANIMALS.get(a[5]) in ALERT_ANIMALS]
        now = time.time()
        if not wanted:
            self._animal_event_start = None
            self._animal_event_captured = False
            return
        if self._animal_event_start is None:
            self._animal_event_start = now
            self._animal_event_captured = False
        if (now - self._animal_event_start >= ANIMAL_SAVE_AFTER
                and not self._animal_event_captured):
            self._animal_event_captured = True
            species = COCO_ANIMALS.get(wanted[0][5], "animal")
            threading.Thread(target=self._capture_animal_clip,
                             args=(len(wanted), species, now), daemon=True).start()

    def _capture_animal_clip(self, count, species, trigger_time):
        """Wait for post-roll, then encode the buffered window into an MP4 and
        save/notify it as an animal event (same clip path as person events)."""
        time.sleep(CLIP_POST)
        data, ext = self._encode_window(trigger_time)
        if not data:
            return
        metrics.inc("vipervision_animal_captures_total")
        # Key kind: cat->c, else the species' first letter (dog->d, bird->b).
        tag = "c" if species == "cat" else species[0]
        emoji = "\U0001F431" if species == "cat" else "\U0001F43E"  # 🐱 / 🐾
        label = f"{emoji} {count} {species}{'s' if count != 1 else ''} detected"
        self.saver.save(data, count, trigger_time, ext=ext,
                        content_type="video/mp4", tag=tag)
        self.notifier.notify(data, count, trigger_time, media="video", label=label)

    def _maybe_save_motion(self, mboxes):
        """Fixed-length clip per motion event (gap-tolerant, like the person
        path), or — when MOTION_EXTEND — an episode-length continuous clip."""
        if not (self.saver.available or self.notifier.available):
            return
        if MOTION_EXTEND:
            self._maybe_record_extended(mboxes)
            return
        now = time.time()
        if not mboxes:
            if self._motion_event_start is not None and now - self._motion_last_seen > EVENT_GRACE:
                self._motion_event_start = None
                self._motion_event_captured = False
            return
        self._motion_last_seen = now
        if self._motion_event_start is None:
            self._motion_event_start = now
            self._motion_event_captured = False
        if (now - self._motion_event_start >= MOTION_SAVE_AFTER
                and not self._motion_event_captured):
            self._motion_event_captured = True
            threading.Thread(target=self._capture_motion_clip,
                             args=(len(mboxes), now), daemon=True).start()

    def _maybe_record_extended(self, mboxes):
        """Keep recording for as long as motion continues; finalize into one
        clip after MOTION_END_GRACE of stillness (or at MOTION_MAX_SECONDS)."""
        now = time.time()
        if mboxes:
            self._motion_last_seen = now
            if not self._rec_active:
                # Start: seed with the recent buffer as pre-roll (the lead-up),
                # then the reader appends live frames until we finalize.
                with self._lock:
                    self._rec_frames = list(self._buffer)
                    self._rec_active = True
                self._rec_start = now
            return
        # No motion this cycle — end the episode after a grace of stillness,
        # or if it has run to the max length.
        if not self._rec_active:
            return
        if (now - self._motion_last_seen > MOTION_END_GRACE
                or now - self._rec_start >= MOTION_MAX_SECONDS):
            with self._lock:
                frames = self._rec_frames
                self._rec_frames = []
                self._rec_active = False
            start = self._rec_start
            threading.Thread(target=self._finalize_extended_clip,
                             args=(frames, start), daemon=True).start()

    def _flush_extended(self):
        """Finalize an in-progress extended recording (motion turned off /
        stream stopping) so the episode isn't lost."""
        if not self._rec_active:
            return
        with self._lock:
            frames = self._rec_frames
            self._rec_frames = []
            self._rec_active = False
        if frames:
            threading.Thread(target=self._finalize_extended_clip,
                             args=(frames, self._rec_start), daemon=True).start()

    def _finalize_extended_clip(self, frames, start_time):
        """Encode a full-episode recording and save/notify it (tag 'm')."""
        data, ext = self._encode_frames(frames)
        if not data:
            return
        dur = int(frames[-1][0] - frames[0][0]) if len(frames) >= 2 else 0
        metrics.inc("vipervision_motion_captures_total")
        self.saver.save(data, 0, start_time, ext=ext,
                        content_type="video/mp4", tag="m")
        self.notifier.notify(data, 0, start_time, media="video",
                             label=f"\U0001F300 Motion — {dur}s episode")

    def _capture_motion_clip(self, count, trigger_time):
        """Wait for post-roll, encode the buffered window, save + notify it as
        a motion event (key tag 'm')."""
        time.sleep(CLIP_POST)
        data, ext = self._encode_window(trigger_time)
        if not data:
            return
        metrics.inc("vipervision_motion_captures_total")
        self.saver.save(data, count, trigger_time, ext=ext,
                        content_type="video/mp4", tag="m")
        self.notifier.notify(data, count, trigger_time, media="video",
                             label="\U0001F300 Motion detected")

    def capture_clip_now(self):
        """Record a clip around the present moment (pre-roll already in the
        buffer + CLIP_POST of post-roll). Blocks ~CLIP_POST seconds. Returns
        (mp4_bytes, ext) or (None, None). For on-demand /clip requests."""
        if self._frame is None:
            return None, None
        trigger = time.time()
        time.sleep(CLIP_POST)
        data, ext = self._encode_window(trigger)
        if data:
            metrics.inc("vipervision_clips_captured_total")
        return data, ext

    def latest_jpeg(self):
        """Return the most recent annotated JPEG frame, or None."""
        with self._lock:
            return self._frame

    def frames(self):
        """Yield JPEG frames as they arrive (blocks between frames)."""
        last = None
        while True:
            with self._new_frame:
                self._new_frame.wait_for(
                    lambda: self._frame is not None and self._frame is not last,
                    timeout=5,
                )
                if self._frame is None or self._frame is last:
                    continue
                last = self._frame
            yield last


stream = RTSPStream()

# PTZ directions -> tokens the camera firmware expects. "DWON" is not a typo
# here: the firmware itself expects that misspelling ("DOWN" is ignored).
# LEFT/RIGHT are crossed: the firmware's LEFT pans the view right, so the
# tokens are swapped to make the on-screen arrows match what the user sees.
PTZ_DIRECTIONS = {"up": "UP", "down": "DWON", "left": "RIGHT", "right": "LEFT"}


def send_ptz(direction):
    """Move the camera one step via its pseudo-RTSP ptzCmd extension.

    Yoosee-style cameras accept an RTSP SET_PARAMETER on the stream port
    carrying `ptzCmd:<DIR>`; each command is one discrete pan/tilt step and
    the motor stops on its own. No authentication is required.
    """
    if not stream.url:
        return False, "no stream connected"
    parsed = urlparse(stream.url)
    host, port = parsed.hostname, parsed.port or 554
    cmd = f"ptzCmd:{PTZ_DIRECTIONS[direction]}"
    msg = (
        f"SET_PARAMETER {stream.url} RTSP/1.0\r\n"
        "CSeq: 0\r\n"
        f"Content-length: {len(cmd)}\r\n"
        f"Content-type: {cmd}\r\n"
        "\r\n"
    )
    try:
        with socket.create_connection((host, port), timeout=3) as s:
            s.sendall(msg.encode())
            s.settimeout(3)
            reply = s.recv(256)
        return b"200 OK" in reply, reply.decode(errors="replace").strip()
    except OSError as e:
        return False, str(e)


# --- Telegram command handlers --------------------------------------------
# Each returns a text string (sent as a message), a ("photo", jpeg, caption)
# tuple, or None. The bot only invokes these for the authorized chat.

def _parse_onoff(args):
    """Map a command argument to True/False/None (None = not recognised)."""
    if not args:
        return None
    v = args[0].lower()
    if v in ("on", "1", "true", "yes", "enable", "enabled"):
        return True
    if v in ("off", "0", "false", "no", "disable", "disabled"):
        return False
    return None


def _cmd_help(args):
    return (
        "ViperVision commands:\n"
        "/status — stream & detection status\n"
        "/snapshot — send the current frame\n"
        "/clip — record & send a short video clip\n"
        "/detect on|off — human detection\n"
        "/motion on|off — motion detection\n"
        "/ptz up|down|left|right — move the camera"
    )


def _cmd_status(args):
    det = "on" if stream.detect_enabled else "off"
    mot = "on" if stream.motion_enabled else "off"
    lines = [
        f"\U0001F4E1 Stream: {stream.status}",
        f"Human detection: {det} ({stream.person_count} now)",
        f"Motion detection: {mot} ({stream.motion_count} now)",
    ]
    if not stream.url:
        lines.append("No RTSP source connected.")
    return "\n".join(lines)


def _cmd_snapshot(args):
    jpeg = stream.latest_jpeg()
    when = time.strftime("%Y-%m-%d %H:%M:%S")
    caption = f"\U0001F4F8 ViperVision snapshot\n{when}"
    return ("photo", jpeg, caption if jpeg else "No frame available yet.")


def _cmd_detect(args):
    if not stream.detector.available:
        return "Human detection model is not loaded."
    want = _parse_onoff(args)
    if want is None:
        return "Usage: /detect on|off  (currently " + \
               ("on)" if stream.detect_enabled else "off)")
    stream.detect_enabled = want
    if not want:
        with stream._lock:
            stream._boxes = []
            stream.person_count = 0
    return "Human detection " + ("enabled." if want else "disabled.")


def _cmd_motion(args):
    if not stream.motion.available:
        return "Motion detection is unavailable."
    want = _parse_onoff(args)
    if want is None:
        return "Usage: /motion on|off  (currently " + \
               ("on)" if stream.motion_enabled else "off)")
    stream.motion_enabled = want
    if not want:
        with stream._lock:
            stream._motion_boxes = []
            stream.motion_count = 0
    return "Motion detection " + ("enabled." if want else "disabled.")


def _cmd_ptz(args):
    direction = args[0].lower() if args else ""
    if direction not in PTZ_DIRECTIONS:
        return "Usage: /ptz up|down|left|right"
    ok, detail = send_ptz(direction)
    return f"PTZ {direction}: " + ("ok" if ok else f"failed ({detail})")


def _record_and_send_clip():
    """Capture a clip and deliver it to the chat (runs off the poll thread so
    the CLIP_POST wait doesn't stall other commands)."""
    count = stream.person_count
    data, ext = stream.capture_clip_now()
    if not data:
        telegram_bot.send_message("Clip capture failed — not enough buffered video.")
        return
    when = time.strftime("%Y-%m-%d %H:%M:%S")
    telegram_bot.send_video(data, f"\U0001F3A5 ViperVision clip\n{when}")
    # Archive it too, so on-demand clips appear in the gallery like auto ones.
    stream.saver.save(data, count, time.time(), ext=ext, content_type="video/mp4")


def _cmd_clip(args):
    if stream.latest_jpeg() is None:
        return "No stream connected — nothing to record."
    # Send the ack synchronously *before* starting the capture thread so it is
    # always delivered ahead of the clip (the thread waits for post-roll, but
    # we don't rely on that race). Returning None means "no further reply".
    secs = int(CLIP_PRE + CLIP_POST)
    telegram_bot.send_message(
        f"\U0001F3A5 Recording a ~{secs}s clip… it'll arrive shortly.")
    threading.Thread(target=_record_and_send_clip, daemon=True).start()
    return None


telegram_bot = TelegramCommandBot({
    "help": _cmd_help,
    "start": _cmd_help,
    "status": _cmd_status,
    "snapshot": _cmd_snapshot,
    "clip": _cmd_clip,
    "detect": _cmd_detect,
    "motion": _cmd_motion,
    "ptz": _cmd_ptz,
})


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("authed"):
        return redirect(".")
    error = None
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if check_credentials(username, password):
            session.clear()
            session["authed"] = True
            return redirect(".")
        # Small delay to blunt password brute-forcing.
        time.sleep(1.0)
        error = "Invalid username or password"
    return render_template("login.html", error=error)


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect("login")


@app.after_request
def no_cache_html(response):
    # Always revalidate HTML so UI updates show up on a normal refresh.
    if response.mimetype == "text/html":
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.route("/")
def index():
    return render_template("index.html", url=stream.url or "", status=stream.status)


@app.route("/connect", methods=["POST"])
def connect():
    url = request.form.get("url", "").strip()
    if url:
        stream.start(url)
    # Relative redirect so it stays behind any proxy path prefix.
    return redirect(".")


@app.route("/disconnect", methods=["POST"])
def disconnect():
    stream.stop()
    return redirect(".")


@app.route("/ptz/<direction>", methods=["POST"])
def ptz(direction):
    if direction not in PTZ_DIRECTIONS:
        return jsonify({"ok": False, "error": "unknown direction"}), 400
    ok, detail = send_ptz(direction)
    return jsonify({"ok": ok, "detail": detail}), (200 if ok else 502)


@app.route("/status")
def status():
    return jsonify({
        "status": stream.status,
        "url": stream.url,
        "detect_available": stream.detector.available,
        "detect_enabled": stream.detect_enabled,
        "person_count": stream.person_count,
        "motion_available": stream.motion.available,
        "motion_enabled": stream.motion_enabled,
        "motion_count": stream.motion_count,
        "animal_available": stream.detector.available,
        "animal_enabled": stream.animal_enabled,
        "animal_count": stream.animal_count,
        "light_available": bool(LIGHT_URL),
        "light_on": light_state["on"],
    })


def _queue_depth(worker):
    """Pending items on a saver/notifier queue (0 when the feature is off)."""
    q = getattr(worker, "_queue", None)
    return q.qsize() if q is not None else 0


@app.route("/metrics", endpoint="metrics")
def metrics_endpoint():
    # Optional bearer-token gate for public exposure.
    if METRICS_TOKEN:
        auth = request.headers.get("Authorization", "")
        token = auth[7:] if auth.startswith("Bearer ") else ""
        if not hmac.compare_digest(token, METRICS_TOKEN):
            return Response("unauthorized\n", status=401,
                            mimetype="text/plain")
    connected = 1 if stream.status == "connected" else 0
    gauges = [
        ("vipervision_up",
         "1 if the RTSP stream is currently connected, else 0.", connected),
        ("vipervision_persons_current",
         "People detected in the most recent frame.", stream.person_count),
        ("vipervision_detection_enabled",
         "1 if person detection is enabled, else 0.",
         1 if stream.detect_enabled else 0),
        ("vipervision_motion_enabled",
         "1 if motion detection is enabled, else 0.",
         1 if stream.motion_enabled else 0),
        ("vipervision_motion_current",
         "Moving regions detected in the most recent frame.",
         stream.motion_count),
        ("vipervision_animal_enabled",
         "1 if animal detection is enabled, else 0.",
         1 if stream.animal_enabled else 0),
        ("vipervision_animal_current",
         "Animals detected in the most recent frame.",
         stream.animal_count),
        ("vipervision_uptime_seconds",
         "Seconds since the process started.", time.time() - START_TIME),
        ("vipervision_storage_queue_depth",
         "Detection clips queued for upload to object storage.",
         _queue_depth(stream.saver)),
        ("vipervision_telegram_queue_depth",
         "Detection alerts queued for delivery to Telegram.",
         _queue_depth(stream.notifier)),
    ]
    body = metrics.render(gauges)
    # 0.0.4 is the Prometheus text exposition format version. Use content_type
    # (not mimetype) so Flask doesn't append a second charset parameter.
    return Response(body,
                    content_type="text/plain; version=0.0.4; charset=utf-8")


@app.route("/gallery")
def gallery():
    return render_template("gallery.html", available=stream.saver.available)


@app.route("/floorplan")
def floorplan():
    # Behind auth (not a PUBLIC_ENDPOINT): it's a map of the user's home.
    return render_template("floorplan.html")


def _clip_prefix(date, base="detections/"):
    """Map an optional ?date=YYYY-MM-DD (under `base`) to a MinIO key prefix.
    Invalid or empty dates fall back to the whole base prefix."""
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", date or "")
    if m:
        return f"{base}{m.group(1)}/{m.group(2)}/{m.group(3)}/"
    return base


def _int_arg(name, default, lo, hi):
    try:
        return max(lo, min(hi, int(request.args.get(name, default))))
    except (TypeError, ValueError):
        return default


@app.route("/clips")
def clips():
    """Paginated JSON list of stored detection clips, newest first.
    Query params: date=YYYY-MM-DD (filter to one day), limit, offset."""
    base = _CATEGORIES.get(request.args.get("scope"), "detections/")
    prefix = _clip_prefix(request.args.get("date", "").strip(), base)
    limit = _int_arg("limit", 30, 1, 200)
    offset = _int_arg("offset", 0, 0, 1_000_000)
    all_clips = stream.saver.list_clips(prefix=prefix)  # newest-first by name
    sort = request.args.get("sort", "newest")
    if sort == "oldest":
        all_clips = all_clips[::-1]
    elif sort == "largest":
        all_clips = sorted(all_clips, key=lambda x: x["size"], reverse=True)
    elif sort == "smallest":
        all_clips = sorted(all_clips, key=lambda x: x["size"])
    # "newest" is the default order returned by list_clips.
    page = all_clips[offset:offset + limit]
    return jsonify({
        "available": stream.saver.available,
        "clips": page,
        "total": len(all_clips),
        "offset": offset,
        "limit": limit,
        "has_more": offset + limit < len(all_clips),
    })


_CATEGORIES = {"detections": "detections/", "saved": "saved/",
               "evidence": "evidence/"}


def _valid_clip_name(name):
    # Only serve objects under the clip prefixes; reject traversal.
    return name.startswith(tuple(_CATEGORIES.values())) and ".." not in name


@app.route("/keep", methods=["POST"])
def keep():
    """Move a clip out of the auto-expiring detections/ area into saved/."""
    if not stream.saver.available:
        return jsonify({"ok": False, "error": "storage disabled"}), 503
    name = request.form.get("name", "")
    if not name.startswith("detections/") or ".." in name:
        return jsonify({"ok": False, "error": "not a detections clip"}), 400
    new_key = stream.saver.keep_clip(name)
    return jsonify({"ok": bool(new_key), "saved_as": new_key}), (200 if new_key else 500)


@app.route("/categorize", methods=["POST"])
def categorize():
    """Move a clip between categories: detections / saved / evidence."""
    if not stream.saver.available:
        return jsonify({"ok": False, "error": "storage disabled"}), 503
    name = request.form.get("name", "")
    dest = _CATEGORIES.get(request.form.get("to", ""))
    if not dest or not _valid_clip_name(name):
        return jsonify({"ok": False, "error": "bad request"}), 400
    new_key = stream.saver.recategorize(name, dest)
    return jsonify({"ok": bool(new_key), "key": new_key}), (200 if new_key else 500)


def _parse_range(header, size):
    """Parse a single-range 'bytes=start-end' header into (start, end)
    inclusive, clamped to the object size. Returns None if unsatisfiable."""
    if not header or not header.startswith("bytes="):
        return None
    spec = header[6:].split(",")[0].strip()
    start_s, _, end_s = spec.partition("-")
    try:
        if start_s == "":  # suffix range: last N bytes
            n = int(end_s)
            if n <= 0:
                return None
            start, end = max(0, size - n), size - 1
        else:
            start = int(start_s)
            end = int(end_s) if end_s else size - 1
    except ValueError:
        return None
    end = min(end, size - 1)
    if start > end or start >= size:
        return None
    return start, end


@app.route("/clip/<path:name>", methods=["GET", "DELETE"])
def clip(name):
    """GET streams one stored clip through the app (MinIO isn't
    browser-reachable), with HTTP Range so <video> can seek and preview a
    poster frame. DELETE removes that one clip."""
    if not stream.saver.available:
        return jsonify({"error": "storage disabled"}), 503
    if not _valid_clip_name(name):
        return jsonify({"error": "not found"}), 404

    if request.method == "DELETE":
        ok = stream.saver.remove_clip(name)
        return jsonify({"ok": ok}), (200 if ok else 500)

    info = stream.saver.stat_clip(name)
    if info is None:
        return jsonify({"error": "not found"}), 404
    size, content_type = info

    rng = _parse_range(request.headers.get("Range"), size)
    if rng is not None:
        start, end = rng
        length = end - start + 1
        resp = Response(stream.saver.stream_clip(name, start, length),
                        status=206, content_type=content_type)
        resp.headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        resp.headers["Content-Length"] = str(length)
    else:
        resp = Response(stream.saver.stream_clip(name), content_type=content_type)
        resp.headers["Content-Length"] = str(size)
    resp.headers["Accept-Ranges"] = "bytes"
    return resp


@app.route("/detect", methods=["POST"])
def detect_toggle():
    if not stream.detector.available:
        return jsonify({"ok": False, "error": "model not loaded"}), 503
    want = request.form.get("enabled", "").lower()
    stream.detect_enabled = want in ("1", "true", "on", "yes")
    if not stream.detect_enabled:
        with stream._lock:
            stream._boxes = []
            stream.person_count = 0
    return jsonify({"ok": True, "detect_enabled": stream.detect_enabled})


@app.route("/motion", methods=["POST"])
def motion_toggle():
    if not stream.motion.available:
        return jsonify({"ok": False, "error": "motion detection unavailable"}), 503
    want = request.form.get("enabled", "").lower()
    stream.motion_enabled = want in ("1", "true", "on", "yes")
    if not stream.motion_enabled:
        with stream._lock:
            stream._motion_boxes = []
            stream.motion_count = 0
    return jsonify({"ok": True, "motion_enabled": stream.motion_enabled})


@app.route("/animal", methods=["POST"])
def animal_toggle():
    if not stream.detector.available:
        return jsonify({"ok": False, "error": "model not loaded"}), 503
    want = request.form.get("enabled", "").lower()
    stream.animal_enabled = want in ("1", "true", "on", "yes")
    if not stream.animal_enabled:
        with stream._lock:
            stream._animal_boxes = []
            stream.animal_count = 0
    return jsonify({"ok": True, "animal_enabled": stream.animal_enabled})


@app.route("/light", methods=["POST"])
def light_toggle():
    """Turn the camera's illuminator on/off via ESPHome's REST API."""
    if not LIGHT_URL:
        return jsonify({"ok": False, "error": "no light configured"}), 503
    want = request.form.get("on", "").lower() in ("1", "true", "on", "yes")
    url = f"{LIGHT_URL}/{'turn_on' if want else 'turn_off'}"
    try:
        req = urllib.request.Request(url, data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=5) as resp:
            ok = resp.status < 300
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:120]}), 502
    if ok:
        light_state["on"] = want
    return jsonify({"ok": ok, "light_on": light_state["on"]})


@app.route("/record", methods=["POST"])
def record():
    """Record a clip on demand and save it to the gallery (pre-roll from the
    buffer + post-roll). Non-blocking: capture runs on a background thread."""
    if stream.latest_jpeg() is None:
        return jsonify({"ok": False, "error": "no stream"}), 409

    def _job():
        count = stream.person_count
        data, ext = stream.capture_clip_now()
        if data:
            stream.saver.save(data, count, time.time(),
                              ext=ext, content_type="video/mp4")

    threading.Thread(target=_job, daemon=True).start()
    return jsonify({"ok": True, "seconds": int(CLIP_PRE + CLIP_POST),
                    "saved": stream.saver.available})


@app.route("/video_feed")
def video_feed():
    def generate():
        for jpeg in stream.frames():
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                + jpeg + b"\r\n"
            )

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

# Chromecast casting. The Chromecast fetches the HLS playlist itself, so it
# needs an address it can reach: CAST_BASE_URL overrides the guessed LAN
# address (e.g. when the app sits behind a reverse proxy).
CAST_BASE_URL = os.environ.get("CAST_BASE_URL", "").strip().rstrip("/")
caster = Caster()


@app.route("/cast/devices")
def cast_devices():
    return jsonify({"devices": discover(), "casting": caster.device})


@app.route("/cast/start", methods=["POST"])
def cast_start():
    device = request.form.get("device", "").strip()
    if not device:
        return jsonify({"ok": False, "error": "no device given"}), 400
    if stream.latest_jpeg() is None:
        return jsonify({"ok": False, "error": "no stream connected"}), 409
    base = CAST_BASE_URL or f"http://{lan_ip()}:{request.environ.get('SERVER_PORT')}"
    try:
        caster.start(device, stream.frames(), base)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 502
    return jsonify({"ok": True, "device": device})


@app.route("/cast/stop", methods=["POST"])
def cast_stop():
    caster.stop()
    return jsonify({"ok": True})


@app.route("/cast/hls/<token>/<path:name>", endpoint="cast_hls")
def cast_hls(token, name):
    # Public (no session) but gated by the per-cast random token in the URL.
    d = caster.hls_dir(token)
    if d is None:
        return Response("not found\n", status=404, mimetype="text/plain")
    resp = send_from_directory(d, name, max_age=0)
    resp.headers["Access-Control-Allow-Origin"] = "*"  # Cast receiver fetches cross-origin
    resp.headers["Cache-Control"] = "no-store"
    return resp


if __name__ == "__main__":
    print("=" * 60)
    print("ViperVision login:")
    print(f"  username: {AUTH_USER}")
    if AUTH_PASSWORD_GENERATED:
        print(f"  password: {AUTH_PASSWORD}   (generated — set VIPERVISION_PASSWORD to choose your own)")
    else:
        print("  password: (from VIPERVISION_PASSWORD)")
    print("=" * 60, flush=True)

    initial_url = os.environ.get("RTSP_URL", "rtsp://192.168.64.106:554/onvif1")
    if initial_url:
        stream.start(initial_url)
    app.run(
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "5000")),
        threaded=True,
        debug=False,
    )
