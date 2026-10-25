# ViperVision

A small Python web app that connects to an RTSP camera stream and displays
the live feed in your browser as an MJPEG stream.

## Login

Every page and endpoint is behind a session login. On startup the app prints
the credentials:

```text
ViperVision login:
  username: admin
  password: <generated>   (set VIPERVISION_PASSWORD to choose your own)
```

If `VIPERVISION_PASSWORD` isn't set, a random password is generated and printed
each start. Set `VIPERVISION_USER` / `VIPERVISION_PASSWORD` to pin your own, and
`SECRET_KEY` to keep sessions valid across restarts (otherwise a restart logs
everyone out). Unauthenticated API/stream calls get `401`; the browser is
redirected to the login page. There's a "Sign out" button in the UI.

> The login protects access, but the dev server speaks plain HTTP. On an
> untrusted network, put it behind a TLS-terminating reverse proxy so the
> password and session cookie aren't sent in the clear.

## How it works

- A background thread opens the RTSP source with OpenCV (FFmpeg backend),
  reads frames, and JPEG-encodes them, keeping only the newest frame.
- Flask serves the frames at `/video_feed` using
  `multipart/x-mixed-replace` (MJPEG), which every browser renders natively
  in an `<img>` tag — no client-side plugins needed.
- If the stream drops or fails to connect, the reader retries every 3 seconds.

## Human detection

The feed is scanned for people using a YOLOX model (COCO-trained, "person"
class only) via OpenCV's DNN module — CPU-only, no GPU or extra services
needed. Detected people get a green box and confidence label, and a live
"Humans: N" count shows in the UI and in `GET /status` (`person_count`).

Inference runs on a background thread throttled to `DETECT_INTERVAL` seconds
(default 0.5), so it never slows the streamed frame rate; the reader just
overlays the most recent result. Toggle it from the UI checkbox or
`POST /detect` with `enabled=true|false`.

The model file (`models/object_detection_yolox.onnx`, ~34 MB) is loaded at
startup. If it's missing, the app still streams and simply reports detection
as unavailable. Override the path with the `PERSON_MODEL` env var.

## Pan/tilt (PTZ)

Arrow controls are overlaid on the bottom-right of the video. A tap moves the
camera one step; press-and-hold repeats steps until released. This uses the
pseudo-RTSP `ptzCmd` extension that Yoosee-style cameras accept on the RTSP
port (a `SET_PARAMETER` request carrying `ptzCmd:UP|DWON|LEFT|RIGHT` — the
`DWON` misspelling is what the firmware expects). No credentials are needed.
The endpoint is `POST /ptz/<up|down|left|right>`.

## Run with Docker

```bash
# 1. Edit docker-compose.yml: set VIPERVISION_PASSWORD, SECRET_KEY, and RTSP_URL
# 2. Build and start
docker compose up -d --build
# 3. Watch logs (shows the login credentials if you didn't set a password)
docker compose logs -f
```

Then open `http://<host>:5000`.

The compose file uses **host networking** on purpose: this camera only speaks
RTP-over-UDP, and UDP return traffic is unreliable through Docker's bridge NAT.
Host mode also lets the container reach a camera on your LAN directly. To use a
bridge network instead, remove `network_mode: host`, uncomment the `ports:`
block, and note that UDP RTSP may not work (set `RTSP_TRANSPORT=tcp` only if
your camera supports TCP — this one does not).

The YOLOX model is baked into the image, so the container is self-contained.

## Setup (without Docker)

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
# Option A: set the stream URL up front
RTSP_URL=rtsp://user:pass@192.168.1.100:554/stream1 .venv/bin/python app.py

# Option B: just start it and enter the URL in the web UI
.venv/bin/python app.py
```

Then open <http://localhost:5000>

## Configuration (environment variables)

| Variable       | Default   | Description                          |
|----------------|-----------|--------------------------------------|
| `RTSP_URL`     | (none)    | Stream to connect to at startup      |
| `HOST`         | `0.0.0.0` | Bind address                         |
| `PORT`         | `5000`    | HTTP port                            |
| `JPEG_QUALITY` | `80`      | JPEG encode quality (1–100)          |

## Notes

- Multiple browser tabs can watch the same stream; frames are decoded and
  encoded once and fanned out to all viewers.
- Uses `opencv-python-headless`, so it runs fine on servers without a display.
- For production use, run behind a real WSGI server that supports streaming
  responses with multiple workers/threads (e.g. `gunicorn --threads 8 app:app`).

## Casting to a Chromecast

The **📺 Cast** button finds Chromecasts on the LAN (mDNS), then streams the
annotated live feed to the one you pick; click it again to stop. Chromecasts
can't play MJPEG, so ffmpeg re-encodes the feed to a rolling H.264 HLS
playlist, served at `/cast/hls/<random-token>/live.m3u8`. The token (new for
each cast) is the only protection on that URL, since the Chromecast can't log
in. Expect a few seconds of latency.

The Chromecast must be able to reach the app: the LAN address is guessed
automatically; set `CAST_BASE_URL` (e.g. `http://192.168.1.10:5001`) to
override it. In Docker, host networking (as in `docker-compose.yml`) is needed
for discovery.
