FROM python:3.14-slim

# opencv-python-headless's wheels link the OpenMP runtime (libgomp); the
# headless build needs no X11/OpenGL system libraries. ffmpeg (with libx264)
# encodes detection clips to H.264 MP4 — OpenCV's bundled ffmpeg lacks a
# software H.264 encoder, so we shell out to the CLI.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first so the layer is cached across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --require-hashes -r requirements.txt

# Application code, templates, and the detection model.
COPY app.py detector.py storage.py notify.py clip.py cast.py metrics.py telegram_bot.py ./
COPY templates ./templates
COPY models ./models

EXPOSE 5000

# The built-in threaded Flask server is sufficient for this single-user app
# and preserves the startup behaviour (auto-connect + printed credentials).
CMD ["python", "app.py"]
