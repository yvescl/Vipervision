"""Detection notifications via Telegram.

Enabled only when TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set; otherwise
every call is a no-op. Sends the annotated snapshot with a caption. Delivery
runs on a background worker thread fed by a bounded queue, so a slow or
unreachable Telegram never blocks the detection loop.

Uses only the standard library (urllib) — no extra dependency.
"""

import os
import queue
import threading
import time
import urllib.request

from metrics import metrics


class TelegramNotifier:
    API = "https://api.telegram.org"

    def __init__(self):
        self._token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        self._chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        self.available = bool(self._token and self._chat)
        self._last_error = None
        if not self.available:
            return
        self._queue = queue.Queue(maxsize=16)
        threading.Thread(target=self._worker, daemon=True).start()
        print("[notify] Telegram notifications enabled", flush=True)

    # Per-media Telegram method, form field, filename and MIME type.
    _MEDIA = {
        "photo": ("sendPhoto", "photo", "detection.jpg", "image/jpeg"),
        "video": ("sendVideo", "video", "detection.mp4", "video/mp4"),
    }

    def notify(self, data, person_count, timestamp=None, media="photo", label=None):
        """Queue a detection photo or video for delivery. `label` overrides the
        default person caption (e.g. "🐱 cat detected"). Non-blocking."""
        if not self.available:
            return
        item = (bytes(data), int(person_count), timestamp or time.time(), media, label)
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            pass  # behind on delivery; drop rather than block detection

    def _worker(self):
        while True:
            data, count, ts, media, label = self._queue.get()
            try:
                self._send(media, data, count, ts, label)
                metrics.inc("vipervision_telegram_sent_total")
                self._last_error = None
            except Exception as e:  # never let a send error kill the worker
                metrics.inc("vipervision_telegram_errors_total")
                if str(e) != self._last_error:
                    print(f"[notify] send failed: {e}", flush=True)
                    self._last_error = str(e)
                time.sleep(2)
            finally:
                self._queue.task_done()

    @staticmethod
    def _multipart(fields, field_name, filename, content_type, file_bytes):
        """Build a multipart/form-data body for text fields + one file."""
        boundary = "----ViperVision" + os.urandom(8).hex()
        pre = []
        for name, value in fields.items():
            pre.append(f"--{boundary}\r\n"
                       f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                       f"{value}\r\n")
        head = "".join(pre).encode()
        file_head = (f"--{boundary}\r\n"
                     f'Content-Disposition: form-data; name="{field_name}"; '
                     f'filename="{filename}"\r\n'
                     f"Content-Type: {content_type}\r\n\r\n").encode()
        tail = f"\r\n--{boundary}--\r\n".encode()
        body = head + file_head + file_bytes + tail
        return boundary, body

    def _send(self, media, data, count, ts, label=None):
        method, field, filename, content_type = self._MEDIA[media]
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        if label:
            caption = f"{label}\n{when}"
        else:
            people = "person" if count == 1 else "people"
            caption = f"\U0001F6A8 ViperVision — {count} {people} detected\n{when}"
        fields = {"chat_id": self._chat, "caption": caption}
        if media == "video":
            fields["supports_streaming"] = "true"
        boundary, body = self._multipart(fields, field, filename,
                                         content_type, data)
        req = urllib.request.Request(
            f"{self.API}/bot{self._token}/{method}", data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}")
