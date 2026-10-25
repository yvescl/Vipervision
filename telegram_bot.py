"""Inbound Telegram command bot for ViperVision.

Long-polls the Telegram Bot API (getUpdates) on a background thread and
dispatches slash-commands to handlers supplied by the app. Uses only the
standard library (urllib), matching notify.py.

Security: commands are accepted ONLY from the configured TELEGRAM_CHAT_ID.
Messages from any other chat are ignored — the bot can move the camera and
toggle detection, so this whitelist is the authorization boundary. Messages
sent before the bot started are skipped so a restart never replays old
commands.

Enabled when TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID are set and
TELEGRAM_COMMANDS is not "0". Delivery of outbound alerts still lives in
notify.py; this module only handles the inbound direction.
"""

import json
import os
import threading
import time
import urllib.parse
import urllib.request

from notify import TelegramNotifier


def _truthy(value):
    return str(value).lower() not in ("0", "false", "no", "off", "")


class TelegramCommandBot:
    API = "https://api.telegram.org"
    POLL_TIMEOUT = 30  # long-poll seconds

    def __init__(self, handlers):
        self._token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        self._chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        self._enabled = _truthy(os.environ.get("TELEGRAM_COMMANDS", "1"))
        self._handlers = handlers  # {command: callable(args) -> reply}
        self.available = bool(self._token and self._chat and self._enabled
                              and handlers)
        self._start = time.time()
        self._last_error = None
        if not self.available:
            return
        self._register_commands()
        threading.Thread(target=self._worker, daemon=True).start()
        print("[telegram] command bot enabled", flush=True)

    # --- Telegram API helpers ---------------------------------------------
    def _api(self, method, params=None, timeout=None):
        url = f"{self.API}/bot{self._token}/{method}"
        data = urllib.parse.urlencode(params).encode() if params else None
        req = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(req, timeout=timeout or 15) as resp:
            return json.loads(resp.read().decode())

    def send_message(self, text):
        try:
            self._api("sendMessage",
                      {"chat_id": self._chat, "text": text,
                       "disable_web_page_preview": "true"})
        except Exception as e:
            self._log_error(e)

    def send_photo(self, jpeg_bytes, caption=""):
        boundary, body = TelegramNotifier._multipart(
            {"chat_id": self._chat, "caption": caption},
            "photo", "snapshot.jpg", "image/jpeg", bytes(jpeg_bytes))
        req = urllib.request.Request(
            f"{self.API}/bot{self._token}/sendPhoto", data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        try:
            with urllib.request.urlopen(req, timeout=60):
                pass
        except Exception as e:
            self._log_error(e)

    def send_video(self, mp4_bytes, caption=""):
        boundary, body = TelegramNotifier._multipart(
            {"chat_id": self._chat, "caption": caption,
             "supports_streaming": "true"},
            "video", "clip.mp4", "video/mp4", bytes(mp4_bytes))
        req = urllib.request.Request(
            f"{self.API}/bot{self._token}/sendVideo", data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        try:
            with urllib.request.urlopen(req, timeout=120):
                pass
        except Exception as e:
            self._log_error(e)

    def _register_commands(self):
        """Publish the command list so Telegram shows an autocomplete menu."""
        commands = [
            {"command": "help", "description": "Show available commands"},
            {"command": "status", "description": "Stream & detection status"},
            {"command": "snapshot", "description": "Send the current frame"},
            {"command": "clip", "description": "Record & send a short video clip"},
            {"command": "detect", "description": "Human detection on|off"},
            {"command": "motion", "description": "Motion detection on|off"},
            {"command": "ptz", "description": "Move camera up|down|left|right"},
        ]
        try:
            self._api("setMyCommands",
                      {"commands": json.dumps(commands)}, timeout=10)
        except Exception as e:
            self._log_error(e)

    def _log_error(self, e):
        if str(e) != self._last_error:
            print(f"[telegram] {e}", flush=True)
            self._last_error = str(e)

    # --- Polling loop ------------------------------------------------------
    def _worker(self):
        offset = None
        while True:
            try:
                params = {"timeout": self.POLL_TIMEOUT}
                if offset is not None:
                    params["offset"] = offset
                resp = self._api("getUpdates", params,
                                 timeout=self.POLL_TIMEOUT + 10)
                self._last_error = None
                for update in resp.get("result", []):
                    offset = update["update_id"] + 1
                    self._handle_update(update)
            except Exception as e:
                # Network blip or 409 (another poller/webhook). Back off.
                self._log_error(e)
                time.sleep(5)

    def _handle_update(self, update):
        msg = update.get("message") or update.get("edited_message")
        if not msg:
            return
        # Authorization: only the configured chat may command the system.
        if str(msg.get("chat", {}).get("id")) != self._chat:
            return
        # Skip messages sent before the bot started (avoid replaying on restart).
        if msg.get("date", 0) < self._start - 5:
            return
        text = (msg.get("text") or "").strip()
        if not text.startswith("/"):
            return
        parts = text.split()
        # Strip a "@botname" suffix Telegram adds in groups: /status@MyBot.
        cmd = parts[0][1:].split("@", 1)[0].lower()
        args = parts[1:]
        handler = self._handlers.get(cmd)
        if handler is None:
            self.send_message(f"Unknown command: /{cmd}\nSend /help for the list.")
            return
        try:
            reply = handler(args)
        except Exception as e:
            self.send_message(f"Command failed: {e}")
            return
        self._dispatch_reply(reply)

    def _dispatch_reply(self, reply):
        """A handler returns either a text string, or a ('photo', bytes,
        caption) tuple, or None (no reply)."""
        if reply is None:
            return
        if isinstance(reply, tuple) and reply and reply[0] == "photo":
            _, data, caption = reply
            if data:
                self.send_photo(data, caption)
            else:
                self.send_message(caption or "No frame available.")
        else:
            self.send_message(str(reply))
