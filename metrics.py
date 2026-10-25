"""Prometheus metrics for ViperVision.

Exposes a handful of counters and gauges in the Prometheus text exposition
format, served at /metrics. Stdlib-only — we only need to increment a few
counters and render live gauges on scrape, so a full prometheus_client
dependency isn't worth it (matches notify.py's no-extra-dependency approach).

Counters are process-cumulative and thread-safe. Gauges are read live from the
stream/saver/notifier at scrape time and passed to render().
"""

import threading

# Counter names + HELP text. Code increments these by name; render() emits a
# `# TYPE ... counter` block for each so unseen counters still report 0.
COUNTERS = {
    "vipervision_frames_total":
        "Video frames read from the RTSP source and streamed.",
    "vipervision_inferences_total":
        "Person-detection inferences run.",
    "vipervision_clips_captured_total":
        "Detection clips encoded after a sustained person event.",
    "vipervision_animal_captures_total":
        "Animal (e.g. cat) snapshots captured after a sustained animal event.",
    "vipervision_motion_captures_total":
        "Clips captured after a sustained motion event.",
    "vipervision_clips_uploaded_total":
        "Detection clips successfully uploaded to object storage.",
    "vipervision_clip_upload_errors_total":
        "Detection clip uploads that failed.",
    "vipervision_telegram_sent_total":
        "Detection alerts successfully delivered to Telegram.",
    "vipervision_telegram_errors_total":
        "Detection alert deliveries to Telegram that failed.",
    "vipervision_reconnects_total":
        "RTSP reconnect attempts (connect failures + dropped streams).",
}


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self._counters = {name: 0.0 for name in COUNTERS}

    def inc(self, name, amount=1.0):
        """Add to a counter. Unknown names are ignored (never raise on a
        hot path)."""
        with self._lock:
            if name in self._counters:
                self._counters[name] += amount

    def render(self, gauges):
        """Render the full exposition. `gauges` is an iterable of
        (name, help_text, value) tuples read live at scrape time."""
        out = []
        for name, help_text, value in gauges:
            out.append(f"# HELP {name} {help_text}")
            out.append(f"# TYPE {name} gauge")
            out.append(f"{name} {_fmt(value)}")
        with self._lock:
            counters = dict(self._counters)
        for name, help_text in COUNTERS.items():
            out.append(f"# HELP {name} {help_text}")
            out.append(f"# TYPE {name} counter")
            out.append(f"{name} {_fmt(counters[name])}")
        return "\n".join(out) + "\n"


def _fmt(value):
    # Emit whole numbers without a trailing .0 for readability; Prometheus
    # accepts either form.
    f = float(value)
    return str(int(f)) if f.is_integer() else repr(f)


# Single shared instance imported by the app and the worker modules.
metrics = Metrics()
