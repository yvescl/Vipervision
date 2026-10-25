"""Detection snapshot storage — uploads annotated JPEGs to MinIO (S3).

Enabled only when MINIO_ENDPOINT and credentials are set; otherwise every
call is a no-op so the app runs fine without object storage. Uploads happen
on a background worker thread fed by a bounded queue, so a slow or unreachable
MinIO never blocks the detection loop (frames are dropped, not buffered
without limit).
"""

import os
import queue
import threading
import time

from metrics import metrics

try:
    from minio import Minio
    _MINIO_IMPORT_OK = True
except ImportError:  # package not installed
    _MINIO_IMPORT_OK = False


def _truthy(value):
    return str(value).lower() in ("1", "true", "yes", "on")


class DetectionSaver:
    def __init__(self):
        self._endpoint = os.environ.get("MINIO_ENDPOINT", "").strip()
        access = os.environ.get("MINIO_ACCESS_KEY", "").strip()
        secret = os.environ.get("MINIO_SECRET_KEY", "").strip()
        self._bucket = os.environ.get("MINIO_BUCKET", "vipervision-detections").strip()
        self._secure = _truthy(os.environ.get("MINIO_SECURE", "0"))

        self.available = bool(_MINIO_IMPORT_OK and self._endpoint and access and secret)
        self._client = None
        self._bucket_ready = False
        self._last_error = None

        if not self.available:
            if self._endpoint and not _MINIO_IMPORT_OK:
                print("[storage] MINIO_ENDPOINT set but 'minio' package missing; "
                      "detection saving disabled.", flush=True)
            return

        self._client = Minio(self._endpoint, access_key=access,
                             secret_key=secret, secure=self._secure)
        # Bounded queue: if uploads can't keep up we drop the newest rather
        # than grow memory without limit.
        self._queue = queue.Queue(maxsize=32)
        threading.Thread(target=self._worker, daemon=True).start()
        print(f"[storage] MinIO detection saving enabled -> "
              f"{'https' if self._secure else 'http'}://{self._endpoint}/{self._bucket}",
              flush=True)

    def save(self, data, person_count, timestamp=None,
             ext="jpg", content_type="image/jpeg", tag="p"):
        """Queue an annotated detection (image or video) for upload.
        `tag` is the one-letter kind in the key (p=person, c=cat). Non-blocking."""
        if not self.available:
            return
        ts = timestamp or time.time()
        item = (bytes(data), int(person_count), ts, ext, content_type, tag)
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            pass  # storage is behind; drop this snapshot rather than block

    def list_clips(self, prefix="detections/", cap=5000):
        """Return stored detections under `prefix`, newest first (capped at
        `cap`): list of {name, size, modified}. The caller paginates. Empty
        list when storage is off/unreachable."""
        if not self.available:
            return []
        try:
            self._ensure_bucket()
            items = []
            for o in self._client.list_objects(self._bucket, prefix=prefix,
                                                recursive=True):
                if o.is_dir:
                    continue
                items.append({
                    "name": o.object_name,
                    "size": o.size,
                    "modified": o.last_modified.timestamp() if o.last_modified else 0,
                })
                if len(items) >= cap:
                    break
            # Sort by the event time encoded in the key (…/YYYYMMDD-HHMMSS-…),
            # newest first. More reliable than last_modified, which reflects
            # upload/copy time — a bulk re-key (e.g. a timezone restamp)
            # collapses every last_modified to the same instant and would
            # scramble the order.
            items.sort(key=lambda x: x["name"], reverse=True)
            return items
        except Exception as e:
            if str(e) != self._last_error:
                print(f"[storage] list failed: {e}", flush=True)
                self._last_error = str(e)
            return []

    def remove_clip(self, name):
        """Delete one stored clip. Returns True on success."""
        if not self.available:
            return False
        try:
            self._client.remove_object(self._bucket, name)
            return True
        except Exception as e:
            print(f"[storage] delete failed: {e}", flush=True)
            return False

    # Clip categories (key prefixes). Only detections/ auto-expires; saved/ and
    # evidence/ are kept until deleted.
    CATEGORIES = ("detections/", "saved/", "evidence/")

    def keep_clip(self, name, base="detections/", dest="saved/"):
        """Move a clip out of the auto-expiring detections/ prefix into saved/.
        Returns the new key or None. (Kept for the /keep endpoint.)"""
        if not name.startswith(base):
            return None
        return self.recategorize(name, dest)

    def recategorize(self, name, dest):
        """Move a clip from its current category prefix to `dest` (one of
        CATEGORIES). Returns the new key, or None on failure."""
        if not self.available or dest not in self.CATEGORIES:
            return None
        for base in self.CATEGORIES:
            if name.startswith(base):
                new_key = dest + name[len(base):]
                if new_key == name:
                    return name
                try:
                    from minio.commonconfig import CopySource
                    self._client.copy_object(self._bucket, new_key,
                                             CopySource(self._bucket, name))
                    self._client.remove_object(self._bucket, name)
                    return new_key
                except Exception as e:
                    print(f"[storage] recategorize failed: {e}", flush=True)
                    return None
        return None

    def stat_clip(self, name):
        """Return (size, content_type) for one object, or None if missing."""
        if not self.available:
            return None
        try:
            st = self._client.stat_object(self._bucket, name)
            return st.size, (st.content_type or "application/octet-stream")
        except Exception:
            return None

    def stream_clip(self, name, offset=0, length=0):
        """Yield an object's bytes in chunks (length=0 means to end-of-object).
        The MinIO connection is always released, even on client disconnect."""
        resp = self._client.get_object(self._bucket, name,
                                       offset=offset, length=length)
        try:
            yield from resp.stream(64 * 1024)
        finally:
            resp.close()
            resp.release_conn()

    def _ensure_bucket(self):
        if self._bucket_ready:
            return True
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)
        self._bucket_ready = True
        return True

    def _worker(self):
        import io
        while True:
            data, count, ts, ext, content_type, tag = self._queue.get()
            try:
                self._ensure_bucket()
                lt = time.localtime(ts)
                name = (f"detections/{time.strftime('%Y/%m/%d', lt)}/"
                        f"{time.strftime('%Y%m%d-%H%M%S', lt)}-{count}{tag}.{ext}")
                self._client.put_object(
                    self._bucket, name, io.BytesIO(data), length=len(data),
                    content_type=content_type,
                    metadata={"X-Amz-Meta-Person-Count": str(count)},
                )
                metrics.inc("vipervision_clips_uploaded_total")
                self._last_error = None
            except Exception as e:  # never let an upload error kill the worker
                metrics.inc("vipervision_clip_upload_errors_total")
                if str(e) != self._last_error:
                    print(f"[storage] upload failed: {e}", flush=True)
                    self._last_error = str(e)
                time.sleep(2)  # back off before the next item
            finally:
                self._queue.task_done()
