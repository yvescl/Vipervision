"""Human (person) detection for ViperVision using YOLOX (OpenCV DNN).

Model: object_detection_yolox from the OpenCV model zoo, COCO-trained.
Only the "person" class (id 0) is reported. Runs on CPU via cv2.dnn.
"""

import os

import cv2
import numpy as np

MODEL_PATH = os.environ.get(
    "PERSON_MODEL",
    os.path.join(os.path.dirname(__file__), "models", "object_detection_yolox.onnx"),
)
INPUT_SIZE = 640
STRIDES = [8, 16, 32]

# COCO animal class ids (the model is COCO-trained) -> display names. Used by
# the optional animal-detection mode; cat is 15.
COCO_ANIMALS = {
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep",
    19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe",
}
ANIMAL_IDS = frozenset(COCO_ANIMALS)


class PersonDetector:
    """Detects people in BGR frames. Returns pixel-space boxes.

    Loads lazily and degrades gracefully: if the model file is missing or
    fails to load, `available` is False and `detect` returns an empty list,
    so the app still streams without detection.
    """

    def __init__(self, conf_threshold=0.5, nms_threshold=0.45):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self._grids, self._strides = self._make_grid()
        self.available = False
        self._net = None
        if os.path.exists(MODEL_PATH):
            try:
                self._net = cv2.dnn.readNetFromONNX(MODEL_PATH)
                self.available = True
            except cv2.error:
                self._net = None

    @staticmethod
    def _make_grid():
        grids, strides = [], []
        for s in STRIDES:
            hs = ws = INPUT_SIZE // s
            xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((hs * ws, 1), s))
        return np.concatenate(grids, 0), np.concatenate(strides, 0)

    def _letterbox(self, img):
        h, w = img.shape[:2]
        r = min(INPUT_SIZE / h, INPUT_SIZE / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, np.uint8)
        padded[:nh, :nw] = cv2.resize(img, (nw, nh))
        return padded, r

    def detect(self, frame):
        """Return a list of (x, y, w, h, confidence) person boxes in pixels."""
        if not self.available:
            return []
        padded, r = self._letterbox(frame)
        blob = cv2.dnn.blobFromImage(padded)
        self._net.setInput(blob)
        out = self._net.forward()[0].copy()  # (8400, 85)

        out[:, :2] = (out[:, :2] + self._grids) * self._strides
        out[:, 2:4] = np.exp(out[:, 2:4]) * self._strides
        scores = out[:, 4:5] * out[:, 5:]
        cls = scores.argmax(1)
        conf = scores.max(1)
        keep = (cls == 0) & (conf > self.conf_threshold)

        boxes, confs = [], []
        for (cx, cy, bw, bh), c in zip(out[keep, :4], conf[keep]):
            boxes.append(
                [int((cx - bw / 2) / r), int((cy - bh / 2) / r),
                 int(bw / r), int(bh / r)]
            )
            confs.append(float(c))
        if not boxes:
            return []
        idx = cv2.dnn.NMSBoxes(boxes, confs, self.conf_threshold, self.nms_threshold)
        if len(idx) == 0:
            return []
        return [(*boxes[i], confs[i]) for i in np.array(idx).flatten()]

    def detect_objects(self, frame, class_ids):
        """Detect any of the given COCO class ids in one forward pass. Returns
        a list of (x, y, w, h, confidence, class_id). NMS is run per class so
        an overlapping person and cat don't suppress each other."""
        if not self.available or not class_ids:
            return []
        padded, r = self._letterbox(frame)
        blob = cv2.dnn.blobFromImage(padded)
        self._net.setInput(blob)
        out = self._net.forward()[0].copy()  # (8400, 85)

        out[:, :2] = (out[:, :2] + self._grids) * self._strides
        out[:, 2:4] = np.exp(out[:, 2:4]) * self._strides
        scores = out[:, 4:5] * out[:, 5:]
        cls = scores.argmax(1)
        conf = scores.max(1)

        results = []
        for cid in class_ids:
            keep = (cls == cid) & (conf > self.conf_threshold)
            if not keep.any():
                continue
            boxes, confs = [], []
            for (cx, cy, bw, bh), c in zip(out[keep, :4], conf[keep]):
                boxes.append(
                    [int((cx - bw / 2) / r), int((cy - bh / 2) / r),
                     int(bw / r), int(bh / r)]
                )
                confs.append(float(c))
            idx = cv2.dnn.NMSBoxes(boxes, confs, self.conf_threshold,
                                   self.nms_threshold)
            for i in (np.array(idx).flatten() if len(idx) else []):
                results.append((*boxes[i], confs[i], cid))
        return results


class RodentDetector:
    """Optional rodent (mouse/rat) detector using a YOLOv8 ONNX model.

    COCO has no rodent class, so this loads a separate single-purpose model
    (e.g. a rat/mouse model exported to ONNX from Ultralytics/Roboflow). YOLOv8
    output differs from YOLOX — shape (1, 4+nc, N), no objectness, and it needs
    a transpose — so it has its own decode. Set RODENT_MODEL to the .onnx path;
    if the file is missing `available` is False and detect() returns [] so the
    app runs unchanged.
    """

    def __init__(self, conf_threshold=0.35, nms_threshold=0.45):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.available = False
        self._net = None
        path = os.environ.get(
            "RODENT_MODEL",
            os.path.join(os.path.dirname(__file__), "models", "rodent.onnx"))
        if os.path.exists(path):
            try:
                self._net = cv2.dnn.readNetFromONNX(path)
                self.available = True
            except cv2.error:
                self._net = None

    def _letterbox(self, img):
        h, w = img.shape[:2]
        r = min(INPUT_SIZE / h, INPUT_SIZE / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, np.uint8)
        padded[:nh, :nw] = cv2.resize(img, (nw, nh))
        return padded, r

    def _decode(self, out, r):
        """Decode a raw YOLOv8 forward output into (x, y, w, h, conf) boxes.
        `out` is the network output; `r` the letterbox scale. Kept separate so
        it can be unit-tested without a model."""
        out = np.squeeze(out)                    # (4+nc, N) or (N, 4+nc)
        if out.ndim != 2:
            return []
        if out.shape[0] < out.shape[1]:          # (4+nc, N) -> (N, 4+nc)
            out = out.T
        xywh = out[:, :4]
        scores = out[:, 4:]
        conf = scores.max(1)
        keep = conf > self.conf_threshold
        boxes, confs = [], []
        for (cx, cy, w, h), c in zip(xywh[keep], conf[keep]):
            boxes.append([int((cx - w / 2) / r), int((cy - h / 2) / r),
                          int(w / r), int(h / r)])
            confs.append(float(c))
        if not boxes:
            return []
        idx = cv2.dnn.NMSBoxes(boxes, confs, self.conf_threshold,
                               self.nms_threshold)
        if len(idx) == 0:
            return []
        return [(*boxes[i], confs[i]) for i in np.array(idx).flatten()]

    def detect(self, frame):
        """Return a list of (x, y, w, h, confidence) rodent boxes in pixels."""
        if not self.available:
            return []
        padded, r = self._letterbox(frame)
        # YOLOv8 wants RGB, 0-1 normalised, 640x640.
        blob = cv2.dnn.blobFromImage(padded, 1 / 255.0,
                                     (INPUT_SIZE, INPUT_SIZE), swapRB=True,
                                     crop=False)
        self._net.setInput(blob)
        return self._decode(self._net.forward(), r)


class MotionDetector:
    """Cheap motion detection via MOG2 background subtraction.

    Pure OpenCV (no model), so it runs anywhere cv2 does and costs a fraction
    of the person detector. Returns bounding boxes of moving regions. It is
    stateful — one instance per stream — because it maintains a running
    background model, so give it frames in order.
    """

    def __init__(self, min_area=None, var_threshold=None):
        # Sensitivity knobs (all tunable via env):
        #  MOTION_VAR_THRESHOLD — higher = ignore subtler pixel changes.
        #  MOTION_MIN_AREA      — larger = ignore smaller moving blobs.
        #  MOTION_HISTORY       — frames the background model averages over.
        # Defaults are deliberately conservative so leaves/shadows/noise don't
        # trip it; lower them if it misses real motion.
        self._var = float(os.environ.get(
            "MOTION_VAR_THRESHOLD",
            var_threshold if var_threshold is not None else 40))
        self.min_area = int(os.environ.get("MOTION_MIN_AREA", min_area or 1500))
        self._history = int(os.environ.get("MOTION_HISTORY", 300))
        # Ignore motion inside these zones (e.g. a switch's blinking LEDs).
        # MOTION_EXCLUDE = "x1,y1,x2,y2;..." in normalized 0-1 coordinates.
        self._exclude = self._parse_zones(os.environ.get("MOTION_EXCLUDE", ""))
        self._kernel = np.ones((3, 3), np.uint8)
        self._bg = self._make_bg()
        self.available = self._bg is not None

    def _make_bg(self):
        try:
            return cv2.createBackgroundSubtractorMOG2(
                history=self._history, varThreshold=self._var,
                detectShadows=True)
        except (AttributeError, cv2.error):
            return None

    @staticmethod
    def _parse_zones(spec):
        """Parse "x1,y1,x2,y2;..." (normalized) into a list of rectangles."""
        zones = []
        for part in spec.split(";"):
            part = part.strip()
            if not part:
                continue
            try:
                x1, y1, x2, y2 = (float(v) for v in part.split(","))
            except ValueError:
                continue
            zones.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
        return zones

    def _excluded(self, x, y, w, h, W, H):
        """True if a box's center falls inside any exclusion zone."""
        cx, cy = (x + w / 2) / W, (y + h / 2) / H
        return any(zx1 <= cx <= zx2 and zy1 <= cy <= zy2
                   for zx1, zy1, zx2, zy2 in self._exclude)

    def reset(self):
        """Drop the learned background so a re-enable starts fresh."""
        if self.available:
            self._bg = self._make_bg()

    def detect(self, frame):
        """Return a list of (x, y, w, h) boxes around moving regions."""
        if not self.available:
            return []
        mask = self._bg.apply(frame)
        # MOG2 marks shadows as 127; keep only hard foreground (255).
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        # Stronger opening removes speckle noise; gentle dilation consolidates
        # without inflating small blobs over the min_area threshold.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel, iterations=2)
        mask = cv2.dilate(mask, self._kernel, iterations=1)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        H, W = frame.shape[:2]
        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if self._exclude and self._excluded(x, y, w, h, W, H):
                continue
            boxes.append((int(x), int(y), int(w), int(h)))
        return boxes
