"""
Air Writer v3 — Graceful pinch drawing
=======================================
Uses MediaPipe Tasks API (HandLandmarker) + OpenCV.

Gesture:
  Pinch  (thumb tip + index tip close)  →  Draw
  Open hand                             →  Hover / pen lifted

  C  →  Clear canvas
  S  →  Save canvas PNG
  Q / Esc  →  Quit

Requirements:
  pip install mediapipe opencv-python numpy
"""

import math
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path("hand_landmarker.task")

# ── Canvas ─────────────────────────────────────────────────────────────────────
CANVAS_H = 480
CANVAS_W = 640

# ── Smoothing ──────────────────────────────────────────────────────────────────
# Two separate EMA smoothers:
#   ALPHA_CURSOR  — very smooth, used for the visible cursor dot
#   ALPHA_STROKE  — slightly more responsive, used for actual canvas drawing
ALPHA_CURSOR = 0.18   # cursor dot — very smooth
ALPHA_STROKE = 0.32   # stroke points — slightly faster
ALPHA_PINCH  = 0.12   # 1-D EMA on the distance value itself

# ── Pinch — raw pixel distance between thumb tip and index tip ─────────────────
# Using pixel distance (not normalised) so behaviour is simple and predictable.
# Tune these two values if needed; they are intentionally generous.
PINCH_CLOSE_PX = 19   # fingers this close  → start drawing
PINCH_OPEN_PX  = 20    # fingers this far    → stop  drawing

# ── Debounce ───────────────────────────────────────────────────────────────────
DEBOUNCE_FRAMES = 5    # consecutive frames needed to flip pinch state

# ── Grace period ───────────────────────────────────────────────────────────────
# If MediaPipe drops the hand for this many frames, keep drawing state alive.
# Prevents flicker when the detector momentarily loses track.
GRACE_FRAMES = 10

# ── Detection upscale ──────────────────────────────────────────────────────────
# MediaPipe struggles with small hands (user far from camera).
# Upscaling the frame before detection makes a distant hand appear larger,
# dramatically improving far-distance tracking.
# Landmarks are returned as normalised [0-1] coords so display mapping is
# unaffected — only the detection input changes.
# 1.5 is a good balance between range and CPU cost. Try 2.0 for even more range.
DETECT_SCALE = 1.5

# ── Stroke quality ─────────────────────────────────────────────────────────────
MIN_MOVE_PX = 2        # ignore movements tinier than this (residual jitter)
PEN_THICK   = 7
PEN_COLOR   = (255, 255, 255)
TIP_RADIUS  = 11


# ══════════════════════════════════════════════════════════════════════════════
class EMA:
    """Single exponential moving average with lazy init and hard reset."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self._x: float | None = None
        self._y: float | None = None

    def update(self, x: float, y: float) -> tuple[int, int]:
        if self._x is None:
            self._x, self._y = float(x), float(y)
        else:
            a = self.alpha
            self._x = a * x + (1 - a) * self._x
            self._y = a * y + (1 - a) * self._y
        return int(round(self._x)), int(round(self._y))

    # Feed without advancing the average — keeps position alive during grace
    def peek(self) -> tuple[int, int] | None:
        if self._x is None:
            return None
        return int(round(self._x)), int(round(self._y))

    def reset(self) -> None:
        self._x = self._y = None


# ══════════════════════════════════════════════════════════════════════════════
class OneEuroFilter:
    """
    One Euro Filter — purpose-built for pointer / hand tracking.

    Principle:
      * Estimates the hand's instantaneous speed each frame.
      * Slow hand  →  low cutoff frequency  →  heavy smoothing, no jitter.
      * Fast hand  →  high cutoff frequency →  minimal lag, strokes keep up.

    Params:
      min_cutoff : Hz — larger = less smoothing when still (try 1–3)
      beta       : speed coefficient — larger = less lag when moving fast (try 0.05–0.2)
      d_cutoff   : Hz for the derivative filter — leave at 1.0
      fps        : expected webcam framerate
    """

    def __init__(self, min_cutoff: float = 1.5, beta: float = 0.07,
                 d_cutoff: float = 1.0, fps: float = 30.0):
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.d_cutoff   = d_cutoff
        self.fps        = fps
        self._x: float | None = None
        self._y: float | None = None
        self._dx = 0.0
        self._dy = 0.0

    @staticmethod
    def _alpha(cutoff: float, fps: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te  = 1.0 / fps
        return 1.0 / (1.0 + tau / te)

    def update(self, x: float, y: float) -> tuple[int, int]:
        if self._x is None:
            self._x, self._y = float(x), float(y)
            return int(round(self._x)), int(round(self._y))

        # 1. Estimate speed via a derivative EMA
        a_d      = self._alpha(self.d_cutoff, self.fps)
        self._dx = a_d * (x - self._x) * self.fps + (1 - a_d) * self._dx
        self._dy = a_d * (y - self._y) * self.fps + (1 - a_d) * self._dy
        speed    = (self._dx ** 2 + self._dy ** 2) ** 0.5

        # 2. Adapt cutoff to speed and apply EMA
        fc      = self.min_cutoff + self.beta * speed
        alpha   = self._alpha(fc, self.fps)
        self._x = alpha * x + (1 - alpha) * self._x
        self._y = alpha * y + (1 - alpha) * self._y
        return int(round(self._x)), int(round(self._y))

    def peek(self) -> tuple[int, int] | None:
        if self._x is None:
            return None
        return int(round(self._x)), int(round(self._y))

    def reset(self) -> None:
        self._x = self._y = None
        self._dx = self._dy = 0.0


# ══════════════════════════════════════════════════════════════════════════════
class AirWriter:

    THUMB_TIP  = 4
    INDEX_TIP  = 8
    MIDDLE_TIP = 12   # used to validate that pinch is thumb↔index, not thumb↔middle

    def __init__(self) -> None:
        self._download_model()
        self.landmarker  = self._build_landmarker()

        self.canvas       = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

        # One Euro Filter smoothers — ideal for hand pointer tracking:
        #   cursor : smooth/jitter-free when still, keeps up on fast movement
        #   stroke : slightly more aggressive to ensure strokes aren't lagged
        self._cur_smooth  = OneEuroFilter(min_cutoff=1.5, beta=0.07)
        self._str_smooth  = OneEuroFilter(min_cutoff=3.0, beta=0.15)
        self._thmb_smooth = OneEuroFilter(min_cutoff=1.5, beta=0.07)
        self._mid_smooth  = OneEuroFilter(min_cutoff=1.5, beta=0.07)
        # Dedicated slow fixed EMAs for pinch DISTANCE ONLY (decoupled from display)
        self._pinch_idx_ema  = EMA(alpha=0.12)
        self._pinch_thmb_ema = EMA(alpha=0.12)
        self._dist_ema: float | None = None   # 1-D EMA on pinch distance

        self._pinching:       bool        = False
        self._prev_draw_pt:   tuple | None = None
        self._start_time:     float        = time.time()

        # Debounce counters
        self._close_frames: int = 0
        self._open_frames:  int = 0

        # Grace period counter (frames since last detection)
        self._lost_frames:  int = 0

    # ── Setup ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _download_model() -> None:
        if not MODEL_PATH.exists():
            print(f"Downloading {MODEL_PATH} … ", end="", flush=True)
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("done.")
        else:
            print(f"Model found: {MODEL_PATH}")

    @staticmethod
    def _build_landmarker() -> mp_vision.HandLandmarker:
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(MODEL_PATH)
            ),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            # Low thresholds + detection upscaling handles far-distance hands
            min_hand_detection_confidence=0.20,
            min_hand_presence_confidence=0.20,
            min_tracking_confidence=0.20,
        )
        return mp_vision.HandLandmarker.create_from_options(opts)

    # ── Pinch geometry ────────────────────────────────────────────────────────
    @staticmethod
    def _pinch_px(lm, w: int, h: int) -> tuple[float, tuple[int,int], tuple[int,int]]:
        """Return (pixel_distance, thumb_pt, index_pt)."""
        tx = lm[AirWriter.THUMB_TIP].x * w
        ty = lm[AirWriter.THUMB_TIP].y * h
        ix = lm[AirWriter.INDEX_TIP].x * w
        iy = lm[AirWriter.INDEX_TIP].y * h
        dist = ((tx - ix) ** 2 + (ty - iy) ** 2) ** 0.5
        return dist, (int(tx), int(ty)), (int(ix), int(iy))

    # ── Pinch state machine ───────────────────────────────────────────────────
    def _update_pinch(self, dist: float) -> None:
        if dist < PINCH_CLOSE_PX:
            self._close_frames += 1
            self._open_frames   = 0
        elif dist > PINCH_OPEN_PX:
            self._open_frames  += 1
            self._close_frames  = 0
        else:
            # Inside hysteresis band — hold both counters
            self._close_frames = 0
            self._open_frames  = 0

        if not self._pinching and self._close_frames >= DEBOUNCE_FRAMES:
            self._pinching     = True
            self._close_frames = 0
            self._prev_draw_pt = self._str_smooth.peek()

        elif self._pinching and self._open_frames >= DEBOUNCE_FRAMES:
            self._pinching    = False
            self._open_frames = 0
            self._prev_draw_pt = None

    # ── Frame processing ──────────────────────────────────────────────────────
    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]

        rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Upscale for detection so distant/small hands are detected reliably.
        # Landmarks are always returned as normalised [0-1] coords, so we can
        # use the original (w, h) below without any coordinate correction.
        if DETECT_SCALE != 1.0:
            det_rgb = cv2.resize(rgb,
                                 (int(w * DETECT_SCALE), int(h * DETECT_SCALE)),
                                 interpolation=cv2.INTER_LINEAR)
        else:
            det_rgb = rgb

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=det_rgb)
        ts_ms  = int((time.time() - self._start_time) * 1_000)

        result  = self.landmarker.detect_for_video(mp_img, ts_ms)
        display = frame_bgr.copy()

        hand_found = bool(result.hand_landmarks)

        if hand_found:
            self._lost_frames = 0
            lm = result.hand_landmarks[0]

            # Raw positions
            ix_raw = lm[self.INDEX_TIP].x * w
            iy_raw = lm[self.INDEX_TIP].y * h
            tx_raw = lm[self.THUMB_TIP].x * w
            ty_raw = lm[self.THUMB_TIP].y * h

            # Velocity-adaptive smooth all tips independently
            cx, cy   = self._cur_smooth.update(ix_raw, iy_raw)    # cursor display
            sx, sy   = self._str_smooth.update(ix_raw, iy_raw)    # stroke drawing
            self._thmb_smooth.update(tx_raw, ty_raw)              # kept for reset symmetry

            # ── Pinch distance (dedicated slow EMAs — stable, no false jumps) ──
            pix, piy = self._pinch_idx_ema.update(ix_raw, iy_raw)
            ptx, pty = self._pinch_thmb_ema.update(tx_raw, ty_raw)
            raw_dist = ((ptx - pix) ** 2 + (pty - piy) ** 2) ** 0.5

            # 1-D EMA on the distance signal for final stability
            if self._dist_ema is None:
                self._dist_ema = raw_dist
            else:
                self._dist_ema = (ALPHA_PINCH * raw_dist
                                  + (1 - ALPHA_PINCH) * self._dist_ema)

            self._update_pinch(self._dist_ema)

            # Raw coords for visual markers only (bridge line, dots)
            _, thumb_pt, index_pt = self._pinch_px(lm, w, h)

            # ── Draw on canvas ────────────────────────────────────────────
            if self._pinching:
                if self._prev_draw_pt is not None:
                    dx = abs(sx - self._prev_draw_pt[0])
                    dy = abs(sy - self._prev_draw_pt[1])
                    if dx > MIN_MOVE_PX or dy > MIN_MOVE_PX:
                        # Line segment
                        cv2.line(self.canvas, self._prev_draw_pt, (sx, sy),
                                 PEN_COLOR, PEN_THICK, lineType=cv2.LINE_AA)
                        # Filled circle caps every segment → no gaps
                        cv2.circle(self.canvas, (sx, sy),
                                   PEN_THICK // 2, PEN_COLOR, -1,
                                   lineType=cv2.LINE_AA)
                self._prev_draw_pt = (sx, sy)

            # ── Visual feedback ───────────────────────────────────────────
            is_drawing   = self._pinching
            cursor_col   = (0, 255, 0)   if is_drawing else (0, 165, 255)
            bridge_col   = (0, 255, 0)   if is_drawing else (80, 80, 255)
            label_col    = (0, 255, 0)   if is_drawing else (255, 255, 255)

            # Cursor at smoothed index tip
            cv2.circle(display, (cx, cy), TIP_RADIUS, cursor_col, 2,
                       lineType=cv2.LINE_AA)
            cv2.circle(display, (cx, cy), 4, cursor_col, -1)

            # Thumb marker
            cv2.circle(display, thumb_pt, 7, cursor_col, 2,
                       lineType=cv2.LINE_AA)

            # Bridge line: thumb ↔ index  (shows the gap clearly)
            cv2.line(display, thumb_pt, index_pt, bridge_col, 2,
                     lineType=cv2.LINE_AA)

            # ── Pinch distance indicator (top-right) ─────────────────────
            d       = self._dist_ema or 0.0
            bar_w   = 150
            bar_h   = 14
            bar_x   = w - bar_w - 10
            bar_y   = 24

            # Color mirrors state: green=pinching, amber=hysteresis, blue=open
            bar_col = (0, 220, 60)   if d < PINCH_CLOSE_PX else (
                      (0, 180, 255) if d < PINCH_OPEN_PX   else (120, 100, 255))

            # Scale bar so it fills at 1.6× open threshold
            bar_max    = max(PINCH_OPEN_PX * 1.6, 1.0)
            fill_px    = int(bar_w * min(d / bar_max, 1.0))
            close_mark = int(bar_w * PINCH_CLOSE_PX / bar_max)
            open_mark  = int(bar_w * PINCH_OPEN_PX  / bar_max)

            # Background → fill → border
            cv2.rectangle(display, (bar_x,          bar_y),
                          (bar_x + bar_w,            bar_y + bar_h), (30, 30, 30), -1)
            cv2.rectangle(display, (bar_x,          bar_y),
                          (bar_x + fill_px,          bar_y + bar_h), bar_col, -1)
            cv2.rectangle(display, (bar_x,          bar_y),
                          (bar_x + bar_w,            bar_y + bar_h), (110, 110, 110), 1)

            # Threshold tick marks
            cv2.line(display, (bar_x + close_mark, bar_y),
                     (bar_x + close_mark, bar_y + bar_h), (0, 255, 80),   1)
            cv2.line(display, (bar_x + open_mark,  bar_y),
                     (bar_x + open_mark,  bar_y + bar_h), (120, 120, 255), 1)

            # Prominent px readout above bar
            cv2.putText(display, f"{d:.1f} px", (bar_x, bar_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, bar_col, 1, cv2.LINE_AA)

            # Threshold labels below bar
            cv2.putText(display, f"C:{PINCH_CLOSE_PX}",
                        (bar_x, bar_y + bar_h + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 200, 80), 1, cv2.LINE_AA)
            cv2.putText(display, f"O:{PINCH_OPEN_PX}",
                        (bar_x + bar_w - 42, bar_y + bar_h + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 220), 1, cv2.LINE_AA)

            # Status text
            status = "DRAWING" if is_drawing else "Hover"
            cv2.putText(display, status, (10, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.80, label_col, 2,
                        cv2.LINE_AA)

        else:
            # ── Hand not detected ────────────────────────────────────────
            self._lost_frames += 1

            if self._lost_frames <= GRACE_FRAMES:
                # Grace period — keep smoothers and pinch state; just show msg
                prev = self._cur_smooth.peek()
                if prev:
                    cv2.circle(display, prev, TIP_RADIUS, (80, 80, 80), 2)
                cv2.putText(display, "Tracking…", (10, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (120, 120, 120),
                            2, cv2.LINE_AA)
            else:
                # Full reset after grace period expires
                self._cur_smooth.reset()
                self._str_smooth.reset()
                self._thmb_smooth.reset()
                self._mid_smooth.reset()
                self._pinch_idx_ema.reset()
                self._pinch_thmb_ema.reset()
                self._dist_ema     = None
                self._pinching     = False
                self._prev_draw_pt = None
                self._close_frames = 0
                self._open_frames  = 0
                cv2.putText(display, "No hand", (10, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (80, 80, 80),
                            2, cv2.LINE_AA)

        # ── Overlay canvas on feed ────────────────────────────────────────
        mask = self.canvas.any(axis=2)
        display[mask] = self.canvas[mask]

        # ── HUD ───────────────────────────────────────────────────────────
        cv2.putText(display, "C=clear  S=save  Q=quit",
                    (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (140, 140, 140), 1,
                    cv2.LINE_AA)

        return display

    # ── Utilities ─────────────────────────────────────────────────────────────
    def clear(self) -> None:
        self.canvas[:] = 0
        self._prev_draw_pt = None
        print("Canvas cleared.")

    def save(self, path: str = "air_canvas.png") -> None:
        cv2.imwrite(path, self.canvas)
        print(f"Canvas saved → '{path}'")

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self) -> None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam (index 0).")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CANVAS_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CANVAS_H)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("\nAir Writer ready.")
        print("  Pinch (thumb + index close)  →  Draw")
        print("  Open hand                    →  Hover")
        print("  C = clear | S = save | Q / Esc = quit")
        print(f"\n  Pinch thresholds:  close < {PINCH_CLOSE_PX}px  |"
              f"  open > {PINCH_OPEN_PX}px\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Camera read failed — exiting.")
                    break

                frame   = cv2.flip(frame, 1)
                display = self.process_frame(frame)

                cv2.imshow("Air Writer", display)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("c"):
                    self.clear()
                elif key == ord("s"):
                    self.save()
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.landmarker.close()
            print("Done.")


if __name__ == "__main__":
    AirWriter().run()
