"""
Air Writer v5 — Finger gesture drawing
=======================================
Uses MediaPipe legacy Solutions API (mp.solutions.hands) + OpenCV.
Compatible with mediapipe 0.10.14.

Gestures:
  Index finger only (middle down)  →  Draw
  Index + middle fingers up        →  Hold / pen lifted

  C  →  Clear canvas
  S  →  Save canvas PNG
  Q / Esc  →  Quit

Requirements:
  pip install mediapipe==0.10.14 opencv-python numpy
"""

import math

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.python.solutions import hands as mp_hands
except ImportError:
    # Fallback for different mediapipe versions
    import mediapipe as mp  # type: ignore
    mp_hands = mp.solutions.hands  # type: ignore

# ── Canvas ─────────────────────────────────────────────────────────────────────
CANVAS_H = 480
CANVAS_W = 640

# ── Smoothing ──────────────────────────────────────────────────────────────────
ALPHA_CURSOR = 0.18   # cursor dot — very smooth
ALPHA_STROKE = 0.32   # stroke points — slightly faster

# ── Debounce ───────────────────────────────────────────────────────────────────
DEBOUNCE_FRAMES = 5   # consecutive frames needed to flip gesture state

# ── Grace period ───────────────────────────────────────────────────────────────
# If MediaPipe drops the hand for this many frames, keep drawing state alive.
GRACE_FRAMES = 10

# ── Detection upscale ──────────────────────────────────────────────────────────
# Upscaling the frame before detection makes a distant hand appear larger,
# improving far-distance tracking. Landmarks are normalised [0-1] so display
# mapping is unaffected — only the detection input changes.
DETECT_SCALE = 1.5

# ── Stroke quality ─────────────────────────────────────────────────────────────
MIN_MOVE_PX = 2
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
            assert self._x is not None and self._y is not None
            self._x = a * x + (1 - a) * self._x
            self._y = a * y + (1 - a) * self._y
        return int(round(self._x)), int(round(self._y))

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

    Slow hand  →  low cutoff frequency  →  heavy smoothing, no jitter.
    Fast hand  →  high cutoff frequency →  minimal lag, strokes keep up.
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

        assert self._x is not None and self._y is not None
        a_d      = self._alpha(self.d_cutoff, self.fps)
        self._dx = a_d * (x - self._x) * self.fps + (1 - a_d) * self._dx
        self._dy = a_d * (y - self._y) * self.fps + (1 - a_d) * self._dy
        speed    = (self._dx ** 2 + self._dy ** 2) ** 0.5

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

    # Landmark indices
    INDEX_TIP  = 8
    INDEX_PIP  = 6
    MIDDLE_TIP = 12
    MIDDLE_PIP = 10

    def __init__(self) -> None:
        # Legacy solutions API — no .task file required
        self._hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.20,
            min_tracking_confidence=0.20,
        )

        self.canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

        # Index tip smoothers: cursor display + stroke drawing
        self._cur_smooth = OneEuroFilter(min_cutoff=1.5, beta=0.07)
        self._str_smooth = OneEuroFilter(min_cutoff=3.0, beta=0.15)

        self._pinching:     bool         = False
        self._prev_draw_pt: tuple | None = None

        # Debounce counters for draw / hold gesture transitions
        self._draw_frames: int = 0   # consecutive frames of draw gesture
        self._hold_frames: int = 0   # consecutive frames of hold gesture

        self._lost_frames: int = 0

    # ── Finger-up detection ───────────────────────────────────────────────────
    @staticmethod
    def _finger_up(lm, tip: int, pip: int) -> bool:
        """True when the fingertip is above its PIP joint — finger is extended."""
        return lm[tip].y < lm[pip].y

    # ── Gesture state machine ─────────────────────────────────────────────────
    def _update_gesture(self, draw_gesture: bool) -> None:
        """
        draw_gesture = True  → index only up  (want to draw)
        draw_gesture = False → index+middle or no clear gesture (want to hold)
        """
        if draw_gesture:
            self._draw_frames += 1
            self._hold_frames  = 0
        else:
            self._hold_frames += 1
            self._draw_frames  = 0

        if not self._pinching and self._draw_frames >= DEBOUNCE_FRAMES:
            self._pinching     = True
            self._draw_frames  = 0
            self._prev_draw_pt = self._str_smooth.peek()

        elif self._pinching and self._hold_frames >= DEBOUNCE_FRAMES:
            self._pinching    = False
            self._hold_frames = 0
            self._prev_draw_pt = None

    # ── Frame processing ──────────────────────────────────────────────────────
    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if DETECT_SCALE != 1.0:
            det_rgb = cv2.resize(rgb,
                                 (int(w * DETECT_SCALE), int(h * DETECT_SCALE)),
                                 interpolation=cv2.INTER_LINEAR)
        else:
            det_rgb = rgb

        result  = self._hands.process(det_rgb)
        display = frame_bgr.copy()

        hand_found = bool(result.multi_hand_landmarks)

        if hand_found:
            self._lost_frames = 0
            lm = result.multi_hand_landmarks[0].landmark

            # ── Cursor position (index tip) ───────────────────────────────
            ix_raw = lm[self.INDEX_TIP].x * w
            iy_raw = lm[self.INDEX_TIP].y * h

            cx, cy = self._cur_smooth.update(ix_raw, iy_raw)   # display
            sx, sy = self._str_smooth.update(ix_raw, iy_raw)   # canvas

            # Middle tip for hold indicator
            mx = int(lm[self.MIDDLE_TIP].x * w)
            my = int(lm[self.MIDDLE_TIP].y * h)

            # ── Gesture detection ─────────────────────────────────────────
            index_up  = self._finger_up(lm, self.INDEX_TIP,  self.INDEX_PIP)
            middle_up = self._finger_up(lm, self.MIDDLE_TIP, self.MIDDLE_PIP)

            # Index alone → draw | index+middle (or no index) → hold
            draw_gesture = index_up and not middle_up
            self._update_gesture(draw_gesture)

            # ── Draw on canvas ────────────────────────────────────────────
            if self._pinching:
                if self._prev_draw_pt is not None:
                    dx = abs(sx - self._prev_draw_pt[0])
                    dy = abs(sy - self._prev_draw_pt[1])
                    if dx > MIN_MOVE_PX or dy > MIN_MOVE_PX:
                        cv2.line(self.canvas, self._prev_draw_pt, (sx, sy),
                                 PEN_COLOR, PEN_THICK, lineType=cv2.LINE_AA)
                        cv2.circle(self.canvas, (sx, sy),
                                   PEN_THICK // 2, PEN_COLOR, -1,
                                   lineType=cv2.LINE_AA)
                self._prev_draw_pt = (sx, sy)

            # ── Visual feedback ───────────────────────────────────────────
            is_drawing = self._pinching
            cursor_col = (0, 255, 0)   if is_drawing else (0, 165, 255)
            label_col  = (0, 255, 0)   if is_drawing else (255, 255, 255)

            # Index tip cursor
            cv2.circle(display, (cx, cy), TIP_RADIUS, cursor_col, 2,
                       lineType=cv2.LINE_AA)
            cv2.circle(display, (cx, cy), 4, cursor_col, -1)

            # Middle tip marker + bridge line (shown when hold gesture active)
            if middle_up:
                cv2.circle(display, (mx, my), 9, (0, 165, 255), 2,
                           lineType=cv2.LINE_AA)
                cv2.line(display, (cx, cy), (mx, my), (0, 140, 220), 2,
                         lineType=cv2.LINE_AA)

            # Status label
            if is_drawing:
                status = "DRAWING"
            elif middle_up:
                status = "Hold"
            else:
                status = "Hover"

            cv2.putText(display, status, (10, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.80, label_col, 2,
                        cv2.LINE_AA)

            # Gesture hint (top-right)
            hint = "index only = draw | index+mid = hold"
            cv2.putText(display, hint, (w - 310, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1,
                        cv2.LINE_AA)

        else:
            # ── Hand not detected ─────────────────────────────────────────
            self._lost_frames += 1

            if self._lost_frames <= GRACE_FRAMES:
                prev = self._cur_smooth.peek()
                if prev:
                    cv2.circle(display, prev, TIP_RADIUS, (80, 80, 80), 2)
                cv2.putText(display, "Tracking…", (10, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (120, 120, 120),
                            2, cv2.LINE_AA)
            else:
                self._cur_smooth.reset()
                self._str_smooth.reset()
                self._pinching     = False
                self._prev_draw_pt = None
                self._draw_frames  = 0
                self._hold_frames  = 0
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
        print("  Index finger only   →  Draw")
        print("  Index + middle up   →  Hold (pen lifted)")
        print("  C = clear | S = save | Q / Esc = quit\n")

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
            self._hands.close()
            print("Done.")


if __name__ == "__main__":
    AirWriter().run()
