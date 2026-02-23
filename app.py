"""
Air Writer App — Full Pipeline
===============================
Connects air-drawing input → CRNN model → text output file.

Workflow:
  1. Pinch to air-draw a word on the canvas
  2. Release pinch — a 2-second countdown starts automatically
  3. CRNN predicts the word when countdown ends
     (press SPACE to predict immediately at any time)
  4. Prediction is shown on screen and logged to output/predictions.txt
  5. Canvas image is saved to output/word_<timestamp>.png
  6. Canvas auto-clears after the result is shown, ready for next word

Keys:
  SPACE  →  Predict now
  C      →  Clear canvas
  S      →  Save canvas manually
  Q/Esc  →  Quit

Requirements:
  pip install mediapipe opencv-python numpy torch
"""

import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from model import CRNN, LabelEncoder, Config
from air_writer import AirWriter

# ── Output ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("output")

# ── Timing ─────────────────────────────────────────────────────────────────────
AUTO_PREDICT_SECS  = 3.0   # idle time after last stroke before auto-predict
SHOW_RESULT_SECS   = 3.5   # how long to display the prediction banner

# ── Vocab — must match training (clean_dataset filters ^[a-zA-Z0-9]+$) ─────────
DEFAULT_CHARS = sorted(
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)


# ══════════════════════════════════════════════════════════════════════════════
def _build_encoder() -> LabelEncoder:
    enc = LabelEncoder()
    enc.char2idx = {ch: idx + 1 for idx, ch in enumerate(DEFAULT_CHARS)}
    enc.idx2char = {idx + 1: ch for idx, ch in enumerate(DEFAULT_CHARS)}
    enc.idx2char[0] = ""
    return enc


def _load_crnn(encoder: LabelEncoder, model_path: str) -> CRNN:
    model = CRNN(
        num_chars=encoder.num_classes(),
        hidden_size=Config.LSTM_HIDDEN,
        dropout=0.0,
    ).to(Config.DEVICE)

    ckpt = Path(model_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"CRNN weights not found: '{ckpt}'")

    model.load_state_dict(
        torch.load(str(ckpt), map_location=Config.DEVICE, weights_only=True)
    )
    model.eval()
    print(f"CRNN loaded from '{ckpt}'  |  device: {Config.DEVICE}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
class AirWriterApp(AirWriter):
    """AirWriter + CRNN prediction bridge + file output."""

    def __init__(self, crnn_path: str = "best_crnn_model.pth") -> None:
        super().__init__()                     # sets up hand tracker + canvas
        OUTPUT_DIR.mkdir(exist_ok=True)

        self._encoder   = _build_encoder()
        self._crnn      = _load_crnn(self._encoder, crnn_path)

        # Prediction UI state
        self._prediction:     str   = ""
        self._pred_show_time: float = 0.0

        # Auto-predict timer
        self._last_stroke_time: float | None = None
        self._waiting_predict:  bool         = False

    # ── Canvas → model tensor ────────────────────────────────────────────────
    def _canvas_to_tensor(self) -> torch.Tensor | None:
        """
        Crop canvas to drawn content, resize to CRNN input format,
        return (1, 1, H, W) float tensor on the correct device, or None
        if the canvas is blank.

        Inversion note: IAM training images are dark ink on white background.
        Our canvas is white strokes on black → invert before feeding the model.
        """
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        pts  = cv2.findNonZero(gray)
        if pts is None:
            return None

        # Tight bounding box + small padding
        x, y, bw, bh = cv2.boundingRect(pts)
        pad = 14
        x1  = max(0, x - pad)
        y1  = max(0, y - pad)
        x2  = min(gray.shape[1], x + bw + pad)
        y2  = min(gray.shape[0], y + bh + pad)
        crop = gray[y1:y2, x1:x2]

        # Invert: match training distribution (dark ink on white)
        crop = cv2.bitwise_not(crop)

        # Resize height → Config.IMG_HEIGHT (64), keep aspect ratio
        h, w  = crop.shape
        tw    = int(w * (Config.IMG_HEIGHT / h))
        tw    = max(Config.MIN_WIDTH, min(tw, Config.MAX_WIDTH))
        img   = cv2.resize(crop, (tw, Config.IMG_HEIGHT))

        img = img.astype(np.float32) / 255.0
        return (
            torch.FloatTensor(img)
            .unsqueeze(0)   # channel
            .unsqueeze(0)   # batch
            .to(Config.DEVICE)
        )

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict_now(self) -> None:
        """Run the CRNN on the current canvas, log result, update display."""
        tensor = self._canvas_to_tensor()
        if tensor is None:
            print("Canvas is empty — nothing to predict.")
            return

        with torch.no_grad():
            log_probs = self._crnn(tensor)                   # (T, 1, C)
            seq       = torch.argmax(log_probs, dim=2)[:, 0].cpu().numpy()

        prediction = self._encoder.decode_greedy(seq) or "?"

        self._prediction     = prediction
        self._pred_show_time = time.time()
        self._waiting_predict = False
        self._save_output(prediction)
        print(f"\n  Prediction: '{prediction}'")

    def _save_output(self, prediction: str) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save canvas image (white strokes on black — original)
        img_path = OUTPUT_DIR / f"word_{ts}.png"
        cv2.imwrite(str(img_path), self.canvas)

        # Append to text log
        txt_path = OUTPUT_DIR / "predictions.txt"
        with open(txt_path, "a", encoding="utf-8") as f:
            f.write(f"{ts}\t{prediction}\n")

        print(f"  Image saved → '{img_path}'")
        print(f"  Text logged → '{txt_path}'")

    # ── HUD overlay on top of AirWriter's display ────────────────────────────
    def _draw_prediction_overlay(self, display: np.ndarray) -> None:
        h, w  = display.shape[:2]
        now   = time.time()

        # ── Countdown bar (shown while waiting to auto-predict) ──────────────
        if self._waiting_predict and self._last_stroke_time is not None:
            elapsed   = now - self._last_stroke_time
            remaining = max(0.0, AUTO_PREDICT_SECS - elapsed)
            progress  = 1.0 - remaining / AUTO_PREDICT_SECS

            bx, by  = 10, 48
            bw2, bh = w - 20, 10
            fill    = int(bw2 * progress)

            cv2.rectangle(display, (bx, by), (bx + bw2, by + bh),
                          (40, 40, 40), -1)
            cv2.rectangle(display, (bx, by), (bx + fill, by + bh),
                          (0, 210, 255), -1)
            cv2.rectangle(display, (bx, by), (bx + bw2, by + bh),
                          (100, 100, 100), 1)
            cv2.putText(display,
                        f"Predicting in {remaining:.1f} s  (SPACE = now)",
                        (bx, by - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 210, 255), 1,
                        cv2.LINE_AA)

        # ── Prediction result banner ─────────────────────────────────────────
        if self._prediction:
            age = now - self._pred_show_time
            if age < SHOW_RESULT_SECS:
                banner_h = 56
                # Semi-transparent dark strip at bottom
                strip = display[h - banner_h:h].copy()
                cv2.rectangle(strip, (0, 0), (w, banner_h), (15, 15, 15), -1)
                cv2.addWeighted(strip, 0.80, display[h - banner_h:h], 0.20,
                                0, display[h - banner_h:h])
                # Prediction text
                cv2.putText(display, self._prediction,
                            (w // 2 - len(self._prediction) * 18, h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 120), 3,
                            cv2.LINE_AA)
                # Label
                cv2.putText(display, "Prediction:",
                            (12, h - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 160, 160),
                            1, cv2.LINE_AA)

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self) -> None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam (index 0).")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("\nReady.")
        print(f"  Output directory: '{OUTPUT_DIR.resolve()}'")
        print("  Pinch → draw word → release → auto-predicts in "
              f"{AUTO_PREDICT_SECS:.0f} s")
        print("  SPACE = predict now | C = clear | S = save | Q = quit\n")

        prev_pinching = False

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Camera read failed — exiting.")
                    break

                frame   = cv2.flip(frame, 1)
                display = self.process_frame(frame)   # AirWriter handles tracking

                now = time.time()

                # ── Track strokes ────────────────────────────────────────────
                # Any frame where the pen is actively drawing, record the time
                # and cancel any pending countdown.
                if self._pinching and self._prev_draw_pt is not None:
                    self._last_stroke_time = now
                    self._waiting_predict  = False

                # ── Pinch-release edge: start countdown ───────────────────────
                if prev_pinching and not self._pinching and self.canvas.any():
                    self._last_stroke_time = now
                    self._waiting_predict  = True

                prev_pinching = self._pinching

                # ── Auto-predict when countdown expires ───────────────────────
                if (self._waiting_predict
                        and self._last_stroke_time is not None
                        and now - self._last_stroke_time >= AUTO_PREDICT_SECS):
                    self.predict_now()

                # ── Auto-clear after result displayed ─────────────────────────
                if (self._prediction
                        and now - self._pred_show_time >= SHOW_RESULT_SECS):
                    self.clear()
                    self._prediction     = ""
                    self._waiting_predict = False

                # ── Draw prediction overlay ───────────────────────────────────
                self._draw_prediction_overlay(display)

                # ── Canvas window: dark ink on white (matches model input) ────
                canvas_view = cv2.bitwise_not(self.canvas)
                # Show last prediction on the canvas window too
                if self._prediction:
                    cv2.putText(canvas_view, self._prediction,
                                (10, canvas_view.shape[0] - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                (30, 120, 30), 2, cv2.LINE_AA)
                cv2.imshow("Canvas", canvas_view)

                cv2.imshow("Air Writer", display)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord(" "):
                    self.predict_now()
                elif key == ord("c"):
                    self.clear()
                    self._prediction      = ""
                    self._waiting_predict = False
                elif key == ord("s"):
                    self.save()

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.landmarker.close()
            print("Done.")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    AirWriterApp().run()
