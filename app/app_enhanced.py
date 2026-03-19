"""
Air Writer Enhanced — Standalone Split-Screen Application
=========================================================
Single window with webcam and canvas side-by-side for seamless writing.

FEATURES:
  ✓ Split-screen UI (webcam + canvas in one window)
  ✓ JSON + image file storage (no database required)
  ✓ Prediction caching for faster responses
  ✓ Gamma correction for better image quality
  ✓ Works offline (no internet required)
  ✓ Auto-save to database with image storage
  ✓ Search and retrieve past predictions
  ✓ Real-time visual feedback

WORKFLOW:
  1. Draw a word in the air with your index finger
  2. Release pinch — 2-second countdown starts
  3. See your strokes on the canvas in real-time
  4. Model predicts the word
  5. Prediction is saved to database automatically
  6. Canvas clears after result shown

KEYS:
  SPACE  →  Predict now (skip countdown)
  D      →  Toggle database save
  H      →  Show help & stats
  C      →  Clear canvas
  S      →  Save canvas manually
  Q/Esc  →  Quit

Requirements:
  pip install mediapipe opencv-python numpy torch
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import time
import hashlib
from datetime import datetime
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from Training.config import Config, SpecialTokens
from Training.data import LabelEncoder
from Training.architecture import Seq2SeqAttention
from air_writer import AirWriter

# Import enhanced modules
from storage_manager import StorageManager
from app_config import (
    MODEL_PATH, MODEL_VERSION, OUTPUT_DIR, CACHE_DIR,
    AUTO_PREDICT_DELAY, SHOW_RESULT_DURATION,
    ENABLE_PREDICTION_CACHE, ENABLE_GAMMA_CORRECTION, GAMMA_VALUE,
    DEFAULT_VOCABULARY, get_storage_config, get_app_info,
    CAMERA_INDEX, CAMERA_RETRY_ATTEMPTS, CAMERA_RETRY_DELAY,
    ENABLE_DEMO_MODE
)

# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _build_encoder() -> LabelEncoder:
    """Build LabelEncoder with vocabulary matching training"""
    enc = LabelEncoder()
    enc.char2idx = {
        SpecialTokens.PAD: 0,
        SpecialTokens.SOS: 1,
        SpecialTokens.EOS: 2,
        SpecialTokens.UNK: 3
    }
    for i, ch in enumerate(DEFAULT_VOCABULARY, start=4):
        enc.char2idx[ch] = i
    enc.idx2char = {v: k for k, v in enc.char2idx.items()}
    return enc


def _load_seq2seq(encoder: LabelEncoder, model_path: str) -> Optional[Seq2SeqAttention]:
    """Load Seq2Seq model with error handling"""
    try:
        model = Seq2SeqAttention(
            num_classes=encoder.num_classes(),
            encoder_hidden=Config.ENCODER_HIDDEN,
            decoder_hidden=Config.DECODER_HIDDEN,
            attention_hidden=Config.ATTENTION_HIDDEN,
            dropout=0.0,
            drop_path=0.0
        ).to(Config.DEVICE)

        ckpt = Path(model_path)
        if not ckpt.exists():
            print(f"✗ Model not found: '{ckpt}'")
            return None

        model.load_state_dict(
            torch.load(str(ckpt), map_location=Config.DEVICE, weights_only=True)
        )
        model.eval()
        print(f"✓ Model loaded: '{ckpt.name}'  |  Device: {Config.DEVICE}")
        return model
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return None


def compute_canvas_hash(canvas: np.ndarray) -> str:
    """Compute hash of canvas for caching"""
    canvas_bytes = canvas.tobytes()
    return hashlib.md5(canvas_bytes).hexdigest()


def apply_gamma_correction(image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """Apply gamma correction to improve image quality"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


# ═══════════════════════════════════════════════════════════════════════════
# ENHANCED AIR WRITER APP
# ═══════════════════════════════════════════════════════════════════════════

class AirWriterEnhanced(AirWriter):
    """
    Enhanced AirWriter with database integration, caching, and robust features.
    Works under any condition with graceful degradation.
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        enable_database: bool = True,
        enable_cache: bool = ENABLE_PREDICTION_CACHE
    ) -> None:
        """
        Initialize enhanced air writer.
        
        Args:
            model_path: Path to model weights
            enable_database: Enable JSON file storage
            enable_cache: Enable prediction caching
        """
        super().__init__()
        
        # Setup directories
        OUTPUT_DIR.mkdir(exist_ok=True)
        CACHE_DIR.mkdir(exist_ok=True)
        
        # Load model
        self._encoder = _build_encoder()
        self._model = _load_seq2seq(self._encoder, model_path)
        self._model_available = self._model is not None
        
        # Storage setup
        self._db_enabled = enable_database
        self._db: Optional[StorageManager] = None
        if enable_database:
            try:
                self._db = StorageManager(**get_storage_config())
                storage_info = self._db.get_database_info()
                print(f"✓ Storage: JSON files @ {storage_info['path']}")
            except Exception as e:
                print(f"✗ Storage initialization failed: {e}")
                print("→ Running without storage (predictions not saved)")
                self._db_enabled = False
        
        # Cache setup
        self._cache_enabled = enable_cache and self._db is not None
        if self._cache_enabled:
            stats = self._db.get_cache_stats()
            print(f"✓ Prediction cache: {stats['count']} entries, {stats['total_hits']} hits")
        
        # UI state
        self._prediction: str = ""
        self._pred_confidence: float = 0.0
        self._pred_show_time: float = 0.0
        self._waiting_predict: bool = False
        self._last_stroke_time: Optional[float] = None
        
        # Stats
        self._prediction_count = 0
        self._cache_hits = 0
        self._total_predict_time = 0.0
        self._auto_save_enabled = True
        
        # App info
        info = get_app_info()
        print(f"✓ {info['name']} {info['version']}")
        if not self._model_available:
            print("⚠️  Running in LIMITED MODE (model not loaded)")
    
    # ───────────────────────────────────────────────────────────────────────
    # CANVAS PROCESSING
    # ───────────────────────────────────────────────────────────────────────
    
    def _canvas_to_tensor(self) -> Tuple[Optional[torch.Tensor], str]:
        """
        Convert canvas to model input tensor.
        
        Returns:
            (tensor, canvas_hash) or (None, "") if canvas is blank
        """
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        pts = cv2.findNonZero(gray)
        
        if pts is None:
            return None, ""
        
        # Compute hash for caching
        canvas_hash = compute_canvas_hash(gray)
        
        # Crop to content
        x, y, bw, bh = cv2.boundingRect(pts)
        pad = 14
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(gray.shape[1], x + bw + pad)
        y2 = min(gray.shape[0], y + bh + pad)
        crop = gray[y1:y2, x1:x2]
        
        # Invert (match training: dark ink on white)
        crop = cv2.bitwise_not(crop)
        
        # Apply gamma correction if enabled
        if ENABLE_GAMMA_CORRECTION:
            crop = apply_gamma_correction(crop, GAMMA_VALUE)
        
        # Resize maintaining aspect ratio
        h, w = crop.shape
        tw = int(w * (Config.IMG_HEIGHT / h))
        tw = max(Config.MIN_WIDTH, min(tw, Config.MAX_WIDTH))
        img = cv2.resize(crop, (tw, Config.IMG_HEIGHT))
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        return (
            torch.FloatTensor(img).unsqueeze(0).unsqueeze(0).to(Config.DEVICE),
            canvas_hash
        )
    
    # ───────────────────────────────────────────────────────────────────────
    # PREDICTION
    # ───────────────────────────────────────────────────────────────────────
    
    def predict_now(self) -> None:
        """Run prediction on current canvas, with caching support"""
        
        if not self._model_available:
            print("⚠️  Model not available, cannot predict")
            self._prediction = "[Model Not Loaded]"
            self._pred_show_time = time.time()
            return
        
        tensor, canvas_hash = self._canvas_to_tensor()
        
        if tensor is None:
            print("⚠️  Canvas is empty")
            return
        
        start_time = time.time()
        
        # Check cache first
        if self._cache_enabled and canvas_hash:
            cached = self._db.get_cached_prediction(canvas_hash)
            if cached:
                prediction = cached['prediction']
                confidence = cached.get('confidence', 0.0)
                self._cache_hits += 1
                print(f"✓ Cache hit! Prediction: '{prediction}' (confidence: {confidence:.2f})")
                self._update_prediction(prediction, confidence, cache_hit=True)
                return
        
        # Model inference
        try:
            with torch.no_grad():
                outputs, _ = self._model(tensor, targets=None, attention_mask=None)
                pred_indices = outputs.argmax(dim=-1)[0].cpu().numpy()
                prediction = self._encoder.decode(pred_indices) or "?"
                
                # Calculate confidence (average max probability)
                probs = torch.softmax(outputs, dim=-1)
                max_probs = probs.max(dim=-1)[0][0].cpu().numpy()
                confidence = float(max_probs.mean())
            
            elapsed = time.time() - start_time
            self._total_predict_time += elapsed
            
            print(f"✓ Prediction: '{prediction}' (confidence: {confidence:.2f}, time: {elapsed:.3f}s)")
            
            # Save to cache
            if self._cache_enabled and canvas_hash:
                self._db.save_cached_prediction(
                    canvas_hash, prediction, confidence, MODEL_VERSION
                )
            
            self._update_prediction(prediction, confidence, cache_hit=False)
            
        except Exception as e:
            print(f"✗ Prediction failed: {e}")
            self._prediction = "[Error]"
            self._pred_show_time = time.time()
    
    def _update_prediction(self, prediction: str, confidence: float, cache_hit: bool) -> None:
        """Update prediction state and save to database"""
        self._prediction = prediction
        self._pred_confidence = confidence
        self._pred_show_time = time.time()
        self._waiting_predict = False
        self._prediction_count += 1
        
        # Auto-save to database (primary storage)
        if self._auto_save_enabled and self._db_enabled:
            self._save_to_database(prediction, confidence)
        
        # Optional: Save canvas image to output folder
        # self._save_output_files(prediction, confidence, cache_hit)
    
    # ───────────────────────────────────────────────────────────────────────
    # SAVING
    # ───────────────────────────────────────────────────────────────────────
    
    def _save_output_files(self, prediction: str, confidence: float, cache_hit: bool) -> None:
        """Save canvas image and text log"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save canvas image
        img_path = OUTPUT_DIR / f"word_{ts}.png"
        cv2.imwrite(str(img_path), self.canvas)
        
        # Append to text log
        txt_path = OUTPUT_DIR / "predictions.txt"
        with open(txt_path, "a", encoding="utf-8") as f:
            cache_flag = "[CACHE]" if cache_hit else "[MODEL]"
            f.write(f"{ts}\t{prediction}\t{confidence:.4f}\t{cache_flag}\n")
        
        print(f"  → Image: {img_path.name}")
        print(f"  → Log: {txt_path.name}")
    
    def _save_to_database(self, prediction: str, confidence: float) -> None:
        """Save prediction and canvas image to database"""
        if not self._db:
            return
        
        try:
            # Convert canvas to PNG bytes for database storage
            image_bytes = None
            try:
                # Encode canvas as PNG in memory
                success, buffer = cv2.imencode('.png', self.canvas)
                if success:
                    image_bytes = buffer.tobytes()
            except Exception as e:
                print(f"  ⚠ Image encoding failed: {e}")
            
            note_id = self._db.save_note(
                title=prediction,
                image_data=image_bytes,
                confidence=confidence
            )
            
            if image_bytes:
                size_kb = len(image_bytes) / 1024
                print(f"  → Saved: '{prediction}' + image ({size_kb:.1f} KB)")
            else:
                print(f"  → Saved: '{prediction}' (no image)")
            
        except Exception as e:
            print(f"  ✗ Storage save failed: {e}")
    
    # ───────────────────────────────────────────────────────────────────────
    # UI OVERLAY
    # ───────────────────────────────────────────────────────────────────────
    
    def _draw_prediction_overlay(self, display: np.ndarray) -> None:
        """Draw prediction UI overlay"""
        h, w = display.shape[:2]
        now = time.time()
        
        # Countdown bar
        if self._waiting_predict and self._last_stroke_time is not None:
            elapsed = now - self._last_stroke_time
            remaining = max(0.0, AUTO_PREDICT_DELAY - elapsed)
            progress = 1.0 - remaining / AUTO_PREDICT_DELAY
            
            bx, by = 10, 48
            bw2, bh = w - 20, 10
            fill = int(bw2 * progress)
            
            cv2.rectangle(display, (bx, by), (bx + bw2, by + bh), (40, 40, 40), -1)
            cv2.rectangle(display, (bx, by), (bx + fill, by + bh), (0, 210, 255), -1)
            cv2.rectangle(display, (bx, by), (bx + bw2, by + bh), (100, 100, 100), 1)
            cv2.putText(
                display,
                f"Predicting in {remaining:.1f}s  (SPACE = now)",
                (bx, by - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 210, 255), 1, cv2.LINE_AA
            )
        
        # Prediction result banner
        if self._prediction:
            age = now - self._pred_show_time
            if age < SHOW_RESULT_DURATION:
                banner_h = 80
                
                # Semi-transparent background
                strip = display[h - banner_h:h].copy()
                cv2.rectangle(strip, (0, 0), (w, banner_h), (15, 15, 15), -1)
                cv2.addWeighted(strip, 0.85, display[h - banner_h:h], 0.15, 0, display[h - banner_h:h])
                
                # Prediction text
                text_size = cv2.getTextSize(self._prediction, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)[0]
                text_x = (w - text_size[0]) // 2
                cv2.putText(
                    display, self._prediction,
                    (text_x, h - 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 120), 3, cv2.LINE_AA
                )
                
                # Confidence and label
                cv2.putText(
                    display, f"Prediction: {self._pred_confidence:.1%} confident",
                    (12, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA
                )
    
    def _draw_stats_overlay(self, display: np.ndarray) -> None:
        """Draw statistics overlay"""
        h, w = display.shape[:2]
        
        # Status indicators
        y = 20
        cv2.putText(display, "Air Writer Enhanced", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        y += 20
        
        # Database status
        db_status = "ON" if self._auto_save_enabled else "OFF"
        db_color = (0, 255, 0) if self._auto_save_enabled else (0, 0, 255)
        cv2.putText(display, f"DB: {db_status}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, db_color, 1, cv2.LINE_AA)
        
        # Cache statistics
        if self._prediction_count > 0:
            cache_rate = (self._cache_hits / self._prediction_count) * 100
            cv2.putText(display, f"Cache: {self._cache_hits}/{self._prediction_count} ({cache_rate:.0f}%)", 
                       (100, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
    
    def _print_help(self) -> None:
        """Print help and statistics"""
        print("\n" + "=" * 70)
        print("AIR WRITER ENHANCED - HELP & STATISTICS")
        print("=" * 70)
        
        print("\nKEYBOARD SHORTCUTS:")
        print("  SPACE    →  Predict now (skip countdown)")
        print("  D        →  Toggle database auto-save")
        print("  H        →  Show this help")
        print("  C        →  Clear canvas")
        print("  S        →  Save canvas manually")
        print("  Q / Esc  →  Quit application")
        
        print("\nSTATISTICS:")
        print(f"  Predictions:      {self._prediction_count}")
        print(f"  Cache hits:       {self._cache_hits}")
        if self._prediction_count > 0:
            cache_rate = (self._cache_hits / self._prediction_count) * 100
            avg_time = self._total_predict_time / (self._prediction_count - self._cache_hits) if self._prediction_count > self._cache_hits else 0
            print(f"  Cache hit rate:   {cache_rate:.1f}%")
            print(f"  Avg predict time: {avg_time:.3f}s")
        print(f"  Storage saving:   {'ON' if self._auto_save_enabled else 'OFF'}")
        
        if self._db:
            print("\nSTORAGE:")
            info = self._db.get_database_info()
            print(f"  Type:       {info['type']}")
            print(f"  Path:       {info['path']}")
            
            stats = self._db.get_cache_stats()
            print(f"  Cache size: {stats['count']} entries")
            print(f"  Total hits: {stats['total_hits']}")
        
        print("=" * 70 + "\n")
    
    # ───────────────────────────────────────────────────────────────────────
    # MAIN LOOP
    # ───────────────────────────────────────────────────────────────────────
    
    def run(self) -> None:
        """Main application loop with robust error handling"""
        
        # Open camera with retry logic
        cap = None
        for attempt in range(CAMERA_RETRY_ATTEMPTS):
            cap = cv2.VideoCapture(CAMERA_INDEX)
            if cap.isOpened():
                break
            print(f"⚠️  Camera attempt {attempt + 1}/{CAMERA_RETRY_ATTEMPTS} failed, retrying...")
            time.sleep(CAMERA_RETRY_DELAY)
        
        if not cap or not cap.isOpened():
            print(f"✗ Cannot open camera (index {CAMERA_INDEX})")
            if ENABLE_DEMO_MODE:
                print("→ Demo mode not implemented yet")
            return
        
        # Configure camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n" + "=" * 70)
        print("✓ AIR WRITER SPLIT-SCREEN MODE READY")
        print("=" * 70)
        print(f"  Output: {OUTPUT_DIR.resolve()}")
        print(f"  Auto-predict: {AUTO_PREDICT_DELAY:.1f}s after last stroke")
        print(f"  Display: Webcam + Canvas side-by-side")
        print("  Press H for help and statistics")
        print("=" * 70 + "\n")
        
        prev_pinching = False
        show_stats = False
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️  Camera read failed")
                    break
                
                frame = cv2.flip(frame, 1)
                display = self.process_frame(frame)
                
                now = time.time()
                
                # Track strokes
                if self._pinching and self._prev_draw_pt is not None:
                    self._last_stroke_time = now
                    self._waiting_predict = False
                
                # Pinch release: start countdown
                if prev_pinching and not self._pinching and self.canvas.any():
                    self._last_stroke_time = now
                    self._waiting_predict = True
                
                prev_pinching = self._pinching
                
                # Auto-predict when countdown expires
                if (self._waiting_predict 
                        and self._last_stroke_time is not None 
                        and now - self._last_stroke_time >= AUTO_PREDICT_DELAY):
                    self.predict_now()
                
                # Auto-clear after showing result
                if (self._prediction 
                        and now - self._pred_show_time >= SHOW_RESULT_DURATION):
                    self.clear()
                    self._prediction = ""
                    self._waiting_predict = False
                
                # Draw overlays
                self._draw_prediction_overlay(display)
                if show_stats:
                    self._draw_stats_overlay(display)
                
                # Prepare canvas view
                canvas_view = cv2.bitwise_not(self.canvas)
                if self._prediction:
                    cv2.putText(
                        canvas_view, self._prediction,
                        (10, canvas_view.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30, 120, 30), 2, cv2.LINE_AA
                    )
                
                # Canvas is already BGR (3 channels), no conversion needed
                canvas_bgr = canvas_view
                
                # Ensure both views have the same height
                h1, w1 = display.shape[:2]
                h2, w2 = canvas_bgr.shape[:2]
                
                if h1 != h2:
                    # Resize canvas to match webcam height
                    canvas_bgr = cv2.resize(canvas_bgr, (int(w2 * h1 / h2), h1))
                
                # Add labels to each view
                cv2.putText(display, "WEBCAM", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas_bgr, "CANVAS", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Create split-screen view (webcam | canvas)
                split_view = np.hstack([display, canvas_bgr])
                
                # Display single window with both views
                cv2.imshow("Air Writer - Split View", split_view)
                
                # Keyboard handling
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # Q or Esc
                    break
                elif key == ord(" "):  # Space
                    self.predict_now()
                elif key == ord("c"):  # C
                    self.clear()
                    self._prediction = ""
                    self._waiting_predict = False
                    print("✓ Canvas cleared")
                elif key == ord("s"):  # S
                    self.save()
                    print("✓ Canvas saved manually")
                elif key == ord("d"):  # D
                    self._auto_save_enabled = not self._auto_save_enabled
                    status = "ON" if self._auto_save_enabled else "OFF"
                    print(f"✓ Storage auto-save: {status}")
                elif key == ord("h"):  # H
                    self._print_help()
                    show_stats = not show_stats
        
        except KeyboardInterrupt:
            print("\n✓ Interrupted by user")
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if hasattr(self, '_hands'):
                self._hands.close()
            if self._db:
                self._db.close()
            print("\n✓ Application closed")
            print(f"  Total predictions: {self._prediction_count}")
            print(f"  Cache hits: {self._cache_hits}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = AirWriterEnhanced()
    app.run()