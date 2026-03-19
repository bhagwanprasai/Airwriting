"""
Air Writer Application Configuration
====================================
Central configuration for the enhanced air writing app.

Features:
- JSON + image file storage (no database required)
- Model paths and caching
- Camera settings
- UI preferences
- Automatic environment detection
"""

from pathlib import Path
import os
from typing import Dict, Any

# Load .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ═══════════════════════════════════════════════════════════════════════════
# PROJECT PATHS
# ═══════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.parent
APP_DIR      = PROJECT_ROOT / "app"
MODEL_DIR    = PROJECT_ROOT / "model"
OUTPUT_DIR   = PROJECT_ROOT / "output"
CACHE_DIR    = PROJECT_ROOT / "cache"

OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# STORAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# All predictions and images live inside OUTPUT_DIR:
#   output/predictions.json   <- prediction records
#   output/cache.json         <- canvas-hash -> prediction cache
#   output/airwrite_<ts>.png  <- one PNG per saved prediction

PREDICTIONS_JSON = OUTPUT_DIR / "predictions.json"
CACHE_JSON       = OUTPUT_DIR / "cache.json"

# ═══════════════════════════════════════════════════════════════════════════
# MODEL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

MODEL_PATH    = str(MODEL_DIR / "ema_model.pth")
MODEL_VERSION = "v1.0"

ENABLE_MODEL_CACHE = True
MODEL_CACHE_DIR    = CACHE_DIR / "models"
MODEL_CACHE_DIR.mkdir(exist_ok=True)

ENABLE_PREDICTION_CACHE = True
PREDICTION_CACHE_SIZE   = 1000
CACHE_EXPIRY_DAYS       = 30

# ═══════════════════════════════════════════════════════════════════════════
# CAMERA CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

CAMERA_INDEX          = int(os.getenv('CAMERA_INDEX', 0))
CAMERA_WIDTH          = 640
CAMERA_HEIGHT         = 480
CAMERA_FPS            = 30
CAMERA_RETRY_ATTEMPTS = 3
CAMERA_RETRY_DELAY    = 1.0
ENABLE_DEMO_MODE      = True

# ═══════════════════════════════════════════════════════════════════════════
# UI CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

WINDOW_TITLE  = "Air Writer - Gesture-Based Text Input"
WINDOW_WIDTH  = 1280
WINDOW_HEIGHT = 720
FULLSCREEN    = False

CANVAS_WIDTH    = 640
CANVAS_HEIGHT   = 480
CANVAS_BG_COLOR = (0, 0, 0)
PEN_COLOR       = (255, 255, 255)
PEN_THICKNESS   = 5

AUTO_PREDICT_DELAY   = 2.0
SHOW_RESULT_DURATION = 3.0

# ═══════════════════════════════════════════════════════════════════════════
# HAND TRACKING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE  = 0.5
MAX_NUM_HANDS            = 1
DETECTION_SCALE          = 1.5
DEBOUNCE_FRAMES          = 5
GRACE_FRAMES             = 10
SMOOTHING_ALPHA_CURSOR   = 0.25
SMOOTHING_ALPHA_STROKE   = 0.45
MIN_MOVEMENT_PIXELS      = 1
STROKE_QUALITY_THRESHOLD = 0.3

# ═══════════════════════════════════════════════════════════════════════════
# LOGGING & DEBUG
# ═══════════════════════════════════════════════════════════════════════════

LOG_LEVEL           = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE            = PROJECT_ROOT / "air_writer.log"
ENABLE_FILE_LOGGING = False
DEBUG_MODE          = os.getenv('DEBUG', 'false').lower() in ('true', '1', 'yes')
SHOW_DEBUG_INFO     = DEBUG_MODE
SAVE_DEBUG_IMAGES   = DEBUG_MODE
ENABLE_PERFORMANCE_STATS = True
SHOW_FPS            = True

# ═══════════════════════════════════════════════════════════════════════════
# EXPORT / SAVE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

AUTO_SAVE_ENABLED  = True
AUTO_SAVE_INTERVAL = 300
ENABLE_IMAGE_EXPORT = True
ENABLE_JSON_EXPORT  = True
IMAGE_FORMAT        = 'PNG'
IMAGE_QUALITY       = 95
TIMESTAMP_FORMAT    = "%Y%m%d_%H%M%S"
FILENAME_PREFIX     = "airwrite"

# ═══════════════════════════════════════════════════════════════════════════
# PERFORMANCE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

USE_MULTITHREADING   = True
WORKER_THREADS       = 2
MAX_UNDO_HISTORY     = 20
CLEAR_CANVAS_ON_PREDICT = False
PREFER_GPU           = True
FALLBACK_TO_CPU      = True

# ═══════════════════════════════════════════════════════════════════════════
# ADVANCED FEATURES
# ═══════════════════════════════════════════════════════════════════════════

ENABLE_GAMMA_CORRECTION   = True
GAMMA_VALUE               = 1.2
ENABLE_ADAPTIVE_THRESHOLD = True
THRESHOLD_BLOCK_SIZE      = 11
THRESHOLD_C               = 2
NORMALIZE_STROKES         = True
TARGET_STROKE_HEIGHT      = 64
MAINTAIN_ASPECT_RATIO     = True

# ═══════════════════════════════════════════════════════════════════════════
# VOCABULARY (must match training)
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_VOCABULARY = sorted(set(
    " !\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ\\]abcdefghijklmnopqrstuvwxyz"
))

# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_storage_config() -> Dict[str, Any]:
    """Return config needed to initialise StorageManager."""
    return {"output_dir": str(OUTPUT_DIR)}


def get_app_info() -> Dict[str, Any]:
    return {
        "name":          "Air Writer Enhanced",
        "version":       "2.0",
        "model_version": MODEL_VERSION,
        "storage_type":  "JSON + PNG files",
        "cache_enabled": ENABLE_PREDICTION_CACHE,
        "debug_mode":    DEBUG_MODE,
    }


def validate_setup() -> Dict[str, bool]:
    return {
        "model_file": Path(MODEL_PATH).exists(),
        "output_dir": OUTPUT_DIR.exists(),
        "cache_dir":  CACHE_DIR.exists(),
    }


def print_config():
    print("=" * 70)
    print("AIR WRITER CONFIGURATION")
    print("=" * 70)
    for key, value in get_app_info().items():
        print(f"  {key:20s}: {value}")
    print("\nPaths:")
    print(f"  Project Root:      {PROJECT_ROOT}")
    print(f"  Model Path:        {MODEL_PATH}")
    print(f"  Output Dir:        {OUTPUT_DIR}")
    print(f"  Predictions JSON:  {PREDICTIONS_JSON}")
    print(f"  Cache JSON:        {CACHE_JSON}")
    print("\nFeatures:")
    print(f"  Prediction Cache:  {ENABLE_PREDICTION_CACHE}")
    print(f"  Gamma Correction:  {ENABLE_GAMMA_CORRECTION}")
    print(f"  Demo Mode:         {ENABLE_DEMO_MODE}")
    print(f"  Multi-threading:   {USE_MULTITHREADING}")
    print("\nValidation:")
    for check, passed in validate_setup().items():
        print(f"  {'✓' if passed else '✗'} {check}")
    print("=" * 70)


if __name__ == "__main__":
    print_config()