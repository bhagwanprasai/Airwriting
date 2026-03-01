"""
Configuration settings for Seq2Seq Attention model
"""

from pathlib import Path
import torch

class Config:
    """Centralized configuration for the entire training pipeline"""
    
    # PATHS - Use LOCAL drive (NOT OneDrive) for speed
    # Copy dataset first: robocopy "C:\Users\Binib\OneDrive\Desktop\Finaltraining" "C:\Finaltraining" /E /MT:8
    LABELS_CSV = Path("C:/Finaltraining/words.csv")
    IMAGES_PATH = Path("C:/Finaltraining/processed_word_dataset")
    
    # Output directories (created automatically)
    CHECKPOINT_DIR = Path("C:/Finaltraining/checkpoints_anti_overfit")
    PLOTS_DIR = Path("C:/Finaltraining/plots_anti_overfit")
    METRICS_DIR = Path("C:/Finaltraining/metrics_anti_overfit")
    
    BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"
    SWA_MODEL_PATH = CHECKPOINT_DIR / "swa_model.pth"
    EMA_MODEL_PATH = CHECKPOINT_DIR / "ema_model.pth"
    LOG_FILE = METRICS_DIR / "training_log.csv"
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # HARDWARE - OPTIMIZED FOR RTX 2060 (6GB VRAM) + R5 5600X
    BATCH_SIZE = 32  # Back to original that worked fast
    USE_AMP = torch.cuda.is_available()  # Essential for 2060
    NUM_WORKERS = 0  # CRITICAL: 0 is often faster on Windows (no multiprocessing overhead)
    PREFETCH_FACTOR = 2  # Load next batches in background
    PERSISTENT_WORKERS = False  # Not used when NUM_WORKERS=0
    
    # REGULARIZATION - TUNED FOR SEQ2SEQ
    EPOCHS = 150
    LEARNING_RATE = 0.001  # Higher LR for seq2seq
    WEIGHT_DECAY = 1e-4  # Lower weight decay
    DROPOUT = 0.3  # REDUCED - seq2seq needs lower dropout
    LABEL_SMOOTHING = 0.1  # Moderate smoothing
    
    # GRADIENT TECHNIQUES
    GRADIENT_ACCUMULATION_STEPS = 1  # No accumulation needed with batch 32
    GRADIENT_NOISE_STD = 0.0  # Disabled - can harm seq2seq training
    
    # EARLY STOPPING
    PATIENCE = 30  # More patience for seq2seq
    MIN_DELTA = 0.001
    
    # COSINE ANNEALING
    WARMUP_EPOCHS = 10  # Longer warmup for seq2seq
    FIRST_CYCLE_EPOCHS = 50
    
    # SWA
    SWA_START = 100
    SWA_LR = 0.0001
    
    # EMA
    USE_EMA = True
    EMA_DECAY = 0.999
    
    # AUGMENTATION - LIGHT for seq2seq (heavy aug hurts learning)
    CUTOUT_PROB = 0.15  # Reduced
    NOISE_PROB = 0.15   # Reduced
    
    # STOCHASTIC DEPTH (DropPath) - helps prevent overfitting
    DROP_PATH = 0.1
    
    # SEQ2SEQ SPECIFIC
    IMG_HEIGHT = 64
    MIN_WIDTH = 32
    MAX_WIDTH = 512
    ENCODER_HIDDEN = 256
    DECODER_HIDDEN = 256
    ATTENTION_HIDDEN = 128
    TEACHER_FORCING_RATIO = 1.0  # Start with full teacher forcing
    TEACHER_FORCING_DECAY = 0.995  # Slow decay
    MIN_TEACHER_FORCING = 0.5  # Don't go below this
    MAX_OUTPUT_LENGTH = 32  # Reduced from 50 (IAM words rarely >25 chars)
    
    # BEAM SEARCH
    BEAM_WIDTH = 5
    USE_BEAM_SEARCH = False  # Disable during early training
    
    # LAYER-WISE LR DECAY
    LAYER_LR_DECAY = 0.95  # Less aggressive layer decay


class SpecialTokens:
    """Special tokens for sequence encoding/decoding"""
    PAD = '<PAD>'
    SOS = '<SOS>'
    EOS = '<EOS>'
    UNK = '<UNK>'
