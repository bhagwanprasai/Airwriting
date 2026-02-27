"""
Configuration Module
Contains all configuration parameters for the CRNN model
"""

from pathlib import Path
import torch


class Config:
    # Base Paths
    WORKSPACE_PATH = Path(r"d:\code\dataset\iam_words")
    LABELS_CSV = Path(r"d:\code\dataset\iam_words\words.csv")
    IMAGES_PATH = Path(r"d:\code\dataset\iam_words\processed_word_dataset")
    
    # Output Directories
    CHECKPOINT_DIR = Path(r"d:\code\dataset\iam_words\checkpoints_final")
    PLOTS_DIR = Path(r"d:\code\dataset\iam_words\plots")
    METRICS_DIR = Path(r"d:\code\dataset\iam_words\metrics")
    
    # File Paths
    BEST_MODEL_PATH = Path(r"d:\code\dataset\iam_words\checkpoints_final\best_crnn_model.pth")
    SWA_MODEL_PATH = Path(r"d:\code\dataset\iam_words\checkpoints_final\swa_crnn_model.pth")
    LOG_FILE = Path(r"d:\code\dataset\iam_words\metrics\training_log.csv")
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hardware
    BATCH_SIZE = 64
    USE_AMP = torch.cuda.is_available()
    NUM_WORKERS = 4
    
    # Regularization
    EPOCHS = 100
    LEARNING_RATE = 0.0003
    WEIGHT_DECAY = 5e-4
    DROPOUT = 0.5
    DROPOUT_PATH = 0.1
    LABEL_SMOOTHING = 0.1
    
    # Gradient Accumulation
    GRADIENT_ACCUMULATION_STEPS = 2
    
    # Early Stopping
    PATIENCE = 30
    MIN_DELTA = 0.001
    
    # Cosine Annealing
    WARMUP_EPOCHS = 5
    FIRST_CYCLE_EPOCHS = 30
    
    # SWA
    SWA_START = 60
    SWA_LR = 0.0001
    
    # TTA
    USE_TTA = True
    TTA_TRANSFORMS = 3
    
    # Augmentation
    CUTOUT_PROB = 0.3
    ELASTIC_PROB = 0.3
    NOISE_PROB = 0.3
    
    # Model
    IMG_HEIGHT = 64
    MIN_WIDTH = 32
    MAX_WIDTH = 512
    LSTM_HIDDEN = 256
    
    # Layer-wise LR Decay
    LAYER_LR_DECAY = 0.95
