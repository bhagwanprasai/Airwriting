"""
CRNN with Seq2Seq Attention: ANTI-OVERFIT Edition
Specifically tuned for IAM Handwriting Dataset

CSV Format: 2 columns
- Column 1: Image filename (e.g., xyz.png)
- Column 2: Transcription text

NEW ANTI-OVERFIT TECHNIQUES:
✅ Smaller batch size (32 instead of 64)
✅ Moderate dropout (0.3 for seq2seq)
✅ Weight decay (1e-4)
✅ Light augmentation (no heavy elastic transform)
✅ Exponential Moving Average (EMA)
✅ Stochastic Depth (DropPath)
✅ Teacher forcing with slow decay
✅ Attention masking for variable-width images

ALL PREVIOUS TECHNIQUES PRESERVED:
✅ Bahdanau Attention, Teacher Forcing, Beam Search
✅ Layer-wise LR Decay, SWA, Cosine Annealing
✅ Gradient Accumulation, 70/15/15 Split
"""

import os
import math
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
from pathlib import Path
import cv2

# Suppress OpenCV warnings (file not found, etc.)
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm import tqdm
from collections import defaultdict, Counter
import copy

# ============================================================================
# Configuration - ANTI-OVERFIT TUNED
# ============================================================================

class Config:
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
    GRADIENT_ACCUMULATION_STEPS = 1  # No accumulation needed with batch 96
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
    # ELASTIC_PROB removed - too heavy, causes errors, not needed for seq2seq
    
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

# ============================================================================
# Special Tokens
# ============================================================================

class SpecialTokens:
    PAD = '<PAD>'
    SOS = '<SOS>'
    EOS = '<EOS>'
    UNK = '<UNK>'

# ============================================================================
# Exponential Moving Average (EMA) - NEW
# ============================================================================

class EMA:
    """Exponential Moving Average of model parameters"""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

# ============================================================================
# Cosine Annealing Scheduler
# ============================================================================

class CosineAnnealingWarmupRestarts(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, first_cycle_steps, warmup_steps, min_lr=0.0, last_epoch=-1):
        self.first_cycle_steps = first_cycle_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_steps) / (self.first_cycle_steps - self.warmup_steps)
            return [self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress)) 
                   for base_lr in self.base_lrs]

# ============================================================================
# Advanced Augmentation
# ============================================================================

def elastic_transform(image, alpha=30, sigma=4, random_state=None):
    """
    Elastic deformation of images as described in [Simard2003].
    Fixed to handle cv2.remap constraints properly.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    
    shape = image.shape
    h, w = shape[:2]
    
    # Check dimensions are within OpenCV limits (SHRT_MAX = 32767)
    if h >= 32767 or w >= 32767:
        return image  # Skip if image is too large
    
    # Generate random displacement fields
    dx = cv2.GaussianBlur(
        (random_state.rand(h, w).astype(np.float32) * 2 - 1), 
        (0, 0), sigma
    ).astype(np.float32) * alpha
    dy = cv2.GaussianBlur(
        (random_state.rand(h, w).astype(np.float32) * 2 - 1), 
        (0, 0), sigma
    ).astype(np.float32) * alpha
    
    # Create coordinate arrays
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Apply displacement and clip to valid range
    map_x = np.clip(x + dx, 0, w - 1).astype(np.float32)
    map_y = np.clip(y + dy, 0, h - 1).astype(np.float32)
    
    # Apply remapping
    return cv2.remap(
        image, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

def cutout_augmentation(image, min_scale=0.02, max_scale=0.15):
    h, w = image.shape
    scale = np.random.uniform(min_scale, max_scale)
    mask_h = int(h * scale)
    mask_w = int(w * scale)
    
    y = np.random.randint(0, h - mask_h + 1) if h > mask_h else 0
    x = np.random.randint(0, w - mask_w + 1) if w > mask_w else 0
    
    image = image.copy()
    image[y:y+mask_h, x:x+mask_w] = 0
    return image

def add_gaussian_noise(image, mean=0, sigma=0.05):
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)

def apply_heavy_aug(image):
    """LIGHT augmentation for seq2seq - focus on common variations"""
    original_image = image.copy()  # Preserve original for error recovery
    try:
        h, w = image.shape
        
        # Small rotation (handwriting tilt)
        if np.random.random() < 0.4:
            angle = np.random.uniform(-5, 5)  # Reduced angle
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            image = cv2.warpAffine(image, M, (w, h), borderValue=(0.0,))
        
        # Light blur (capture image quality variation)
        if np.random.random() < 0.3:
            kernel = 3  # Only small kernel
            image = cv2.GaussianBlur(image, (kernel, kernel), 0)
        
        # Brightness variation
        if np.random.random() < 0.4:
            scale = np.random.uniform(0.85, 1.15)  # Narrower range
            image = np.clip(image * scale, 0, 1)
        
        # Cutout (random occlusion)
        if np.random.random() < Config.CUTOUT_PROB:
            image = cutout_augmentation(image, min_scale=0.02, max_scale=0.1)  # Smaller cutouts
        
        # Light gaussian noise
        if np.random.random() < Config.NOISE_PROB:
            image = add_gaussian_noise(image, sigma=0.05)  # Reduced noise
        
        # Contrast variation
        if np.random.random() < 0.2:  # Reduced probability
            alpha = np.random.uniform(0.9, 1.1)  # Narrower rangeto
            image = np.clip(alpha * image, 0, 1)
        
        return image
    
    except Exception as e:
        # If augmentation fails, return original image
        import warnings
        warnings.warn(f"Augmentation failed: {e}, returning original image")
        return original_image

# ============================================================================
# Label Encoder with Special Tokens
# ============================================================================

class LabelEncoder:
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        
    def build_vocab(self, texts):
        chars = sorted(set(''.join(texts)))
        
        self.char2idx = {
            SpecialTokens.PAD: 0,
            SpecialTokens.SOS: 1,
            SpecialTokens.EOS: 2,
            SpecialTokens.UNK: 3
        }
        
        for i, char in enumerate(chars, start=4):
            self.char2idx[char] = i
        
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        
        print(f"Vocab size: {len(self.char2idx)} (including special tokens)")
    
    def encode(self, text):
        indices = [self.char2idx[SpecialTokens.SOS]]
        for char in text:
            indices.append(self.char2idx.get(char, self.char2idx[SpecialTokens.UNK]))
        indices.append(self.char2idx[SpecialTokens.EOS])
        return indices
    
    def decode(self, indices):
        chars = []
        for idx in indices:
            if idx == self.char2idx[SpecialTokens.EOS]:
                break
            if idx not in [self.char2idx[SpecialTokens.PAD], 
                          self.char2idx[SpecialTokens.SOS]]:
                chars.append(self.idx2char.get(idx, ''))
        return ''.join(chars)
    
    def num_classes(self):
        return len(self.char2idx)

# ============================================================================
# Dataset - FIXED FOR 2-COLUMN CSV
# ============================================================================

def clean_dataset(df):
    """
    Clean dataset - works with any column names
    Assumes first column = filename, second column = transcription
    """
    # Get column names
    cols = df.columns.tolist()
    
    # Rename to standard names
    if len(cols) >= 2:
        df = df.rename(columns={cols[0]: 'filename', cols[1]: 'transcription'})
    else:
        raise ValueError(f"CSV must have at least 2 columns, found {len(cols)}")
    
    # Clean
    df = df.dropna(subset=['transcription'])
    df = df[df['transcription'].str.len() > 0]
    df = df[df['transcription'].str.len() <= Config.MAX_OUTPUT_LENGTH - 2]
    
    print(f"Dataset cleaned: {len(df)} samples")
    print(f"Sample row: {df.iloc[0]['filename']} -> '{df.iloc[0]['transcription']}'")
    
    return df.reset_index(drop=True)

def validate_dataset_files(df, images_path, skip_if_not_exists=False):
    """
    Validate that image directory exists. Skip file-by-file validation on slow filesystems.
    
    Args:
        df: DataFrame with 'filename' column
        images_path: Path to images directory
        skip_if_not_exists: If True, skip validation if directory doesn't exist
    """
    images_path = Path(images_path)
    
    if not images_path.exists():
        if skip_if_not_exists:
            print(f"\n⚠ Image directory not found: {images_path}")
            print(f"  Skipping validation (will be validated on training machine)")
            return df
        else:
            raise FileNotFoundError(f"Images directory does not exist: {images_path}")
    
    print(f"\n✓ Image directory exists: {images_path}")
    print(f"  Skipping file-by-file validation (slow on OneDrive)")
    print(f"  Missing files will use blank images during training")
    
    # Return full dataframe - we'll handle missing files during training
    return df

class WordDataset(Dataset):
    # Class-level counter for failed loads
    _load_failures = 0
    # Class-level shared path map (built once, shared across train/val/test datasets)
    _shared_path_map = None
    _shared_path_map_dir = None
    
    @classmethod
    def build_shared_path_map(cls, images_path):
        """Scan directory tree ONCE using os.scandir (fastest on Windows).
        Call this ONCE in main() before creating any datasets."""
        images_path = str(images_path)
        if cls._shared_path_map is not None and cls._shared_path_map_dir == images_path:
            return  # Already built for this directory
        
        path_map = {}
        
        def scan_dir(directory):
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        if entry.is_file() and entry.name.endswith('.png'):
                            path_map[entry.name] = entry.path
                        elif entry.is_dir():
                            scan_dir(entry.path)
            except PermissionError:
                pass
        
        print(f"  Scanning image directory (one-time)...")
        scan_dir(images_path)
        print(f"  Found {len(path_map)} images")
        cls._shared_path_map = path_map
        cls._shared_path_map_dir = images_path
    
    def __init__(self, df, images_path, encoder, augment=False):
        self.df = df
        self.images_path = Path(images_path)
        self.encoder = encoder
        self.augment = augment
        
        # Use shared path map (built once in main, shared across all datasets)
        if WordDataset._shared_path_map is None:
            WordDataset.build_shared_path_map(images_path)
        self._path_map = WordDataset._shared_path_map
        
        found = sum(1 for _, row in df.iterrows() if row['filename'] in self._path_map)
        print(f"  Dataset: {len(df)} samples, {found} images found on disk")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename']
        
        # Use pre-built path map - ZERO filesystem calls per image
        if filename in self._path_map:
            image = cv2.imread(self._path_map[filename], cv2.IMREAD_GRAYSCALE)
        else:
            image = None
        
        if image is None:
            WordDataset._load_failures += 1
            if WordDataset._load_failures == 10:
                print(f"\n⚠ {WordDataset._load_failures}+ images failed to load")
            image = np.zeros((Config.IMG_HEIGHT, Config.MIN_WIDTH), dtype=np.uint8)
        
        image = image.astype(np.float32) / 255.0
        
        # Apply augmentation BEFORE resize
        if self.augment:
            image = apply_heavy_aug(image)
        
        # Resize height to IMG_HEIGHT, keep aspect ratio
        h, w = image.shape
        scale = Config.IMG_HEIGHT / h
        new_w = int(w * scale)
        new_w = max(Config.MIN_WIDTH, min(new_w, Config.MAX_WIDTH))
        image = cv2.resize(image, (new_w, Config.IMG_HEIGHT))
        
        # DON'T pad here - let collate_fn handle dynamic padding
        image = torch.FloatTensor(image).unsqueeze(0)  # [1, H, W]
        
        text = row['transcription']
        target = self.encoder.encode(text)
        
        # Return width for dynamic padding
        return image, target, text, new_w

def collate_fn(batch):
    """Dynamically pad images to batch max width with BUCKET padding.
    Rounds up to nearest multiple of 32 so cudnn.benchmark only sees ~16 distinct
    sizes instead of hundreds (prevents constant CUDA algorithm re-tuning)."""
    images, targets, texts, widths = zip(*batch)
    
    # Find max width in this batch, then round up to bucket boundary
    max_w = max(widths)
    BUCKET = 32  # Only ~16 possible sizes (32..512), cudnn.benchmark caches all
    max_w = ((max_w + BUCKET - 1) // BUCKET) * BUCKET
    max_w = min(max_w, Config.MAX_WIDTH)  # Cap at MAX_WIDTH
    batch_size = len(images)
    
    # Pad images to bucketed width
    padded_imgs = torch.zeros(batch_size, 1, Config.IMG_HEIGHT, max_w)
    for i, (img, w) in enumerate(zip(images, widths)):
        padded_imgs[i, :, :, :w] = img
    
    # Pad targets
    max_target_len = max(len(t) for t in targets)
    padded_targets = []
    target_lengths = []
    
    for target in targets:
        target_lengths.append(len(target))
        padded = target + [0] * (max_target_len - len(target))
        padded_targets.append(padded)
    
    targets = torch.LongTensor(padded_targets)
    target_lengths = torch.LongTensor(target_lengths)
    
    # CRITICAL: Create attention mask based on actual widths
    # After CNN: width becomes W/4 (due to 2x MaxPool2d(2,2))
    encoder_seq_len = max_w // 4
    attention_mask = torch.zeros(batch_size, encoder_seq_len, dtype=torch.bool)
    for i, w in enumerate(widths):
        valid_len = w // 4  # After CNN pooling
        attention_mask[i, :valid_len] = True
    
    return padded_imgs, targets, target_lengths, texts, attention_mask, torch.LongTensor(list(widths))

# ============================================================================
# DropPath (Stochastic Depth) for Anti-Overfitting
# ============================================================================

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

# ============================================================================
# Residual Block for CNN
# ============================================================================

class ResidualBlock(nn.Module):
    """Residual block with batch normalization and optional DropPath"""
    def __init__(self, in_channels, out_channels, drop_path=0.0):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.drop_path(out)  # Apply stochastic depth
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out

# ============================================================================
# Bahdanau Attention Mechanism - FIXED with Layer Normalization
# ============================================================================

class BahdanauAttention(nn.Module):
    def __init__(self, encoder_hidden, decoder_hidden, attention_hidden):
        super(BahdanauAttention, self).__init__()
        
        self.attention_hidden = attention_hidden
        self.encoder_projection = nn.Linear(encoder_hidden * 2, attention_hidden, bias=False)
        self.decoder_projection = nn.Linear(decoder_hidden, attention_hidden, bias=False)
        self.energy = nn.Linear(attention_hidden, 1, bias=False)
        
        # Add layer normalization for stability
        self.layer_norm = nn.LayerNorm(attention_hidden)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        batch_size, seq_len, encoder_dim = encoder_outputs.size()
        
        # Project encoder outputs: [batch, seq_len, attention_hidden]
        encoder_proj = self.encoder_projection(encoder_outputs)
        
        # Project decoder hidden: [batch, 1, attention_hidden]
        decoder_proj = self.decoder_projection(decoder_hidden).unsqueeze(1)
        
        # Add and apply layer norm
        combined = encoder_proj + decoder_proj
        combined = self.layer_norm(combined)
        
        # Calculate energy: [batch, seq_len]
        energy = self.energy(torch.tanh(combined))
        energy = energy.squeeze(2)
        
        # Apply mask if provided
        if mask is not None:
            energy = energy.masked_fill(~mask, float('-inf'))
        
        # Calculate attention weights: [batch, seq_len]
        attention_weights = F.softmax(energy, dim=1)
        attention_weights = self.dropout(attention_weights)
        
        # Calculate context vector: [batch, encoder_dim]
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)
        
        return context, attention_weights

# ============================================================================
# CNN Encoder - With DropPath for Anti-Overfitting
# ============================================================================

class CNNEncoder(nn.Module):
    """CNN encoder with clear pooling strategy for seq2seq"""
    def __init__(self, drop_path=0.1):
        super(CNNEncoder, self).__init__()
        
        # Block 1: 64 -> H/2, W/2
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # H/2, W/2
        )
        
        # Block 2: 128 -> H/4, W/4
        self.block2 = nn.Sequential(
            ResidualBlock(64, 128, drop_path=drop_path),
            nn.MaxPool2d(2, 2)  # H/4, W/4
        )
        
        # Block 3: 256 -> H/8, W/4 (only pool height)
        self.block3 = nn.Sequential(
            ResidualBlock(128, 256, drop_path=drop_path),
            ResidualBlock(256, 256, drop_path=drop_path),
            nn.MaxPool2d((2, 1), (2, 1))  # H/8, same W
        )
        
        # Block 4: 512 -> H/16, W/4
        self.block4 = nn.Sequential(
            ResidualBlock(256, 512, drop_path=drop_path),
            nn.MaxPool2d((2, 1), (2, 1))  # H/16, same W
        )
        
        # For height 64: after block4 = 64/16 = 4
        # Adaptive pool to height 1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        
        self.dropout = nn.Dropout(Config.DROPOUT)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # x: [batch, 1, 64, width]
        out = self.block1(x)   # [batch, 64, 32, W/2]
        out = self.block2(out)  # [batch, 128, 16, W/4]
        out = self.block3(out)  # [batch, 256, 8, W/4]
        out = self.block4(out)  # [batch, 512, 4, W/4]
        
        # Adaptive pool to get height = 1
        batch, channels, height, width = out.size()
        out = self.adaptive_pool(out)  # [batch, 512, 1, W/4]
        out = self.dropout(out)
        
        # Reshape for RNN: [batch, seq_len, features]
        out = out.squeeze(2)  # [batch, 512, W/4]
        out = out.permute(0, 2, 1)  # [batch, W/4, 512]
        
        return out

# ============================================================================
# LSTM Encoder
# ============================================================================

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.3):
        super(LSTMEncoder, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0,
            bidirectional=True
        )
        
        # Initialize LSTM weights
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
        
    def forward(self, x):
        outputs, hidden = self.lstm(x)
        return outputs, hidden

# ============================================================================
# LSTM Decoder with Attention and Residual Connection - FIXED
# ============================================================================

class LSTMDecoder(nn.Module):
    def __init__(self, num_classes, encoder_hidden, decoder_hidden, attention_hidden, dropout=0.3):
        super(LSTMDecoder, self).__init__()
        
        self.num_classes = num_classes
        self.decoder_hidden = decoder_hidden
        self.encoder_hidden = encoder_hidden
        
        # Embedding with proper initialization
        self.embedding = nn.Embedding(num_classes, decoder_hidden, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Attention mechanism
        self.attention = BahdanauAttention(encoder_hidden, decoder_hidden, attention_hidden)
        
        # LSTM cell - takes embedded + context
        self.lstm = nn.LSTMCell(
            input_size=decoder_hidden + encoder_hidden * 2,
            hidden_size=decoder_hidden
        )
        
        # Output projection: hidden + context -> num_classes
        self.fc_out = nn.Sequential(
            nn.Linear(decoder_hidden + encoder_hidden * 2, decoder_hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Initialize embedding
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        self.embedding.weight.data[0].fill_(0)  # PAD token
        
        # Initialize LSTM cell
        nn.init.xavier_uniform_(self.lstm.weight_ih)
        nn.init.orthogonal_(self.lstm.weight_hh)
        nn.init.zeros_(self.lstm.bias_ih)
        nn.init.zeros_(self.lstm.bias_hh)
        # Set forget gate bias to 1
        self.lstm.bias_hh.data[self.decoder_hidden:2*self.decoder_hidden].fill_(1.0)
        
        # Initialize output layers
        for m in self.fc_out.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, input_token, hidden, cell, encoder_outputs, mask=None):
        # Embedding: [batch, decoder_hidden]
        embedded = self.embedding(input_token)
        embedded = self.embedding_dropout(embedded)
        
        # Attention: [batch, encoder_hidden * 2]
        context, attention_weights = self.attention(hidden, encoder_outputs, mask)
        
        # LSTM input: concatenate embedded and context
        lstm_input = torch.cat([embedded, context], dim=1)
        
        # LSTM step
        new_hidden, new_cell = self.lstm(lstm_input, (hidden, cell))
        new_hidden = self.dropout(new_hidden)
        
        # Output: concatenate hidden and context
        output_input = torch.cat([new_hidden, context], dim=1)
        output = self.fc_out(output_input)
        
        return output, new_hidden, new_cell, attention_weights

# ============================================================================
# Seq2Seq Model
# ============================================================================

class Seq2SeqAttention(nn.Module):
    def __init__(self, num_classes, encoder_hidden, decoder_hidden, attention_hidden, dropout=0.5, drop_path=0.1):
        super(Seq2SeqAttention, self).__init__()
        
        self.num_classes = num_classes
        self.encoder_hidden = encoder_hidden
        self.decoder_hidden = decoder_hidden
        
        self.cnn_encoder = CNNEncoder(drop_path=drop_path)  # With stochastic depth
        self.lstm_encoder = LSTMEncoder(512, encoder_hidden, dropout)
        self.decoder = LSTMDecoder(num_classes, encoder_hidden, decoder_hidden, attention_hidden, dropout)
        
        # Bridge layers: LSTM encoder is bidirectional with 2 layers
        # h_n shape: [num_layers * 2, batch, hidden] = [4, batch, hidden]
        # We'll use only the top layer's hidden states (last 2 directions)
        bridge_input_dim = encoder_hidden * 2  # Just top layer, both directions
        
        self.encoder_to_decoder_h = nn.Linear(bridge_input_dim, decoder_hidden)
        self.encoder_to_decoder_c = nn.Linear(bridge_input_dim, decoder_hidden)
        
        # Initialize bridge
        nn.init.xavier_uniform_(self.encoder_to_decoder_h.weight)
        nn.init.xavier_uniform_(self.encoder_to_decoder_c.weight)
        nn.init.zeros_(self.encoder_to_decoder_h.bias)
        nn.init.zeros_(self.encoder_to_decoder_c.bias)
        
    def forward(self, images, targets=None, teacher_forcing_ratio=0.5, attention_mask=None):
        """Forward pass with optional attention mask for padding"""
        batch_size = images.size(0)
        device = images.device
        
        # CNN encoding: [batch, seq_len, 512]
        cnn_features = self.cnn_encoder(images)
        
        # LSTM encoding: encoder_outputs [batch, seq_len, encoder_hidden*2]
        # h_n, c_n: [num_layers*2, batch, encoder_hidden] = [4, batch, hidden]
        encoder_outputs, (h_n, c_n) = self.lstm_encoder(cnn_features)
        
        # Use only top layer hidden states (last 2 are from top layer: forward and backward)
        # h_n[-2] = top layer forward, h_n[-1] = top layer backward
        top_h = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [batch, encoder_hidden * 2]
        top_c = torch.cat([c_n[-2], c_n[-1]], dim=1)  # [batch, encoder_hidden * 2]
        
        # Bridge to decoder initial states: [batch, decoder_hidden]
        decoder_hidden = torch.tanh(self.encoder_to_decoder_h(top_h))
        decoder_cell = torch.tanh(self.encoder_to_decoder_c(top_c))
        
        seq_len = encoder_outputs.size(1)
        
        # Use provided mask or create default (all True)
        if attention_mask is None:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        else:
            # Ensure mask matches encoder output sequence length
            mask = attention_mask.to(device)
            if mask.size(1) != seq_len:
                # Adjust mask size if CNN output differs
                mask = mask[:, :seq_len] if mask.size(1) > seq_len else F.pad(mask, (0, seq_len - mask.size(1)), value=False)
        
        if targets is not None:
            max_len = targets.size(1)
            outputs = torch.zeros(batch_size, max_len, self.num_classes, device=device)
            attention_weights_list = []
            
            input_token = targets[:, 0]
            
            for t in range(1, max_len):
                output, decoder_hidden, decoder_cell, attn_weights = self.decoder(
                    input_token, decoder_hidden, decoder_cell, encoder_outputs, mask
                )
                
                outputs[:, t] = output
                attention_weights_list.append(attn_weights)
                
                use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
                if use_teacher_forcing:
                    input_token = targets[:, t]
                else:
                    input_token = output.argmax(dim=1)
            
            attention_weights = torch.stack(attention_weights_list, dim=1)
        else:
            max_len = Config.MAX_OUTPUT_LENGTH
            outputs = torch.zeros(batch_size, max_len, self.num_classes, device=device)
            attention_weights_list = []
            
            input_token = torch.full((batch_size,), 1, dtype=torch.long, device=device)
            
            for t in range(max_len):
                output, decoder_hidden, decoder_cell, attn_weights = self.decoder(
                    input_token, decoder_hidden, decoder_cell, encoder_outputs, mask
                )
                
                outputs[:, t] = output
                attention_weights_list.append(attn_weights)
                
                input_token = output.argmax(dim=1)
                
                if (input_token == 2).all():
                    break
            
            attention_weights = torch.stack(attention_weights_list, dim=1) if attention_weights_list else None
        
        return outputs, attention_weights

# ============================================================================
# Beam Search Decoder - FIXED
# ============================================================================

class BeamSearchDecoder:
    def __init__(self, model, beam_width=5):
        self.model = model
        self.beam_width = beam_width
        
    def decode(self, images, attention_mask=None):
        """Beam search with optional attention mask"""
        self.model.eval()
        batch_size = images.size(0)
        device = images.device
        
        predictions = []
        
        with torch.no_grad():
            cnn_features = self.model.cnn_encoder(images)
            encoder_outputs, (h_n, c_n) = self.model.lstm_encoder(cnn_features)
            
            seq_len = encoder_outputs.size(1)
            
            for b in range(batch_size):
                # Extract top layer hidden states for this sample
                # h_n[-2] = top layer forward, h_n[-1] = top layer backward
                top_h = torch.cat([h_n[-2, b:b+1], h_n[-1, b:b+1]], dim=1)  # [1, encoder_hidden * 2]
                top_c = torch.cat([c_n[-2, b:b+1], c_n[-1, b:b+1]], dim=1)  # [1, encoder_hidden * 2]
                
                decoder_hidden = torch.tanh(self.model.encoder_to_decoder_h(top_h))
                decoder_cell = torch.tanh(self.model.encoder_to_decoder_c(top_c))
                
                encoder_outputs_b = encoder_outputs[b:b+1]
                
                # Use attention mask if provided
                if attention_mask is not None:
                    mask_b = attention_mask[b:b+1].to(device)
                    # Adjust mask size if needed
                    if mask_b.size(1) != seq_len:
                        mask_b = mask_b[:, :seq_len] if mask_b.size(1) > seq_len else F.pad(mask_b, (0, seq_len - mask_b.size(1)), value=False)
                else:
                    mask_b = torch.ones(1, seq_len, dtype=torch.bool, device=device)
                
                # beams: (log_score, sequence, hidden, cell)
                # Start with log(1) = 0
                beams = [(0.0, [1], decoder_hidden, decoder_cell)]
                
                for _ in range(Config.MAX_OUTPUT_LENGTH):
                    candidates = []
                    
                    for log_score, sequence, hidden, cell in beams:
                        if sequence[-1] == 2:  # EOS token
                            candidates.append((log_score, sequence, hidden, cell))
                            continue
                        
                        input_token = torch.tensor([sequence[-1]], device=device)
                        output, new_hidden, new_cell, _ = self.model.decoder(
                            input_token, hidden, cell, encoder_outputs_b, mask_b
                        )
                        
                        log_probs = F.log_softmax(output, dim=1)
                        top_log_probs, top_indices = log_probs.topk(self.beam_width)
                        
                        for log_prob, idx in zip(top_log_probs[0], top_indices[0]):
                            # ADD log probabilities (not multiply)
                            new_log_score = log_score + log_prob.item()
                            new_sequence = sequence + [idx.item()]
                            candidates.append((new_log_score, new_sequence, new_hidden, new_cell))
                    
                    # Sort by log score (higher is better)
                    beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:self.beam_width]
                    
                    if all(seq[-1] == 2 for _, seq, _, _ in beams):
                        break
                
                best_sequence = beams[0][1]
                predictions.append(best_sequence)
        
        return predictions

# ============================================================================
# Training Functions with Gradient Noise - NEW
# ============================================================================

def add_gradient_noise(model, std=0.01):
    """Add Gaussian noise to gradients to improve generalization"""
    for param in model.parameters():
        if param.grad is not None:
            noise = torch.randn_like(param.grad) * std
            param.grad.add_(noise)

def train_one_epoch(model, loader, criterion, optimizer, scaler, teacher_forcing_ratio, epoch, ema=None):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    
    optimizer.zero_grad()
    
    for batch_idx, (images, targets, target_lengths, texts, attention_mask, widths) in enumerate(pbar):
        images = images.to(Config.DEVICE, non_blocking=True)
        targets = targets.to(Config.DEVICE, non_blocking=True)
        attention_mask = attention_mask.to(Config.DEVICE, non_blocking=True)
        
        if Config.USE_AMP:
            with torch.amp.autocast('cuda'):
                outputs, _ = model(images, targets, teacher_forcing_ratio, attention_mask)
                
                outputs = outputs[:, 1:, :].contiguous()
                targets = targets[:, 1:].contiguous()
                
                outputs = outputs.view(-1, outputs.size(2))
                targets = targets.view(-1)
                
                loss = criterion(outputs, targets)
        else:
            outputs, _ = model(images, targets, teacher_forcing_ratio, attention_mask)
            
            outputs = outputs[:, 1:, :].contiguous()
            targets = targets[:, 1:].contiguous()
            
            outputs = outputs.view(-1, outputs.size(2))
            targets = targets.view(-1)
            
            loss = criterion(outputs, targets)
        
        loss = loss / Config.GRADIENT_ACCUMULATION_STEPS
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            
            # Add gradient noise - NEW
            if Config.GRADIENT_NOISE_STD > 0:
                add_gradient_noise(model, Config.GRADIENT_NOISE_STD)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Update EMA - NEW
            if ema is not None:
                ema.update(model)
        
        total_loss += loss.item() * Config.GRADIENT_ACCUMULATION_STEPS
        pbar.set_postfix({'loss': f'{loss.item() * Config.GRADIENT_ACCUMULATION_STEPS:.4f}'})
    
    return total_loss / len(loader)

def validate(model, loader, criterion, encoder, use_beam_search=False):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_ground_truths = []
    num_batches = 0
    
    # SPEED: Only run greedy on subset, skip beam search during training validation
    max_metric_batches = 20  # Only decode 20 batches for WER/CER (~2000 samples)
    
    with torch.no_grad():
        for batch_idx, (images, targets, target_lengths, texts, attention_mask, widths) in enumerate(tqdm(loader, desc="[Validation]")):
            images = images.to(Config.DEVICE, non_blocking=True)
            targets = targets.to(Config.DEVICE, non_blocking=True)
            attention_mask = attention_mask.to(Config.DEVICE, non_blocking=True)
            
            # Calculate loss using teacher-forced outputs (FAST)
            if Config.USE_AMP:
                with torch.amp.autocast('cuda'):
                    outputs, _ = model(images, targets, teacher_forcing_ratio=1.0, attention_mask=attention_mask)
                    
                    outputs_loss = outputs[:, 1:, :].contiguous()
                    targets_loss = targets[:, 1:].contiguous()
                    
                    outputs_loss = outputs_loss.view(-1, outputs_loss.size(2))
                    targets_loss = targets_loss.view(-1)
                    
                    loss = criterion(outputs_loss, targets_loss)
            else:
                outputs, _ = model(images, targets, teacher_forcing_ratio=1.0, attention_mask=attention_mask)
                
                outputs_loss = outputs[:, 1:, :].contiguous()
                targets_loss = targets[:, 1:].contiguous()
                
                outputs_loss = outputs_loss.view(-1, outputs_loss.size(2))
                targets_loss = targets_loss.view(-1)
                
                loss = criterion(outputs_loss, targets_loss)
            
            total_loss += loss.item()
            num_batches += 1
            
            # SPEED: Only decode a subset of batches for metrics
            if batch_idx < max_metric_batches:
                # Use greedy from teacher-forced outputs (FAST - no extra forward pass)
                pred_indices = outputs.argmax(dim=2).cpu().numpy()
                for seq in pred_indices:
                    all_predictions.append(encoder.decode(seq.tolist()))
                all_ground_truths.extend(texts)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    total_words = len(all_ground_truths)
    correct_words = sum([1 for pred, gt in zip(all_predictions, all_ground_truths) if pred == gt])
    wer = 1 - (correct_words / total_words) if total_words > 0 else 1.0
    
    total_chars = sum(len(gt) for gt in all_ground_truths)
    char_errors = sum(
        levenshtein_distance(pred, gt)
        for pred, gt in zip(all_predictions, all_ground_truths)
    )
    cer = char_errors / total_chars if total_chars > 0 else 1.0
    
    print(f"WER: {wer:.2%} | CER: {cer:.2%}")
    if len(all_ground_truths) > 0:
        print(f"Sample: GT='{all_ground_truths[0]}' | Pred='{all_predictions[0]}'")
    
    return avg_loss, wer, cer


def levenshtein_distance(s1, s2):
    """Compute edit distance between two sequences"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

# ============================================================================
# History Tracker
# ============================================================================

class HistoryTracker:
    def __init__(self):
        self.history = defaultdict(list)
        
    def update(self, epoch, train_loss, val_loss, wer, cer, lr):
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['wer'].append(wer)
        self.history['cer'].append(cer)
        self.history['lr'].append(lr)
        
        df = pd.DataFrame(self.history)
        df.to_csv(Config.LOG_FILE, index=False)
    
    def plot(self):
        df = pd.DataFrame(self.history)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
        axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val Loss', marker='s')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(df['epoch'], df['wer'], label='WER', color='red', marker='o')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('WER')
        axes[0, 1].set_title('Word Error Rate')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(df['epoch'], df['cer'], label='CER', color='green', marker='o')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('CER')
        axes[1, 0].set_title('Character Error Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(df['epoch'], df['lr'], label='Learning Rate', color='purple', marker='o')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(str(Config.PLOTS_DIR / 'training_curves.png'), dpi=300)
        plt.close()
        
    def plot_overfitting_gap(self):
        """Plot train vs val loss gap to detect overfitting"""
        df = pd.DataFrame(self.history)
        if len(df) < 2:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate gap
        gap = df['train_loss'] - df['val_loss']
        
        ax.fill_between(df['epoch'], df['train_loss'], df['val_loss'], 
                       alpha=0.3, label='Gap (Train - Val)')
        ax.plot(df['epoch'], df['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax.plot(df['epoch'], df['val_loss'], 'r-', label='Val Loss', linewidth=2)
        
        # Mark overfitting region (where train < val significantly)
        overfit_threshold = 0.1
        overfit_epochs = df[gap < -overfit_threshold]['epoch'].values
        if len(overfit_epochs) > 0:
            ax.axvspan(overfit_epochs[0], df['epoch'].max(), alpha=0.1, color='red', 
                      label='Potential Overfitting')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Overfitting Detection: Train vs Validation Loss Gap')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(str(Config.PLOTS_DIR / 'overfitting_gap.png'), dpi=300)
        plt.close()

# ============================================================================
# Visualization Functions
# ============================================================================

def visualize_attention(model, images, encoder, attention_mask=None, save_path=None, num_samples=4):
    """
    Visualize attention weights as heatmaps overlaid on input images.
    Shows where the model "looks" when predicting each character.
    """
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        outputs, attention_weights = model(images.to(device), attention_mask=attention_mask)
        predictions = outputs.argmax(dim=-1)
    
    if attention_weights is None:
        print("No attention weights available")
        return
    
    num_samples = min(num_samples, images.size(0))
    fig, axes = plt.subplots(num_samples, 2, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Original image
        img = images[i, 0].cpu().numpy()
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f'Input Image')
        axes[i, 0].axis('off')
        
        # Attention heatmap
        attn = attention_weights[i].cpu().numpy()  # (output_len, input_len)
        
        # Decode prediction
        pred_indices = predictions[i].cpu().numpy()
        pred_text = encoder.decode(pred_indices)
        
        # Plot attention heatmap
        im = axes[i, 1].imshow(attn, aspect='auto', cmap='hot')
        axes[i, 1].set_xlabel('Input Position (Image Features)')
        axes[i, 1].set_ylabel('Output Position (Characters)')
        axes[i, 1].set_title(f'Attention Map | Pred: "{pred_text}"')
        plt.colorbar(im, ax=axes[i, 1])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_predictions(model, dataloader, encoder, save_path=None, num_samples=10, use_beam=False):
    """
    Visualize sample predictions vs ground truth.
    Saves to file and returns results for analysis.
    """
    model.eval()
    device = next(model.parameters()).device
    
    results = []
    beam_decoder = BeamSearchDecoder(model, beam_width=5) if use_beam else None
    
    with torch.no_grad():
        for batch in dataloader:
            images, targets, target_lengths, input_lengths, attention_mask, widths = batch
            images = images.to(device)
            attention_mask = attention_mask.to(device)
            
            if use_beam:
                predictions = beam_decoder.decode(images, attention_mask)
            else:
                outputs, _ = model(images, attention_mask=attention_mask)
                pred_indices = outputs.argmax(dim=-1).cpu().numpy()
                predictions = [encoder.decode(p) for p in pred_indices]
            
            # Get ground truth
            for i in range(len(predictions)):
                gt_indices = targets[i, :target_lengths[i]].cpu().numpy()
                gt_text = encoder.decode(gt_indices)
                pred_text = predictions[i] if use_beam else predictions[i]
                
                results.append({
                    'ground_truth': gt_text,
                    'prediction': pred_text,
                    'correct': gt_text == pred_text,
                    'cer': levenshtein_distance(pred_text, gt_text) / max(len(gt_text), 1)
                })
                
                if len(results) >= num_samples:
                    break
            
            if len(results) >= num_samples:
                break
    
    # Save to file
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("Sample Predictions Report\n")
            f.write("=" * 60 + "\n\n")
            
            correct_count = sum(1 for r in results if r['correct'])
            f.write(f"Accuracy: {correct_count}/{len(results)} ({100*correct_count/len(results):.1f}%)\n\n")
            
            for i, r in enumerate(results):
                status = "✓" if r['correct'] else "✗"
                f.write(f"{i+1}. {status}\n")
                f.write(f"   GT:   '{r['ground_truth']}'\n")
                f.write(f"   Pred: '{r['prediction']}'\n")
                f.write(f"   CER:  {r['cer']:.2%}\n\n")
    
    return results

def plot_confusion_matrix(model, dataloader, encoder, save_path=None, max_samples=1000):
    """
    Generate character-level confusion matrix to identify common errors.
    """
    from collections import Counter
    
    model.eval()
    device = next(model.parameters()).device
    
    # Collect character-level errors
    substitutions = Counter()  # (gt_char, pred_char) -> count
    insertions = Counter()     # pred_char -> count
    deletions = Counter()      # gt_char -> count
    
    sample_count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images, targets, target_lengths, input_lengths, attention_mask, widths = batch
            images = images.to(device)
            attention_mask = attention_mask.to(device)
            
            outputs, _ = model(images, attention_mask=attention_mask)
            pred_indices = outputs.argmax(dim=-1).cpu().numpy()
            
            for i in range(len(pred_indices)):
                gt_indices = targets[i, :target_lengths[i]].cpu().numpy()
                gt_text = encoder.decode(gt_indices)
                pred_text = encoder.decode(pred_indices[i])
                
                # Simple alignment using edit operations
                gt_chars = list(gt_text)
                pred_chars = list(pred_text)
                
                # Count mismatches at each position
                for j in range(min(len(gt_chars), len(pred_chars))):
                    if gt_chars[j] != pred_chars[j]:
                        substitutions[(gt_chars[j], pred_chars[j])] += 1
                
                # Extra predictions = insertions
                if len(pred_chars) > len(gt_chars):
                    for c in pred_chars[len(gt_chars):]:
                        insertions[c] += 1
                
                # Missing predictions = deletions  
                if len(gt_chars) > len(pred_chars):
                    for c in gt_chars[len(pred_chars):]:
                        deletions[c] += 1
                
                sample_count += 1
                if sample_count >= max_samples:
                    break
            
            if sample_count >= max_samples:
                break
    
    # Generate report
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("Character-Level Error Analysis\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Analyzed {sample_count} samples\n\n")
            
            f.write("Top 20 Substitution Errors (GT -> Pred):\n")
            f.write("-" * 40 + "\n")
            for (gt, pred), count in substitutions.most_common(20):
                f.write(f"  '{gt}' -> '{pred}': {count}\n")
            
            f.write("\nTop 10 Insertion Errors (Extra chars):\n")
            f.write("-" * 40 + "\n")
            for char, count in insertions.most_common(10):
                f.write(f"  '{char}': {count}\n")
            
            f.write("\nTop 10 Deletion Errors (Missing chars):\n")
            f.write("-" * 40 + "\n")
            for char, count in deletions.most_common(10):
                f.write(f"  '{char}': {count}\n")
    
    # Visual confusion matrix for top confused pairs
    if save_path and len(substitutions) > 0:
        top_chars = set()
        for (gt, pred), _ in substitutions.most_common(50):
            top_chars.add(gt)
            top_chars.add(pred)
        top_chars = sorted(list(top_chars))[:20]  # Limit to 20 chars
        
        if len(top_chars) > 1:
            matrix = np.zeros((len(top_chars), len(top_chars)))
            char_to_idx = {c: i for i, c in enumerate(top_chars)}
            
            for (gt, pred), count in substitutions.items():
                if gt in char_to_idx and pred in char_to_idx:
                    matrix[char_to_idx[gt], char_to_idx[pred]] = count
            
            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(matrix, cmap='Reds')
            
            ax.set_xticks(range(len(top_chars)))
            ax.set_yticks(range(len(top_chars)))
            ax.set_xticklabels(top_chars)
            ax.set_yticklabels(top_chars)
            
            ax.set_xlabel('Predicted Character')
            ax.set_ylabel('Ground Truth Character')
            ax.set_title('Character Confusion Matrix (Top Errors)')
            
            plt.colorbar(im)
            plt.tight_layout()
            plt.savefig(str(Config.PLOTS_DIR / 'confusion_matrix.png'), dpi=300)
            plt.close()
    
    return {'substitutions': substitutions, 'insertions': insertions, 'deletions': deletions}

# ============================================================================
# Layer-wise Learning Rate
# ============================================================================

def get_layer_wise_params(model):
    params = []
    
    params.append({
        'params': model.cnn_encoder.parameters(),
        'lr': Config.LEARNING_RATE * (Config.LAYER_LR_DECAY ** 3),
        'name': 'cnn_encoder'
    })
    
    params.append({
        'params': model.lstm_encoder.parameters(),
        'lr': Config.LEARNING_RATE * (Config.LAYER_LR_DECAY ** 2),
        'name': 'lstm_encoder'
    })
    
    params.append({
        'params': model.decoder.parameters(),
        'lr': Config.LEARNING_RATE,
        'name': 'decoder'
    })
    
    params.append({
        'params': list(model.encoder_to_decoder_h.parameters()) + 
                 list(model.encoder_to_decoder_c.parameters()),
        'lr': Config.LEARNING_RATE * Config.LAYER_LR_DECAY,
        'name': 'bridge'
    })
    
    return params

# ============================================================================
# Main
# ============================================================================

def main():
    # GPU optimizations for RTX 2060
    torch.backends.cudnn.benchmark = True  # Auto-tune convolution algorithms
    torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 for matmul (faster)
    torch.backends.cudnn.allow_tf32 = True  # Use TF32 for cudnn (faster)
    
    # Print GPU info and config
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"Config: BATCH_SIZE={Config.BATCH_SIZE}, NUM_WORKERS={Config.NUM_WORKERS}")
    print(f"Loading from: {Config.IMAGES_PATH}")
    
    # Validate paths before starting
    print("=" * 60)
    print("Path Validation")
    print("=" * 60)
    
    if not Config.LABELS_CSV.exists():
        print(f"❌ ERROR: CSV file not found at: {Config.LABELS_CSV}")
        print(f"   Please update Config.LABELS_CSV to point to your words.csv file")
        return
    
    if not Config.IMAGES_PATH.exists():
        print(f"❌ ERROR: Images directory not found at: {Config.IMAGES_PATH}")
        print(f"   Please update Config.IMAGES_PATH to point to your processed_word_dataset folder")
        return
    
    print(f"✓ CSV file: {Config.LABELS_CSV}")
    print(f"✓ Images directory: {Config.IMAGES_PATH}")
    
    # Create directories
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.PLOTS_DIR, exist_ok=True)
    os.makedirs(Config.METRICS_DIR, exist_ok=True)
    
    print("=" * 60)
    print("Seq2Seq Attention Training")
    print("✅ Batch size: 32 (effective 64 with grad accumulation)")
    print("✅ Dropout: 0.3 (seq2seq optimized)")
    print("✅ Light augmentation (rotation, blur, noise)")
    print("✅ Attention masking for variable-width images")
    print("✅ EMA + SWA + DropPath for anti-overfitting")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading and cleaning dataset...")
    df = clean_dataset(pd.read_csv(Config.LABELS_CSV))
    
    # Validate that all image files exist (skip if on different machine)
    df = validate_dataset_files(df, Config.IMAGES_PATH, skip_if_not_exists=True)
    
    if len(df) == 0:
        print("\n❌ ERROR: No valid samples found in dataset!")
        print("   This usually means:")
        print("   1. Image directory path is incorrect")
        print("   2. Images are in subdirectories (IAM uses a01/, a01-000u/, etc.)")
        print("   3. Running on different machine - update Config.IMAGES_PATH")
        print("\n   The code will work when run on the training machine with correct paths.")
        return
    
    encoder = LabelEncoder()
    encoder.build_vocab(df['transcription'].values)
    
    # 70/15/15 split
    train_df = df.sample(frac=0.7, random_state=42)
    remaining_df = df.drop(train_df.index)
    val_df = remaining_df.sample(frac=0.5, random_state=42)
    test_df = remaining_df.drop(val_df.index)
    
    print(f"Train samples: {len(train_df)} (70%)")
    print(f"Val samples: {len(val_df)} (15%)")
    print(f"Test samples: {len(test_df)} (15%)")
    
    # Build path map ONCE - shared across all 3 datasets (avoids 3x dir scan)
    WordDataset.build_shared_path_map(Config.IMAGES_PATH)
    
    # Create dataloaders with optimized settings
    loader_kwargs = {
        'batch_size': Config.BATCH_SIZE,
        'num_workers': Config.NUM_WORKERS,
        'collate_fn': collate_fn,
        'pin_memory': True,
        'prefetch_factor': Config.PREFETCH_FACTOR if Config.NUM_WORKERS > 0 else None,
        'persistent_workers': Config.PERSISTENT_WORKERS if Config.NUM_WORKERS > 0 else False
    }
    
    train_loader = DataLoader(
        WordDataset(train_df, Config.IMAGES_PATH, encoder, augment=True),
        shuffle=True,
        **loader_kwargs
    )
    val_loader = DataLoader(
        WordDataset(val_df, Config.IMAGES_PATH, encoder, augment=False),
        shuffle=False,
        **loader_kwargs
    )
    test_loader = DataLoader(
        WordDataset(test_df, Config.IMAGES_PATH, encoder, augment=False),
        shuffle=False,
        **loader_kwargs
    )
    
    # Create model
    print("\n2. Creating Seq2Seq model...")
    model = Seq2SeqAttention(
        num_classes=encoder.num_classes(),
        encoder_hidden=Config.ENCODER_HIDDEN,
        decoder_hidden=Config.DECODER_HIDDEN,
        attention_hidden=Config.ATTENTION_HIDDEN,
        dropout=Config.DROPOUT,
        drop_path=Config.DROP_PATH  # Stochastic depth for anti-overfitting
    ).to(Config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Setup with layer-wise LR
    print("\n3. Setting up Layer-wise Learning Rates...")
    param_groups = get_layer_wise_params(model)
    for group in param_groups:
        print(f"  {group['name']}: LR = {group['lr']:.6f}")
    
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=Config.LABEL_SMOOTHING)
    optimizer = optim.AdamW(param_groups, weight_decay=Config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=Config.USE_AMP)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=Config.FIRST_CYCLE_EPOCHS,
        warmup_steps=Config.WARMUP_EPOCHS,
        min_lr=float(Config.LEARNING_RATE * 0.01)
    )
    
    # EMA Setup - NEW
    ema = EMA(model, decay=Config.EMA_DECAY) if Config.USE_EMA else None
    if ema:
        print(f"\n4. EMA enabled (decay={Config.EMA_DECAY})")
    
    # SWA Setup
    print(f"5. SWA will start at epoch {Config.SWA_START}")
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=Config.SWA_LR)
    swa_started = False
    
    tracker = HistoryTracker()
    
    best_wer = 1.0
    best_model_state = None
    patience_counter = 0
    teacher_forcing_ratio = Config.TEACHER_FORCING_RATIO
    
    print(f"\n6. Starting training...")
    print("=" * 60)
    
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        print(f"Teacher Forcing: {teacher_forcing_ratio:.3f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if epoch >= Config.SWA_START and not swa_started:
            print(f"🔄 Starting SWA")
            swa_started = True
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, 
                                     teacher_forcing_ratio, epoch+1, ema)
        
        # Update SWA
        if swa_started:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        
        # Validate
        use_beam = Config.USE_BEAM_SEARCH and epoch >= Config.WARMUP_EPOCHS
        val_loss, wer, cer = validate(model, val_loader, criterion, encoder, use_beam_search=use_beam)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Track
        tracker.update(epoch+1, train_loss, val_loss, wer, cer, optimizer.param_groups[0]['lr'])
        tracker.plot()
        
        # Save best
        if wer < (best_wer - Config.MIN_DELTA):
            best_wer = wer
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"✓ NEW BEST MODEL (WER: {wer:.2%})")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter}/{Config.PATIENCE} epochs. Best WER: {best_wer:.2%}")
        
        # Early stopping
        if patience_counter >= Config.PATIENCE:
            print(f"\n⚠ EARLY STOPPING")
            # Restore best weights
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                print("✓ Restored best model weights")
            break
        
        # Decay teacher forcing (with minimum)
        teacher_forcing_ratio = max(
            Config.MIN_TEACHER_FORCING, 
            teacher_forcing_ratio * Config.TEACHER_FORCING_DECAY
        )
        
        # Step scheduler
        if not swa_started:
            scheduler.step()
    
    # Update SWA BN
    if swa_started:
        print("\n7. Updating SWA Batch Normalization...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=Config.DEVICE)
        torch.save(swa_model.state_dict(), Config.SWA_MODEL_PATH)
        print(f"✓ SWA model saved")
    
    # Save EMA model
    if ema:
        print("\n8. Saving EMA model...")
        ema.apply_shadow(model)
        torch.save(model.state_dict(), Config.EMA_MODEL_PATH)
        ema.restore(model)
        print(f"✓ EMA model saved")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("9. Final Evaluation on Test Set")
    print("=" * 60)
    
    models_to_test = [
        ("Best Checkpoint", Config.BEST_MODEL_PATH),
    ]
    if swa_started:
        models_to_test.append(("SWA Model", Config.SWA_MODEL_PATH))
    if ema:
        models_to_test.append(("EMA Model", Config.EMA_MODEL_PATH))
    
    results = []
    for name, path in models_to_test:
        print(f"\nEvaluating {name}...")
        model.load_state_dict(torch.load(path, map_location=Config.DEVICE, weights_only=True))
        test_loss, test_wer, test_cer = validate(model, test_loader, criterion, encoder, use_beam_search=True)
        results.append((name, test_wer, test_cer))
        print(f"{name} - Test WER: {test_wer:.2%} | Test CER: {test_cer:.2%}")
    
    # Find best
    best_result = min(results, key=lambda x: x[1])
    print(f"\n✨ Best model: {best_result[0]} (WER: {best_result[1]:.2%})")
    
    # Load best model for visualizations
    best_model_map = {
        "Best Checkpoint": Config.BEST_MODEL_PATH,
        "SWA Model": Config.SWA_MODEL_PATH,
        "EMA Model": Config.EMA_MODEL_PATH
    }
    model.load_state_dict(torch.load(best_model_map[best_result[0]], 
                                      map_location=Config.DEVICE, weights_only=True))
    
    # Generate comprehensive visualizations
    print("\n" + "=" * 60)
    print("10. Generating Visualizations")
    print("=" * 60)
    
    # Overfitting gap plot
    print("  • Plotting overfitting analysis...")
    tracker.plot_overfitting_gap()
    
    # Sample predictions
    print("  • Generating sample predictions...")
    visualize_predictions(
        model, test_loader, encoder, 
        save_path=str(Config.METRICS_DIR / 'sample_predictions.txt'),
        num_samples=50, use_beam=True
    )
    
    # Attention visualization (on a few test samples)
    print("  • Visualizing attention maps...")
    test_batch = next(iter(test_loader))
    images, targets, target_lengths, input_lengths, attention_mask, widths = test_batch
    visualize_attention(
        model, images[:4], encoder, 
        attention_mask=attention_mask[:4].to(Config.DEVICE),
        save_path=str(Config.PLOTS_DIR / 'attention_maps.png'),
        num_samples=4
    )
    
    # Character-level error analysis
    print("  • Analyzing character-level errors...")
    plot_confusion_matrix(
        model, test_loader, encoder,
        save_path=str(Config.METRICS_DIR / 'character_errors.txt'),
        max_samples=500
    )
    
    print(f"\n✓ Visualizations saved to:")
    print(f"  Plots: {Config.PLOTS_DIR}")
    print(f"  Metrics: {Config.METRICS_DIR}")
    
    print("\n" + "=" * 60)
    print("Training Completed!")
    print("=" * 60)

if __name__ == '__main__':
    main()