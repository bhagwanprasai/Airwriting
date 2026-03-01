"""
Data loading and preprocessing utilities
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset

from Training.config import Config, SpecialTokens


# ============================================================================
# Augmentation Functions
# ============================================================================

def cutout_augmentation(image, min_scale=0.02, max_scale=0.15):
    """Apply cutout augmentation to simulate occlusions"""
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
    """Add Gaussian noise to image"""
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
            alpha = np.random.uniform(0.9, 1.1)  # Narrower range
            image = np.clip(alpha * image, 0, 1)
        
        return image
    
    except Exception as e:
        # If augmentation fails, return original image
        import warnings
        warnings.warn(f"Augmentation failed: {e}, returning original image")
        return original_image


# ============================================================================
# Label Encoder
# ============================================================================

class LabelEncoder:
    """Encode/decode text to/from token indices"""
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        
    def build_vocab(self, texts):
        """Build vocabulary from list of texts"""
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
        """Encode text to list of indices"""
        indices = [self.char2idx[SpecialTokens.SOS]]
        for char in text:
            indices.append(self.char2idx.get(char, self.char2idx[SpecialTokens.UNK]))
        indices.append(self.char2idx[SpecialTokens.EOS])
        return indices
    
    def decode(self, indices):
        """Decode indices to text"""
        chars = []
        for idx in indices:
            if idx == self.char2idx[SpecialTokens.EOS]:
                break
            if idx not in [self.char2idx[SpecialTokens.PAD], 
                          self.char2idx[SpecialTokens.SOS]]:
                chars.append(self.idx2char.get(idx, ''))
        return ''.join(chars)
    
    def num_classes(self):
        """Return number of classes (vocab size)"""
        return len(self.char2idx)


# ============================================================================
# Dataset Utilities
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


# ============================================================================
# Dataset Class
# ============================================================================

class WordDataset(Dataset):
    """Dataset for handwriting word images"""
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


# ============================================================================
# Collate Function
# ============================================================================

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
