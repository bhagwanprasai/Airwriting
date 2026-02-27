"""
Data Loading Module
Contains dataset classes and data loading utilities
"""

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path

from config import Config
from augmentation import apply_heavy_aug, cutout_augmentation, add_gaussian_noise


def clean_dataset(df):
    """Clean dataset"""
    df = df.dropna(subset=['transcription'])
    mask = df['transcription'].astype(str).str.match(r'^[a-zA-Z0-9]+$')
    return df[mask].copy()


class WordDataset(Dataset):
    """Dataset with enhanced augmentation"""
    def __init__(self, df, images_path, label_encoder, augment=False):
        self.df = df.reset_index(drop=True)
        self.images_path = Path(images_path)
        self.label_encoder = label_encoder
        self.augment = augment

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = str(row['filename'])
        if fname.endswith('.png'):
            fname = fname[:-4]
        text = str(row['transcription'])
        
        parts = fname.split('-')
        possible_paths = [self.images_path / f"{fname}.png"]
        if len(parts) >= 2:
            possible_paths.append(
                self.images_path / parts[0] / f"{parts[0]}-{parts[1]}" / f"{fname}.png"
            )
        
        img = None
        for p in possible_paths:
            if p.exists():
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                break
        
        if img is None:
            img = np.zeros((Config.IMG_HEIGHT, 64), dtype=np.uint8)
            text = ""
            
        if self.augment:
            img = apply_heavy_aug(img)
        
        h, w = img.shape
        target_w = int(w * (Config.IMG_HEIGHT / h))
        target_w = max(Config.MIN_WIDTH, min(target_w, Config.MAX_WIDTH))
        img = cv2.resize(img, (target_w, Config.IMG_HEIGHT))
        img = img.astype(np.float32) / 255.0
        
        if self.augment:
            if np.random.random() < 0.5:
                img = img * np.random.uniform(0.8, 1.2)
                img = np.clip(img, 0, 1)
            
            if np.random.random() < Config.CUTOUT_PROB:
                img = cutout_augmentation(img)
            
            if np.random.random() < Config.NOISE_PROB:
                img = add_gaussian_noise(img)

        img_tensor = torch.FloatTensor(img).unsqueeze(0)
        label = self.label_encoder.encode(text)
        return img_tensor, torch.LongTensor(label), text, target_w


def collate_fn(batch):
    """Collate function"""
    images, labels, texts, widths = zip(*batch)
    max_w = max(widths)
    padded_imgs = torch.zeros(len(images), 1, Config.IMG_HEIGHT, max_w)
    for i, (img, w) in enumerate(zip(images, widths)):
        padded_imgs[i, :, :, :w] = img
    targets = torch.cat(labels)
    target_lens = torch.LongTensor([len(l) for l in labels])
    input_lens = torch.LongTensor([w // 4 for w in widths])
    return padded_imgs, targets, input_lens, target_lens, texts
