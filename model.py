"""
CRNN with Seq2Seq Attention: Complete Implementation
✅ Bahdanau Attention Mechanism
✅ Teacher Forcing with Scheduled Sampling
✅ Beam Search Decoding
✅ Layer-wise Learning Rate Decay (CNN < Encoder < Decoder)
✅ Stochastic Weight Averaging (SWA) - starts at epoch 60
✅ Cosine Annealing with Warmup
✅ Gradient Accumulation (effective batch = 128)
✅ Label Smoothing (0.1)
✅ Enhanced Augmentation (Elastic, Cutout, Noise)
✅ 70/15/15 Train/Val/Test Split
✅ All Anti-Overfit Techniques from Original
"""

import os
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np
from pathlib import Path
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm import tqdm
from collections import defaultdict, Counter
from difflib import SequenceMatcher
import copy

# ============================================================================
# Configuration (ABSOLUTE PATHS)
# ============================================================================

class Config:
    # 1. Base Paths
    WORKSPACE_PATH = Path(r"d:\code\dataset\iam_words")
    LABELS_CSV = Path(r"d:\code\dataset\iam_words\words.csv")
    IMAGES_PATH = Path(r"d:\code\dataset\iam_words\processed_word_dataset")
    
    # 2. Dedicated Output Directories
    CHECKPOINT_DIR = Path(r"d:\code\dataset\iam_words\checkpoints_final")
    PLOTS_DIR = Path(r"d:\code\dataset\iam_words\plots")
    METRICS_DIR = Path(r"d:\code\dataset\iam_words\metrics")
    
    # 3. Specific File Paths
    BEST_MODEL_PATH = Path(r"d:\code\dataset\iam_words\checkpoints_final\best_seq2seq_model.pth")
    SWA_MODEL_PATH = Path(r"d:\code\dataset\iam_words\checkpoints_final\swa_seq2seq_model.pth")
    LOG_FILE = Path(r"d:\code\dataset\iam_words\metrics\training_log.csv")
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # HARDWARE
    BATCH_SIZE = 64
    USE_AMP = torch.cuda.is_available()
    NUM_WORKERS = 4      
    
    # REGULARIZATION
    EPOCHS = 100
    LEARNING_RATE = 0.0003
    WEIGHT_DECAY = 5e-4
    DROPOUT = 0.5
    DROPOUT_PATH = 0.1
    LABEL_SMOOTHING = 0.1
    
    # GRADIENT ACCUMULATION
    GRADIENT_ACCUMULATION_STEPS = 2
    
    # EARLY STOPPING
    PATIENCE = 30
    MIN_DELTA = 0.001    
    
    # COSINE ANNEALING
    WARMUP_EPOCHS = 5
    FIRST_CYCLE_EPOCHS = 30
    
    # SWA
    SWA_START = 60
    SWA_LR = 0.0001
    
    # AUGMENTATION
    CUTOUT_PROB = 0.3
    ELASTIC_PROB = 0.3
    NOISE_PROB = 0.3
    
    IMG_HEIGHT = 64
    MIN_WIDTH = 32
    MAX_WIDTH = 512
    
    # SEQ2SEQ SPECIFIC PARAMETERS
    ENCODER_HIDDEN = 256
    DECODER_HIDDEN = 256
    ATTENTION_HIDDEN = 128
    TEACHER_FORCING_RATIO = 0.5  # Start with 50% teacher forcing
    TEACHER_FORCING_DECAY = 0.99  # Decay per epoch
    MAX_OUTPUT_LENGTH = 50  # Maximum word length
    
    # BEAM SEARCH
    BEAM_WIDTH = 5
    USE_BEAM_SEARCH = True
    
    # LAYER-WISE LR DECAY
    LAYER_LR_DECAY = 0.95

# ============================================================================
# Special Tokens
# ============================================================================

class SpecialTokens:
    PAD = '<PAD>'
    SOS = '<SOS>'  # Start of sequence
    EOS = '<EOS>'  # End of sequence
    UNK = '<UNK>'  # Unknown character

# ============================================================================
# Cosine Annealing Scheduler with Warmup
# ============================================================================

class CosineAnnealingWarmupRestarts(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, first_cycle_steps, warmup_steps, min_lr=0, last_epoch=-1):
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
    if random_state is None:
        random_state = np.random.RandomState(None)
    shape = image.shape
    dx = np.array(cv2.GaussianBlur((random_state.rand(*shape).astype(np.float32) * 2 - 1), (0, 0), sigma)) * float(alpha)
    dy = np.array(cv2.GaussianBlur((random_state.rand(*shape).astype(np.float32) * 2 - 1), (0, 0), sigma)) * float(alpha)
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    return cv2.remap(image, indices[1].astype(np.float32), indices[0].astype(np.float32), 
                    interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

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
    h, w = image.shape
    
    if np.random.random() < Config.ELASTIC_PROB:
        image = elastic_transform(image, alpha=h*0.4, sigma=h*0.08)
    
    if np.random.random() < 0.5:
        angle = np.random.uniform(-7, 7)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        image = cv2.warpAffine(image, M, (w, h), borderValue=0)
    
    if np.random.random() < 0.4:
        kernel = np.random.choice([3, 5])
        image = cv2.GaussianBlur(image, (kernel, kernel), 0)
    
    if np.random.random() < 0.4:
        scale = np.random.uniform(0.9, 1.1)
        image = np.clip(image * scale, 0, 1)
    
    if np.random.random() < Config.CUTOUT_PROB:
        image = cutout_augmentation(image)
    
    if np.random.random() < Config.NOISE_PROB:
        image = add_gaussian_noise(image)
    
    return image

# ============================================================================
# Label Encoder with Special Tokens
# ============================================================================

class LabelEncoder:
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        
    def build_vocab(self, texts):
        chars = sorted(set(''.join(texts)))
        
        # Add special tokens
        self.char2idx = {
            SpecialTokens.PAD: 0,
            SpecialTokens.SOS: 1,
            SpecialTokens.EOS: 2,
            SpecialTokens.UNK: 3
        }
        
        # Add regular characters
        for i, char in enumerate(chars, start=4):
            self.char2idx[char] = i
        
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        
        print(f"Vocab size: {len(self.char2idx)} (including special tokens)")
        print(f"Special tokens: PAD={self.char2idx[SpecialTokens.PAD]}, "
              f"SOS={self.char2idx[SpecialTokens.SOS]}, "
              f"EOS={self.char2idx[SpecialTokens.EOS]}, "
              f"UNK={self.char2idx[SpecialTokens.UNK]}")
    
    def encode(self, text):
        """Encode text to indices (with SOS and EOS)"""
        indices = [self.char2idx[SpecialTokens.SOS]]
        for char in text:
            indices.append(self.char2idx.get(char, self.char2idx[SpecialTokens.UNK]))
        indices.append(self.char2idx[SpecialTokens.EOS])
        return indices
    
    def decode(self, indices):
        """Decode indices to text (remove special tokens)"""
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
# Dataset
# ============================================================================

def clean_dataset(df):
    df = df.dropna(subset=['transcription'])
    df = df[df['transcription'].str.len() > 0]
    df = df[df['transcription'].str.len() <= Config.MAX_OUTPUT_LENGTH - 2]  # Account for SOS/EOS
    return df.reset_index(drop=True)

class WordDataset(Dataset):
    def __init__(self, df, images_path, encoder, augment=False):
        self.df = df
        self.images_path = Path(images_path)
        self.encoder = encoder
        self.augment = augment
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.images_path / row['image_path']
        
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            image = np.zeros((Config.IMG_HEIGHT, Config.MIN_WIDTH), dtype=np.uint8)
        
        image = image.astype(np.float32) / 255.0
        
        if self.augment:
            image = apply_heavy_aug(image)
        
        h, w = image.shape
        scale = Config.IMG_HEIGHT / h
        new_w = int(w * scale)
        new_w = max(Config.MIN_WIDTH, min(new_w, Config.MAX_WIDTH))
        image = cv2.resize(image, (new_w, Config.IMG_HEIGHT))
        
        # Pad to MAX_WIDTH
        if new_w < Config.MAX_WIDTH:
            pad_w = Config.MAX_WIDTH - new_w
            image = np.pad(image, ((0, 0), (0, pad_w)), mode='constant', constant_values=0)
        
        image = torch.FloatTensor(image).unsqueeze(0)
        
        text = row['transcription']
        target = self.encoder.encode(text)
        
        return image, target, text

def collate_fn(batch):
    images, targets, texts = zip(*batch)
    images = torch.stack(images, dim=0)
    
    # Pad targets to same length
    max_target_len = max(len(t) for t in targets)
    padded_targets = []
    target_lengths = []
    
    for target in targets:
        target_lengths.append(len(target))
        padded = target + [0] * (max_target_len - len(target))  # Pad with PAD token
        padded_targets.append(padded)
    
    targets = torch.LongTensor(padded_targets)
    target_lengths = torch.LongTensor(target_lengths)
    
    return images, targets, target_lengths, texts

# ============================================================================
# Bahdanau Attention Mechanism
# ============================================================================

class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention Mechanism
    Computes attention weights and context vector
    """
    def __init__(self, encoder_hidden, decoder_hidden, attention_hidden):
        super(BahdanauAttention, self).__init__()
        
        self.encoder_projection = nn.Linear(encoder_hidden * 2, attention_hidden)  # *2 for bidirectional
        self.decoder_projection = nn.Linear(decoder_hidden, attention_hidden)
        self.energy = nn.Linear(attention_hidden, 1)
        
    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        Args:
            decoder_hidden: (batch, decoder_hidden)
            encoder_outputs: (batch, seq_len, encoder_hidden*2)
            mask: (batch, seq_len) - True for valid positions
        Returns:
            context: (batch, encoder_hidden*2)
            attention_weights: (batch, seq_len)
        """
        batch_size, seq_len, _ = encoder_outputs.size()
        
        # Project encoder outputs: (batch, seq_len, attention_hidden)
        encoder_proj = self.encoder_projection(encoder_outputs)
        
        # Project decoder hidden: (batch, 1, attention_hidden)
        decoder_proj = self.decoder_projection(decoder_hidden).unsqueeze(1)
        
        # Compute energy: (batch, seq_len, 1)
        energy = self.energy(torch.tanh(encoder_proj + decoder_proj))
        energy = energy.squeeze(2)  # (batch, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            energy = energy.masked_fill(~mask, float('-inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(energy, dim=1)  # (batch, seq_len)
        
        # Compute context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  # (batch, 1, encoder_hidden*2)
        context = context.squeeze(1)  # (batch, encoder_hidden*2)
        
        return context, attention_weights

# ============================================================================
# CNN Encoder
# ============================================================================

class CNNEncoder(nn.Module):
    """
    CNN Feature Extractor
    Reduces height to 1 and outputs sequence of features
    """
    def __init__(self):
        super(CNNEncoder, self).__init__()
        
        self.cnn = nn.Sequential(
            # Block 1: 64x512 -> 32x256
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 32x256 -> 16x128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 16x128 -> 8x64
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            
            # Block 4: 8x64 -> 4x64
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            
            # Block 5: 4x64 -> 1x64
            nn.Conv2d(512, 512, kernel_size=2, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.dropout = nn.Dropout(Config.DROPOUT * 0.5)
        
    def forward(self, x):
        """
        Args:
            x: (batch, 1, height, width)
        Returns:
            features: (batch, seq_len, 512)
        """
        conv = self.cnn(x)  # (batch, 512, 1, width')
        conv = self.dropout(conv)
        
        batch, channels, height, width = conv.size()
        assert height == 1, f"Height should be 1, got {height}"
        
        conv = conv.squeeze(2)  # (batch, 512, width')
        conv = conv.permute(0, 2, 1)  # (batch, width', 512)
        
        return conv

# ============================================================================
# LSTM Encoder
# ============================================================================

class LSTMEncoder(nn.Module):
    """
    Bidirectional LSTM Encoder
    Processes CNN features sequentially
    """
    def __init__(self, input_size, hidden_size, dropout=0.5):
        super(LSTMEncoder, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            outputs: (batch, seq_len, hidden_size*2)
            hidden: tuple of (h, c)
        """
        outputs, hidden = self.lstm(x)
        return outputs, hidden

# ============================================================================
# LSTM Decoder with Attention
# ============================================================================

class LSTMDecoder(nn.Module):
    """
    LSTM Decoder with Bahdanau Attention
    Generates output sequence one character at a time
    """
    def __init__(self, num_classes, encoder_hidden, decoder_hidden, attention_hidden, dropout=0.5):
        super(LSTMDecoder, self).__init__()
        
        self.num_classes = num_classes
        self.decoder_hidden = decoder_hidden
        
        # Embedding layer
        self.embedding = nn.Embedding(num_classes, decoder_hidden)
        
        # Attention mechanism
        self.attention = BahdanauAttention(encoder_hidden, decoder_hidden, attention_hidden)
        
        # LSTM cell (input: embedding + context)
        self.lstm = nn.LSTMCell(
            input_size=decoder_hidden + encoder_hidden * 2,
            hidden_size=decoder_hidden
        )
        
        # Output projection
        self.fc_out = nn.Linear(decoder_hidden + encoder_hidden * 2, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_token, hidden, cell, encoder_outputs, mask=None):
        """
        Single decoding step
        Args:
            input_token: (batch,) - previous token
            hidden: (batch, decoder_hidden)
            cell: (batch, decoder_hidden)
            encoder_outputs: (batch, seq_len, encoder_hidden*2)
            mask: (batch, seq_len)
        Returns:
            output: (batch, num_classes)
            hidden: (batch, decoder_hidden)
            cell: (batch, decoder_hidden)
            attention_weights: (batch, seq_len)
        """
        # Embed input token
        embedded = self.embedding(input_token)  # (batch, decoder_hidden)
        embedded = self.dropout(embedded)
        
        # Compute attention
        context, attention_weights = self.attention(hidden, encoder_outputs, mask)
        
        # Concatenate embedding and context
        lstm_input = torch.cat([embedded, context], dim=1)
        
        # LSTM step
        hidden, cell = self.lstm(lstm_input, (hidden, cell))
        
        # Output projection
        output_input = torch.cat([hidden, context], dim=1)
        output = self.fc_out(output_input)
        
        return output, hidden, cell, attention_weights

# ============================================================================
# Seq2Seq Model
# ============================================================================

class Seq2SeqAttention(nn.Module):
    """
    Complete Seq2Seq model with Attention
    CNN -> LSTM Encoder -> LSTM Decoder with Attention
    """
    def __init__(self, num_classes, encoder_hidden, decoder_hidden, attention_hidden, dropout=0.5):
        super(Seq2SeqAttention, self).__init__()
        
        self.num_classes = num_classes
        self.encoder_hidden = encoder_hidden
        self.decoder_hidden = decoder_hidden
        
        # Encoder
        self.cnn_encoder = CNNEncoder()
        self.lstm_encoder = LSTMEncoder(512, encoder_hidden, dropout)
        
        # Decoder
        self.decoder = LSTMDecoder(num_classes, encoder_hidden, decoder_hidden, attention_hidden, dropout)
        
        # Initialize decoder hidden state from encoder
        self.encoder_to_decoder_h = nn.Linear(encoder_hidden * 2 * 2, decoder_hidden)  # *2 for bidirectional, *2 for 2 layers
        self.encoder_to_decoder_c = nn.Linear(encoder_hidden * 2 * 2, decoder_hidden)
        
    def forward(self, images, targets=None, teacher_forcing_ratio=0.5):
        """
        Args:
            images: (batch, 1, height, width)
            targets: (batch, target_len) - for training
            teacher_forcing_ratio: probability of using teacher forcing
        Returns:
            outputs: (batch, max_len, num_classes)
            attention_weights: (batch, max_len, seq_len)
        """
        batch_size = images.size(0)
        device = images.device
        
        # Encode
        cnn_features = self.cnn_encoder(images)  # (batch, seq_len, 512)
        encoder_outputs, (h_n, c_n) = self.lstm_encoder(cnn_features)  # (batch, seq_len, hidden*2)
        
        # Initialize decoder hidden state
        # h_n: (4, batch, hidden) -> reshape to (batch, hidden*4)
        h_n = h_n.permute(1, 0, 2).contiguous().view(batch_size, -1)
        c_n = c_n.permute(1, 0, 2).contiguous().view(batch_size, -1)
        
        decoder_hidden = self.encoder_to_decoder_h(h_n)
        decoder_cell = self.encoder_to_decoder_c(c_n)
        
        # Create mask for encoder outputs (all valid since padded with zeros)
        seq_len = encoder_outputs.size(1)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        
        # Decode
        if targets is not None:
            # Training mode
            max_len = targets.size(1)
            outputs = torch.zeros(batch_size, max_len, self.num_classes, device=device)
            attention_weights_list = []
            
            # Start with SOS token
            input_token = targets[:, 0]  # First token is SOS
            
            for t in range(1, max_len):
                output, decoder_hidden, decoder_cell, attn_weights = self.decoder(
                    input_token, decoder_hidden, decoder_cell, encoder_outputs, mask
                )
                
                outputs[:, t] = output
                attention_weights_list.append(attn_weights)
                
                # Teacher forcing
                use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
                if use_teacher_forcing:
                    input_token = targets[:, t]
                else:
                    input_token = output.argmax(dim=1)
            
            attention_weights = torch.stack(attention_weights_list, dim=1)  # (batch, max_len-1, seq_len)
        else:
            # Inference mode
            max_len = Config.MAX_OUTPUT_LENGTH
            outputs = torch.zeros(batch_size, max_len, self.num_classes, device=device)
            attention_weights_list = []
            
            # Start with SOS token
            input_token = torch.full((batch_size,), 1, dtype=torch.long, device=device)  # SOS=1
            
            for t in range(max_len):
                output, decoder_hidden, decoder_cell, attn_weights = self.decoder(
                    input_token, decoder_hidden, decoder_cell, encoder_outputs, mask
                )
                
                outputs[:, t] = output
                attention_weights_list.append(attn_weights)
                
                # Greedy decoding
                input_token = output.argmax(dim=1)
                
                # Stop if all sequences have generated EOS
                if (input_token == 2).all():  # EOS=2
                    break
            
            attention_weights = torch.stack(attention_weights_list, dim=1) if attention_weights_list else None
        
        return outputs, attention_weights

# ============================================================================
# Beam Search Decoder
# ============================================================================

class BeamSearchDecoder:
    """
    Beam Search for better inference
    """
    def __init__(self, model, beam_width=5):
        self.model = model
        self.beam_width = beam_width
        
    def decode(self, images):
        """
        Args:
            images: (batch, 1, height, width)
        Returns:
            predictions: list of strings
        """
        self.model.eval()
        batch_size = images.size(0)
        device = images.device
        
        predictions = []
        
        with torch.no_grad():
            # Encode
            cnn_features = self.model.cnn_encoder(images)
            encoder_outputs, (h_n, c_n) = self.model.lstm_encoder(cnn_features)
            
            for b in range(batch_size):
                # Initialize decoder for this sample
                h_n_b = h_n[:, b:b+1, :].permute(1, 0, 2).contiguous().view(1, -1)
                c_n_b = c_n[:, b:b+1, :].permute(1, 0, 2).contiguous().view(1, -1)
                
                decoder_hidden = self.model.encoder_to_decoder_h(h_n_b)
                decoder_cell = self.model.encoder_to_decoder_c(c_n_b)
                
                encoder_outputs_b = encoder_outputs[b:b+1]
                seq_len = encoder_outputs_b.size(1)
                mask = torch.ones(1, seq_len, dtype=torch.bool, device=device)
                
                # Initialize beam
                beams = [(1.0, [1], decoder_hidden, decoder_cell)]  # (score, sequence, hidden, cell)
                
                for _ in range(Config.MAX_OUTPUT_LENGTH):
                    candidates = []
                    
                    for score, sequence, hidden, cell in beams:
                        # If sequence ends with EOS, keep it
                        if sequence[-1] == 2:
                            candidates.append((score, sequence, hidden, cell))
                            continue
                        
                        # Decode one step
                        input_token = torch.tensor([sequence[-1]], device=device)
                        output, new_hidden, new_cell, _ = self.model.decoder(
                            input_token, hidden, cell, encoder_outputs_b, mask
                        )
                        
                        log_probs = F.log_softmax(output, dim=1)
                        top_probs, top_indices = log_probs.topk(self.beam_width)
                        
                        for prob, idx in zip(top_probs[0], top_indices[0]):
                            new_score = score * prob.item()
                            new_sequence = sequence + [idx.item()]
                            candidates.append((new_score, new_sequence, new_hidden, new_cell))
                    
                    # Select top beam_width candidates
                    beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:self.beam_width]
                    
                    # Stop if all beams end with EOS
                    if all(seq[-1] == 2 for _, seq, _, _ in beams):
                        break
                
                # Select best sequence
                best_sequence = beams[0][1]
                predictions.append(best_sequence)
        
        return predictions

# ============================================================================
# Training Functions
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, teacher_forcing_ratio, epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    
    optimizer.zero_grad()
    
    for batch_idx, (images, targets, target_lengths, texts) in enumerate(pbar):
        images = images.to(Config.DEVICE)
        targets = targets.to(Config.DEVICE)
        
        with torch.amp.autocast(device_type=Config.DEVICE.type, enabled=Config.USE_AMP):
            outputs, _ = model(images, targets, teacher_forcing_ratio)
            
            # Reshape for loss: (batch * seq_len, num_classes) and (batch * seq_len)
            outputs = outputs[:, 1:, :].contiguous()  # Skip first position (SOS)
            targets = targets[:, 1:].contiguous()  # Skip first position (SOS)
            
            outputs = outputs.view(-1, outputs.size(2))
            targets = targets.view(-1)
            
            # Ignore PAD tokens in loss
            loss = criterion(outputs, targets)
        
        # Gradient accumulation
        loss = loss / Config.GRADIENT_ACCUMULATION_STEPS
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * Config.GRADIENT_ACCUMULATION_STEPS
        pbar.set_postfix({'loss': f'{loss.item() * Config.GRADIENT_ACCUMULATION_STEPS:.4f}'})
    
    return total_loss / len(loader)

def validate(model, loader, criterion, encoder, use_beam_search=False):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_ground_truths = []
    
    beam_decoder = BeamSearchDecoder(model, Config.BEAM_WIDTH) if use_beam_search else None
    
    with torch.no_grad():
        for images, targets, target_lengths, texts in tqdm(loader, desc="[Validation]"):
            images = images.to(Config.DEVICE)
            targets = targets.to(Config.DEVICE)
            
            with torch.amp.autocast(device_type=Config.DEVICE.type, enabled=Config.USE_AMP):
                outputs, _ = model(images, targets, teacher_forcing_ratio=0.0)
                
                outputs_loss = outputs[:, 1:, :].contiguous()
                targets_loss = targets[:, 1:].contiguous()
                
                outputs_loss = outputs_loss.view(-1, outputs_loss.size(2))
                targets_loss = targets_loss.view(-1)
                
                loss = criterion(outputs_loss, targets_loss)
            
            total_loss += loss.item()
            
            # Decode predictions
            if use_beam_search and beam_decoder:
                pred_sequences = beam_decoder.decode(images)
                for seq in pred_sequences:
                    all_predictions.append(encoder.decode(seq))
            else:
                pred_indices = outputs.argmax(dim=2).cpu().numpy()
                for seq in pred_indices:
                    all_predictions.append(encoder.decode(seq))
            
            all_ground_truths.extend(texts)
    
    # Calculate WER and CER
    total_words = len(all_ground_truths)
    correct_words = sum([1 for pred, gt in zip(all_predictions, all_ground_truths) if pred == gt])
    wer = 1 - (correct_words / total_words) if total_words > 0 else 1.0
    
    total_chars = sum(len(gt) for gt in all_ground_truths)
    char_errors = sum(
        sum(1 for a, b in zip(pred, gt) if a != b) + abs(len(pred) - len(gt))
        for pred, gt in zip(all_predictions, all_ground_truths)
    )
    cer = char_errors / total_chars if total_chars > 0 else 1.0
    
    print(f"WER: {wer:.2%} | CER: {cer:.2%}")
    print(f"Sample: GT='{all_ground_truths[0]}' | Pred='{all_predictions[0]}'")
    
    return total_loss / len(loader), wer, cer

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
        
        # Loss
        axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
        axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val Loss', marker='s')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # WER
        axes[0, 1].plot(df['epoch'], df['wer'], label='WER', color='red', marker='o')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('WER')
        axes[0, 1].set_title('Word Error Rate')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # CER
        axes[1, 0].plot(df['epoch'], df['cer'], label='CER', color='green', marker='o')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('CER')
        axes[1, 0].set_title('Character Error Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 1].plot(df['epoch'], df['lr'], label='Learning Rate', color='purple', marker='o')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(str(Config.PLOTS_DIR / 'training_curves.png'), dpi=300)
        plt.close()

# ============================================================================
# Layer-wise Learning Rate Decay
# ============================================================================

def get_layer_wise_params(model):
    """
    Group parameters by layer for layer-wise LR decay
    Earlier layers (CNN) learn slower than later layers (Decoder)
    """
    params = []
    
    # CNN Encoder (lowest LR)
    params.append({
        'params': model.cnn_encoder.parameters(),
        'lr': Config.LEARNING_RATE * (Config.LAYER_LR_DECAY ** 3),
        'name': 'cnn_encoder'
    })
    
    # LSTM Encoder (medium LR)
    params.append({
        'params': model.lstm_encoder.parameters(),
        'lr': Config.LEARNING_RATE * (Config.LAYER_LR_DECAY ** 2),
        'name': 'lstm_encoder'
    })
    
    # Decoder (highest LR - needs to learn most)
    params.append({
        'params': model.decoder.parameters(),
        'lr': Config.LEARNING_RATE,
        'name': 'decoder'
    })
    
    # Bridge layers (medium-high LR)
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
    torch.backends.cudnn.benchmark = True
    
    # Create directories
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.PLOTS_DIR, exist_ok=True)
    os.makedirs(Config.METRICS_DIR, exist_ok=True)
    
    print("=" * 60)
    print("Seq2Seq with Attention Training")
    print("✅ Layer-wise Learning Rate Decay")
    print("✅ Stochastic Weight Averaging (SWA)")
    print("✅ 70/15/15 Train/Val/Test Split")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading and cleaning dataset...")
    df = clean_dataset(pd.read_csv(Config.LABELS_CSV))
    encoder = LabelEncoder()
    encoder.build_vocab(df['transcription'].values)
    
    # Split data: 70% train, 15% val, 15% test
    train_df = df.sample(frac=0.7, random_state=42)
    remaining_df = df.drop(train_df.index)
    val_df = remaining_df.sample(frac=0.5, random_state=42)  # 15% of total
    test_df = remaining_df.drop(val_df.index)  # Remaining 15%
    
    print(f"Train samples: {len(train_df)} (70%)")
    print(f"Val samples: {len(val_df)} (15%)")
    print(f"Test samples: {len(test_df)} (15%)")
    
    # Create dataloaders
    train_loader = DataLoader(
        WordDataset(train_df, Config.IMAGES_PATH, encoder, augment=True),
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn, 
        pin_memory=True
    )
    val_loader = DataLoader(
        WordDataset(val_df, Config.IMAGES_PATH, encoder, augment=False),
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn, 
        pin_memory=True
    )
    test_loader = DataLoader(
        WordDataset(test_df, Config.IMAGES_PATH, encoder, augment=False),
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn, 
        pin_memory=True
    )
    
    # Create model
    print("\n2. Creating Seq2Seq model with Attention...")
    model = Seq2SeqAttention(
        num_classes=encoder.num_classes(),
        encoder_hidden=Config.ENCODER_HIDDEN,
        decoder_hidden=Config.DECODER_HIDDEN,
        attention_hidden=Config.ATTENTION_HIDDEN,
        dropout=Config.DROPOUT
    ).to(Config.DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Training setup with Layer-wise LR
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
        min_lr=Config.LEARNING_RATE * 0.01
    )
    
    # SWA Setup
    print("\n4. Setting up Stochastic Weight Averaging (SWA)...")
    print(f"  SWA will start at epoch {Config.SWA_START}")
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=Config.SWA_LR)
    swa_started = False
    
    tracker = HistoryTracker()
    
    best_wer = 1.0
    patience_counter = 0
    teacher_forcing_ratio = Config.TEACHER_FORCING_RATIO
    
    print(f"\n5. Starting training with teacher forcing ratio: {teacher_forcing_ratio:.2f}")
    print("=" * 60)
    
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        print(f"Teacher Forcing Ratio: {teacher_forcing_ratio:.3f}")
        
        # Start SWA after specified epoch
        if epoch >= Config.SWA_START and not swa_started:
            print(f"\n🔄 Starting SWA at epoch {epoch+1}")
            swa_started = True
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, teacher_forcing_ratio, epoch+1)
        
        # Update SWA model
        if swa_started:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        
        # Validate (use beam search after warmup)
        use_beam = Config.USE_BEAM_SEARCH and epoch >= Config.WARMUP_EPOCHS
        val_loss, wer, cer = validate(model, val_loader, criterion, encoder, use_beam_search=use_beam)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Update tracker
        tracker.update(epoch+1, train_loss, val_loss, wer, cer, optimizer.param_groups[0]['lr'])
        tracker.plot()
        
        # Save best model
        if wer < (best_wer - Config.MIN_DELTA):
            best_wer = wer
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"✓ NEW BEST MODEL (WER: {wer:.2%})")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs. Best WER: {best_wer:.2%}")
        
        # Early stopping
        if patience_counter >= Config.PATIENCE:
            print(f"\n⚠ EARLY STOPPING: No improvement for {Config.PATIENCE} epochs")
            break
        
        # Decay teacher forcing
        teacher_forcing_ratio *= Config.TEACHER_FORCING_DECAY
        
        # Step scheduler (only if not using SWA scheduler)
        if not swa_started:
            scheduler.step()
    
    # Update SWA batch normalization statistics
    if swa_started:
        print("\n6. Updating SWA Batch Normalization statistics...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=Config.DEVICE)
        torch.save(swa_model.state_dict(), Config.SWA_MODEL_PATH)
        print(f"✓ SWA model saved to: {Config.SWA_MODEL_PATH}")
    
    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("7. Final Evaluation on Test Set")
    print("=" * 60)
    
    # Evaluate best model
    print("\nEvaluating best checkpoint model...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE, weights_only=True))
    test_loss, test_wer, test_cer = validate(model, test_loader, criterion, encoder, use_beam_search=True)
    print(f"Best Model - Test WER: {test_wer:.2%} | Test CER: {test_cer:.2%}")
    
    # Evaluate SWA model if available
    if swa_started:
        print("\nEvaluating SWA model...")
        swa_model.eval()
        test_loss_swa, test_wer_swa, test_cer_swa = validate(swa_model, test_loader, criterion, encoder, use_beam_search=True)
        print(f"SWA Model - Test WER: {test_wer_swa:.2%} | Test CER: {test_cer_swa:.2%}")
        
        if test_wer_swa < test_wer:
            print("\n✨ SWA model performs better! Using SWA for final results.")
            final_wer, final_cer = test_wer_swa, test_cer_swa
        else:
            print("\n✨ Best checkpoint model performs better! Using checkpoint for final results.")
            final_wer, final_cer = test_wer, test_cer
    else:
        final_wer, final_cer = test_wer, test_cer
    
    print("\n" + "=" * 60)
    print("Training Completed!")
    print("=" * 60)
    print(f"Best Validation WER: {best_wer:.2%}")
    print(f"Final Test WER: {final_wer:.2%}")
    print(f"Final Test CER: {final_cer:.2%}")
    print(f"Best model: {Config.BEST_MODEL_PATH}")
    if swa_started:
        print(f"SWA model: {Config.SWA_MODEL_PATH}")
    print("=" * 60)

if __name__ == '__main__':
    main()