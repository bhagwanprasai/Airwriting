"""
CRNN Anti-Overfit: Elastic Distortion, High Dropout & Regularization
Includes: Absolute Paths, Separated Directories for Plots/Metrics
"""

import os
import matplotlib
matplotlib.use('Agg') # Headless mode to prevent thread crashes
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np
from pathlib import Path
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict, Counter
from difflib import SequenceMatcher

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
    BEST_MODEL_PATH = Path(r"d:\code\dataset\iam_words\checkpoints_final\best_crnn_model.pth")
    LOG_FILE = Path(r"d:\code\dataset\iam_words\metrics\training_log.csv")
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # HARDWARE
    BATCH_SIZE = 96      
    USE_AMP = torch.cuda.is_available()
    NUM_WORKERS = 4      
    
    # REGULARIZATION
    EPOCHS = 100
    LEARNING_RATE = 0.0005
    WEIGHT_DECAY = 1e-3  
    
    # EARLY STOPPING
    PATIENCE = 20        
    MIN_DELTA = 0.001    
    
    IMG_HEIGHT = 64
    MIN_WIDTH = 32
    MAX_WIDTH = 512
    LSTM_HIDDEN = 256
    DROPOUT = 0.5        

# ============================================================================
# Advanced Augmentation (Elastic)
# ============================================================================

def elastic_transform(image, alpha=30, sigma=4, random_state=None):
    if random_state is None: random_state = np.random.RandomState(None)
    shape = image.shape
    dx = cv2.GaussianBlur((random_state.rand(*shape).astype(np.float32) * 2 - 1), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((random_state.rand(*shape).astype(np.float32) * 2 - 1), (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    return cv2.remap(image, indices[1].astype(np.float32), indices[0].astype(np.float32),interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def apply_heavy_aug(image):
    h, w = image.shape
    if np.random.random() < 0.4: image = elastic_transform(image, alpha=h*0.5, sigma=h*0.1)
    if np.random.random() < 0.5:
        angle = np.random.uniform(-7, 7)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        image = cv2.warpAffine(image, M, (w, h), borderValue=0)
    if np.random.random() < 0.5:
        shear_factor = np.random.uniform(-0.3, 0.3)
        M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
        new_w = w + int(abs(shear_factor*h))
        image = cv2.warpAffine(image, M, (new_w, h), borderValue=0)
    if np.random.random() < 0.3:
        kernel = np.ones((2,2), np.uint8)
        if np.random.random() < 0.5: image = cv2.erode(image, kernel, iterations=1)
        else: image = cv2.dilate(image, kernel, iterations=1)
    return image

# ============================================================================
# Utils & Encoding
# ============================================================================

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2): return levenshtein_distance(s2, s1)
    if len(s2) == 0: return len(s1)
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

class HistoryTracker:
    def __init__(self):
        self.history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'wer': [], 'cer': [], 'lr': []}
    
    def update(self, epoch, train_loss, val_loss, wer, cer, lr):
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['wer'].append(wer)
        self.history['cer'].append(cer)
        self.history['lr'].append(lr)
        pd.DataFrame(self.history).to_csv(str(Config.LOG_FILE), index=False)

    def plot(self):
        epochs = self.history['epoch']
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.history['train_loss'], label='Train Loss')
        plt.plot(epochs, self.history['val_loss'], label='Val Loss')
        plt.legend(); plt.grid(True); plt.title("Loss")
        plt.savefig(str(Config.PLOTS_DIR / "loss_curve.png")); plt.close()
        
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.history['wer'], label='WER')
        plt.plot(epochs, self.history['cer'], label='CER')
        plt.legend(); plt.grid(True); plt.title("Error Rate")
        plt.savefig(str(Config.PLOTS_DIR / "error_metrics.png")); plt.close()

class LabelEncoder:
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
    def build_vocab(self, texts):
        chars = sorted(list(set("".join(texts))))
        self.char2idx = {char: idx + 1 for idx, char in enumerate(chars)}
        self.idx2char = {idx + 1: char for idx, char in enumerate(chars)}
        self.idx2char[0] = ''
        print(f"Vocab Size: {len(self.char2idx)}")
    def encode(self, text):
        return [self.char2idx[char] for char in text if char in self.char2idx]
    def decode_greedy(self, preds):
        res = []
        prev = None
        for idx in preds:
            if idx != 0 and idx != prev:
                res.append(self.idx2char[idx])
            prev = idx
        return ''.join(res)
    def num_classes(self): return len(self.char2idx) + 1

def clean_dataset(df):
    df = df.dropna(subset=['transcription'])
    mask = df['transcription'].astype(str).str.match(r'^[a-zA-Z0-9]+$')
    return df[mask].copy()

class WordDataset(Dataset):
    def __init__(self, df, images_path, label_encoder, augment=False):
        self.df = df.reset_index(drop=True)
        self.images_path = Path(images_path)
        self.label_encoder = label_encoder
        self.augment = augment

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = str(row['filename'])
        if fname.endswith('.png'): fname = fname[:-4]
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
            
        if self.augment: img = apply_heavy_aug(img)
        
        h, w = img.shape
        target_w = int(w * (Config.IMG_HEIGHT / h))
        target_w = max(Config.MIN_WIDTH, min(target_w, Config.MAX_WIDTH))
        img = cv2.resize(img, (target_w, Config.IMG_HEIGHT))
        img = img.astype(np.float32) / 255.0
        
        if self.augment and np.random.random() < 0.5:
             img = img * np.random.uniform(0.8, 1.2)
             img = np.clip(img, 0, 1)

        img_tensor = torch.FloatTensor(img).unsqueeze(0)
        label = self.label_encoder.encode(text)
        return img_tensor, torch.LongTensor(label), text, target_w

def collate_fn(batch):
    images, labels, texts, widths = zip(*batch)
    max_w = max(widths)
    padded_imgs = torch.zeros(len(images), 1, Config.IMG_HEIGHT, max_w)
    for i, (img, w) in enumerate(zip(images, widths)): padded_imgs[i, :, :, :w] = img
    targets = torch.cat(labels)
    target_lens = torch.LongTensor([len(l) for l in labels])
    input_lens = torch.LongTensor([w // 4 for w in widths])
    return padded_imgs, targets, input_lens, target_lens, texts

# ============================================================================
# Model
# ============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride, bias=False), nn.BatchNorm2d(out_c))

    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))

class CRNN(nn.Module):
    def __init__(self, num_chars, hidden_size=256, dropout=0.2):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(ResidualBlock(64, 128), nn.MaxPool2d(2, 2))
        self.layer3 = nn.Sequential(ResidualBlock(128, 256), ResidualBlock(256, 256), nn.MaxPool2d((2, 1)))
        
        self.rnn = nn.LSTM(
            input_size=256 * 8, 
            hidden_size=hidden_size, 
            num_layers=2, 
            bidirectional=True, 
            batch_first=True, 
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size * 2, num_chars)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.layer3(self.layer2(self.layer1(x)))
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).reshape(b, w, c * h)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return torch.nn.functional.log_softmax(x, dim=2).permute(1, 0, 2)

# ============================================================================
# Engine
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc="Training")
    
    for images, targets, input_lens, target_lens, _ in pbar:
        images, targets = images.to(Config.DEVICE), targets.to(Config.DEVICE)
        input_lens, target_lens = input_lens.to(Config.DEVICE), target_lens.to(Config.DEVICE)
        
        if not (input_lens >= target_lens).all(): continue
            
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type=Config.DEVICE.type, enabled=Config.USE_AMP):
            preds = model(images)
            loss = criterion(preds, targets, input_lens, target_lens)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    return total_loss / len(loader)

def validate(model, loader, criterion, encoder):
    model.eval()
    total_loss = 0; total_dist = 0; total_chars = 0; total_words = 0; correct_words = 0
    
    print("\nValidating (Fast Greedy)...")
    
    with torch.no_grad():
        for images, targets, input_lens, target_lens, texts in tqdm(loader):
            images, targets = images.to(Config.DEVICE), targets.to(Config.DEVICE)
            input_lens, target_lens = input_lens.to(Config.DEVICE), target_lens.to(Config.DEVICE)
            
            with torch.amp.autocast(device_type=Config.DEVICE.type, enabled=Config.USE_AMP):
                log_probs = model(images)
                loss = criterion(log_probs, targets, input_lens, target_lens)
            
            total_loss += loss.item()
            
            preds = torch.argmax(log_probs, dim=2).cpu().numpy()
            preds = preds.transpose(1, 0) 
            
            for i, sequence in enumerate(preds):
                target_text = texts[i]
                pred_text = encoder.decode_greedy(sequence)
                
                if pred_text == target_text: correct_words += 1
                total_words += 1
                total_dist += levenshtein_distance(pred_text, target_text)
                total_chars += len(target_text)
                
    wer = 1 - (correct_words / total_words) if total_words > 0 else 1.0
    cer = total_dist / total_chars if total_chars > 0 else 1.0
    print(f"WER: {wer:.2%} | CER: {cer:.2%}")
    return total_loss / len(loader), wer, cer

# ============================================================================
# Final Plotting & Reporting Functions
# ============================================================================

def plot_detailed_metrics(all_predictions, all_ground_truths, save_dir):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    similarities = []
    edit_distances = []
    for pred, gt in zip(all_predictions, all_ground_truths):
        sim = SequenceMatcher(None, pred, gt).ratio()
        similarities.append(sim)
        if len(pred) == 0 and len(gt) == 0: edit_distances.append(0)
        else: edit_distances.append(1 - sim)
    
    ax1.hist(similarities, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    ax1.set_xlabel('Similarity Score', fontsize=12)
    ax1.set_title('Prediction Similarity Distribution', fontsize=14, fontweight='bold')
    ax1.axvline(x=np.mean(similarities), color='red', linestyle='--', label=f'Mean: {np.mean(similarities):.3f}', linewidth=2)
    ax1.legend(); ax1.grid(True, alpha=0.3)
    
    pred_lengths = [len(p) for p in all_predictions]
    gt_lengths = [len(g) for g in all_ground_truths]
    ax2.scatter(gt_lengths, pred_lengths, alpha=0.3, s=10)
    max_len = max(max(pred_lengths) if pred_lengths else 0, max(gt_lengths) if gt_lengths else 0)
    ax2.plot([0, max_len], [0, max_len], 'r--', label='Perfect Prediction', linewidth=2)
    ax2.set_xlabel('Ground Truth Length', fontsize=12); ax2.set_ylabel('Predicted Length', fontsize=12)
    ax2.set_title('Sequence Length Comparison', fontsize=14, fontweight='bold')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    
    length_acc = {}
    for pred, gt in zip(all_predictions, all_ground_truths):
        length = len(gt)
        if length not in length_acc: length_acc[length] = {'correct': 0, 'total': 0}
        length_acc[length]['total'] += 1
        if pred == gt: length_acc[length]['correct'] += 1
    
    lengths = sorted(length_acc.keys())
    accuracies = [length_acc[l]['correct'] / length_acc[l]['total'] for l in lengths]
    counts = [length_acc[l]['total'] for l in lengths]
    
    ax3.scatter(lengths, accuracies, s=[c/2 for c in counts], alpha=0.6)
    ax3.set_xlabel('Sequence Length', fontsize=12); ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_title('Accuracy by Sequence Length', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3); ax3.set_ylim([0, 1])
    
    correct = sum([1 for p, g in zip(all_predictions, all_ground_truths) if p == g])
    incorrect = len(all_predictions) - correct
    
    categories = ['Correct', 'Incorrect']
    values = [correct, incorrect]
    ax4.bar(categories, values, color=['green', 'red'], alpha=0.7, edgecolor='black')
    overall_acc = correct/len(all_predictions) if all_predictions else 0
    ax4.set_title(f'Overall Performance (Acc: {overall_acc:.2%})', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    for i, (cat, val) in enumerate(zip(categories, values)):
        ax4.text(i, val, str(val), ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(str(Path(save_dir) / 'detailed_metrics.png'), dpi=300)
    plt.close()

def plot_confusion_matrix(all_predictions, all_ground_truths, save_dir):
    char_predictions = []
    char_ground_truths = []
    
    for pred, gt in zip(all_predictions, all_ground_truths):
        max_len = max(len(pred), len(gt))
        pred_padded = pred.ljust(max_len, ' ')
        gt_padded = gt.ljust(max_len, ' ')
        char_predictions.extend(list(pred_padded))
        char_ground_truths.extend(list(gt_padded))
    
    char_freq = Counter(char_ground_truths)
    top_chars = [char for char, _ in char_freq.most_common(30) if char != ' ']
    
    filtered_preds = []
    filtered_gts = []
    for p, g in zip(char_predictions, char_ground_truths):
        if g in top_chars:
            filtered_preds.append(p if p in top_chars else 'OTHER')
            filtered_gts.append(g)
            
    labels = top_chars + ['OTHER'] if 'OTHER' in filtered_preds else top_chars
    cm = confusion_matrix(filtered_gts, filtered_preds, labels=labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues', 
               xticklabels=labels, yticklabels=labels, ax=ax, cbar_kws={'label': 'Normalized Frequency'})
    ax.set_xlabel('Predicted Character', fontsize=14); ax.set_ylabel('True Character', fontsize=14)
    ax.set_title('Character-Level Confusion Matrix (Top 30 Characters)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(Path(save_dir) / 'confusion_matrix.png'), dpi=300)
    plt.close()

def plot_sequence_length_analysis(all_predictions, all_ground_truths, save_dir):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    gt_lengths = [len(g) for g in all_ground_truths]
    ax1.hist(gt_lengths, bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
    ax1.set_title('Ground Truth Length Distribution', fontsize=14, fontweight='bold')
    mean_len = np.mean(gt_lengths) if gt_lengths else 0
    ax1.axvline(x=mean_len, color='red', linestyle='--', label=f'Mean: {mean_len:.1f}', linewidth=2)
    ax1.legend(); ax1.grid(True, alpha=0.3)
    
    length_errors = defaultdict(list)
    for pred, gt in zip(all_predictions, all_ground_truths):
        length_errors[len(gt)].append(abs(len(pred) - len(gt)))
    
    lengths = sorted(length_errors.keys())
    avg_errors = [np.mean(length_errors[l]) for l in lengths]
    ax2.scatter(lengths, avg_errors, s=50, alpha=0.6)
    ax2.set_xlabel('Ground Truth Length', fontsize=12); ax2.set_title('Average Length Prediction Error', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    all_chars = ''.join(all_ground_truths)
    char_freq = Counter(all_chars)
    top_20 = char_freq.most_common(20)
    if top_20:
        chars, freqs = zip(*top_20)
        ax3.barh(range(len(chars)), freqs, color='coral', alpha=0.7, edgecolor='black')
        ax3.set_yticks(range(len(chars))); ax3.set_yticklabels([repr(c) for c in chars])
        ax3.set_title('Top 20 Character Frequencies', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        
        char_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
        for pred, gt in zip(all_predictions, all_ground_truths):
            for i, gt_char in enumerate(gt):
                char_accuracy[gt_char]['total'] += 1
                if i < len(pred) and pred[i] == gt_char: char_accuracy[gt_char]['correct'] += 1
        
        top_char_acc = [(c, char_accuracy[c]['correct'] / char_accuracy[c]['total']) for c, _ in top_20 if c in char_accuracy]
        if top_char_acc:
            chars_acc, accs = zip(*top_char_acc)
            ax4.barh(range(len(chars_acc)), accs, color='lightblue', alpha=0.7, edgecolor='black')
            ax4.set_yticks(range(len(chars_acc))); ax4.set_yticklabels([repr(c) for c in chars_acc])
            ax4.set_title('Per-Character Accuracy (Top 20)', fontsize=14, fontweight='bold')
            ax4.set_xlim([0, 1]); ax4.grid(True, alpha=0.3, axis='x')
            
    plt.tight_layout()
    plt.savefig(str(Path(save_dir) / 'sequence_analysis.png'), dpi=300)
    plt.close()

def generate_classification_report(all_predictions, all_ground_truths, save_dir):
    report_path = Path(save_dir) / 'evaluation_report.txt'
    with open(str(report_path), 'w') as f:
        f.write("=" * 80 + "\nCRNN MODEL EVALUATION REPORT\n" + "=" * 80 + "\n\n")
        total_samples = len(all_predictions)
        word_accuracy = sum([1 for p, g in zip(all_predictions, all_ground_truths) if p == g]) / total_samples if total_samples > 0 else 0
        f.write(f"Total Samples: {total_samples}\nWord-Level Accuracy: {word_accuracy:.4f} ({word_accuracy*100:.2f}%)\n\n")
        
        char_correct, char_total = 0, 0
        for pred, gt in zip(all_predictions, all_ground_truths):
            for i in range(max(len(pred), len(gt))):
                p_char, g_char = pred[i] if i < len(pred) else '', gt[i] if i < len(gt) else ''
                if p_char == g_char and g_char != '': char_correct += 1
                if g_char != '': char_total += 1
        
        char_accuracy = char_correct / char_total if char_total > 0 else 0
        f.write(f"Character-Level Accuracy: {char_accuracy:.4f} ({char_accuracy*100:.2f}%)\nTotal Characters: {char_total}\n\n")
        
        f.write("Sample Errors (First 20):\n" + "-" * 80 + "\n")
        error_count = 0
        for pred, gt in zip(all_predictions, all_ground_truths):
            if pred != gt and error_count < 20:
                f.write(f"Ground Truth: '{gt}'\nPredicted:    '{pred}'\n" + "-" * 80 + "\n")
                error_count += 1
        f.write("\nEND OF REPORT\n")

def generate_final_reports(model, loader, encoder):
    print("\n" + "=" * 60)
    print("Generating Final Detailed Metrics & Plots...")
    print("=" * 60)
    
    # Load Best Model
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE, weights_only=True))
    model.eval()
    
    all_preds, all_gts = [], []
    
    with torch.no_grad():
        for images, targets, input_lens, target_lens, texts in tqdm(loader, desc="Collecting Predictions"):
            images = images.to(Config.DEVICE)
            with torch.amp.autocast(device_type=Config.DEVICE.type, enabled=Config.USE_AMP):
                log_probs = model(images)
            preds = torch.argmax(log_probs, dim=2).cpu().numpy().transpose(1, 0)
            
            for i, sequence in enumerate(preds):
                all_gts.append(texts[i])
                all_preds.append(encoder.decode_greedy(sequence))
                
    # Direct output to specific Plot and Metrics folders
    plot_detailed_metrics(all_preds, all_gts, Config.PLOTS_DIR)
    plot_confusion_matrix(all_preds, all_gts, Config.PLOTS_DIR)
    plot_sequence_length_analysis(all_preds, all_gts, Config.PLOTS_DIR)
    generate_classification_report(all_preds, all_gts, Config.METRICS_DIR)
    print(f"\n✓ All detailed plots have been saved to '{Config.PLOTS_DIR}'")
    print(f"✓ Final evaluation report saved to '{Config.METRICS_DIR}'")

# ============================================================================
# Main
# ============================================================================

def main():
    torch.backends.cudnn.benchmark = True
    
    # 1. Create dedicated folders automatically
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.PLOTS_DIR, exist_ok=True)
    os.makedirs(Config.METRICS_DIR, exist_ok=True)
    
    print("1. Cleaning Data...")
    df = clean_dataset(pd.read_csv(Config.LABELS_CSV))
    encoder = LabelEncoder()
    encoder.build_vocab(df['transcription'].values)
    
    train_df = df.sample(frac=0.9, random_state=42)
    val_df = df.drop(train_df.index)
    
    train_loader = DataLoader(
        WordDataset(train_df, Config.IMAGES_PATH, encoder, augment=True),
        batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, 
        collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        WordDataset(val_df, Config.IMAGES_PATH, encoder, augment=False),
        batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS, 
        collate_fn=collate_fn, pin_memory=True
    )
    
    model = CRNN(encoder.num_classes(), Config.LSTM_HIDDEN, dropout=Config.DROPOUT).to(Config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=Config.USE_AMP)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    tracker = HistoryTracker()
    
    best_wer = 1.0
    patience_counter = 0
    
    print(f"3. Starting Anti-Overfit Training (Dropout: {Config.DROPOUT}, Decay: {Config.WEIGHT_DECAY})...")
    for epoch in range(Config.EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{Config.EPOCHS} ---")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, wer, cer = validate(model, val_loader, criterion, encoder)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        tracker.update(epoch+1, train_loss, val_loss, wer, cer, optimizer.param_groups[0]['lr'])
        tracker.plot()
        
        if wer < (best_wer - Config.MIN_DELTA):
            best_wer = wer
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"--> NEW BEST MODEL (WER: {wer:.2%})")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs. Best WER: {best_wer:.2%}")
            
        if patience_counter >= Config.PATIENCE:
            print(f"\nSTOPPING EARLY: No improvement for {Config.PATIENCE} epochs.")
            break
            
        scheduler.step(val_loss)

    # Automatically generate all final detailed plots and matrices
    generate_final_reports(model, val_loader, encoder)

if __name__ == '__main__':
    main()