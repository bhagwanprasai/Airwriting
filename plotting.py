"""
Plotting and Visualization Module
Contains functions for generating plots and metrics visualization
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher
import torch
from tqdm import tqdm

from config import Config


class HistoryTracker:
    """Track and plot training metrics"""
    def __init__(self):
        self.history = {
            'epoch': [], 'train_loss': [], 'val_loss': [], 'test_loss': [],
            'wer': [], 'cer': [], 'lr': []
        }
    
    def update(self, epoch, train_loss, val_loss, test_loss, wer, cer, lr):
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['test_loss'].append(test_loss)
        self.history['wer'].append(wer)
        self.history['cer'].append(cer)
        self.history['lr'].append(lr)
        pd.DataFrame(self.history).to_csv(str(Config.LOG_FILE), index=False)

    def plot(self):
        epochs = self.history['epoch']
        
        plt.figure(figsize=(12, 5))
        plt.plot(epochs, self.history['train_loss'], label='Train Loss', linewidth=2)
        plt.plot(epochs, self.history['val_loss'], label='Val Loss', linewidth=2)
        plt.plot(epochs, self.history['test_loss'], label='Test Loss', linewidth=2, linestyle='--')
        plt.legend(); plt.grid(True); plt.title("Loss Curves")
        plt.xlabel('Epoch'); plt.ylabel('Loss')
        plt.savefig(str(Config.PLOTS_DIR / "loss_curve.png"), dpi=150); plt.close()
        
        plt.figure(figsize=(12, 5))
        plt.plot(epochs, self.history['wer'], label='WER', linewidth=2)
        plt.plot(epochs, self.history['cer'], label='CER', linewidth=2)
        plt.legend(); plt.grid(True); plt.title("Error Metrics")
        plt.xlabel('Epoch'); plt.ylabel('Error Rate')
        plt.savefig(str(Config.PLOTS_DIR / "error_metrics.png"), dpi=150); plt.close()
        
        plt.figure(figsize=(12, 5))
        plt.plot(epochs, self.history['lr'], linewidth=2, color='orange')
        plt.grid(True); plt.title("Learning Rate Schedule")
        plt.xlabel('Epoch'); plt.ylabel('Learning Rate')
        plt.savefig(str(Config.PLOTS_DIR / "learning_rate.png"), dpi=150); plt.close()


def plot_detailed_metrics(all_predictions, all_ground_truths, save_dir):
    """Generate detailed metrics plots"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    similarities = []
    for pred, gt in zip(all_predictions, all_ground_truths):
        sim = SequenceMatcher(None, pred, gt).ratio()
        similarities.append(sim)
    
    ax1.hist(similarities, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    ax1.set_xlabel('Similarity Score')
    ax1.set_title('Prediction Similarity Distribution', fontweight='bold')
    ax1.axvline(x=np.mean(similarities), color='red', linestyle='--', label=f'Mean: {np.mean(similarities):.3f}')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    
    pred_lengths = [len(p) for p in all_predictions]
    gt_lengths = [len(g) for g in all_ground_truths]
    ax2.scatter(gt_lengths, pred_lengths, alpha=0.3, s=10)
    max_len = max(max(pred_lengths) if pred_lengths else 0, max(gt_lengths) if gt_lengths else 0)
    ax2.plot([0, max_len], [0, max_len], 'r--', label='Perfect')
    ax2.set_xlabel('Ground Truth Length'); ax2.set_ylabel('Predicted Length')
    ax2.set_title('Sequence Length Comparison', fontweight='bold')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    
    length_acc = {}
    for pred, gt in zip(all_predictions, all_ground_truths):
        length = len(gt)
        if length not in length_acc:
            length_acc[length] = {'correct': 0, 'total': 0}
        length_acc[length]['total'] += 1
        if pred == gt:
            length_acc[length]['correct'] += 1
    
    lengths = sorted(length_acc.keys())
    accuracies = [length_acc[l]['correct'] / length_acc[l]['total'] for l in lengths]
    counts = [length_acc[l]['total'] for l in lengths]
    
    ax3.scatter(lengths, accuracies, s=[c/2 for c in counts], alpha=0.6)
    ax3.set_xlabel('Sequence Length'); ax3.set_ylabel('Accuracy')
    ax3.set_title('Accuracy by Sequence Length', fontweight='bold')
    ax3.grid(True, alpha=0.3); ax3.set_ylim([0, 1])
    
    correct = sum([1 for p, g in zip(all_predictions, all_ground_truths) if p == g])
    incorrect = len(all_predictions) - correct
    
    ax4.bar(['Correct', 'Incorrect'], [correct, incorrect], color=['green', 'red'], alpha=0.7)
    overall_acc = correct/len(all_predictions) if all_predictions else 0
    ax4.set_title(f'Overall Performance (Acc: {overall_acc:.2%})', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(str(Path(save_dir) / 'detailed_metrics.png'), dpi=300)
    plt.close()


def generate_final_reports(model, loader, encoder):
    """Generate final evaluation reports"""
    print("\nGenerating Final Reports...")
    
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE, weights_only=True))
    model.eval()
    
    all_preds, all_gts = [], []
    
    with torch.no_grad():
        for images, targets, input_lens, target_lens, texts in tqdm(loader, desc="Collecting Predictions"):
            images = images.to(Config.DEVICE)
            with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
                log_probs = model(images)
            preds = torch.argmax(log_probs, dim=2).cpu().numpy().transpose(1, 0)
            
            for i, sequence in enumerate(preds):
                all_gts.append(texts[i])
                all_preds.append(encoder.decode_greedy(sequence))
                
    plot_detailed_metrics(all_preds, all_gts, Config.PLOTS_DIR)
    print(f"✓ Plots saved to '{Config.PLOTS_DIR}'")
