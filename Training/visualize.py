"""
Visualization and plotting utilities
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import torch

from Training.config import Config
from Training.utils import levenshtein_distance
from Training.architecture import BeamSearchDecoder


class HistoryTracker:
    """Track and visualize training history"""
    def __init__(self):
        self.history = defaultdict(list)
        
    def update(self, epoch, train_loss, val_loss, wer, cer, lr):
        """Update history with metrics from current epoch"""
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['wer'].append(wer)
        self.history['cer'].append(cer)
        self.history['lr'].append(lr)
        
        df = pd.DataFrame(self.history)
        df.to_csv(Config.LOG_FILE, index=False)
    
    def plot(self):
        """Plot training curves"""
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
            images, targets, target_lengths, texts, attention_mask, widths = batch
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
                # Decode ground truth from targets
                gt_indices = targets[i].cpu().numpy()
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
    model.eval()
    device = next(model.parameters()).device
    
    # Collect character-level errors
    substitutions = Counter()  # (gt_char, pred_char) -> count
    insertions = Counter()     # pred_char -> count
    deletions = Counter()      # gt_char -> count
    
    sample_count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images, targets, target_lengths, texts, attention_mask, widths = batch
            images = images.to(device)
            attention_mask = attention_mask.to(device)
            
            outputs, _ = model(images, attention_mask=attention_mask)
            pred_indices = outputs.argmax(dim=-1).cpu().numpy()
            
            for i in range(len(pred_indices)):
                gt_indices = targets[i].cpu().numpy()
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
