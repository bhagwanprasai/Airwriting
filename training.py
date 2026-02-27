"""
Training Module
Contains training and validation functions
"""

import math
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from config import Config
from utils import levenshtein_distance


class CosineAnnealingWarmupRestarts(optim.lr_scheduler._LRScheduler):
    """Cosine annealing with warmup"""
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


def train_one_epoch(model, loader, criterion, optimizer, scaler, accumulation_steps=1):
    """Training with gradient accumulation"""
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc="Training")
    
    optimizer.zero_grad()
    batch_idx = 0
    
    for batch_idx, (images, targets, input_lens, target_lens, _) in enumerate(pbar):
        images, targets = images.to(Config.DEVICE), targets.to(Config.DEVICE)
        input_lens, target_lens = input_lens.to(Config.DEVICE), target_lens.to(Config.DEVICE)
        
        if not (input_lens >= target_lens).all():
            continue
        
        with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
            preds = model(images)
            loss = criterion(preds, targets, input_lens, target_lens)
            loss = loss / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        pbar.set_postfix({'loss': f"{loss.item() * accumulation_steps:.4f}"})
    
    if (batch_idx + 1) % accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
    return total_loss / len(loader)


def validate(model, loader, criterion, encoder, use_tta=False):
    """Validation with optional TTA"""
    model.eval()
    total_loss = 0
    total_dist = 0
    total_chars = 0
    total_words = 0
    correct_words = 0
    
    with torch.no_grad():
        for images, targets, input_lens, target_lens, texts in tqdm(loader, desc="Validating"):
            images, targets = images.to(Config.DEVICE), targets.to(Config.DEVICE)
            input_lens, target_lens = input_lens.to(Config.DEVICE), target_lens.to(Config.DEVICE)
            
            with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
                if use_tta and Config.USE_TTA:
                    log_probs_list = []
                    log_probs_list.append(model(images))
                    
                    for _ in range(Config.TTA_TRANSFORMS - 1):
                        aug_images = images * torch.FloatTensor([np.random.uniform(0.9, 1.1)]).to(Config.DEVICE)
                        aug_images = torch.clamp(aug_images, 0, 1)
                        log_probs_list.append(model(aug_images))
                    
                    log_probs = torch.stack(log_probs_list).mean(dim=0)
                else:
                    log_probs = model(images)
                
                loss = criterion(log_probs, targets, input_lens, target_lens)
            
            total_loss += loss.item()
            
            preds = torch.argmax(log_probs, dim=2).cpu().numpy()
            preds = preds.transpose(1, 0)
            
            for i, sequence in enumerate(preds):
                target_text = texts[i]
                pred_text = encoder.decode_greedy(sequence)
                
                if pred_text == target_text:
                    correct_words += 1
                total_words += 1
                total_dist += levenshtein_distance(pred_text, target_text)
                total_chars += len(target_text)
                
    wer = 1 - (correct_words / total_words) if total_words > 0 else 1.0
    cer = total_dist / total_chars if total_chars > 0 else 1.0
    
    return total_loss / len(loader), wer, cer


def get_layer_wise_lr_params(model, base_lr, decay_rate=0.95):
    """Layer-wise learning rate decay"""
    params = []
    
    params.append({
        'params': model.layer1.parameters(),
        'lr': base_lr * (decay_rate ** 3),
        'name': 'layer1'
    })
    
    params.append({
        'params': model.layer2.parameters(),
        'lr': base_lr * (decay_rate ** 2),
        'name': 'layer2'
    })
    
    params.append({
        'params': model.layer3.parameters(),
        'lr': base_lr * decay_rate,
        'name': 'layer3'
    })
    
    params.append({
        'params': list(model.rnn.parameters()) + list(model.fc.parameters()),
        'lr': base_lr,
        'name': 'rnn_fc'
    })
    
    return params
