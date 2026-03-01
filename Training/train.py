"""
Training and validation functions
"""

import torch
from tqdm import tqdm

from Training.config import Config
from Training.utils import add_gradient_noise, levenshtein_distance


def train_one_epoch(model, loader, criterion, optimizer, scaler, teacher_forcing_ratio, epoch, ema=None):
    """Train for one epoch"""
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
            
            # Add gradient noise
            if Config.GRADIENT_NOISE_STD > 0:
                add_gradient_noise(model, Config.GRADIENT_NOISE_STD)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Update EMA
            if ema is not None:
                ema.update(model)
        
        total_loss += loss.item() * Config.GRADIENT_ACCUMULATION_STEPS
        pbar.set_postfix({'loss': f'{loss.item() * Config.GRADIENT_ACCUMULATION_STEPS:.4f}'})
    
    return total_loss / len(loader)


def validate(model, loader, criterion, encoder, use_beam_search=False):
    """Validate model"""
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
