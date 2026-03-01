"""
Utility functions and classes for training
"""

import torch
import torch.optim as optim
import math


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


class CosineAnnealingWarmupRestarts(optim.lr_scheduler._LRScheduler):
    """Cosine annealing scheduler with warmup"""
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


def add_gradient_noise(model, std=0.01):
    """Add Gaussian noise to gradients to improve generalization"""
    for param in model.parameters():
        if param.grad is not None:
            noise = torch.randn_like(param.grad) * std
            param.grad.add_(noise)


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


def get_layer_wise_params(model, learning_rate, layer_lr_decay):
    """Get parameters with layer-wise learning rates"""
    params = []
    
    params.append({
        'params': model.cnn_encoder.parameters(),
        'lr': learning_rate * (layer_lr_decay ** 3),
        'name': 'cnn_encoder'
    })
    
    params.append({
        'params': model.lstm_encoder.parameters(),
        'lr': learning_rate * (layer_lr_decay ** 2),
        'name': 'lstm_encoder'
    })
    
    params.append({
        'params': model.decoder.parameters(),
        'lr': learning_rate,
        'name': 'decoder'
    })
    
    params.append({
        'params': list(model.encoder_to_decoder_h.parameters()) + 
                 list(model.encoder_to_decoder_c.parameters()),
        'lr': learning_rate * layer_lr_decay,
        'name': 'bridge'
    })
    
    return params
