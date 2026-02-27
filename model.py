"""
CRNN Anti-Overfit: Modularized Implementation  
✅ Cosine Annealing with Warmup
✅ Gradient Accumulation
✅ Layer-wise Learning Rate Decay
✅ Stochastic Weight Averaging (SWA)
✅ Label Smoothing
✅ Stochastic Depth (DropPath)
✅ Enhanced Augmentation (Cutout, Elastic, Noise)
✅ Test-Time Augmentation (TTA)
✅ 70/15/15 Train/Val/Test Split
"""

import os
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR

# Import from modularized files
from config import Config
from utils import LabelEncoder
from data_loader import clean_dataset, WordDataset, collate_fn
from model_architecture import CRNN, LabelSmoothingCTCLoss
from training import (
    CosineAnnealingWarmupRestarts, 
    train_one_epoch, 
    validate, 
    get_layer_wise_lr_params
)
from plotting import HistoryTracker, generate_final_reports

# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    torch.backends.cudnn.benchmark = True
    
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.PLOTS_DIR, exist_ok=True)
    os.makedirs(Config.METRICS_DIR, exist_ok=True)
    
    print("=" * 80)
    print("CRNN ANTI-OVERFITTING TRAINING")
    print("=" * 80)
    print(f"✓ Cosine Annealing with Warmup")
    print(f"✓ Gradient Accumulation (steps={Config.GRADIENT_ACCUMULATION_STEPS})")
    print(f"✓ Layer-wise LR Decay")
    print(f"✓ Stochastic Weight Averaging (start={Config.SWA_START})")
    print(f"✓ Label Smoothing")
    print(f"✓ Stochastic Depth/DropPath")
    print(f"✓ Enhanced Augmentation")
    print(f"✓ Test-Time Augmentation")
    print("=" * 80)
    
    print("\n1. Loading Data...")
    df = clean_dataset(pd.read_csv(Config.LABELS_CSV))
    print(f"   Total samples: {len(df)}")
    
    encoder = LabelEncoder()
    encoder.build_vocab(df['transcription'].values)
    
    # 70/15/15 split
    print("\n2. Creating 70/15/15 Train/Val/Test Split...")
    train_df = df.sample(frac=0.7, random_state=42)
    remaining_df = df.drop(train_df.index)
    val_df = remaining_df.sample(frac=0.5, random_state=42)
    test_df = remaining_df.drop(val_df.index)
    
    print(f"   Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    # Create loaders
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
    test_loader = DataLoader(
        WordDataset(test_df, Config.IMAGES_PATH, encoder, augment=False),
        batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS, 
        collate_fn=collate_fn, pin_memory=True
    )
    
    # Initialize model
    print("\n3. Initializing Model...")
    model = CRNN(
        encoder.num_classes(), 
        Config.LSTM_HIDDEN, 
        dropout=Config.DROPOUT,
        drop_path=Config.DROPOUT_PATH
    ).to(Config.DEVICE)
    
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Layer-wise LR
    print("\n4. Setting up Layer-wise Learning Rates...")
    param_groups = get_layer_wise_lr_params(model, Config.LEARNING_RATE, Config.LAYER_LR_DECAY)
    for pg in param_groups:
        print(f"   {pg['name']}: LR = {pg['lr']:.6f}")
    
    optimizer = optim.AdamW(param_groups, weight_decay=Config.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=Config.USE_AMP)
    
    # Scheduler
    print("\n5. Setting up Cosine Annealing Scheduler...")
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=Config.FIRST_CYCLE_EPOCHS,
        warmup_steps=Config.WARMUP_EPOCHS,
        min_lr=Config.LEARNING_RATE * 0.01
    )
    
    # SWA
    print("\n6. Setting up Stochastic Weight Averaging...")
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=Config.SWA_LR)
    swa_started = False
    
    criterion = LabelSmoothingCTCLoss(blank=0, smoothing=Config.LABEL_SMOOTHING, zero_infinity=True)
    tracker = HistoryTracker()
    
    best_wer = 1.0
    patience_counter = 0
    
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    
    for epoch in range(Config.EPOCHS):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        print(f"{'='*80}")
        
        if epoch == Config.SWA_START and not swa_started:
            print(f"🔄 Starting SWA at epoch {epoch+1}")
            swa_started = True
        
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS
        )
        
        val_loss, val_wer, val_cer = validate(model, val_loader, criterion, encoder, use_tta=False)
        test_loss, test_wer, test_cer = validate(model, test_loader, criterion, encoder, use_tta=False)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n📊 Results:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss:   {val_loss:.4f} | WER: {val_wer:.2%} | CER: {val_cer:.2%}")
        print(f"   Test Loss:  {test_loss:.4f} | WER: {test_wer:.2%} | CER: {test_cer:.2%}")
        print(f"   LR: {current_lr:.6f}")
        
        tracker.update(epoch+1, train_loss, val_loss, test_loss, val_wer, val_cer, current_lr)
        tracker.plot()
        
        if val_wer < (best_wer - Config.MIN_DELTA):
            best_wer = val_wer
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"   ✅ NEW BEST MODEL (Val WER: {val_wer:.2%})")
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement for {patience_counter} epochs (Best: {best_wer:.2%})")
        
        if swa_started:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
        
        if patience_counter >= Config.PATIENCE:
            print(f"\n🛑 EARLY STOPPING after {Config.PATIENCE} epochs")  
            break
    
    # Finalize SWA
    if swa_started:
        print("\n" + "=" * 80)
        print("Finalizing SWA...")
        print("=" * 80)
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=Config.DEVICE)
        torch.save(swa_model.state_dict(), Config.SWA_MODEL_PATH)
        print(f"✓ SWA model saved")
        
        print("\n📊 Evaluating SWA Model (with TTA)...")
        swa_val_loss, swa_val_wer, swa_val_cer = validate(
            swa_model.module, val_loader, criterion, encoder, use_tta=True
        )
        swa_test_loss, swa_test_wer, swa_test_cer = validate(
            swa_model.module, test_loader, criterion, encoder, use_tta=True
        )
        
        print(f"   Val (SWA+TTA):  WER: {swa_val_wer:.2%} | CER: {swa_val_cer:.2%}")
        print(f"   Test (SWA+TTA): WER: {swa_test_wer:.2%} | CER: {swa_test_cer:.2%}")
    
    generate_final_reports(model, test_loader, encoder)
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Best Val WER: {best_wer:.2%}")
    print(f"Checkpoints: {Config.CHECKPOINT_DIR}")
    print(f"Plots: {Config.PLOTS_DIR}")
    print("=" * 80)

if __name__ == '__main__':
    main()
