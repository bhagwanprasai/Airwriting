"""
CRNN with Seq2Seq Attention: ANTI-OVERFIT Edition
Specifically tuned for IAM Handwriting Dataset

This is the main entry point. All components are modularized:
- config.py: Configuration and constants
- utils.py: Utility functions (EMA, schedulers, etc.)
- data.py: Data loading and augmentation
- architecture.py: Model architecture components
- train.py: Training and validation loops
- visualize.py: Visualization and plotting

CSV Format: 2 columns
- Column 1: Image filename (e.g., xyz.png)
- Column 2: Transcription text
"""

import os
import warnings
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
import copy

# Suppress OpenCV warnings
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

# Import all modular components
from Training.config import Config
from Training.utils import (
    EMA, CosineAnnealingWarmupRestarts, get_layer_wise_params
)
from Training.data import (
    LabelEncoder, clean_dataset, validate_dataset_files, 
    WordDataset, collate_fn
)
from Training.architecture import Seq2SeqAttention
from Training.train import train_one_epoch, validate
from Training.visualize import (
    HistoryTracker, visualize_attention, visualize_predictions, 
    plot_confusion_matrix
)


def main():
    """Main training pipeline"""
    
    # GPU optimizations for RTX 2060
    torch.backends.cudnn.benchmark = True  # Auto-tune convolution algorithms
    torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 for matmul (faster)
    torch.backends.cudnn.allow_tf32 = True  # Use TF32 for cudnn (faster)
    
    # Print GPU info and config
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"Config: BATCH_SIZE={Config.BATCH_SIZE}, NUM_WORKERS={Config.NUM_WORKERS}")
    print(f"Loading from: {Config.IMAGES_PATH}")
    
    # Validate paths before starting
    print("=" * 60)
    print("Path Validation")
    print("=" * 60)
    
    if not Config.LABELS_CSV.exists():
        print(f"❌ ERROR: CSV file not found at: {Config.LABELS_CSV}")
        print(f"   Please update Config.LABELS_CSV to point to your words.csv file")
        return
    
    if not Config.IMAGES_PATH.exists():
        print(f"❌ ERROR: Images directory not found at: {Config.IMAGES_PATH}")
        print(f"   Please update Config.IMAGES_PATH to point to your processed_word_dataset folder")
        return
    
    print(f"✓ CSV file: {Config.LABELS_CSV}")
    print(f"✓ Images directory: {Config.IMAGES_PATH}")
    
    # Create directories
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.PLOTS_DIR, exist_ok=True)
    os.makedirs(Config.METRICS_DIR, exist_ok=True)
    
    print("=" * 60)
    print("Seq2Seq Attention Training")
    print("✅ Batch size: 32")
    print("✅ Dropout: 0.3 (seq2seq optimized)")
    print("✅ Light augmentation (rotation, blur, noise)")
    print("✅ Attention masking for variable-width images")
    print("✅ EMA + SWA + DropPath for anti-overfitting")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading and cleaning dataset...")
    df = clean_dataset(pd.read_csv(Config.LABELS_CSV))
    
    # Validate that all image files exist (skip if on different machine)
    df = validate_dataset_files(df, Config.IMAGES_PATH, skip_if_not_exists=True)
    
    if len(df) == 0:
        print("\n❌ ERROR: No valid samples found in dataset!")
        print("   This usually means:")
        print("   1. Image directory path is incorrect")
        print("   2. Images are in subdirectories (IAM uses a01/, a01-000u/, etc.)")
        print("   3. Running on different machine - update Config.IMAGES_PATH")
        print("\n   The code will work when run on the training machine with correct paths.")
        return
    
    encoder = LabelEncoder()
    encoder.build_vocab(df['transcription'].values)
    
    # 70/15/15 split
    train_df = df.sample(frac=0.7, random_state=42)
    remaining_df = df.drop(train_df.index)
    val_df = remaining_df.sample(frac=0.5, random_state=42)
    test_df = remaining_df.drop(val_df.index)
    
    print(f"Train samples: {len(train_df)} (70%)")
    print(f"Val samples: {len(val_df)} (15%)")
    print(f"Test samples: {len(test_df)} (15%)")
    
    # Build path map ONCE - shared across all 3 datasets (avoids 3x dir scan)
    WordDataset.build_shared_path_map(Config.IMAGES_PATH)
    
    # Create dataloaders with optimized settings
    loader_kwargs = {
        'batch_size': Config.BATCH_SIZE,
        'num_workers': Config.NUM_WORKERS,
        'collate_fn': collate_fn,
        'pin_memory': True,
        'prefetch_factor': Config.PREFETCH_FACTOR if Config.NUM_WORKERS > 0 else None,
        'persistent_workers': Config.PERSISTENT_WORKERS if Config.NUM_WORKERS > 0 else False
    }
    
    train_loader = DataLoader(
        WordDataset(train_df, Config.IMAGES_PATH, encoder, augment=True),
        shuffle=True,
        **loader_kwargs
    )
    val_loader = DataLoader(
        WordDataset(val_df, Config.IMAGES_PATH, encoder, augment=False),
        shuffle=False,
        **loader_kwargs
    )
    test_loader = DataLoader(
        WordDataset(test_df, Config.IMAGES_PATH, encoder, augment=False),
        shuffle=False,
        **loader_kwargs
    )
    
    # Create model
    print("\n2. Creating Seq2Seq model...")
    model = Seq2SeqAttention(
        num_classes=encoder.num_classes(),
        encoder_hidden=Config.ENCODER_HIDDEN,
        decoder_hidden=Config.DECODER_HIDDEN,
        attention_hidden=Config.ATTENTION_HIDDEN,
        dropout=Config.DROPOUT,
        drop_path=Config.DROP_PATH  # Stochastic depth for anti-overfitting
    ).to(Config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Setup with layer-wise LR
    print("\n3. Setting up Layer-wise Learning Rates...")
    param_groups = get_layer_wise_params(model, Config.LEARNING_RATE, Config.LAYER_LR_DECAY)
    for group in param_groups:
        print(f"  {group['name']}: LR = {group['lr']:.6f}")
    
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=Config.LABEL_SMOOTHING)
    optimizer = optim.AdamW(param_groups, weight_decay=Config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=Config.USE_AMP)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=Config.FIRST_CYCLE_EPOCHS,
        warmup_steps=Config.WARMUP_EPOCHS,
        min_lr=float(Config.LEARNING_RATE * 0.01)
    )
    
    # EMA Setup
    ema = EMA(model, decay=Config.EMA_DECAY) if Config.USE_EMA else None
    if ema:
        print(f"\n4. EMA enabled (decay={Config.EMA_DECAY})")
    
    # SWA Setup
    print(f"5. SWA will start at epoch {Config.SWA_START}")
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=Config.SWA_LR)
    swa_started = False
    
    tracker = HistoryTracker()
    
    best_wer = 1.0
    best_model_state = None
    patience_counter = 0
    teacher_forcing_ratio = Config.TEACHER_FORCING_RATIO
    
    print(f"\n6. Starting training...")
    print("=" * 60)
    
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        print(f"Teacher Forcing: {teacher_forcing_ratio:.3f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if epoch >= Config.SWA_START and not swa_started:
            print(f"🔄 Starting SWA")
            swa_started = True
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, 
                                     teacher_forcing_ratio, epoch+1, ema)
        
        # Update SWA
        if swa_started:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        
        # Validate
        use_beam = Config.USE_BEAM_SEARCH and epoch >= Config.WARMUP_EPOCHS
        val_loss, wer, cer = validate(model, val_loader, criterion, encoder, use_beam_search=use_beam)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Track
        tracker.update(epoch+1, train_loss, val_loss, wer, cer, optimizer.param_groups[0]['lr'])
        tracker.plot()
        
        # Save best
        if wer < (best_wer - Config.MIN_DELTA):
            best_wer = wer
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"✓ NEW BEST MODEL (WER: {wer:.2%})")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter}/{Config.PATIENCE} epochs. Best WER: {best_wer:.2%}")
        
        # Early stopping
        if patience_counter >= Config.PATIENCE:
            print(f"\n⚠ EARLY STOPPING")
            # Restore best weights
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                print("✓ Restored best model weights")
            break
        
        # Decay teacher forcing (with minimum)
        teacher_forcing_ratio = max(
            Config.MIN_TEACHER_FORCING, 
            teacher_forcing_ratio * Config.TEACHER_FORCING_DECAY
        )
        
        # Step scheduler
        if not swa_started:
            scheduler.step()
    
    # Update SWA BN
    if swa_started:
        print("\n7. Updating SWA Batch Normalization...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=Config.DEVICE)
        torch.save(swa_model.state_dict(), Config.SWA_MODEL_PATH)
        print(f"✓ SWA model saved")
    
    # Save EMA model
    if ema:
        print("\n8. Saving EMA model...")
        ema.apply_shadow(model)
        torch.save(model.state_dict(), Config.EMA_MODEL_PATH)
        ema.restore(model)
        print(f"✓ EMA model saved")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("9. Final Evaluation on Test Set")
    print("=" * 60)
    
    models_to_test = [
        ("Best Checkpoint", Config.BEST_MODEL_PATH),
    ]
    if swa_started:
        models_to_test.append(("SWA Model", Config.SWA_MODEL_PATH))
    if ema:
        models_to_test.append(("EMA Model", Config.EMA_MODEL_PATH))
    
    results = []
    for name, path in models_to_test:
        print(f"\nEvaluating {name}...")
        model.load_state_dict(torch.load(path, map_location=Config.DEVICE, weights_only=True))
        test_loss, test_wer, test_cer = validate(model, test_loader, criterion, encoder, use_beam_search=True)
        results.append((name, test_wer, test_cer))
        print(f"{name} - Test WER: {test_wer:.2%} | Test CER: {test_cer:.2%}")
    
    # Find best
    best_result = min(results, key=lambda x: x[1])
    print(f"\n✨ Best model: {best_result[0]} (WER: {best_result[1]:.2%})")
    
    # Load best model for visualizations
    best_model_map = {
        "Best Checkpoint": Config.BEST_MODEL_PATH,
        "SWA Model": Config.SWA_MODEL_PATH,
        "EMA Model": Config.EMA_MODEL_PATH
    }
    model.load_state_dict(torch.load(best_model_map[best_result[0]], 
                                      map_location=Config.DEVICE, weights_only=True))
    
    # Generate comprehensive visualizations
    print("\n" + "=" * 60)
    print("10. Generating Visualizations")
    print("=" * 60)
    
    # Overfitting gap plot
    print("  • Plotting overfitting analysis...")
    tracker.plot_overfitting_gap()
    
    # Sample predictions
    print("  • Generating sample predictions...")
    visualize_predictions(
        model, test_loader, encoder, 
        save_path=str(Config.METRICS_DIR / 'sample_predictions.txt'),
        num_samples=50, use_beam=True
    )
    
    # Attention visualization (on a few test samples)
    print("  • Visualizing attention maps...")
    test_batch = next(iter(test_loader))
    images, targets, target_lengths, texts, attention_mask, widths = test_batch
    visualize_attention(
        model, images[:4], encoder, 
        attention_mask=attention_mask[:4].to(Config.DEVICE),
        save_path=str(Config.PLOTS_DIR / 'attention_maps.png'),
        num_samples=4
    )
    
    # Character-level error analysis
    print("  • Analyzing character-level errors...")
    plot_confusion_matrix(
        model, test_loader, encoder,
        save_path=str(Config.METRICS_DIR / 'character_errors.txt'),
        max_samples=500
    )
    
    print(f"\n✓ Visualizations saved to:")
    print(f"  Plots: {Config.PLOTS_DIR}")
    print(f"  Metrics: {Config.METRICS_DIR}")
    
    print("\n" + "=" * 60)
    print("Training Completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
