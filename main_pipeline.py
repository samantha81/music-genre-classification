import os
import sys
import time
import random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.config import (
    SAMPLE_RATE, DURATION, SEG_SAMPLES, SEG_STEP, SEG_DURATION, SEG_OVERLAP,
    DATA_DIR, OUTPUT_ROOT, DEVICE, BATCH_SIZE, BATCH_SIZE_SEG, NUM_WORKERS,
    NUM_EPOCHS, PATIENCE, TUNE_EPOCHS, N_TRIALS, TUNE_EPOCHS_SEG, N_TRIALS_SEG,
    CHECKPOINT, CHECKPOINT_SEG, HISTORY_PATH, HISTORY_SEG_PATH
)
from src.utils import pprint, print_header, HAS_KAGGLE
from src.audio.dataset import GTZANDataset, GTZANSegmentDataset, collate_fn, collate_segments
from src.audio.processing import build_file_index, load_or_compute_segments
from src.models.cnn_attention import MusicGenreCNN
from src.models.cnn_lstm import FrozenCNNLSTM
from src.training.loops import (
    train_one_epoch, evaluate, majority_vote_accuracy,
    train_one_epoch_seg, evaluate_seg
)
from src.training.tuning import sample_config, sample_config_seg
from src.evaluation.plots import (
    plot_spectrogram_grid, plot_training_curves,
    plot_hp_loss_curves, plot_hp_results_table
)
from src.evaluation.suite import run_full_evaluation
from src.prediction.predictor import (
    predict_genre, display_prediction, run_demo,
    _check_eval_ready, _load_saved_models
)

def build_data(raw_root):
    """Build file index, split, cache, return all datasets + loaders."""
    print(f"\nRaw Dataset Path: {raw_root}")
    print(f"Output Folder:    {OUTPUT_ROOT}")
    
    segments_per_song = (SAMPLE_RATE * DURATION - SEG_SAMPLES) // SEG_STEP + 1
    print(f"Segments per song: ~{segments_per_song}  (seg={SEG_DURATION}s, overlap={SEG_OVERLAP*100:.0f}%)")
    print(f"Effective training samples ~ {700 * segments_per_song}  (vs 700 before)")

    all_files, all_labels = build_file_index(raw_root)

    print("\nDataset structure:")
    for root, dirs, files in os.walk(str(raw_root)):
        level = root.replace(str(raw_root), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        if level == 1:
            for f in files[:3]: 
                print(f'{indent}  {f}')
            if len(files) > 3:  
                print(f'{indent}  ... ({len(files)} total)')

    # Stratified 70/15/15 split
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        all_files, all_labels, test_size=0.30, stratify=all_labels, random_state=42
    )
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=0.50, stratify=temp_labels, random_state=42
    )
    print(f"Split: Train={len(train_files)} | Val={len(val_files)} | Test={len(test_files)}")

    # Compute local feature cache
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("Pre-computing segments (skips already-cached files)...")
    for file_path in all_files:
        load_or_compute_segments(file_path, OUTPUT_ROOT)
    print("Done.")

    # Segment-level loader setups (for CNN path)
    train_dataset = GTZANDataset(train_files, train_labels, OUTPUT_ROOT, augment=True)
    val_dataset = GTZANDataset(val_files, val_labels, OUTPUT_ROOT, augment=False)
    test_dataset = GTZANDataset(test_files, test_labels, OUTPUT_ROOT, augment=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn
    )
    print(f"Segments - Train:{len(train_dataset)} | Val:{len(val_dataset)} | Test:{len(test_dataset)}")

    # Show sample spectrograms
    plot_spectrogram_grid(train_loader)

    # Song-level loader setups (for CNN+LSTM path)
    train_song_ds = GTZANSegmentDataset(train_files, train_labels, OUTPUT_ROOT, augment=True)
    val_song_ds = GTZANSegmentDataset(val_files, val_labels, OUTPUT_ROOT, augment=False)
    test_song_ds = GTZANSegmentDataset(test_files, test_labels, OUTPUT_ROOT, augment=False)
    
    train_song_dl = DataLoader(
        train_song_ds, batch_size=BATCH_SIZE_SEG, shuffle=True, 
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_segments
    )
    val_song_dl = DataLoader(
        val_song_ds, batch_size=BATCH_SIZE_SEG, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_segments
    )
    test_song_dl = DataLoader(
        test_song_ds, batch_size=BATCH_SIZE_SEG, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_segments
    )
    print(f"Songs   - Train:{len(train_song_ds)} | Val:{len(val_song_ds)} | Test:{len(test_song_ds)}")

    return (train_files, val_files, test_files, train_labels, val_labels, test_labels,
            train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader,
            train_song_ds, val_song_ds, test_song_ds,
            train_song_dl, val_song_dl, test_song_dl)


def run_training(skip_hp_search: bool = False):
    """Full pipeline execution orchestrator."""
    print_header()
    print(f"\nDevice: {DEVICE}")

    # Set up data directories
    raw_root = DATA_DIR / "genres_original"
    if not raw_root.exists():
        if HAS_KAGGLE:
            print("[yellow]Downloading GTZAN from Kaggle...[/yellow]")
            import kagglehub
            download_path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")
            raw_root = Path(download_path) / "Data" / "genres_original"
        else:
            print(f"[red]Dataset not found at {raw_root}[/red]\n"
                  "Place GTZAN 'genres_original' there, or: pip install kagglehub")
            sys.exit(1)

    (train_files, val_files, test_files, train_labels, val_labels, test_labels,
     train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader,
     train_song_ds, val_song_ds, test_song_ds,
     train_song_dl, val_song_dl, test_song_dl) = build_data(raw_root)

    random.seed(42)
    torch.manual_seed(42)

    # CNN Hyperparameter Search
    if not skip_hp_search:
        print(f"\nCNN Random Search: {N_TRIALS} trials x {TUNE_EPOCHS} epochs\n")
        header = f"  {'Trial':<10} {'LR':>9} {'Dropout':>8} {'WD':>9} {'Best Val Loss':>14} {'Best Val Acc':>13}"
        print(header)
        print("  " + "-" * 68)
        
        tune_results = []
        for trial_idx in range(N_TRIALS):
            config = sample_config(trial_idx + 1)
            trial_model = MusicGenreCNN(num_classes=10, dropout=config["dropout"]).to(DEVICE)
            loss_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            optimizer = torch.optim.AdamW(trial_model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TUNE_EPOCHS)
            
            train_losses = []
            val_losses = []
            best_val_loss = float("inf")
            best_val_acc = 0.0
            
            for epoch in range(1, TUNE_EPOCHS + 1):
                train_loss, _ = train_one_epoch(trial_model, train_loader, loss_criterion, optimizer, DEVICE)
                val_loss, val_acc = evaluate(trial_model, val_loader, loss_criterion, DEVICE)
                
                lr_scheduler.step()
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    
            tune_results.append({
                "name": config["name"], "lr": config["lr"], "dropout": config["dropout"],
                "wd": config["weight_decay"], "train_loss": train_losses, "val_loss": val_losses,
                "best_val_loss": best_val_loss, "best_val_acc": best_val_acc,
            })
            print(f"  {config['name']:<10} {config['lr']:>9.2e} {config['dropout']:>8.2f} "
                  f"{config['weight_decay']:>9.2e} {best_val_loss:>14.4f} {best_val_acc:>13.4f}")

        best_config = min(tune_results, key=lambda r: r["best_val_loss"])
        print(f"\n-> Best: {best_config['name']}  "
              f"lr={best_config['lr']:.2e}  dropout={best_config['dropout']:.2f}  wd={best_config['wd']:.2e}")
              
        sorted_results = sorted(tune_results, key=lambda r: r["best_val_loss"])
        plot_hp_loss_curves(tune_results, best_config, "Loss Curves - CNN Random Search Trials", "hp_loss_curves.png")
        plot_hp_results_table(
            tune_results, best_config, sorted_results,
            ["Rank", "Trial", "LR", "Dropout", "Weight Decay", "Best Val Loss", "Best Val Acc"],
            "hp_results_table.png"
        )
        
        worst_config = max(tune_results, key=lambda r: r["best_val_loss"])
        reduction_percentage = (worst_config["best_val_loss"] - best_config["best_val_loss"]) / worst_config["best_val_loss"] * 100
        print(f"Loss reduced by {reduction_percentage:.1f}%  ({worst_config['name']} -> {best_config['name']})")
        
        final_lr = best_config["lr"]
        final_wd = best_config["wd"]
        final_do = best_config["dropout"]
    else:
        final_lr = 5e-4
        final_wd = 1e-4
        final_do = 0.40
        print(f"Skipping HP search. Defaults: lr={final_lr}  wd={final_wd}  dropout={final_do}")

    # Final CNN Training Phase
    print(f"\nFinal CNN training - up to {NUM_EPOCHS} epochs (patience={PATIENCE})")
    print(f"lr={final_lr:.2e}  dropout={final_do:.2f}  wd={final_wd:.2e}\n")

    # Quick shape sanity check
    output_shape_check = MusicGenreCNN()(torch.zeros(2, 1, 128, 128))
    assert output_shape_check.shape == (2, 10), "CNN output shape mismatch!"
    total_cnn_params = sum(p.numel() for p in MusicGenreCNN().parameters() if p.requires_grad)
    print(f"MusicGenreCNN params: {total_cnn_params:,}")

    cnn_model = MusicGenreCNN(num_classes=10, dropout=final_do).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(cnn_model.parameters(), lr=final_lr, weight_decay=final_wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>8}  {'Val Acc':>8}  {'LR':>9}  {'Time':>6}")
    print("-" * 68)

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        train_loss, train_acc = train_one_epoch(cnn_model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = evaluate(cnn_model, val_loader, criterion, DEVICE)
        scheduler.step()
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        current_lr = scheduler.get_last_lr()[0]
        epoch_elapsed_time = time.time() - epoch_start_time
        print(f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>9.4f}  "
              f"{val_loss:>8.4f}  {val_acc:>8.4f}  {current_lr:>9.2e}  {epoch_elapsed_time:>5.1f}s")
              
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(cnn_model.state_dict(), CHECKPOINT)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    print(f"\nBest val loss: {best_val_loss:.4f}  ->  saved to '{CHECKPOINT}'")
    import numpy as np
    np.save(str(HISTORY_PATH), history)
    plot_training_curves(history, "CNN Final Training Curves", "fig_cnn_training_curves.png")

    cnn_model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True))
    test_loss, seg_acc = evaluate(cnn_model, test_loader, criterion, DEVICE)
    vote_acc = majority_vote_accuracy(cnn_model, test_dataset, DEVICE)
    print(f"Segment-level Acc : {seg_acc*100:.2f}%")
    print(f"Song-level Acc    : {vote_acc*100:.2f}%  (+{(vote_acc-seg_acc)*100:.2f}pp from voting)")

    # Frozen CNN + LSTM Phase
    print("\n" + "="*60)
    print("  TRAINING FROZEN CNN + LSTM MODEL")
    print("="*60)

    cnn_backbone = MusicGenreCNN(num_classes=10, dropout=0.0).to(DEVICE)
    cnn_backbone.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True))
    cnn_backbone.eval()
    for p in cnn_backbone.parameters():
        p.requires_grad = False
    print(f"CNN backbone loaded from '{CHECKPOINT}' (frozen).")

    # Sequential model shape validation
    dummy_segment_tensor = torch.zeros(2, 19, 1, 128, 128).to(DEVICE)
    frozen_lstm_check = FrozenCNNLSTM(cnn_backbone).to(DEVICE)
    assert frozen_lstm_check(dummy_segment_tensor).shape == (2, 10), "FrozenCNNLSTM output shape mismatch!"
    trainable_lstm_params = sum(p.numel() for p in frozen_lstm_check.parameters() if p.requires_grad)
    print(f"FrozenCNNLSTM trainable params: {trainable_lstm_params:,}")

    # Hyperparameter Search for CNN + LSTM
    if not skip_hp_search:
        print(f"\nCNN+LSTM Random Search: {N_TRIALS_SEG} trials x {TUNE_EPOCHS_SEG} epochs\n")
        header_seg = f"  {'Trial':<10} {'LR':>9} {'Drop':>6} {'Hidden':>7} {'WD':>9} {'Best Val Loss':>14} {'Best Val Acc':>13}"
        print(header_seg)
        print("  " + "-" * 78)
        
        tune_results_seg = []
        for trial_idx in range(N_TRIALS_SEG):
            config = sample_config_seg(trial_idx + 1)
            trial_model_seg = FrozenCNNLSTM(
                cnn_backbone, num_classes=10, lstm_hidden=config["lstm_hidden"], dropout=config["dropout"]
            ).to(DEVICE)
            loss_criterion_seg = nn.CrossEntropyLoss(label_smoothing=0.1)
            optimizer_seg = torch.optim.AdamW(trial_model_seg.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
            lr_scheduler_seg = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_seg, T_max=TUNE_EPOCHS_SEG)
            
            train_losses_seg = []
            val_losses_seg = []
            best_val_loss_seg = float("inf")
            best_val_acc_seg = 0.0
            
            for epoch in range(1, TUNE_EPOCHS_SEG + 1):
                train_loss, _ = train_one_epoch_seg(trial_model_seg, train_song_dl, loss_criterion_seg, optimizer_seg, DEVICE)
                val_loss, val_acc = evaluate_seg(trial_model_seg, val_song_dl, loss_criterion_seg, DEVICE)
                
                lr_scheduler_seg.step()
                train_losses_seg.append(train_loss)
                val_losses_seg.append(val_loss)
                
                if val_loss < best_val_loss_seg:
                    best_val_loss_seg = val_loss
                    best_val_acc_seg = val_acc
                    
            tune_results_seg.append({
                "name": config["name"], "lr": config["lr"], "dropout": config["dropout"],
                "lstm_hidden": config["lstm_hidden"], "wd": config["weight_decay"],
                "train_loss": train_losses_seg, "val_loss": val_losses_seg,
                "best_val_loss": best_val_loss_seg, "best_val_acc": best_val_acc_seg,
            })
            print(f"  {config['name']:<10} {config['lr']:>9.2e} {config['dropout']:>6.2f} "
                  f"{config['lstm_hidden']:>7} {config['weight_decay']:>9.2e} "
                  f"{best_val_loss_seg:>14.4f} {best_val_acc_seg:>13.4f}")

        best_config_seg = min(tune_results_seg, key=lambda r: r["best_val_loss"])
        print(f"\n-> Best: {best_config_seg['name']}  lr={best_config_seg['lr']:.2e}  "
              f"dropout={best_config_seg['dropout']:.2f}  hidden={best_config_seg['lstm_hidden']}  "
              f"wd={best_config_seg['wd']:.2e}")
              
        sorted_results_seg = sorted(tune_results_seg, key=lambda r: r["best_val_loss"])
        plot_hp_loss_curves(tune_results_seg, best_config_seg, "Frozen CNN+LSTM Loss Curves - Random Search", "seg_hp_loss_curves.png")
        plot_hp_results_table(
            tune_results_seg, best_config_seg, sorted_results_seg,
            ["Rank", "Trial", "LR", "Dropout", "Hidden", "Weight Decay", "Best Val Loss", "Best Val Acc"],
            "seg_hp_results_table.png", extra_col="lstm_hidden"
        )
        
        worst_config_seg = max(tune_results_seg, key=lambda r: r["best_val_loss"])
        reduction_percentage_seg = (worst_config_seg["best_val_loss"] - best_config_seg["best_val_loss"]) / worst_config_seg["best_val_loss"] * 100
        print(f"Loss reduced by {reduction_percentage_seg:.1f}%  ({worst_config_seg['name']} -> {best_config_seg['name']})")
        
        final_lr_seg = best_config_seg["lr"]
        final_wd_seg = best_config_seg["wd"]
        final_do_seg = best_config_seg["dropout"]
        final_hidden_seg = best_config_seg["lstm_hidden"]
    else:
        final_lr_seg = 5e-4
        final_wd_seg = 1e-4
        final_do_seg = 0.60
        final_hidden_seg = 48
        print(f"Skipping HP search. Defaults: lr={final_lr_seg}  wd={final_wd_seg}  dropout={final_do_seg}  hidden={final_hidden_seg}")

    # Final CNN + LSTM Training Phase
    print(f"\nFinal CNN+LSTM training - up to {NUM_EPOCHS} epochs (patience={PATIENCE})")
    print(f"lr={final_lr_seg:.2e}  dropout={final_do_seg:.2f}  hidden={final_hidden_seg}  wd={final_wd_seg:.2e}\n")

    seg_model = FrozenCNNLSTM(cnn_backbone, num_classes=10, lstm_hidden=final_hidden_seg, dropout=final_do_seg).to(DEVICE)
    criterion_seg = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer_seg = torch.optim.AdamW(seg_model.parameters(), lr=final_lr_seg, weight_decay=final_wd_seg)
    scheduler_seg = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_seg, T_max=NUM_EPOCHS, eta_min=1e-6)

    history_seg = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss_seg = float("inf")
    patience_counter_seg = 0

    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>8}  {'Val Acc':>8}  {'LR':>9}  {'Time':>6}")
    print("-" * 68)

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        train_loss, train_acc = train_one_epoch_seg(seg_model, train_song_dl, criterion_seg, optimizer_seg, DEVICE)
        val_loss, val_acc = evaluate_seg(seg_model, val_song_dl, criterion_seg, DEVICE)
        scheduler_seg.step()
        
        history_seg["train_loss"].append(train_loss)
        history_seg["val_loss"].append(val_loss)
        history_seg["train_acc"].append(train_acc)
        history_seg["val_acc"].append(val_acc)
        
        current_lr = scheduler_seg.get_last_lr()[0]
        epoch_elapsed_time = time.time() - epoch_start_time
        print(f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>9.4f}  "
              f"{val_loss:>8.4f}  {val_acc:>8.4f}  {current_lr:>9.2e}  {epoch_elapsed_time:>5.1f}s")
              
        if val_loss < best_val_loss_seg:
            best_val_loss_seg = val_loss
            patience_counter_seg = 0
            torch.save(seg_model.state_dict(), CHECKPOINT_SEG)
        else:
            patience_counter_seg += 1
            if patience_counter_seg >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    print(f"\nBest val loss: {best_val_loss_seg:.4f}  ->  saved to '{CHECKPOINT_SEG}'")
    np.save(str(HISTORY_SEG_PATH), history_seg)
    plot_training_curves(history_seg, "Frozen CNN+LSTM Training Curves", "fig_lstm_training_curves.png")

    seg_model.load_state_dict(torch.load(CHECKPOINT_SEG, map_location=DEVICE, weights_only=True))
    test_loss_s, test_acc_s = evaluate_seg(seg_model, test_song_dl, criterion_seg, DEVICE)
    print(f"CNN+LSTM Test Loss: {test_loss_s:.4f}  |  Test Acc: {test_acc_s*100:.2f}%")
    print(f"Best val acc : {max(history_seg['val_acc']):.4f}")

    # Generate full visual metric reports
    run_full_evaluation(
        cnn_model, seg_model,
        test_dataset, test_loader,
        test_song_ds, test_song_dl,
        history, history_seg,
        criterion, criterion_seg,
    )


def interactive_menu():
    print_header()
    while True:
        pprint("\n[bold cyan]What would you like to do?[/bold cyan]")
        print("  1. Train both models - full pipeline (CNN + Frozen CNN+LSTM)")
        print("  2. Train - skip hyperparameter search (faster)")
        print("  3. Evaluate saved models - all 12 plot sections")
        print("  4. Predict genre of an audio file")
        print("  5. Demo mode (random test samples)")
        print("  6. Exit")
        choice = input("\n> ").strip()

        if choice in ("1", "2"):
            run_training(skip_hp_search=(choice == "2"))

        elif choice == "3":
            _ok, _msg = _check_eval_ready()
            if not _ok:
                pprint(f"[red]{_msg}[/red]")
                continue
            raw_root = DATA_DIR / "genres_original"
            (_, _, _, _, _, _,
             _, _, test_ds, _, _, test_dl,
             _, _, test_seg_ds, _, _, test_seg_dl) = build_data(raw_root)
            cnn_model, seg_model, history, history_seg = _load_saved_models()
            run_full_evaluation(
                cnn_model, seg_model,
                test_ds, test_dl, test_seg_ds, test_seg_dl,
                history, history_seg,
                nn.CrossEntropyLoss(label_smoothing=0.1),
                nn.CrossEntropyLoss(label_smoothing=0.1),
            )

        elif choice == "4":
            path = input("Path to audio file (.wav / .mp3): ").strip()
            pprint("\n[bold cyan]Select model for prediction:[/bold cyan]")
            print("  1. CNN (Segment-level soft voting)")
            print("  2. CNN+LSTM (Sequence-level attention classifier)")
            model_choice = input("> ").strip()
            model_type = "lstm" if model_choice == "2" else "cnn"
            try:
                result = predict_genre(path, model_type=model_type)
                display_prediction(result, path)
            except Exception as e:
                pprint(f"[red]Error: {e}[/red]")

        elif choice == "5":
            run_demo()
            
        elif choice == "6":
            pprint("[dim]Goodbye! [/dim]")
            break
            
        else:
            pprint("[yellow]Invalid choice. Enter 1-6.[/yellow]")

