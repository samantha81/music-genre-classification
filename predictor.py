import numpy as np
import torch
import torch.nn as nn
import random
from pathlib import Path
import sys

from src.config import (
    CHECKPOINT, CHECKPOINT_SEG, HISTORY_PATH, HISTORY_SEG_PATH,
    DEVICE, GENRES, IDX_TO_GENRE, GENRE_EMOJI, DATA_DIR, OUTPUT_ROOT
)
from src.utils import pprint, print_header, RICH, console
from src.audio.processing import load_audio, slice_waveform, waveform_to_melspectrogram, normalise_spectrogram
from src.models.cnn_attention import MusicGenreCNN
from src.models.cnn_lstm import FrozenCNNLSTM

try:
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
except ImportError:
    pass

@torch.inference_mode()
def predict_genre(audio_path: str, model_type: str = "cnn", model=None, top_k: int = 3) -> dict:
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Not found: {audio_path}")

    # Load audio and prepare segments
    waveform = load_audio(audio_path)
    segments = slice_waveform(waveform)
    if not segments:
        raise ValueError("Audio too short to extract segments.")
    mels = torch.stack([normalise_spectrogram(waveform_to_melspectrogram(s)) for s in segments]).to(DEVICE)

    if model_type == "cnn":
        if model is None:
            if not Path(CHECKPOINT).exists():
                raise RuntimeError(f"No saved model at {CHECKPOINT}. Run --train first.")
            model = MusicGenreCNN().to(DEVICE)
            model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True))
        model.eval()
        logits = model(mels)
        avg_probs = torch.softmax(logits, dim=1).mean(dim=0).cpu().numpy()
    elif model_type == "lstm":
        if model is None:
            if not Path(CHECKPOINT).exists():
                raise RuntimeError(f"CNN backbone checkpoint not found at {CHECKPOINT}. Train CNN backbone first.")
            if not Path(CHECKPOINT_SEG).exists():
                raise RuntimeError(f"No saved LSTM model at {CHECKPOINT_SEG}. Run training first.")
            bb = MusicGenreCNN(dropout=0.0).to(DEVICE)
            bb.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True))
            for p in bb.parameters():
                p.requires_grad = False
            model = FrozenCNNLSTM(bb, lstm_hidden=48).to(DEVICE)
            model.load_state_dict(torch.load(CHECKPOINT_SEG, map_location=DEVICE, weights_only=True))
        model.eval()
        lstm_input = mels.unsqueeze(0)  # (1, N, 1, 128, 128)
        logits = model(lstm_input)
        avg_probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Choose 'cnn' or 'lstm'.")

    top_indices = np.argsort(avg_probs)[::-1][:top_k]
    
    return {
        "predicted_genre": IDX_TO_GENRE[top_indices[0]],
        "confidence": float(avg_probs[top_indices[0]]),
        "top_k": [(IDX_TO_GENRE[i], float(avg_probs[i])) for i in top_indices],
        "all_probs": {IDX_TO_GENRE[i]: float(avg_probs[i]) for i in range(10)},
        "n_segments": len(segments),
        "model_used": "CNN (soft voting)" if model_type == "cnn" else "CNN+LSTM (attention classifier)"
    }


def display_prediction(result: dict, audio_path: str):
    genre = result["predicted_genre"]
    confidence = result["confidence"]
    genre_label = GENRE_EMOJI.get(genre, genre.title())
    n_segments = result["n_segments"]
    model_used = result.get("model_used", "CNN")
    
    if RICH:
        console.print(Panel(
            f"[bold]{genre_label}[/bold]\n"
            f"[dim]Confidence: {confidence*100:.1f}%  |  Segments: {n_segments}  |  Model: {model_used}[/dim]",
            title=f"[cyan]{Path(audio_path).name}[/cyan]", border_style="green"
        ))
        
        table = Table(title="Top Predictions", box=box.SIMPLE)
        table.add_column("Rank", style="dim")
        table.add_column("Genre", style="bold")
        table.add_column("Confidence")
        table.add_column("Bar")
        
        for rank, (g, p) in enumerate(result["top_k"], 1):
            bar_length = int(p * 30)
            bar = "#" * bar_length + "-" * (30 - bar_length)
            color = "green" if rank == 1 else "dim"
            table.add_row(
                str(rank), GENRE_EMOJI.get(g, g.title()), f"{p*100:.1f}%", f"[{color}]{bar}[/]"
            )
        console.print(table)
    else:
        print(f"\nFile: {Path(audio_path).name}")
        print(f"Predicted: {genre_label}  ({confidence*100:.1f}%)")
        print(f"Segments : {n_segments}")
        print(f"Model    : {model_used}")
        
        for rank, (g, p) in enumerate(result["top_k"], 1):
            bar = "#" * int(p * 30)
            print(f"  {rank}. {g:<12} {p*100:.1f}%  {bar}")


def run_demo():
    print_header()
    pprint("\n[bold]== DEMO MODE ==[/bold]")
    raw_root = DATA_DIR / "genres_original"
    if not raw_root.exists():
        pprint(f"[red]Dataset not found at {raw_root}[/red]")
        sys.exit(1)
    if not Path(CHECKPOINT).exists():
        pprint(f"[red]No saved model at {CHECKPOINT}[/red]")
        sys.exit(1)
    model = MusicGenreCNN().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True))

    demo_genres = random.sample(list(GENRES.keys()), 5)
    demo_files = []
    for genre in demo_genres:
        files = sorted((raw_root / genre).iterdir())[:10]
        if files:
            demo_files.append((random.choice(files), genre))

    correct = 0
    for audio_path, true_genre in demo_files:
        try:
            result = predict_genre(str(audio_path), model=model)
            display_prediction(result, str(audio_path))
            hit = result["predicted_genre"] == true_genre
            correct += int(hit)
            status = "[green]OK CORRECT[/green]" if hit else f"[red]MISS True: {true_genre}[/red]"
            pprint(f"  {status}\n")
        except Exception as e:
            pprint(f"[yellow]  Skipped {audio_path.name}: {e}[/yellow]")

    pprint(f"[bold]Demo accuracy: {correct}/{len(demo_files)}[/bold]")


def _check_eval_ready():
    missing = []

    if not Path(CHECKPOINT).exists():
        missing.append(f"CNN model not found: {CHECKPOINT}")

    raw_root = DATA_DIR / "genres_original"
    if not raw_root.exists():
        missing.append(f"Dataset not found: {raw_root} (run --train to download, or place GTZAN folder there)")

    cache_exists = any(OUTPUT_ROOT.glob("**/*.pt")) if OUTPUT_ROOT.exists() else False
    if not cache_exists and not raw_root.exists():
        missing.append(f"No spectrogram cache found at {OUTPUT_ROOT}")

    if missing:
        msg = ("Cannot run evaluation - the following are missing:\n  "
               + "\n  ".join(f"- {m}" for m in missing)
               + "\n\nRun option 1 or 2 (train) first.")
        return False, msg

    return True, "OK"


def _load_saved_models():
    cnn_model = MusicGenreCNN().to(DEVICE)
    cnn_model.load_state_dict(
        torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True))
    print(f"Loaded CNN from {CHECKPOINT}")

    seg_model = None
    if Path(CHECKPOINT_SEG).exists():
        try:
            bb = MusicGenreCNN(dropout=0.0).to(DEVICE)
            bb.load_state_dict(
                torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True))
            for p in bb.parameters():
                p.requires_grad = False
            seg_model = FrozenCNNLSTM(bb, lstm_hidden=48).to(DEVICE)
            seg_model.load_state_dict(
                torch.load(CHECKPOINT_SEG, map_location=DEVICE, weights_only=True))
            print(f"Loaded CNN+LSTM from {CHECKPOINT_SEG}")
        except Exception as e:
            print(f"[WARN] Could not load CNN+LSTM: {e} (evaluation will skip LSTM sections)")
            seg_model = None
    else:
        print(f"[INFO] CNN+LSTM checkpoint not found ({CHECKPOINT_SEG}) - LSTM sections will be skipped.")

    history = None
    if HISTORY_PATH.exists():
        try:
            history = np.load(str(HISTORY_PATH), allow_pickle=True).item()
        except Exception:
            print("[WARN] Could not load CNN training history.")

    history_seg = None
    if HISTORY_SEG_PATH.exists():
        try:
            history_seg = np.load(str(HISTORY_SEG_PATH), allow_pickle=True).item()
        except Exception:
            print("[WARN] Could not load CNN+LSTM training history.")

    return cnn_model, seg_model, history, history_seg

