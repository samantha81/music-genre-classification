import streamlit as st
import numpy as np
import threading
import torch
import torch.nn as nn
from pathlib import Path
import time
import os
import random
import pickle
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pandas as pd

# Import project utilities and models
from src.config import (
    CHECKPOINT, CHECKPOINT_SEG, HISTORY_PATH, HISTORY_SEG_PATH,
    DATA_DIR, PLOT_DIR, GENRES, IDX_TO_GENRE, GENRE_EMOJI, DEVICE,
    SAMPLE_RATE, BATCH_SIZE, BATCH_SIZE_SEG, NUM_WORKERS,
    PATIENCE, OUTPUT_ROOT, DURATION, SEG_DURATION, SEG_OVERLAP,
    SEG_SAMPLES, SEG_STEP
)
from src.utils import count_params, model_size_mb, measure_inference, PALETTE, save_fig
from src.audio.processing import load_audio, slice_waveform, waveform_to_melspectrogram, normalise_spectrogram, build_file_index, load_or_compute_segments
from src.models.cnn_attention import MusicGenreCNN
from src.models.cnn_lstm import FrozenCNNLSTM
from src.main_pipeline import build_data
from src.evaluation.metrics import majority_vote, compute_metrics
from src.evaluation.plots import plot_confusion_matrix, plot_roc, plot_embeddings
from sklearn.metrics import classification_report, confusion_matrix as sk_confusion_matrix, roc_auc_score
from src.audio.dataset import GTZANDataset, GTZANSegmentDataset, collate_fn, collate_segments
from src.training.loops import (
    train_one_epoch, evaluate, majority_vote_accuracy,
    train_one_epoch_seg, evaluate_seg
)

# Set page config and aesthetic theme
st.set_page_config(
    page_title="Music Genre Classifier Studio",
    page_icon="music",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main app background and text colors */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
        font-family: 'Inter', -apple-system, system-ui, sans-serif;
    }
    
    /* Headers styling */
    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc !important;
        font-weight: 600 !important;
        margin-top: 0.5rem !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid #334155;
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #94a3b8;
    }
    
    /* Card design */
    .studio-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #10b981;
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.2rem;
    }
    
    /* Dynamic status badges */
    .cuda-badge {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .cpu-badge {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .model-info-badge {
        background-color: rgba(59, 130, 246, 0.15);
        color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.3);
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }

    /* Tab styling overrides */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8 !important;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        color: #10b981 !important;
        border-bottom-color: #10b981 !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Caching Model Loaders
# ----------------------------------------------------
@st.cache_resource
def load_saved_models_cached():
    cnn_model = None
    seg_model = None
    history = None
    history_seg = None
    
    if Path(CHECKPOINT).exists():
        try:
            cnn_model = MusicGenreCNN().to(DEVICE)
            cnn_model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True))
            cnn_model.eval()
        except Exception as e:
            st.sidebar.error(f"Error loading CNN backbone: {e}")
            
    if Path(CHECKPOINT_SEG).exists() and cnn_model is not None:
        try:
            bb = MusicGenreCNN(dropout=0.0).to(DEVICE)
            bb.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True))
            for p in bb.parameters():
                p.requires_grad = False
            seg_model = FrozenCNNLSTM(bb, lstm_hidden=48).to(DEVICE)
            seg_model.load_state_dict(torch.load(CHECKPOINT_SEG, map_location=DEVICE, weights_only=True))
            seg_model.eval()
        except Exception as e:
            st.sidebar.error(f"Error loading LSTM: {e}")
            
    if HISTORY_PATH.exists():
        try:
            history = np.load(str(HISTORY_PATH), allow_pickle=True).item()
        except Exception:
            pass
            
    if HISTORY_SEG_PATH.exists():
        try:
            history_seg = np.load(str(HISTORY_SEG_PATH), allow_pickle=True).item()
        except Exception:
            pass
            
    return cnn_model, seg_model, history, history_seg

# Initialize cache
cnn_model, seg_model, history, history_seg = load_saved_models_cached()
EVAL_CACHE_PATH = PLOT_DIR / "streamlit_evaluation_cache.pkl"

# Background evaluation runner
def _save_evaluation_cache(result):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": time.time(),
        "result": result,
    }
    with EVAL_CACHE_PATH.open("wb") as f:
        pickle.dump(payload, f)


def _load_evaluation_cache():
    if not EVAL_CACHE_PATH.exists():
        return None
    try:
        with EVAL_CACHE_PATH.open("rb") as f:
            payload = pickle.load(f)
        return payload.get("result")
    except Exception:
        return None


def _compute_full_evaluation_in_background(_cnn_model, _seg_model, state):
    try:
        result = compute_full_evaluation(_cnn_model, _seg_model)
        _save_evaluation_cache(result)
        state["result"] = result
        state["status"] = "done"
    except Exception as e:
        state["error"] = str(e)
        state["status"] = "error"

# Evaluation state
@st.cache_resource(show_spinner=False)
def get_eval_state():
    """A single shared, mutable dict that tracks evaluation progress and
    detects when model checkpoints change so it can trigger a re-run."""
    return {
        "status": "idle", "result": None, "error": None,
        "lock": threading.Lock(),
        "cnn_mtime": None,  # last evaluated CNN checkpoint mtime
        "seg_mtime": None,  # last evaluated LSTM checkpoint mtime
    }

eval_state = get_eval_state()

def _most_confused_pair(y_true, y_pred, class_names):
    """Mirrors the off-diagonal-argmax logic inside plots.plot_confusion_matrix
    (which only prints this, it doesn't return it) so the dashboard can show
    the same 'most confused pair' figure next to the saved confusion matrix
    image."""
    cm = sk_confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm = cm.copy()
    np.fill_diagonal(cm, 0)
    if cm.max() == 0:
        return None
    idx = np.unravel_index(np.argmax(cm), cm.shape)
    return class_names[idx[0]], class_names[idx[1]], int(cm[idx])


class _SimpleLoader:
    """Simple batch iterator for background evaluation on Windows."""
    def __init__(self, dataset, batch_size, collate_fn, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            import random
            random.shuffle(indices)
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]
            yield self.collate_fn(batch)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def _build_test_datasets_only(raw_root):
    """Build only test datasets needed for evaluation, without creating any
    DataLoaders (avoids spawning worker processes, which breaks on Windows
    when called from a Streamlit background thread)."""
    from sklearn.model_selection import train_test_split
    from src.audio.processing import build_file_index, load_or_compute_segments

    all_files, all_labels = build_file_index(raw_root)

    # Stratified 70/15/15 split (same seed as build_data)
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        all_files, all_labels, test_size=0.30, stratify=all_labels, random_state=42
    )
    _, test_files, _, test_labels = train_test_split(
        temp_files, temp_labels, test_size=0.50, stratify=temp_labels, random_state=42
    )

    # Ensure test segments are cached
    for file_path in test_files:
        load_or_compute_segments(file_path, OUTPUT_ROOT)

    test_dataset = GTZANDataset(test_files, test_labels, OUTPUT_ROOT, augment=False)
    test_seg_dataset = GTZANSegmentDataset(test_files, test_labels, OUTPUT_ROOT, augment=False)
    return test_dataset, test_seg_dataset


def _get_predictions_and_embeddings(model, loader, device, return_probs=True):
    """Collect predictions and penultimate-layer embeddings in one model pass."""
    model.eval()
    all_preds, all_labels, all_probs, embeddings = [], [], [], []
    linear_layers = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    target_layer = linear_layers[-2][1] if len(linear_layers) >= 2 else (linear_layers[-1][1] if linear_layers else None)

    def hook_fn(module, input, output):
        embeddings.append(input[0].detach().cpu().numpy())

    handle = target_layer.register_forward_hook(hook_fn) if target_layer is not None else None
    try:
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(device)
                labels = batch[1]
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()

                all_probs.append(probs)
                all_preds.extend(np.argmax(probs, axis=1))
                all_labels.extend(labels.numpy())
    finally:
        if handle is not None:
            handle.remove()

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.vstack(all_probs) if return_probs and all_probs else None,
        np.vstack(embeddings) if embeddings else None,
    )

def compute_full_evaluation(_cnn_model, _seg_model):
    """Run the kept evaluation sections and return tables plus plot paths."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    gn = list(GENRES.keys())

    raw_root = DATA_DIR / "genres_original"
    test_dataset, test_seg_dataset = _build_test_datasets_only(raw_root)

    # Manual batching avoids the
    # multiprocessing and pickling issues that arise on Windows when
    # PyTorch's DataLoader runs from a background thread.
    test_loader = _SimpleLoader(test_dataset, BATCH_SIZE, collate_fn)
    test_seg_loader = _SimpleLoader(test_seg_dataset, BATCH_SIZE_SEG, collate_segments)

    # ---- Predictions and embeddings (single pass per model) ----
    cnn_seg_true, cnn_seg_pred, cnn_seg_prob, cnn_emb = _get_predictions_and_embeddings(_cnn_model, test_loader, DEVICE)
    cnn_song_true, cnn_song_pred = majority_vote(cnn_seg_true, cnn_seg_pred, test_dataset)
    lstm_true, lstm_pred, lstm_prob, lstm_emb = _get_predictions_and_embeddings(_seg_model, test_seg_loader, DEVICE)

    # ---- Section 1: Model Performance Summary (real compute_metrics) ----
    rows = [
        compute_metrics(cnn_seg_true, cnn_seg_pred, cnn_seg_prob, "CNN (Segment)"),
        compute_metrics(cnn_song_true, cnn_song_pred, None, "CNN (Song, Majority Vote)"),
        compute_metrics(lstm_true, lstm_pred, lstm_prob, "CNN+LSTM (Song)"),
    ]
    perf_summary = pd.DataFrame(rows).set_index("Model")

    # ---- Section 12: Final Comparison Dashboard (subset of columns) ----
    perf_cols = [
        "Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)",
        "Balanced Accuracy", "Cohen's Kappa", "MCC", "ROC-AUC (macro OvR)"
    ]
    final_dashboard = perf_summary[[c for c in perf_cols if c in perf_summary.columns]]

    radar_cols = [c for c in ["Accuracy", "F1 (macro)", "F1 (weighted)", "Balanced Accuracy", "Cohen's Kappa", "MCC"]
                  if c in perf_summary.columns]
    fig_final_bar, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(radar_cols))
    w = 0.8 / len(perf_summary)
    for i, (model_lbl, row) in enumerate(perf_summary.iterrows()):
        ax.bar(x + i * w, row[radar_cols].values, w, label=model_lbl,
               color=PALETTE[i % len(PALETTE)], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x + w * (len(perf_summary) - 1) / 2)
    ax.set_xticklabels(radar_cols, rotation=20, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Final Model Comparison - All Key Metrics")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig_final_bar.tight_layout()
    save_fig("fig_s12_final_bar.png")  # also persisted to PLOT_DIR, matching the CLI pipeline

    # ---- Section 2: Classification Reports (sklearn, same call as suite.py) ----
    cls_reports = {
        "CNN (Segment-level)": pd.DataFrame(classification_report(
            cnn_seg_true, cnn_seg_pred, target_names=gn, output_dict=True, zero_division=0)).T,
        "CNN (Song-level, Majority Vote)": pd.DataFrame(classification_report(
            cnn_song_true, cnn_song_pred, target_names=gn, output_dict=True, zero_division=0)).T,
        "CNN+LSTM (Song-level)": pd.DataFrame(classification_report(
            lstm_true, lstm_pred, target_names=gn, output_dict=True, zero_division=0)).T,
    }

    # ---- Section 3: Confusion Matrix Analysis (real plot_confusion_matrix) ----
    cm_specs = [
        ("CNN Segment-level", cnn_seg_true, cnn_seg_pred, "fig_s3_cnn_segment_cm.png"),
        ("CNN Song-level (Majority Vote)", cnn_song_true, cnn_song_pred, "fig_s3_cnn_song_cm.png"),
        ("CNN+LSTM Song-level", lstm_true, lstm_pred, "fig_s3_lstm_cm.png"),
    ]
    cm_results = {}
    for title, y_true, y_pred, fname in cm_specs:
        plot_confusion_matrix(y_true, y_pred, gn, title, fname)
        cm_results[title] = {
            "path": PLOT_DIR / fname,
            "most_confused": _most_confused_pair(y_true, y_pred, gn),
        }

    # ---- Section 4: ROC-AUC Analysis (real plot_roc) ----
    roc_specs = [
        ("CNN (Segment)", cnn_seg_true, cnn_seg_prob, "fig_s4_roc_cnn_seg.png"),
        ("CNN+LSTM (Song)", lstm_true, lstm_prob, "fig_s4_roc_lstm.png"),
    ]
    roc_results = {}
    for title, y_true, y_prob, fname in roc_specs:
        plot_roc(y_true, y_prob, gn, title, fname)
        try:
            n_classes = len(gn)
            y_true_bin = np.eye(n_classes)[y_true]
            macro_auc = float(roc_auc_score(y_true_bin, y_prob, average="macro"))
            micro_auc = float(roc_auc_score(y_true_bin, y_prob, average="micro"))
        except Exception:
            macro_auc = micro_auc = float("nan")
        roc_results[title] = {
            "path": PLOT_DIR / fname,
            "macro_auc": macro_auc,
            "micro_auc": micro_auc,
        }

    # ---- Section 7: Genre-Wise Performance (same as suite.py Section 7, but
    # Skip the sorted horizontal-bar variant. ----
    def _genre_report_df(y_true, y_pred):
        r = classification_report(
            y_true, y_pred, labels=list(range(len(gn))),
            target_names=gn, output_dict=True, zero_division=0
        )
        return pd.DataFrame(r).T.loc[gn, ["precision", "recall", "f1-score"]]

    cnn_rep = _genre_report_df(cnn_seg_true, cnn_seg_pred)
    lstm_rep = _genre_report_df(lstm_true, lstm_pred)

    genre_df = pd.DataFrame({
        "CNN Precision": cnn_rep["precision"].values,
        "CNN Recall": cnn_rep["recall"].values,
        "CNN F1": cnn_rep["f1-score"].values,
        "LSTM Precision": lstm_rep["precision"].values,
        "LSTM Recall": lstm_rep["recall"].values,
        "LSTM F1": lstm_rep["f1-score"].values,
    }, index=gn)

    genre_improved = genre_df[genre_df["LSTM F1"] > genre_df["CNN F1"]].index.tolist()
    genre_degraded = genre_df[genre_df["LSTM F1"] < genre_df["CNN F1"]].index.tolist()

    fig_genre_f1, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(gn)); w = 0.35
    ax.bar(x - w / 2, genre_df["CNN F1"], w, label="CNN (segment)", color=PALETTE[0])
    ax.bar(x + w / 2, genre_df["LSTM F1"], w, label="CNN+LSTM (song)", color=PALETTE[1])
    ax.set_xticks(x); ax.set_xticklabels(gn, rotation=35, ha="right")
    ax.set_ylabel("F1-score")
    ax.set_title("Genre-wise F1: CNN vs CNN+LSTM")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, 1.05)
    fig_genre_f1.tight_layout()
    save_fig("fig_s7_genre_f1.png")
    # The sorted F1 plot is excluded from the dashboard.

    # ---- Section 8: Feature Embedding Analysis (PCA & t-SNE) ----
    emb_results = {}
    for model_name, emb, lbl in [
        ("CNN", cnn_emb, cnn_seg_true),
        ("CNN+LSTM", lstm_emb, lstm_true),
    ]:
        try:
            if emb is None:
                emb_results[model_name] = None
                continue
            safe_name = model_name.replace("+", "")
            fname = f"fig_s8_embeddings_{safe_name}.png"
            plot_embeddings(emb, lbl, gn, model_name, fname)
            emb_results[model_name] = PLOT_DIR / fname
        except Exception:
            emb_results[model_name] = None

    return {
        "perf_summary": perf_summary,
        "final_dashboard": final_dashboard,
        "fig_final_bar_path": PLOT_DIR / "fig_s12_final_bar.png",
        "cls_reports": cls_reports,
        "cm_results": cm_results,
        "roc_results": roc_results,
        "genre_df": genre_df,
        "genre_improved": genre_improved,
        "genre_degraded": genre_degraded,
        "genre_best_cnn": (genre_df["CNN F1"].idxmax(), float(genre_df["CNN F1"].max())),
        "genre_worst_cnn": (genre_df["CNN F1"].idxmin(), float(genre_df["CNN F1"].min())),
        "genre_best_lstm": (genre_df["LSTM F1"].idxmax(), float(genre_df["LSTM F1"].max())),
        "genre_worst_lstm": (genre_df["LSTM F1"].idxmin(), float(genre_df["LSTM F1"].min())),
        "fig_genre_f1_path": PLOT_DIR / "fig_s7_genre_f1.png",
        "emb_paths": emb_results,
        # Expose sample counts for the dashboard.
        # how many predictions were actually collected.
        "cnn_segments": len(cnn_seg_true),
        "cnn_songs": len(cnn_song_true),
        "lstm_songs": len(lstm_true),
        "seg_dataset_len": len(test_dataset),
        "seg_dataset_songs": len(set(s[2] for s in test_dataset.segments)),
    }


def _start_evaluation_if_needed(_cnn_model, _seg_model, force=False):
    if _cnn_model is None or _seg_model is None:
        return

    with eval_state["lock"]:
        if eval_state["status"] == "running":
            return

        if not force and eval_state["result"] is not None:
            eval_state["status"] = "done"
            return

        if not force:
            cached_result = _load_evaluation_cache()
            if cached_result is not None:
                eval_state["result"] = cached_result
                eval_state["status"] = "done"
                eval_state["error"] = None
                return

        cnn_mtime = os.path.getmtime(CHECKPOINT) if Path(CHECKPOINT).exists() else 0
        seg_mtime = os.path.getmtime(CHECKPOINT_SEG) if Path(CHECKPOINT_SEG).exists() else 0
        eval_state["status"] = "running"
        eval_state["result"] = None
        eval_state["error"] = None
        eval_state["cnn_mtime"] = cnn_mtime
        eval_state["seg_mtime"] = seg_mtime
        threading.Thread(
            target=_compute_full_evaluation_in_background,
            args=(_cnn_model, _seg_model, eval_state),
            daemon=True,
        ).start()

# Load saved evaluation on startup, or compute it once if no saved result exists.
_start_evaluation_if_needed(cnn_model, seg_model)

# ----------------------------------------------------
# Sidebar Setup
# ----------------------------------------------------
st.sidebar.markdown("""
<div style='text-align: center; margin-bottom: 1.5rem;'>
    <h2 style='margin: 0; font-size: 1.6rem;'>Studio Classifier</h2>
    <p style='color: #64748b; font-size: 0.85rem; margin-top: 0.2rem;'>GTZAN Music Classification Suite</p>
</div>
""", unsafe_allow_html=True)

# Device configuration info
device_str = "GPU (CUDA)" if torch.cuda.is_available() else "CPU (Fallback)"
badge_class = "cuda-badge" if torch.cuda.is_available() else "cpu-badge"
st.sidebar.markdown(f"""
<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom: 1.5rem;'>
    <span style='color: #94a3b8; font-size: 0.85rem;'>Hardware Context:</span>
    <span class='{badge_class}'>{device_str}</span>
</div>
""", unsafe_allow_html=True)

# Navigation
page = st.sidebar.radio(
    "NAVIGATION",
    ["Predict & Playground", "Model Evaluation", "Training Panel"],
    index=0
)

st.sidebar.markdown("---")

# Global training status banner
if st.session_state.get("train_running"):
    _status = st.session_state.get("train_status", "")
    _prog   = st.session_state.get("train_progress", 0)
    st.sidebar.markdown(f"""
    <div style='background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.3);
                border-radius:8px; padding:0.75rem; margin-bottom:0.75rem;'>
        <div style='font-size:0.75rem; font-weight:700; color:#10b981;
                    text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.4rem;'>
            Training Active
        </div>
        <div style='font-size:0.78rem; color:#94a3b8; margin-bottom:0.5rem;'>{_status}</div>
        <div style='background:#1e293b; border-radius:4px; height:6px; overflow:hidden;'>
            <div style='background:#10b981; width:{_prog}%; height:100%; border-radius:4px;'></div>
        </div>
        <div style='font-size:0.7rem; color:#475569; margin-top:0.3rem; text-align:right;'>{_prog}%</div>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("### CHECKPOINT ENGINE")

if cnn_model is not None:
    st.sidebar.markdown("<span style='color:#10b981; font-weight:600;'>Status:</span> CNN Backbone: **Active**", unsafe_allow_html=True)
else:
    st.sidebar.markdown("<span style='color:#ef4444; font-weight:600;'>Status:</span> CNN Backbone: **Missing**", unsafe_allow_html=True)
    
if seg_model is not None:
    st.sidebar.markdown("<span style='color:#10b981; font-weight:600;'>Status:</span> CNN+LSTM Head: **Active**", unsafe_allow_html=True)
else:
    st.sidebar.markdown("<span style='color:#f59e0b; font-weight:600;'>Status:</span> CNN+LSTM Head: **Missing**", unsafe_allow_html=True)

# ----------------------------------------------------
# Predictor Detailed Inference
# ----------------------------------------------------
@torch.inference_mode()
def predict_genre_detailed(audio_path, cnn_m, lstm_m, model_type="cnn"):
    # Load and slice audio waveform
    waveform = load_audio(audio_path)
    segments = slice_waveform(waveform)
    if not segments:
        raise ValueError("Audio file is too short to extract segments.")
        
    mels = torch.stack([normalise_spectrogram(waveform_to_melspectrogram(s)) for s in segments]).to(DEVICE)
    
    if model_type == "cnn":
        cnn_m.eval()
        logits = cnn_m(mels)
        probs = torch.softmax(logits, dim=1).cpu().numpy() # Shape: (N_segments, 10)
        avg_probs = probs.mean(axis=0) # Shape: (10,)
    else: # lstm
        lstm_m.eval()
        lstm_input = mels.unsqueeze(0) # Shape: (1, N_segments, 1, 128, 128)
        logits = lstm_m(lstm_input)
        avg_probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy() # Shape: (10,)
        
        # Also compute segment-level outputs using CNN backbone for detailed breakdown
        cnn_m.eval()
        cnn_logits = cnn_m(mels)
        probs = torch.softmax(cnn_logits, dim=1).cpu().numpy()
        
    top_indices = np.argsort(avg_probs)[::-1]
    top_k = [(IDX_TO_GENRE[idx], float(avg_probs[idx])) for idx in top_indices[:3]]
    
    return {
        "predicted_genre": IDX_TO_GENRE[top_indices[0]],
        "confidence": float(avg_probs[top_indices[0]]),
        "top_k": top_k,
        "probs": probs,
        "avg_probs": avg_probs,
        "n_segments": len(segments)
    }

# ----------------------------------------------------
# Page 1: Predict & Playground
# ----------------------------------------------------
if page == "Predict & Playground":
    st.markdown("<h1>Predict & Feature Playground</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:1.05rem;'>Upload music files or inspect random test samples to explore the model focus.</p>", unsafe_allow_html=True)
    
    if cnn_model is None:
        st.warning("Baseline CNN checkpoint (`models/best_cnn.pt`) is missing. Please go to the **Training Panel** first to train the network.")
    else:
        # Columns for input configuration
        col_setup, col_meta = st.columns([2, 1])
        
        with col_setup:
            audio_source = st.radio("Select Audio Source", ["Upload Custom File", "Choose GTZAN Test Sample"])
            
            uploaded_file = None
            sample_path = None
            
            if audio_source == "Upload Custom File":
                uploaded_file = st.file_uploader("Upload Audio File (.wav or .mp3)", type=["wav", "mp3"])
                if uploaded_file is not None:
                    # Save file locally in a temporary directory
                    temp_dir = Path("cache/temp_uploads")
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    uploaded_path = temp_dir / uploaded_file.name
                    with open(uploaded_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    sample_path = str(uploaded_path)
            else:
                raw_root = Path(DATA_DIR) / "genres_original"
                if not raw_root.exists():
                    st.info("GTZAN raw dataset not found in `data/genres_original`. Run training first to download it automatically, or place it there.")
                else:
                    # Get genres list
                    available_genres = sorted([d.name for d in raw_root.iterdir() if d.is_dir()])
                    selected_genre = st.selectbox("Genre Category", available_genres)
                    
                    if selected_genre:
                        genre_dir = raw_root / selected_genre
                        wav_files = sorted(list(genre_dir.glob("*.wav")))
                        wav_names = [f.name for f in wav_files]
                        
                        selected_file_name = st.selectbox("Select Test File", wav_names)
                        if selected_file_name:
                            sample_path = str(genre_dir / selected_file_name)
                            
        with col_meta:
            with st.container(border=True):
                st.markdown("<h3 style='margin-top:0; margin-bottom:1rem; color:#f8fafc;'>Inference Settings</h3>", unsafe_allow_html=True)
                model_selection = st.selectbox(
                    "Classification Architecture",
                    ["CNN Baseline (Segment Voting)", "CNN+LSTM Attention (Sequence Model)"]
                )
                
                model_type = "cnn" if "CNN Baseline" in model_selection else "lstm"
                
                if model_type == "lstm" and seg_model is None:
                    st.info("CNN+LSTM checkpoint missing. Using CNN Baseline fallback.")
                    model_type = "cnn" 
            
        # Trigger Inference
        if sample_path is not None:
            # Display Audio player
            st.markdown("### Audio Clip")
            st.audio(sample_path)
            
            if st.button("Run Inference", type="primary"):
                with st.spinner("Processing audio features and computing activations..."):
                    try:
                        res = predict_genre_detailed(sample_path, cnn_model, seg_model, model_type)
                        
                        # Layout results
                        col_results, col_plots = st.columns([1, 1])
                        
                        with col_results:
                            pred_genre = res["predicted_genre"]
                            predicted_label = GENRE_EMOJI.get(pred_genre, pred_genre.title())

                            # Build each confidence bar as a flat HTML fragment
                            # without internal indentation. Streamlit passes this
                            # through Markdown before rendering the HTML.
                            bar_chunks = []
                            for genre, confidence in res["top_k"]:
                                option_label = GENRE_EMOJI.get(genre, genre.title())
                                bar_chunks.append(
                                    f"<div style='margin-bottom:1rem;'>"
                                    f"<div style='display:flex;justify-content:space-between;margin-bottom:0.25rem;font-size:0.9rem;'>"
                                    f"<span><strong>{option_label}</strong></span>"
                                    f"<span>{confidence*100:.1f}%</span>"
                                    f"</div>"
                                    f"<div style='background-color:#334155;height:8px;border-radius:4px;overflow:hidden;'>"
                                    f"<div style='background-color:#10b981;width:{confidence*100}%;height:100%;border-radius:4px;'></div>"
                                    f"</div>"
                                    f"</div>"
                                )
                            bars_html = "".join(bar_chunks)

                            model_label = "CNN Baseline" if model_type == "cnn" else "CNN+LSTM Attention"

                            # Assemble the card as one string for the same reason.
                            card_html = (
                                f"<div class='studio-card'>"
                                f"<div class='metric-label'>Predicted Genre</div>"
                                f"<div class='metric-value'>{predicted_label}</div>"
                                f"<p style='color:#94a3b8;font-size:0.9rem;margin-top:0.3rem;'>"
                                f"Model: <strong>{model_label}</strong> | "
                                f"Segments: <strong>{res['n_segments']}</strong>"
                                f"</p>"
                                f"<h3 style='margin-top:1.5rem;margin-bottom:1rem;color:#f8fafc;font-size:1.1rem;'>TOP CLASSIFICATION CONFIDENCES</h3>"
                                f"{bars_html}"
                                f"</div>"
                            )
                            st.markdown(card_html, unsafe_allow_html=True)
                                
                        with col_plots:
                            # 1. Mel Spectrogram Plot
                            y, sr = librosa.load(sample_path, sr=SAMPLE_RATE)
                            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
                            S_dB = librosa.power_to_db(S, ref=np.max)
                            
                            fig, ax = plt.subplots(figsize=(10, 4.2), facecolor='#0f172a')
                            ax.set_facecolor('#0f172a')
                            img = librosa.display.specshow(S_dB, sr=sr, hop_length=512, x_axis='time', y_axis='mel', ax=ax, cmap='viridis')
                            ax.tick_params(colors='#f8fafc', labelsize=8)
                            ax.xaxis.label.set_color('#94a3b8')
                            ax.yaxis.label.set_color('#94a3b8')
                            ax.set_title("Audio Mel Spectrogram", color='#f8fafc', fontsize=11, fontweight='semibold')
                            for spine in ['top', 'right']:
                                ax.spines[spine].set_visible(False)
                            ax.spines['left'].set_color('#334155')
                            ax.spines['bottom'].set_color('#334155')
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # 2. Segment-level probability heatmap
                            probs = res["probs"]  # (N_segments, 10)
                            
                            fig_heat, ax_heat = plt.subplots(figsize=(10, 4.5), facecolor='#0f172a')
                            ax_heat.set_facecolor('#0f172a')
                            
                            # Transpose to 10 x N_segments
                            genre_labels = list(GENRES.keys())
                            im = ax_heat.imshow(probs.T, aspect='auto', cmap='plasma', origin='lower', extent=[0, res['n_segments'] * 1.5 + 1.5, 0, 10])
                            
                            ax_heat.set_yticks(np.arange(10) + 0.5)
                            ax_heat.set_yticklabels([g.upper() for g in genre_labels], color='#f8fafc', fontsize=8)
                            ax_heat.tick_params(colors='#f8fafc', labelsize=8)
                            ax_heat.set_xlabel("Time (seconds)", color='#94a3b8', fontsize=9)
                            ax_heat.set_ylabel("Genre Probabilities", color='#94a3b8', fontsize=9)
                            ax_heat.set_title("Temporal Model Focus Heatmap", color='#f8fafc', fontsize=11, fontweight='semibold')
                            
                            ax_heat.spines['left'].set_color('#334155')
                            ax_heat.spines['bottom'].set_color('#334155')
                            for spine in ['top', 'right']:
                                ax_heat.spines[spine].set_visible(False)
                                
                            cbar = fig_heat.colorbar(im, ax=ax_heat)
                            cbar.ax.yaxis.set_tick_params(color='#f8fafc', labelcolor='#f8fafc', labelsize=8)
                            cbar.outline.set_edgecolor('#334155')
                            
                            plt.tight_layout()
                            st.pyplot(fig_heat)
                            
                    except Exception as e:
                        st.error(f"Inference error: {e}")

# ----------------------------------------------------
# Page 2: Model Evaluation Dashboard
# ----------------------------------------------------
elif page == "Model Evaluation":
    st.markdown("<h1>Model Evaluation Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:1.05rem;'>Live model evaluation results, classification diagnostics, and embedding visualizations.</p>", unsafe_allow_html=True)
    
    if cnn_model is None:
        st.warning("No trained models detected. Please run the training pipeline first.")
    else:
        # Summary metrics cards
        st.markdown("### KEY ARCHITECTURAL METRICS")
        c1, c2 = st.columns(2)

        # Calculate params and sizes
        total_p_cnn, train_p_cnn = count_params(cnn_model)
        size_cnn = model_size_mb(cnn_model)

        with c1:
            st.markdown(f"""
                <div class='studio-card'>
                    <div style='display:flex; align-items:center; gap:0.6rem; margin-bottom:1rem;'>
                        <div style='width:10px; height:10px; border-radius:50%; background:#10b981;'></div>
                        <span style='font-size:0.75rem; font-weight:700; letter-spacing:0.1em; color:#94a3b8; text-transform:uppercase;'>CNN Backbone</span>
                    </div>
                    <div style='display:flex; gap:2rem;'>
                        <div>
                            <div class='metric-label'>Model Size</div>
                            <div class='metric-value'>{size_cnn} <span style='font-size:1.1rem;'>MB</span></div>
                        </div>
                        <div style='width:1px; background:#334155;'></div>
                        <div>
                            <div class='metric-label'>Parameters</div>
                            <div class='metric-value'>{total_p_cnn:,}</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        if seg_model is not None:
            total_p_lstm, train_p_lstm = count_params(seg_model)
            size_lstm = model_size_mb(seg_model)
            with c2:
                st.markdown(f"""
                    <div class='studio-card'>
                        <div style='display:flex; align-items:center; gap:0.6rem; margin-bottom:1rem;'>
                            <div style='width:10px; height:10px; border-radius:50%; background:#3b82f6;'></div>
                            <span style='font-size:0.75rem; font-weight:700; letter-spacing:0.1em; color:#94a3b8; text-transform:uppercase;'>CNN+LSTM Head</span>
                        </div>
                        <div style='display:flex; gap:2rem;'>
                            <div>
                                <div class='metric-label'>Model Size</div>
                                <div class='metric-value'>{size_lstm} <span style='font-size:1.1rem;'>MB</span></div>
                            </div>
                            <div style='width:1px; background:#334155;'></div>
                            <div>
                                <div class='metric-label'>Parameters</div>
                                <div class='metric-value'>{total_p_lstm:,}</div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            with c2:
                st.markdown(f"""
                    <div class='studio-card'>
                        <div style='display:flex; align-items:center; gap:0.6rem; margin-bottom:1rem;'>
                            <div style='width:10px; height:10px; border-radius:50%; background:#475569;'></div>
                            <span style='font-size:0.75rem; font-weight:700; letter-spacing:0.1em; color:#475569; text-transform:uppercase;'>CNN+LSTM Head</span>
                        </div>
                        <div style='display:flex; gap:2rem;'>
                            <div>
                                <div class='metric-label'>Model Size</div>
                                <div class='metric-value' style='color:#475569;'>N/A</div>
                            </div>
                            <div style='width:1px; background:#334155;'></div>
                            <div>
                                <div class='metric-label'>Parameters</div>
                                <div class='metric-value' style='color:#475569;'>N/A</div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
        st.markdown("---")

        if seg_model is None:
            st.warning("CNN+LSTM checkpoint (`models/best_frozen_cnn_lstm.pt`) is missing. Train the sequence model before running the full comparison dashboard.")
            st.stop()

        _start_evaluation_if_needed(cnn_model, seg_model)

        # Re-evaluate button
        col_refresh, col_status = st.columns([1, 3])
        with col_refresh:
            if st.button("Re-evaluate", type="secondary", use_container_width=True):
                with eval_state["lock"]:
                    eval_state["status"] = "idle"
                    eval_state["result"] = None
                    eval_state["error"] = None
                    eval_state["cnn_mtime"] = None
                    eval_state["seg_mtime"] = None
                    if EVAL_CACHE_PATH.exists():
                        EVAL_CACHE_PATH.unlink()
                    load_saved_models_cached.clear()
                _start_evaluation_if_needed(cnn_model, seg_model, force=True)
                st.rerun()
        with col_status:
            status_badge = {
                "running": "Evaluation running...",
                "idle": "Evaluation idle",
                "done": "Evaluation complete",
                "error": "Evaluation failed",
            }.get(eval_state["status"], eval_state["status"])
            st.markdown(f"<span style='color:#94a3b8;font-size:0.85rem;'>{status_badge}</span>", unsafe_allow_html=True)

        # Debug info card
        if eval_state["status"] == "done" and eval_state["result"]:
            _r = eval_state["result"]
            st.markdown(
                f"<div style='display:flex;gap:2rem;padding:0.3rem 1rem;margin-bottom:0.5rem;"
                f"background:#1e293b;border-radius:6px;font-size:0.78rem;color:#94a3b8;'>"
                f"<span>CNN segments: <strong style='color:#f8fafc'>{_r['cnn_segments']}</strong></span>"
                f"<span>CNN songs (majority vote): <strong style='color:#f8fafc'>{_r['cnn_songs']}</strong></span>"
                f"<span>LSTM songs: <strong style='color:#f8fafc'>{_r['lstm_songs']}</strong></span>"
                f"<span>Unique songs in seg dataset: <strong style='color:#f8fafc'>{_r['seg_dataset_songs']}</strong></span>"
                f"</div>",
                unsafe_allow_html=True
            )

        if eval_state["status"] in ("running", "idle"):
            with st.spinner("Running the full evaluation suite in the background against the test split... this page will update automatically."):
                time.sleep(2)
            st.rerun()

        elif eval_state["status"] == "error":
            st.error(f"Error executing evaluation suite: {eval_state['error']}")

        else:
            res = eval_state["result"]

            # ---- Section 1: Model Performance Summary ----
            st.markdown("### MODEL PERFORMANCE SUMMARY")
            st.dataframe(res["perf_summary"].style.format("{:.4f}", na_rep="-"), use_container_width=True)

            # ---- Section 2: Classification Reports ----
            st.markdown("---")
            st.markdown("### CLASSIFICATION REPORTS")
            cls_tabs = st.tabs(list(res["cls_reports"].keys()))
            for tab, (name, df) in zip(cls_tabs, res["cls_reports"].items()):
                with tab:
                    st.dataframe(df.style.format("{:.4f}", na_rep="-"), use_container_width=True)

            # ---- Section 3: Confusion Matrix Analysis ----
            st.markdown("---")
            st.markdown("### CONFUSION MATRIX ANALYSIS")
            cm_tabs = st.tabs(list(res["cm_results"].keys()))
            for tab, (name, info) in zip(cm_tabs, res["cm_results"].items()):
                with tab:
                    if info["path"].exists():
                        st.image(str(info["path"]), use_container_width=True)
                    if info["most_confused"] is not None:
                        g_true, g_pred, count = info["most_confused"]
                        st.markdown(f"**Most confused pair:** {g_true} -> {g_pred} ({count} misclassifications)")

            # ---- Section 4: ROC-AUC Analysis ----
            st.markdown("---")
            st.markdown("### ROC-AUC ANALYSIS")
            roc_tabs = st.tabs(list(res["roc_results"].keys()))
            for tab, (name, info) in zip(roc_tabs, res["roc_results"].items()):
                with tab:
                    if info["path"].exists():
                        st.image(str(info["path"]), use_container_width=True)
                    st.markdown(f"**Macro AUC = {info['macro_auc']:.4f}**  |  **Micro AUC = {info['micro_auc']:.4f}**")

            # ---- Section 7: Genre-Wise Performance Analysis ----
            st.markdown("---")
            st.markdown("### GENRE-WISE PERFORMANCE ANALYSIS")
            st.dataframe(res["genre_df"].style.format("{:.4f}"), use_container_width=True)
            if res["fig_genre_f1_path"].exists():
                st.image(str(res["fig_genre_f1_path"]), use_container_width=True)

            gi1, gi2 = st.columns(2)
            with gi1:
                st.markdown(f"**Genres improved by LSTM:** {', '.join(res['genre_improved']) or 'None'}")
                bc_name, bc_val = res["genre_best_cnn"]
                wc_name, wc_val = res["genre_worst_cnn"]
                st.markdown(f"**Best CNN F1:** {bc_name} ({bc_val:.4f})")
                st.markdown(f"**Worst CNN F1:** {wc_name} ({wc_val:.4f})")
            with gi2:
                st.markdown(f"**Genres degraded by LSTM:** {', '.join(res['genre_degraded']) or 'None'}")
                bl_name, bl_val = res["genre_best_lstm"]
                wl_name, wl_val = res["genre_worst_lstm"]
                st.markdown(f"**Best LSTM F1:** {bl_name} ({bl_val:.4f})")
                st.markdown(f"**Worst LSTM F1:** {wl_name} ({wl_val:.4f})")

            # ---- Section 8: Feature Embedding Analysis (PCA & t-SNE) ----
            st.markdown("---")
            st.markdown("### FEATURE EMBEDDING ANALYSIS (PCA & t-SNE)")
            ec1, ec2 = st.columns(2)
            for col, model_name in [(ec1, "CNN"), (ec2, "CNN+LSTM")]:
                with col:
                    emb_path = res["emb_paths"].get(model_name)
                    if emb_path is not None and emb_path.exists():
                        st.image(str(emb_path), use_container_width=True)
                    else:
                        st.info(f"Embeddings could not be extracted from the {model_name} model.")

            # ---- Section 12: Final Comparison Dashboard ----
            st.markdown("---")
            st.markdown("### FINAL COMPARISON DASHBOARD")
            st.dataframe(res["final_dashboard"].style.format("{:.4f}", na_rep="-"), use_container_width=True)
            if res["fig_final_bar_path"].exists():
                st.image(str(res["fig_final_bar_path"]), use_container_width=True)



# ----------------------------------------------------
# Page 3: Training Control Panel
# ----------------------------------------------------
elif page == "Training Panel":
    st.markdown("<h1>Model Training Panel</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:1.05rem;'>Configure hyperparameters, manage neural network layers, and trigger training with live feedback.</p>", unsafe_allow_html=True)
    
    st.markdown("### PIPELINE HYPERPARAMETERS")
    
    col_hyper, col_actions = st.columns([2, 1])
    
    with col_hyper:
        lr = st.slider("Learning Rate (Optimiser step size)", min_value=1e-5, max_value=1e-3, value=5e-4, step=5e-5, format="%e")
        dropout = st.slider("Dropout Rate (CNN dense layer regularisation)", min_value=0.0, max_value=0.8, value=0.6, step=0.05)
        lstm_hidden = st.selectbox("LSTM Hidden Units", [16, 32, 48, 64, 128], index=2)
        epochs = st.slider("Maximum Training Epochs", min_value=2, max_value=100, value=20, step=1)
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=2)
        
    with col_actions:
        with st.container(border=True):
            st.markdown("""
            <h3 style='margin-top:0; margin-bottom:0.5rem; color:#f8fafc;'>Training Run</h3>
            <p style='color:#94a3b8; font-size:0.9rem; margin-bottom:1rem;'>Start training from scratch. Live charts will display training progression.</p>
            """, unsafe_allow_html=True)
            
            train_baseline = st.checkbox("Train Baseline CNN Backbone", value=True)
            train_lstm = st.checkbox("Train Frozen CNN+LSTM Head", value=True)
            
            start_trigger = st.button("Start Training Pipeline", type="primary", use_container_width=True)
        
    # Background training worker
    def _training_worker(cfg):
        """Runs in a daemon thread; writes progress to st.session_state only."""
        ss = st.session_state
        try:
            raw_root = DATA_DIR / "genres_original"
            ss["train_status"] = "Preparing data splits..."
            (train_files, val_files, test_files, train_labels, val_labels, test_labels,
             train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader,
             train_song_ds, val_song_ds, test_song_ds,
             train_song_dl, val_song_dl, test_song_dl) = build_data(raw_root)

            # Phase 1: CNN
            if cfg["train_baseline"]:
                _cnn = MusicGenreCNN(dropout=cfg["dropout"]).to(DEVICE)
                criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
                optimizer = torch.optim.AdamW(_cnn.parameters(), lr=cfg["lr"], weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
                cnn_history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
                best_val_loss = float("inf")
                patience_counter = 0

                for epoch in range(1, cfg["epochs"] + 1):
                    ss["train_status"] = f"CNN - Epoch {epoch}/{cfg['epochs']} | training..."
                    train_loss, train_acc = train_one_epoch(_cnn, train_loader, criterion, optimizer, DEVICE)
                    ss["train_status"] = f"CNN - Epoch {epoch}/{cfg['epochs']} | validating..."
                    val_loss, val_acc = evaluate(_cnn, val_loader, criterion, DEVICE)
                    scheduler.step()

                    cnn_history["train_loss"].append(train_loss)
                    cnn_history["val_loss"].append(val_loss)
                    cnn_history["train_acc"].append(train_acc)
                    cnn_history["val_acc"].append(val_acc)
                    ss["train_cnn_history"] = cnn_history
                    ss["train_progress"] = int((epoch / cfg["epochs"]) * 50)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        torch.save(_cnn.state_dict(), CHECKPOINT)
                    else:
                        patience_counter += 1
                        if patience_counter >= PATIENCE:
                            ss["train_status"] = f"Early stopping CNN at epoch {epoch}"
                            break

                np.save(str(HISTORY_PATH), cnn_history)

            # Phase 2: LSTM
            if cfg["train_lstm"]:
                if not Path(CHECKPOINT).exists():
                    ss["train_status"] = "No CNN checkpoint found - LSTM skipped."
                    ss["train_done"] = True
                    return
                backbone = MusicGenreCNN(dropout=0.0).to(DEVICE)
                backbone.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True))
                for p in backbone.parameters():
                    p.requires_grad = False
                _seg = FrozenCNNLSTM(backbone, num_classes=10,
                                     lstm_hidden=cfg["lstm_hidden"], dropout=cfg["dropout"]).to(DEVICE)
                criterion_seg = nn.CrossEntropyLoss(label_smoothing=0.1)
                optimizer_seg = torch.optim.AdamW(_seg.parameters(), lr=cfg["lr"], weight_decay=1e-4)
                scheduler_seg = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_seg, T_max=cfg["epochs"])
                lstm_history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
                best_val_loss_seg = float("inf")
                patience_counter_seg = 0

                for epoch in range(1, cfg["epochs"] + 1):
                    ss["train_status"] = f"LSTM - Epoch {epoch}/{cfg['epochs']} | training..."
                    train_loss, train_acc = train_one_epoch_seg(_seg, train_song_dl, criterion_seg, optimizer_seg, DEVICE)
                    ss["train_status"] = f"LSTM - Epoch {epoch}/{cfg['epochs']} | validating..."
                    val_loss, val_acc = evaluate_seg(_seg, val_song_dl, criterion_seg, DEVICE)
                    scheduler_seg.step()

                    lstm_history["train_loss"].append(train_loss)
                    lstm_history["val_loss"].append(val_loss)
                    lstm_history["train_acc"].append(train_acc)
                    lstm_history["val_acc"].append(val_acc)
                    ss["train_lstm_history"] = lstm_history
                    ss["train_progress"] = 50 + int((epoch / cfg["epochs"]) * 50)

                    if val_loss < best_val_loss_seg:
                        best_val_loss_seg = val_loss
                        patience_counter_seg = 0
                        torch.save(_seg.state_dict(), CHECKPOINT_SEG)
                    else:
                        patience_counter_seg += 1
                        if patience_counter_seg >= PATIENCE:
                            ss["train_status"] = f"Early stopping LSTM at epoch {epoch}"
                            break

                np.save(str(HISTORY_SEG_PATH), lstm_history)

            ss["train_status"] = "Training complete. Checkpoints saved."
            ss["train_done"] = True
            load_saved_models_cached.clear()

        except Exception as exc:
            ss["train_status"] = f"Error: {exc}"
            ss["train_done"] = True

    # Session-state initialisation
    for _k, _v in [
        ("train_running", False), ("train_done", False),
        ("train_progress", 0), ("train_status", ""),
        ("train_cnn_history", None), ("train_lstm_history", None),
    ]:
        if _k not in st.session_state:
            st.session_state[_k] = _v

    # Start button
    if start_trigger:
        raw_root = DATA_DIR / "genres_original"
        if not raw_root.exists():
            st.error(f"GTZAN dataset not found at {raw_root}. Please download or place it there.")
        elif not st.session_state["train_running"]:
            # Reset state for a fresh run
            st.session_state.update({
                "train_running": True, "train_done": False,
                "train_progress": 0, "train_status": "Starting...",
                "train_cnn_history": None, "train_lstm_history": None,
            })
            cfg = dict(lr=lr, dropout=dropout, lstm_hidden=lstm_hidden,
                       epochs=epochs, train_baseline=train_baseline, train_lstm=train_lstm)
            t = threading.Thread(target=_training_worker, args=(cfg,), daemon=True)
            t.start()
            st.rerun()

    # Live status display
    if st.session_state["train_running"]:
        st.markdown("---")
        st.markdown("### TRAINING PROGRESS")

        # Mark finished when worker sets the flag
        if st.session_state["train_done"]:
            st.session_state["train_running"] = False
            load_saved_models_cached.clear()

        status_msg = st.session_state["train_status"]
        if "Error:" in status_msg:
            st.error(status_msg)
        elif "Training complete" in status_msg:
            st.success(status_msg)
        else:
            st.info(status_msg)

        st.progress(st.session_state["train_progress"])

        # Loss curves
        cnn_h = st.session_state["train_cnn_history"]
        lstm_h = st.session_state["train_lstm_history"]
        if cnn_h and cnn_h["train_loss"]:
            st.markdown("#### CNN Loss Curves")
            st.line_chart(pd.DataFrame({
                "Train Loss": cnn_h["train_loss"],
                "Val Loss":   cnn_h["val_loss"],
            }))
        if lstm_h and lstm_h["train_loss"]:
            st.markdown("#### CNN+LSTM Loss Curves")
            st.line_chart(pd.DataFrame({
                "Train Loss": lstm_h["train_loss"],
                "Val Loss":   lstm_h["val_loss"],
            }))

        # Auto-refresh every 2 s while training is active
        if not st.session_state["train_done"]:
            time.sleep(2)
            st.rerun()
