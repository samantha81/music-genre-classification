import os
import sys
import time
import warnings
from pathlib import Path
import matplotlib
import torch
from src.config import BASE_DIR, PLOT_DIR, DEVICE

warnings.filterwarnings('ignore')

def _setup_matplotlib():
    matplotlib.use("Agg")
    import matplotlib.pyplot as _p
    return "Agg", _p

_MPLBACKEND, plt = _setup_matplotlib()
_INTERACTIVE_DISPLAY = False

matplotlib.rcParams.update({
    'font.size': 13, 'axes.titlesize': 14, 'axes.labelsize': 13,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'legend.fontsize': 11, 'figure.dpi': 150,
    'savefig.bbox': 'tight', 'savefig.dpi': 150,
})

try:
    import seaborn as sns
    PALETTE = sns.color_palette("deep")
    HAS_SNS = True
except ImportError:
    PALETTE = list(plt.cm.tab10.colors[:10])
    HAS_SNS = False

try:
    import pandas as pd
    HAS_PD = True
except ImportError:
    HAS_PD = False

try:
    import librosa
    import librosa.display as lrd
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    from statsmodels.stats.contingency_tables import mcnemar
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    import kagglehub
    HAS_KAGGLE = True
except ImportError:
    HAS_KAGGLE = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    RICH = True
    console = Console()
except ImportError:
    RICH = False
    console = None

def pprint(msg, style=""):
    if RICH:
        console.print(msg, style=style)
    else:
        import re
        plain = re.sub(r"\[/?[a-zA-Z_ ]*\]", "", msg)
        print(plain)

def print_header():
    if RICH:
        console.print(Panel.fit(
            "[bold cyan]Music Genre Classifier[/bold cyan]\n"
            "[dim]CNN + Frozen CNN+LSTM | GTZAN Dataset | 10 Genres[/dim]",
            border_style="cyan"
        ))
    else:
        print("=" * 60)
        print("   MUSIC GENRE CLASSIFIER")
        print("   CNN + Frozen CNN+LSTM | GTZAN | 10 Genres")
        print("=" * 60)

def save_fig(fname, fig=None):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / fname
    fig = fig or plt.gcf()
    fig.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Saved] {path}")

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def model_size_mb(model):
    import tempfile
    tmp = tempfile.mktemp(suffix=".pt")
    torch.save(model.state_dict(), tmp)
    size = os.path.getsize(tmp) / 1e6
    os.remove(tmp)
    return round(size, 2)

def measure_inference(model, loader, device, n_batches=50):
    model.eval()
    times, n_samples = [], 0
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            inputs = batch[0].to(device)
            
            t0 = time.perf_counter()
            _ = model(inputs)
            t1 = time.perf_counter()
            
            times.append(t1 - t0)
            n_samples += inputs.size(0)
            
    import numpy as np
    mean_latency_ms = np.mean(times) * 1000
    throughput_fps = n_samples / np.sum(times)
    
    return round(mean_latency_ms, 2), round(throughput_fps, 1)

