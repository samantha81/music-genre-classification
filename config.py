import torch
from pathlib import Path

# Audio & spectrogram configurations
SAMPLE_RATE = 22050
DURATION = 30
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
TARGET_HEIGHT = 128
TARGET_WIDTH = 128

# Segment sliding controls (3s windows with 50% overlap)
SEG_DURATION = 3
SEG_OVERLAP = 0.5
SEG_SAMPLES = int(SAMPLE_RATE * SEG_DURATION)
SEG_STEP = int(SEG_SAMPLES * (1.0 - SEG_OVERLAP))

# Training parameters
BATCH_SIZE = 64
BATCH_SIZE_SEG = 16
NUM_WORKERS = 2
NUM_EPOCHS = 50
PATIENCE = 12
TUNE_EPOCHS = 20
N_TRIALS = 12
TUNE_EPOCHS_SEG = 20
N_TRIALS_SEG = 12

# File engine paths (relative to application root)
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_ROOT = BASE_DIR / "cache" / "spectrograms"
PLOT_DIR = BASE_DIR / "plots"
CHECKPOINT = str(BASE_DIR / "models" / "best_cnn.pt")
CHECKPOINT_SEG = str(BASE_DIR / "models" / "best_frozen_cnn_lstm.pt")
HISTORY_PATH = BASE_DIR / "train_history.npy"
HISTORY_SEG_PATH = BASE_DIR / "train_history_seg.npy"

# Dataset mappings
GENRES = {
    'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4,
    'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9,
}
IDX_TO_GENRE = {v: k for k, v in GENRES.items()}
GENRE_EMOJI = {
    "blues": "Blues", "classical": "Classical", "country": "Country", "disco": "Disco",
    "hiphop": "Hip-hop", "jazz": "Jazz", "metal": "Metal", "pop": "Pop",
    "reggae": "Reggae", "rock": "Rock",
}

# Mapping relationships for visual validation checks
MUSICAL_SIMILARITIES = {
    frozenset(['rock', 'metal']): "Rock and Metal share distorted guitar timbres and high energy.",
    frozenset(['blues', 'jazz']): "Blues and Jazz share improvisational phrasing and similar chord progressions.",
    frozenset(['disco', 'pop']): "Disco and Pop share danceable rhythms and polished production.",
    frozenset(['country', 'rock']): "Country and Rock share guitar-driven arrangements and similar tempi.",
    frozenset(['classical', 'jazz']): "Classical and Jazz share complex harmonic language.",
    frozenset(['hiphop', 'reggae']): "Hip-Hop and Reggae share rhythmic groove and off-beat emphasis.",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

