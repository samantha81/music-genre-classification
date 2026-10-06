import torch
import torch.nn as nn
import torchaudio.transforms as T
import soundfile as sf
from pathlib import Path
from src.config import (
    SAMPLE_RATE, DURATION, N_MELS, N_FFT, HOP_LENGTH,
    TARGET_HEIGHT, TARGET_WIDTH, SEG_SAMPLES, SEG_STEP, GENRES
)

# Augmentation processing configuration
specaugment = nn.Sequential(
    T.FrequencyMasking(freq_mask_param=15),
    T.TimeMasking(time_mask_param=35),
)

def apply_augmentation(mel: torch.Tensor) -> torch.Tensor:
    return specaugment(mel)

def load_audio(filepath):
    data, sr = sf.read(str(filepath))
    waveform = torch.from_numpy(data).float()

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.t()

    if sr != SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    target_len = SAMPLE_RATE * DURATION
    current_len = waveform.shape[1]
    if current_len >= target_len:
        start = (current_len - target_len) // 2
        waveform = waveform[:, start:start + target_len]
    else:
        waveform = torch.nn.functional.pad(waveform, (0, target_len - current_len))

    return waveform

def waveform_to_melspectrogram(waveform):
    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, f_min=20, f_max=8000
    )
    db_transform = T.AmplitudeToDB(stype='power', top_db=80)
    mel = mel_transform(waveform)
    mel = db_transform(mel)
    mel = torch.nn.functional.interpolate(
        mel.unsqueeze(0), size=(TARGET_HEIGHT, TARGET_WIDTH),
        mode='bilinear', align_corners=False
    ).squeeze(0)
    return mel

def normalise_spectrogram(mel):
    mean = mel.mean()
    std = mel.std() + 1e-8
    return (mel - mean) / std

def build_file_index(root: Path):
    filepaths, labels = [], []
    CORRUPTED = {"jazz.00054.wav"}
    for genre, label in GENRES.items():
        genre_dir = root / genre
        if not genre_dir.exists():
            print(f"Warning: {genre_dir} not found, skipping.")
            continue
        for audio_file in sorted(genre_dir.iterdir()):
            if audio_file.suffix in ('.wav', '.au', '.mp3') and audio_file.name not in CORRUPTED:
                filepaths.append(audio_file)
                labels.append(label)
    print(f"Total files found: {len(filepaths)}")
    return filepaths, labels

def get_seg_cache_path(audio_path: Path, seg_idx: int, cache_root: Path) -> Path:
    return cache_root / audio_path.parent.name / f"{audio_path.stem}_seg{seg_idx:02d}.pt"

def slice_waveform(waveform: torch.Tensor) -> list:
    segments = []
    total = waveform.shape[1]
    start = 0
    while start + SEG_SAMPLES <= total:
        segments.append(waveform[:, start : start + SEG_SAMPLES])
        start += SEG_STEP
    return segments

def load_or_compute_segments(audio_path: Path, cache_root: Path):
    seg0 = get_seg_cache_path(audio_path, 0, cache_root)
    seg0.parent.mkdir(parents=True, exist_ok=True)
    existing = sorted(seg0.parent.glob(f"{audio_path.stem}_seg*.pt"))
    if existing:
        return [torch.load(p, weights_only=True) for p in existing]
    try:
        waveform = load_audio(audio_path)
        segments = slice_waveform(waveform)
        mel_segs = []
        for idx, seg_wav in enumerate(segments):
            mel = waveform_to_melspectrogram(seg_wav)
            mel = normalise_spectrogram(mel)
            cp = get_seg_cache_path(audio_path, idx, cache_root)
            torch.save(mel, cp)
            mel_segs.append(mel)
        return mel_segs
    except Exception as e:
        print(f"Skipping {audio_path.name}: {e}")
        return []
