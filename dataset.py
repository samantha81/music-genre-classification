import torch
from torch.utils.data import Dataset
from src.audio.processing import load_or_compute_segments, apply_augmentation

class GTZANDataset(Dataset):
    def __init__(self, filepaths, labels, cache_root, augment=False):
        self.cache_root = cache_root
        self.augment = augment
        self.segments = []

        for song_id, (fp, lbl) in enumerate(zip(filepaths, labels)):
            seg_paths = sorted(
                (cache_root / fp.parent.name).glob(f"{fp.stem}_seg*.pt")
            )
            if not seg_paths:
                load_or_compute_segments(fp, cache_root)
                seg_paths = sorted(
                    (cache_root / fp.parent.name).glob(f"{fp.stem}_seg*.pt")
                )
            for sp in seg_paths:
                self.segments.append((sp, lbl, song_id))

        print(f"Dataset ready: {len(self.segments)} segments from "
              f"{len(filepaths)} songs "
              f"({'train+aug' if augment else 'eval'})")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        seg_path, label, song_id = self.segments[idx]
        mel = torch.load(seg_path, weights_only=True)
        if self.augment:
            mel = apply_augmentation(mel)
        return mel, torch.tensor(label, dtype=torch.long), song_id


class GTZANSegmentDataset(Dataset):
    def __init__(self, filepaths, labels, cache_root, augment=False):
        self.cache_root = cache_root
        self.augment = augment
        self.songs = []

        for song_id, (fp, lbl) in enumerate(zip(filepaths, labels)):
            seg_paths = sorted(
                (cache_root / fp.parent.name).glob(f"{fp.stem}_seg*.pt")
            )
            if not seg_paths:
                continue
            segments = []
            for sp in seg_paths:
                mel = torch.load(sp, weights_only=True)
                if self.augment:
                    mel = apply_augmentation(mel)
                segments.append(mel)
            if segments:
                self.songs.append((torch.stack(segments), lbl, song_id))

        print(f"SegmentDataset ready: {len(self.songs)} songs "
              f"({'train+aug' if augment else 'eval'})")

    def __len__(self):
        return len(self.songs)

    def __getitem__(self, idx):
        segments, label, song_id = self.songs[idx]
        return segments, torch.tensor(label, dtype=torch.long), song_id


def collate_fn(batch):
    mels, labels, song_ids = zip(*batch)
    return torch.stack(mels), torch.stack(labels), torch.tensor(song_ids)


def collate_segments(batch):
    segments, labels, song_ids = zip(*batch)
    max_segs = max(seg.size(0) for seg in segments)
    padded = []
    for seg in segments:
        n = seg.size(0)
        if n < max_segs:
            padding = torch.zeros(max_segs - n, 1, 128, 128)
            seg = torch.cat([seg, padding], dim=0)
        padded.append(seg)
    return torch.stack(padded), torch.stack(labels), torch.tensor(song_ids)
