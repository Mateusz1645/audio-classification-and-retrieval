# src/audio_classification/data.py
import os

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.audio_classification.features import MelSpectrogramExtractor


def load_fma_metadata(metadata_dir: str = "data/raw/fma_metadata") -> pd.DataFrame:
    """Parses official FMA tracks.csv metadata for fma_small tracks and official splits."""
    tracks_path = os.path.join(metadata_dir, "tracks.csv")

    # Read multi-index CSV
    tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1])

    # Select fma_small subset
    small_tracks = tracks[tracks[("set", "subset")] == "small"]

    df = pd.DataFrame(
        {
            "track_id": small_tracks.index,
            "genre": small_tracks[("track", "genre_top")],
            "artist_id": small_tracks[("artist", "id")],
            "split": small_tracks[("set", "split")],  # 'training', 'validation', 'test'
        }
    ).reset_index(drop=True)

    return df


def get_audio_path(audio_dir: str, track_id: int) -> str:
    """Converts integer track_id to FMA nested directory path (e.g., 2 -> 'data/raw/fma_small/000/000002.mp3')."""
    tid_str = f"{track_id:06d}"
    folder = tid_str[:3]
    return os.path.join(audio_dir, folder, f"{tid_str}.mp3")


class FMASpectrogramDataset(Dataset):
    """Load one FMA split as variable-length sequences of log-Mel segments.

    Build genre IDs from the complete metadata table to keep all splits
    consistent. Optional per-track caching stores unnormalized spectrograms.
    Use a new cache directory when preprocessing parameters change.
    With cache_only=True, missing tensor files raise before decoding audio.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        audio_dir: str = "data/raw/fma_small",
        target_split: str = "training",
        cache_dir: str | None = None,
        cache_only: bool = False,
    ):
        self.df = df[df["split"] == target_split].reset_index(drop=True)
        self.audio_dir = audio_dir
        self.extractor = MelSpectrogramExtractor()
        self.cache_dir = cache_dir
        self.cache_only = cache_only
        if cache_only and cache_dir is None:
            raise ValueError("cache_only requires cache_dir")

        # Use the full metadata table so genre IDs match across all splits.
        self.genres = sorted(df["genre"].unique().tolist())
        self.genre_to_idx = {g: i for i, g in enumerate(self.genres)}

    def __len__(self):
        """Return the number of tracks in the selected split."""
        return len(self.df)

    def __getitem__(self, idx):
        """Return [segments, 1, mel_bins, frames] data and one integer genre label."""
        row = self.df.iloc[idx]
        track_id = row["track_id"]
        label = self.genre_to_idx[row["genre"]]

        audio_path = get_audio_path(self.audio_dir, track_id)
        cache_path = (
            os.path.join(self.cache_dir, f"{track_id:06d}.pt")
            if self.cache_dir is not None
            else None
        )

        if self.cache_only and (cache_path is None or not os.path.isfile(cache_path)):
            raise FileNotFoundError(
                f"Missing cached spectrogram: {cache_path}. "
                "Run prepare_spectrogram_cache before training."
            )

        try:
            if cache_path is not None and os.path.isfile(cache_path):
                # The training loop moves cached CPU tensors to its compute device.
                spectrograms = torch.load(
                    cache_path, map_location="cpu", weights_only=True
                )
            else:
                spectrograms = self.extractor.process_file(audio_path)
                if spectrograms is None:
                    raise ValueError("Audio clip too short or empty")
                if cache_path is not None:
                    os.makedirs(self.cache_dir, exist_ok=True)
                    torch.save(spectrograms, cache_path)
            return spectrograms, torch.tensor(label, dtype=torch.long)
        except Exception as exc:
            raise RuntimeError(f"Cannot load audio: {audio_path}") from exc


def collate_spectrograms(batch):
    """Flatten segments while retaining the boundaries between tracks.

    Return spectrograms [sum(counts), 1, mel_bins, frames], labels [tracks]
    and segment counts [tracks]. Counts allow track-level aggregation later.
    """
    spectrograms, labels = zip(*batch)
    counts = torch.tensor([spec.shape[0] for spec in spectrograms])
    return torch.cat(spectrograms), torch.stack(labels), counts
