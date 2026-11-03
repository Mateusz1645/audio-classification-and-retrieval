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
    """PyTorch Dataset for Log-Mel Spectrogram audio classification."""

    def __init__(
        self,
        df: pd.DataFrame,
        audio_dir: str = "data/raw/fma_small",
        target_split: str = "training",
    ):
        self.df = df[df["split"] == target_split].reset_index(drop=True)
        self.audio_dir = audio_dir
        self.extractor = MelSpectrogramExtractor()

        # Build consistent label mapping
        self.genres = sorted(self.df["genre"].unique().tolist())
        self.genre_to_idx = {g: i for i, g in enumerate(self.genres)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        track_id = row["track_id"]
        label = self.genre_to_idx[row["genre"]]

        audio_path = get_audio_path(self.audio_dir, track_id)

        try:
            spectrograms = self.extractor.process_file(audio_path)
            if spectrograms is None:
                raise ValueError("Audio clip too short or empty")
            return spectrograms, torch.tensor(label, dtype=torch.long)
        except Exception:
            # Fallback zero tensor for corrupted audio files
            return torch.zeros((10, 1, 128, 130)), torch.tensor(
                label, dtype=torch.long
            )