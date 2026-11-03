# src/audio_classification/features.py
import warnings
import librosa
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

# Suppress minor audio decoding warnings
warnings.filterwarnings("ignore", category=UserWarning)


def extract_classical_features(
    file_path: str, sample_rate: int = 22050
) -> np.ndarray:
    """Extracts classical tabular audio descriptors for ML baselines.

    Computes mean & std of 20 MFCCs, Spectral Centroid, and Zero-Crossing Rate.
    Uses torchaudio loading to bypass libmpg123 MP3 frame corruption crashes.
    Returns 1D feature vector of shape (44,).
    """
    try:
        # Load audio via torchaudio to prevent low-level decoder crashes
        waveform, sr = torchaudio.load(file_path)

        # Convert stereo/multi-channel to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample to target rate (22,050 Hz)
        if sr != sample_rate:
            resampler = T.Resample(sr, sample_rate)
            waveform = resampler(waveform)

        # Convert PyTorch tensor to 1D NumPy array for librosa feature extraction
        y = waveform.squeeze().numpy()

        # 1. 20 MFCCs (mean & std)
        mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=20)
        mfcc_mean, mfcc_std = np.mean(mfcc, axis=1), np.std(mfcc, axis=1)

        # 2. Spectral Centroid (mean & std)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sample_rate)
        cent_mean, cent_std = np.mean(centroid), np.std(centroid)

        # 3. Zero Crossing Rate (mean & std)
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean, zcr_std = np.mean(zcr), np.std(zcr)

        return np.hstack(
            [mfcc_mean, mfcc_std, cent_mean, cent_std, zcr_mean, zcr_std]
        )
    except Exception:
        # Fallback zero array if audio clip is corrupt or missing
        return np.zeros(44, dtype=np.float32)


class MelSpectrogramExtractor:
    """Converts 1D audio waveforms into 2D Log-Mel Spectrograms for CNN backbones."""

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        segment_duration: float = 3.0,
    ):
        self.sample_rate = sample_rate
        self.segment_samples = int(sample_rate * segment_duration)

        # Torchaudio MelSpectrogram & dB scale transforms
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.db_transform = T.AmplitudeToDB(top_db=80.0)

    def process_file(self, file_path: str) -> torch.Tensor:
        """Loads MP3, converts to mono 22.05kHz, slices into 3s non-overlapping chunks,

        and computes Log-Mel Spectrograms tensor of shape [num_chunks, 1, n_mels, time_frames].
        """
        waveform, sr = torchaudio.load(file_path)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        total_samples = waveform.shape[-1]
        chunks = []

        # Slice 30-second audio into 3-second non-overlapping segments
        for start in range(
            0, total_samples - self.segment_samples + 1, self.segment_samples
        ):
            chunk = waveform[:, start : start + self.segment_samples]
            mel_spec = self.mel_transform(chunk)
            log_mel_spec = self.db_transform(mel_spec)
            chunks.append(log_mel_spec)

        if not chunks:
            return None

        # Output shape: [num_chunks, 1, 128, 130]
        return torch.stack(chunks)