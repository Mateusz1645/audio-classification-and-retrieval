# src/audio_classification/features.py
import warnings

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T

# Suppress minor audio decoding warnings
warnings.filterwarnings("ignore", category=UserWarning)


def extract_classical_features(file_path: str, sample_rate: int = 22050) -> np.ndarray:
    """Extracts classical tabular audio descriptors for ML baselines.

    Computes mean & std of 20 MFCCs, Spectral Centroid, and Zero-Crossing Rate.
    Uses torchaudio loading to bypass libmpg123 MP3 frame corruption crashes.
    Returns 1D feature vector of shape (44,).
    """
    try:
        # Load audio via torchaudio to prevent low-level decoder crashes
        waveform, sr = sf.read(file_path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(waveform.T)

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

        return np.hstack([mfcc_mean, mfcc_std, cent_mean, cent_std, zcr_mean, zcr_std])
    except Exception as exc:
        raise RuntimeError(f"Cannot extract audio features: {file_path}") from exc


class MelSpectrogramExtractor:
    """Create batched log-Mel segments on a selected CPU or CUDA device."""

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        segment_duration: float = 3.0,
        device: str | torch.device = "cpu",
    ):
        self.sample_rate = sample_rate
        self.segment_samples = int(sample_rate * segment_duration)
        self.device = torch.device(device)

        # Torchaudio MelSpectrogram & dB scale transforms
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.mel_transform = self.mel_transform.to(self.device)
        self.db_transform = T.AmplitudeToDB(top_db=80.0).to(self.device)

    def process_file(self, file_path: str) -> torch.Tensor:
        """Loads MP3, converts to mono 22.05kHz, slices into 3s non-overlapping chunks,

        and computes Log-Mel Spectrograms tensor of shape [num_chunks, 1, n_mels, time_frames].
        All complete segments are transformed together on self.device.
        """
        waveform, sr = sf.read(file_path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(waveform.T)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform = waveform.to(self.device)
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate).to(self.device)
            waveform = resampler(waveform)

        if waveform.shape[-1] < self.segment_samples:
            return None

        # Transform all complete segments together instead of launching one FFT per loop.
        chunks = waveform.unfold(-1, self.segment_samples, self.segment_samples)
        chunks = chunks.transpose(0, 1)
        return self.db_transform(self.mel_transform(chunks))
