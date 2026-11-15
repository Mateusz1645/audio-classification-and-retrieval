from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf
import torch

from src.audio_classification import features
from src.audio_classification.features import (
    MelSpectrogramExtractor,
    extract_classical_features,
)


def test_classical_feature_order_statistics_and_mono_conversion(monkeypatch):
    stereo = np.column_stack(
        [np.ones(4096, dtype=np.float32), np.full(4096, 3, dtype=np.float32)]
    )
    monkeypatch.setattr(features.sf, "read", Mock(return_value=(stereo, 22050)))
    mfcc = Mock(return_value=np.arange(60, dtype=np.float32).reshape(20, 3))
    monkeypatch.setattr(features.librosa.feature, "mfcc", mfcc)
    monkeypatch.setattr(
        features.librosa.feature,
        "spectral_centroid",
        Mock(return_value=np.array([[100.0, 200.0, 300.0]])),
    )
    monkeypatch.setattr(
        features.librosa.feature,
        "zero_crossing_rate",
        Mock(return_value=np.array([[0.1, 0.2, 0.3]])),
    )

    result = extract_classical_features("stereo.wav")

    expected = np.concatenate(
        [
            np.arange(20) * 3 + 1,
            np.full(20, np.sqrt(2 / 3)),
            [200, 100 * np.sqrt(2 / 3), 0.2, 0.1 * np.sqrt(2 / 3)],
        ]
    )
    np.testing.assert_allclose(result, expected, rtol=1e-6)
    np.testing.assert_array_equal(mfcc.call_args.kwargs["y"], np.full(4096, 2))
    assert mfcc.call_args.kwargs["sr"] == 22050
    assert mfcc.call_args.kwargs["n_mfcc"] == 20


@pytest.mark.parametrize("sample_rate", [8000, 22050])
@pytest.mark.parametrize("silent", [False, True])
def test_classical_features_are_finite_for_real_wav_files(
    tmp_path, sample_rate, silent
):
    samples = np.arange(sample_rate // 2, dtype=np.float32)
    audio = (
        np.zeros_like(samples)
        if silent
        else np.sin(2 * np.pi * 440 * samples / sample_rate).astype(np.float32)
    )
    path = tmp_path / "audio.wav"
    sf.write(path, audio, sample_rate, subtype="FLOAT")

    result = extract_classical_features(str(path))

    assert result.shape == (44,)
    assert np.isfinite(result).all()
    if not silent:
        assert np.any(result != 0)


def test_classical_features_raise_on_decode_failure_instead_of_returning_zeros(
    monkeypatch,
):
    original = sf.LibsndfileError(1, "Broken recording")
    monkeypatch.setattr(features.sf, "read", Mock(side_effect=original))

    with pytest.raises(RuntimeError, match="broken.mp3") as error:
        extract_classical_features("broken.mp3")

    assert error.value.__cause__ is original


@pytest.mark.parametrize(("samples", "segments"), [(799, 0), (800, 1), (2000, 2)])
def test_mel_extractor_uses_only_complete_segments(tmp_path, samples, segments):
    path = tmp_path / "silence.wav"
    sf.write(path, np.zeros(samples, dtype=np.float32), 8000, subtype="FLOAT")
    extractor = MelSpectrogramExtractor(
        sample_rate=8000, n_fft=256, hop_length=64, n_mels=16, segment_duration=0.1
    )

    result = extractor.process_file(str(path))

    if segments == 0:
        assert result is None
    else:
        assert result.shape == (segments, 1, 16, 13)
        assert result.device.type == "cpu"
        assert torch.isfinite(result).all()


@pytest.mark.parametrize("sample_rate", [8000, 16000])
def test_mel_extractor_downmixes_stereo_and_resamples(tmp_path, sample_rate):
    samples = np.arange(sample_rate // 4, dtype=np.float32)
    tone = np.sin(2 * np.pi * 440 * samples / sample_rate).astype(np.float32)
    stereo = np.column_stack([tone * 0.25, tone * 0.75])
    mono_path = tmp_path / "mono.wav"
    stereo_path = tmp_path / "stereo.wav"
    sf.write(mono_path, stereo.mean(axis=1), sample_rate, subtype="FLOAT")
    sf.write(stereo_path, stereo, sample_rate, subtype="FLOAT")
    extractor = MelSpectrogramExtractor(
        sample_rate=8000, n_fft=256, hop_length=64, n_mels=16, segment_duration=0.1
    )

    mono = extractor.process_file(str(mono_path))
    downmixed = extractor.process_file(str(stereo_path))

    assert downmixed.shape == (2, 1, 16, 13)
    torch.testing.assert_close(downmixed, mono, rtol=1e-4, atol=1e-3)


def test_mel_extractor_propagates_invalid_audio_errors(tmp_path):
    path = tmp_path / "invalid.wav"
    path.write_bytes(b"This is not an audio recording")
    extractor = MelSpectrogramExtractor()

    with pytest.raises(sf.LibsndfileError):
        extractor.process_file(str(path))
