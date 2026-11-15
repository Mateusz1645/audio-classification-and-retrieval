from pathlib import Path
from unittest.mock import Mock

import mlflow
import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch
import torch.nn.functional as F
from mlflow.tracking import MlflowClient
from torch import nn
from torch.utils.data import DataLoader

from src.audio_classification.data import FMASpectrogramDataset, collate_spectrograms
from src.audio_classification.evaluate import evaluate_model, grad_cam
from src.audio_classification.features import MelSpectrogramExtractor
from src.audio_classification.models import AudioCNN
from src.audio_classification.prepare_data import prepare_spectrogram_cache
from src.audio_classification.train import train_model


@pytest.fixture(autouse=True)
def deterministic_cpu():
    """Keep synthetic CNN tests deterministic and inexpensive on CPU."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            yield
    finally:
        torch.set_num_threads(previous_threads)


@pytest.fixture
def metadata():
    return pd.DataFrame(
        {
            "track_id": [1, 2, 3],
            "genre": ["Rock", "Folk", "Rock"],
            "artist_id": [11, 22, 33],
            "split": ["training", "validation", "test"],
        }
    )


@pytest.mark.parametrize("constant", [False, True])
def test_cnn_outputs_are_finite_for_audio_and_silence(constant):
    model = AudioCNN().eval()
    specs = torch.zeros(2, 1, 32, 32) if constant else torch.randn(2, 1, 32, 32)
    with torch.no_grad():
        logits = model(specs)
        embeddings = model.extract_embeddings(specs)
    assert logits.shape == (2, 8)
    assert embeddings.shape == (2, 512)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(embeddings).all()


def test_cnn_backpropagates_to_convolution_weights():
    model = AudioCNN()
    loss = F.cross_entropy(model(torch.randn(2, 1, 32, 32)), torch.tensor([0, 1]))
    loss.backward()
    gradient = model.features[0].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_collate_preserves_variable_track_boundaries():
    specs, labels, counts = collate_spectrograms(
        [
            (torch.ones(2, 1, 32, 32), torch.tensor(1)),
            (torch.zeros(3, 1, 32, 32), torch.tensor(0)),
        ]
    )
    assert specs.shape == (5, 1, 32, 32)
    assert labels.tolist() == [1, 0]
    assert counts.tolist() == [2, 3]
    assert torch.all(specs[:2] == 1)
    assert torch.all(specs[2:] == 0)


def test_dataset_uses_one_genre_mapping_for_all_splits(metadata):
    training = FMASpectrogramDataset(metadata, target_split="training")
    validation = FMASpectrogramDataset(metadata, target_split="validation")
    assert training.genre_to_idx == validation.genre_to_idx == {"Folk": 0, "Rock": 1}
    assert training.df["track_id"].tolist() == [1]
    assert validation.df["track_id"].tolist() == [2]


def test_cache_only_never_decodes_missing_audio(metadata, tmp_path, monkeypatch):
    dataset = FMASpectrogramDataset(metadata, cache_dir=str(tmp_path), cache_only=True)
    decoder = Mock(side_effect=AssertionError("Raw audio must not be decoded"))
    monkeypatch.setattr(dataset.extractor, "process_file", decoder)
    with pytest.raises(FileNotFoundError, match="prepare_spectrogram_cache"):
        dataset[0]
    decoder.assert_not_called()


def test_preparation_reuses_cache_and_reports_decoder_failures(
    metadata, tmp_path, monkeypatch
):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    cache_dir = tmp_path / "cache"
    specs = torch.randn(2, 1, 128, 130)

    def decode(self, path):
        if Path(path).stem == "000003":
            raise sf.LibsndfileError(1, "Broken recording")
        return specs

    monkeypatch.setattr(MelSpectrogramExtractor, "process_file", decode)
    valid, report = prepare_spectrogram_cache(
        metadata, audio_dir, cache_dir, num_workers=2
    )
    assert valid["track_id"].tolist() == [1, 2]
    assert report["created_tracks"] == 2
    assert report["excluded"][0]["track_id"] == 3
    assert (cache_dir / "preprocessing_report.json").is_file()

    decoder = Mock(side_effect=AssertionError("Cached tracks must not be decoded"))
    monkeypatch.setattr(MelSpectrogramExtractor, "process_file", decoder)
    valid, report = prepare_spectrogram_cache(
        metadata, audio_dir, cache_dir, num_workers=2
    )
    assert valid["track_id"].tolist() == [1, 2]
    assert report["cached_tracks"] == 2
    assert report["created_tracks"] == 0
    assert report["excluded"][0]["track_id"] == 3
    decoder.assert_not_called()

    dataset = FMASpectrogramDataset(valid, cache_dir=str(cache_dir), cache_only=True)
    cached, label = dataset[0]
    torch.testing.assert_close(cached, specs)
    assert label.item() == 1


def test_preparation_does_not_hide_infrastructure_errors(
    metadata, tmp_path, monkeypatch
):
    decoder = Mock(side_effect=RuntimeError("CUDA out of memory"))
    monkeypatch.setattr(MelSpectrogramExtractor, "process_file", decoder)
    with pytest.raises(RuntimeError, match="out of memory"):
        prepare_spectrogram_cache(metadata, tmp_path, tmp_path / "cache", num_workers=1)


def test_batched_spectrograms_match_individual_segments(tmp_path):
    sample_rate = 22050
    samples = np.arange(sample_rate * 6, dtype=np.float32)
    audio = np.sin(2 * np.pi * 440 * samples / sample_rate).astype(np.float32)
    path = tmp_path / "tone.wav"
    sf.write(path, audio, sample_rate, subtype="FLOAT")
    extractor = MelSpectrogramExtractor()
    batched = extractor.process_file(str(path))
    waveform = torch.from_numpy(audio).unsqueeze(0)
    individual = torch.stack(
        [
            extractor.db_transform(
                extractor.mel_transform(
                    waveform[:, start : start + extractor.segment_samples]
                )
            )
            for start in (0, extractor.segment_samples)
        ]
    )
    assert batched.shape == (2, 1, 128, 130)
    torch.testing.assert_close(batched, individual, rtol=1e-4, atol=1e-3)


class LogitTracks(list):
    """Provide known segment logits to test aggregation independently of training."""

    genres = ["Folk", "Rock"]


class LogitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Identity()

    def extract_embeddings(self, specs):
        return specs.flatten(1)


def test_evaluation_averages_probabilities_per_track():
    dataset = LogitTracks(
        [
            (
                torch.tensor([[8.0, 0.0], [0.0, 2.0]]).reshape(2, 1, 1, 2),
                torch.tensor(0),
            ),
            (torch.tensor([0.0, 4.0]).reshape(1, 1, 1, 2), torch.tensor(1)),
        ]
    )
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_spectrograms)
    result = evaluate_model(LogitModel(), loader, torch.device("cpu"))
    np.testing.assert_allclose(
        result["probabilities"][0], [0.5594338, 0.4405662], rtol=1e-5
    )
    np.testing.assert_array_equal(result["y_true"], [0, 1])
    np.testing.assert_array_equal(result["y_pred"], [0, 1])
    np.testing.assert_array_equal(result["confusion_matrix"], np.eye(2, dtype=int))
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["loss"] == pytest.approx(0.5408908, rel=1e-5)


def test_evaluation_rejects_an_empty_split():
    loader = DataLoader(LogitTracks(), collate_fn=collate_spectrograms)
    with pytest.raises(ValueError, match="empty"):
        evaluate_model(LogitModel(), loader, torch.device("cpu"))


def test_grad_cam_returns_finite_maps_and_restores_model_mode():
    model = AudioCNN().train()
    heatmaps, probabilities = grad_cam(
        model, torch.randn(2, 1, 32, 32), 0, torch.device("cpu")
    )
    assert heatmaps.shape == (2, 32, 32)
    assert probabilities.shape == (2, 8)
    assert torch.isfinite(heatmaps).all()
    assert torch.all((heatmaps >= 0) & (heatmaps <= 1))
    torch.testing.assert_close(probabilities.sum(1), torch.ones(2))
    assert model.training
    assert not model.features[-4]._forward_hooks


def test_training_saves_a_reloadable_checkpoint_and_mlflow_run(
    metadata, tmp_path, monkeypatch
):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    for track_id in (1, 2):
        torch.save(torch.randn(2, 1, 32, 32), cache_dir / f"{track_id:06d}.pt")
    loaders = [
        DataLoader(
            FMASpectrogramDataset(
                metadata,
                target_split=split,
                cache_dir=str(cache_dir),
                cache_only=True,
            ),
            batch_size=1,
            collate_fn=collate_spectrograms,
        )
        for split in ("training", "validation")
    ]
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    try:
        client = MlflowClient()
        experiment_id = client.create_experiment(
            "cnn-tests", artifact_location=(tmp_path / "artifacts").as_uri()
        )
        monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", experiment_id)
        model = AudioCNN(num_classes=2, embedding_dim=16)
        checkpoint_path = tmp_path / "model.pt"
        checkpoint = train_model(
            model, *loaders, torch.device("cpu"), checkpoint_path, epochs=1
        )
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        reloaded = AudioCNN(**saved["model_config"]).eval()
        reloaded.load_state_dict(saved["model_state_dict"])
        specs = torch.randn(2, 1, 32, 32)
        with torch.no_grad():
            torch.testing.assert_close(model(specs), reloaded(specs))
        assert len(checkpoint["history"]) == 1
        assert checkpoint["best_epoch"] == 1
        assert checkpoint["genres"] == ["Folk", "Rock"]
        run = client.get_run(checkpoint["run_id"])
        assert run.info.status == "FINISHED"
        assert run.data.params["model"] == "AudioCNN"
        assert run.data.metrics["best_val_macro_f1"] == checkpoint["best_val_macro_f1"]
    finally:
        mlflow.set_tracking_uri(previous_uri)
