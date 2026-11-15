from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.audio_classification.data import (
    FMASpectrogramDataset,
    collate_spectrograms,
    get_audio_path,
    load_fma_metadata,
)


@pytest.fixture
def metadata():
    return pd.DataFrame(
        {
            "track_id": [10, 20, 30],
            "genre": ["Folk", "Rock", "Rock"],
            "artist_id": [1, 2, 3],
            "split": ["training", "training", "validation"],
        }
    )


def test_metadata_preserves_small_tracks_and_official_splits(tmp_path):
    columns = pd.MultiIndex.from_tuples(
        [("set", "subset"), ("set", "split"), ("track", "genre_top"), ("artist", "id")]
    )
    tracks = pd.DataFrame(
        [
            ["small", "training", "Rock", 10],
            ["medium", "training", "Jazz", 20],
            ["small", "validation", "Folk", 30],
            ["small", "test", "Rock", 40],
        ],
        index=pd.Index([2, 1000, 3000, 4000], name="track_id"),
        columns=columns,
    )
    tracks.to_csv(tmp_path / "tracks.csv")

    result = load_fma_metadata(str(tmp_path))

    expected = pd.DataFrame(
        {
            "track_id": [2, 3000, 4000],
            "genre": ["Rock", "Folk", "Rock"],
            "artist_id": [10, 30, 40],
            "split": ["training", "validation", "test"],
        }
    )
    pd.testing.assert_frame_equal(result, expected)


def test_missing_metadata_raises_instead_of_returning_an_empty_dataset(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_fma_metadata(str(tmp_path))


@pytest.mark.parametrize(
    ("track_id", "folder", "filename"),
    [
        (2, "000", "000002.mp3"),
        (999, "000", "000999.mp3"),
        (1000, "001", "001000.mp3"),
        (123456, "123", "123456.mp3"),
    ],
)
def test_audio_paths_follow_fma_directory_boundaries(
    tmp_path, track_id, folder, filename
):
    assert Path(get_audio_path(str(tmp_path), track_id)) == tmp_path / folder / filename


def test_cache_only_requires_a_cache_directory(metadata):
    with pytest.raises(ValueError, match="cache_only requires cache_dir"):
        FMASpectrogramDataset(metadata, cache_only=True)


def test_dataset_caches_extracted_segments_and_reuses_them(
    metadata, tmp_path, monkeypatch
):
    cache_dir = tmp_path / "cache"
    dataset = FMASpectrogramDataset(metadata, cache_dir=str(cache_dir))
    expected = torch.arange(256, dtype=torch.float32).reshape(2, 1, 16, 8)
    decoder = Mock(return_value=expected)
    monkeypatch.setattr(dataset.extractor, "process_file", decoder)

    first, first_label = dataset[0]
    second, second_label = dataset[0]

    decoder.assert_called_once_with(get_audio_path(dataset.audio_dir, 10))
    assert (cache_dir / "000010.pt").is_file()
    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(second, expected)
    assert first_label.dtype == second_label.dtype == torch.long
    assert first_label.item() == second_label.item() == 0


def test_dataset_reports_short_audio_with_its_path(metadata, monkeypatch):
    dataset = FMASpectrogramDataset(metadata)
    monkeypatch.setattr(dataset.extractor, "process_file", Mock(return_value=None))

    with pytest.raises(RuntimeError, match="000010.mp3") as error:
        dataset[0]

    assert isinstance(error.value.__cause__, ValueError)
    assert "too short or empty" in str(error.value.__cause__)


def test_dataset_keeps_the_original_decoder_error(metadata, monkeypatch):
    dataset = FMASpectrogramDataset(metadata)
    original = OSError("Decoder failed")
    monkeypatch.setattr(dataset.extractor, "process_file", Mock(side_effect=original))

    with pytest.raises(RuntimeError, match="000010.mp3") as error:
        dataset[0]

    assert error.value.__cause__ is original


def test_dataloader_reads_cached_tracks_without_decoding(
    metadata, tmp_path, monkeypatch
):
    torch.save(torch.ones(2, 1, 16, 8), tmp_path / "000010.pt")
    torch.save(torch.zeros(1, 1, 16, 8), tmp_path / "000020.pt")
    dataset = FMASpectrogramDataset(metadata, cache_dir=str(tmp_path), cache_only=True)
    decoder = Mock(side_effect=AssertionError("Cached tracks must not be decoded"))
    monkeypatch.setattr(dataset.extractor, "process_file", decoder)
    loader = DataLoader(
        dataset, batch_size=2, num_workers=0, collate_fn=collate_spectrograms
    )

    specs, labels, counts = next(iter(loader))

    assert specs.shape == (3, 1, 16, 8)
    assert counts.tolist() == [2, 1]
    assert labels.tolist() == [0, 1]
    assert torch.all(specs[:2] == 1)
    assert torch.all(specs[2:] == 0)
    decoder.assert_not_called()
