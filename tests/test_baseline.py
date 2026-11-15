import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from src.audio_classification.data import get_audio_path


@pytest.fixture
def metadata():
    return pd.DataFrame(
        {
            "track_id": [1, 2, 3, 4, 5, 6, 7],
            "genre": ["Rock", "Folk", "Rock", "Folk", "Rock", "Folk", "Rock"],
            "artist_id": [11, 22, 33, 44, 55, 66, 77],
            "split": ["training"] * 3 + ["validation"] * 3 + ["test"],
        },
        index=[10, 20, 30, 40, 50, 60, 70],
    )


def run_baseline(project_root, metadata, extractor):
    """Execute the actual baseline cell with isolated data and temporary caches."""
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "02_features_and_baseline.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "rf_baseline.fit(" in "".join(cell["source"])
    )
    namespace = {
        "project_root": project_root,
        "df_metadata": metadata,
        "np": np,
        "extract_classical_features": extractor,
        "get_audio_path": get_audio_path,
        "RandomForestClassifier": RandomForestClassifier,
        "classification_report": classification_report,
    }
    exec(compile(source, str(notebook_path), "exec"), namespace)
    return namespace


def test_baseline_saves_features_and_preserves_labels_after_decode_failures(
    tmp_path, metadata
):
    def extract(path):
        track_id = int(Path(path).stem)
        if track_id in (3, 6):
            raise RuntimeError("Broken recording")
        return np.full(44, track_id, dtype=np.float32)

    extractor = Mock(side_effect=extract)

    result = run_baseline(tmp_path, metadata, extractor)

    processed = tmp_path / "data" / "processed"
    train_cache = np.load(processed / "classical_train.npy", allow_pickle=False)
    val_cache = np.load(processed / "classical_val.npy", allow_pickle=False)
    assert train_cache.shape == val_cache.shape == (3, 44)
    assert np.isnan(train_cache[2]).all()
    assert np.isnan(val_cache[2]).all()
    np.testing.assert_array_equal(result["X_train"][:, 0], [1, 2])
    np.testing.assert_array_equal(result["y_train"], ["Rock", "Folk"])
    np.testing.assert_array_equal(result["X_val"][:, 0], [4, 5])
    np.testing.assert_array_equal(result["y_val"], ["Folk", "Rock"])
    assert result["y_pred"].shape == (2,)
    assert set(result["y_pred"]) <= {"Folk", "Rock"}
    assert [int(Path(call.args[0]).stem) for call in extractor.call_args_list] == [
        1,
        2,
        3,
        4,
        5,
        6,
    ]


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_baseline_reuses_cache_and_filters_nonfinite_rows_with_their_labels(
    tmp_path, metadata, invalid
):
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    train_features = np.repeat(np.array([[1.0], [2.0], [3.0]]), 44, axis=1)
    val_features = np.repeat(np.array([[4.0], [5.0], [6.0]]), 44, axis=1)
    train_features[2, 17] = invalid
    val_features[2, 9] = invalid
    np.save(processed / "classical_train.npy", train_features, allow_pickle=False)
    np.save(processed / "classical_val.npy", val_features, allow_pickle=False)
    extractor = Mock(
        side_effect=AssertionError("Cached features must not be recomputed")
    )

    result = run_baseline(tmp_path, metadata, extractor)

    extractor.assert_not_called()
    np.testing.assert_array_equal(result["X_train"], train_features[:2])
    np.testing.assert_array_equal(result["y_train"], ["Rock", "Folk"])
    np.testing.assert_array_equal(result["X_val"], val_features[:2])
    np.testing.assert_array_equal(result["y_val"], ["Folk", "Rock"])
    assert np.isfinite(result["X_train"]).all()
    assert np.isfinite(result["X_val"]).all()
    np.testing.assert_array_equal(result["rf_baseline"].classes_, ["Folk", "Rock"])
