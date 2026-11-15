import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf
import torch

from src.audio_classification.data import get_audio_path
from src.audio_classification.features import MelSpectrogramExtractor


def prepare_spectrogram_cache(
    df, audio_dir, cache_dir, device="cpu", num_workers=4, retry_failed=False
):
    """Reuse cached spectrograms and prepare missing tracks before training.

    Decode files concurrently and run resampling and log-Mel transforms on
    device. Return filtered metadata and a report of reused, created and
    excluded tracks. Decoder failures are recorded, never replaced by zeros.
    GPU, filesystem and other infrastructure errors remain fatal.
    """
    if num_workers < 1:
        raise ValueError("num_workers must be positive")
    audio_dir = Path(audio_dir)
    cache_dir = Path(cache_dir)
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory does not exist: {audio_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    report_path = cache_dir / "preprocessing_report.json"
    previous = json.loads(report_path.read_text()) if report_path.exists() else {}
    failures = previous.get("failures", {})
    extractor = MelSpectrogramExtractor(device=device)
    report = {
        "requested_tracks": len(df),
        "cached_tracks": 0,
        "created_tracks": 0,
        "excluded": [],
        "failures": failures,
    }

    @torch.inference_mode()
    def prepare(row):
        track_id = int(row.track_id)
        cache_path = cache_dir / f"{track_id:06d}.pt"
        if cache_path.is_file():
            try:
                specs = torch.load(cache_path, map_location="cpu", weights_only=True)
                if (
                    not isinstance(specs, torch.Tensor)
                    or specs.ndim != 4
                    or specs.shape[0] < 1
                    or specs.shape[1:] != (1, 128, 130)
                    or not torch.isfinite(specs).all()
                ):
                    raise ValueError("Invalid cached spectrogram")
                return track_id, "cached", None
            except (RuntimeError, EOFError, pickle.UnpicklingError, ValueError):
                pass

        if str(track_id) in failures and not retry_failed:
            return track_id, "excluded", failures[str(track_id)]

        try:
            specs = extractor.process_file(get_audio_path(str(audio_dir), track_id))
            if specs is None:
                raise ValueError("Audio clip has no complete three-second segment")
            if not torch.isfinite(specs).all():
                raise ValueError("Non-finite spectrogram")
        except (sf.LibsndfileError, ValueError) as exc:
            return track_id, "excluded", f"{type(exc).__name__}: {exc}"

        # Publish complete files atomically so an interruption leaves no broken .pt.
        temporary = cache_path.with_suffix(".tmp")
        torch.save(specs.cpu(), temporary)
        temporary.replace(cache_path)
        return track_id, "created", None

    def save_report():
        temporary = report_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(report_path)

    valid_ids = set()
    rows = {int(row.track_id): row for row in df.itertuples(index=False)}
    previous_threads = torch.get_num_threads()
    # Multiple decoding workers must not each start a full CPU thread pool.
    torch.set_num_threads(1)
    executor = ThreadPoolExecutor(max_workers=num_workers)
    try:
        for index, (track_id, status, error) in enumerate(
            executor.map(prepare, rows.values()), 1
        ):
            if error is None:
                valid_ids.add(track_id)
                failures.pop(str(track_id), None)
                report[f"{status}_tracks"] += 1
            else:
                failures[str(track_id)] = error
                row = rows[track_id]
                report["excluded"].append(
                    {
                        "track_id": track_id,
                        "split": row.split,
                        "genre": row.genre,
                        "error": error,
                    }
                )
            if index == 1 or index % 100 == 0 or index == len(rows):
                save_report()
                print(
                    f"Spectrogram cache: {index}/{len(rows)}, "
                    f"reused={report['cached_tracks']}, "
                    f"created={report['created_tracks']}, "
                    f"excluded={len(report['excluded'])}",
                    flush=True,
                )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        torch.set_num_threads(previous_threads)
        save_report()

    if not valid_ids:
        raise ValueError(f"No usable tracks; see {report_path}")
    return df[df["track_id"].isin(valid_ids)].copy(), report
