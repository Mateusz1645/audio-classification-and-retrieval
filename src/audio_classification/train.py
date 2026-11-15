from pathlib import Path
from time import perf_counter

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from src.audio_classification.evaluate import evaluate_model
from src.audio_classification.utils import set_seed


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    checkpoint_path,
    epochs=30,
    learning_rate=1e-3,
    patience=5,
    seed=42,
    use_amp=True,
    data_report=None,
):
    """Train the CNN and restore the best validation macro-F1 checkpoint.

    Each track contributes equally to the loss, regardless of segment count.
    Log metrics, preprocessing parameters and split IDs to the active MLflow
    experiment. Save the best weights to checkpoint_path and return the
    checkpoint with its training history. Validation controls early stopping.
    CUDA training optionally uses mixed precision and gradient scaling.
    data_report records preprocessing exclusions in the same MLflow run.
    """
    if not len(train_loader.dataset) or not len(val_loader.dataset):
        raise ValueError("Training and validation splits must be non-empty")
    if epochs < 1 or patience < 1:
        raise ValueError("epochs and patience must be positive")
    if train_loader.dataset.genres != val_loader.dataset.genres:
        raise ValueError("Training and validation class mappings differ")

    set_seed(seed)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.to(device)
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    extractor = train_loader.dataset.extractor
    preprocessing = {
        "sample_rate": extractor.sample_rate,
        "n_fft": extractor.mel_transform.n_fft,
        "hop_length": extractor.mel_transform.hop_length,
        "n_mels": extractor.mel_transform.n_mels,
        "segment_duration": extractor.segment_samples / extractor.sample_rate,
        "normalization": "per-segment z-score",
        "aggregation": "mean segment probabilities",
    }
    parameters = {
        **preprocessing,
        "model": "AudioCNN",
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": 1e-4,
        "patience": patience,
        "batch_size_tracks": train_loader.batch_size,
        "embedding_dim": model.embedding.out_features,
        "dropout": model.dropout.p,
        "seed": seed,
        "train_tracks": len(train_loader.dataset),
        "validation_tracks": len(val_loader.dataset),
        "device": str(device),
        "mixed_precision": amp_enabled,
        "torch_version": str(torch.__version__),
    }
    best_f1, stale_epochs = -1.0, 0
    history = []
    started = perf_counter()

    with mlflow.start_run(run_name=checkpoint_path.stem) as run:
        mlflow.log_params(parameters)
        if data_report is not None:
            mlflow.log_dict(data_report, "preprocessing_report.json")
        mlflow.log_dict(
            {
                "training": train_loader.dataset.df["track_id"].astype(int).tolist(),
                "validation": val_loader.dataset.df["track_id"].astype(int).tolist(),
                "genres": train_loader.dataset.genres,
            },
            "splits.json",
        )
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            targets, predictions = [], []
            print(f"Epoch {epoch}/{epochs}", flush=True)
            for batch_index, (specs, labels, counts) in enumerate(train_loader, 1):
                specs = specs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=amp_enabled
                ):
                    logits = model(specs)
                logits = logits.float()
                sizes = counts.tolist()
                segment_labels = labels.repeat_interleave(counts.to(device))
                segment_losses = F.cross_entropy(
                    logits, segment_labels, reduction="none"
                )
                # Average within each track before averaging across the batch.
                loss = torch.stack(
                    [part.mean() for part in segment_losses.split(sizes)]
                ).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Compute training metrics for tracks, not individual segments.
                track_probs = torch.stack(
                    [part.mean(0) for part in logits.detach().softmax(1).split(sizes)]
                )
                total_loss += loss.item() * len(labels)
                targets.append(labels.cpu().numpy())
                predictions.append(track_probs.argmax(1).cpu().numpy())
                if batch_index % 100 == 0:
                    print(
                        f"  {batch_index}/{len(train_loader)} training batches",
                        flush=True,
                    )

            y_true = np.concatenate(targets)
            y_pred = np.concatenate(predictions)
            validation = evaluate_model(model, val_loader, device, use_amp=amp_enabled)
            metrics = {
                "train_loss": total_loss / len(y_true),
                "train_accuracy": float(accuracy_score(y_true, y_pred)),
                "train_macro_f1": float(
                    f1_score(
                        y_true,
                        y_pred,
                        labels=list(range(len(train_loader.dataset.genres))),
                        average="macro",
                        zero_division=0,
                    )
                ),
                **{
                    f"val_{key}": validation[key]
                    for key in ("loss", "accuracy", "macro_f1")
                },
            }
            history.append({"epoch": epoch, **metrics})
            mlflow.log_metrics(metrics, step=epoch)
            print(
                f"Epoch {epoch}/{epochs}: "
                f"train_loss={metrics['train_loss']:.4f}, "
                f"val_loss={metrics['val_loss']:.4f}, "
                f"val_macro_f1={metrics['val_macro_f1']:.4f}",
                flush=True,
            )

            # Select the checkpoint using validation data only.
            if validation["macro_f1"] > best_f1:
                best_f1 = validation["macro_f1"]
                stale_epochs = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_config": {
                            "num_classes": model.classifier.out_features,
                            "embedding_dim": model.embedding.out_features,
                        },
                        "genres": train_loader.dataset.genres,
                        "parameters": parameters,
                        "best_epoch": epoch,
                        "best_val_macro_f1": best_f1,
                        "run_id": run.info.run_id,
                        "history": history,
                        "training_seconds": perf_counter() - started,
                    },
                    checkpoint_path,
                )
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    print("Early stopping", flush=True)
                    break

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        checkpoint["history"] = history
        checkpoint["training_seconds"] = perf_counter() - started
        torch.save(checkpoint, checkpoint_path)
        # Restore the best epoch rather than the final training epoch.
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        mlflow.log_metrics(
            {
                "best_val_macro_f1": best_f1,
                "best_epoch": checkpoint["best_epoch"],
                "training_seconds": checkpoint["training_seconds"],
            }
        )
        mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")

    return checkpoint
