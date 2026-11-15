from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


@torch.inference_mode()
def evaluate_model(model, loader, device, use_amp=True):
    """Evaluate complete tracks from batches of concatenated segments.

    Average segment probabilities and embeddings within each track. Loss is
    segment cross-entropy averaged per track. Returned arrays follow loader
    order; inference time includes data loading and spectrogram preparation.
    CUDA forward passes optionally use AMP; metrics remain in float32.
    """
    model.eval()
    targets, probabilities, embeddings = [], [], []
    total_loss = 0.0
    started = perf_counter()

    for specs, labels, counts in loader:
        specs = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp and device.type == "cuda",
        ):
            features = model.extract_embeddings(specs)
            logits = model.classifier(features)
        features = features.float()
        logits = logits.float()
        sizes = counts.tolist()
        segment_labels = labels.repeat_interleave(counts.to(device))
        segment_losses = F.cross_entropy(logits, segment_labels, reduction="none")
        total_loss += sum(part.mean().item() for part in segment_losses.split(sizes))
        # Recover track-level predictions from the concatenated segment batch.
        track_probs = torch.stack(
            [part.mean(0) for part in logits.softmax(1).split(sizes)]
        )
        track_embeddings = torch.stack([part.mean(0) for part in features.split(sizes)])
        targets.append(labels.cpu().numpy())
        probabilities.append(track_probs.cpu().numpy())
        embeddings.append(track_embeddings.cpu().numpy())

    if not targets:
        raise ValueError("Evaluation split is empty")

    y_true = np.concatenate(targets)
    probs = np.concatenate(probabilities)
    y_pred = probs.argmax(1)
    class_indices = list(range(len(loader.dataset.genres)))
    return {
        "loss": total_loss / len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true, y_pred, labels=class_indices, average="macro", zero_division=0
            )
        ),
        "report": classification_report(
            y_true,
            y_pred,
            labels=class_indices,
            target_names=loader.dataset.genres,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=class_indices),
        "y_true": y_true,
        "y_pred": y_pred,
        "probabilities": probs,
        "embeddings": np.concatenate(embeddings),
        "inference_seconds": perf_counter() - started,
    }


def grad_cam(model, specs, class_index, device):
    """Explain one track's mean class probability using its final convolution.

    specs contains all segments of one track, shaped [segments, 1, mel, time].
    Return heatmaps [segments, mel, time] and segment class probabilities,
    both detached and moved to CPU.
    """
    activations, gradients = [], []

    def capture(module, inputs, output):
        """Capture convolution activations and register their gradient hook."""
        activations.append(output)
        output.register_hook(gradients.append)

    was_training = model.training
    model.eval()
    # The final convolution is followed by BatchNorm, ReLU and MaxPool.
    handle = model.features[-4].register_forward_hook(capture)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(specs.to(device))
        probabilities = logits.softmax(1)
        # Differentiate the same mean probability used for track predictions.
        probabilities.mean(0)[class_index].backward()
        weights = gradients[0].mean(dim=(-2, -1), keepdim=True)
        heatmaps = (weights * activations[0]).sum(1, keepdim=True).relu()
        heatmaps = F.interpolate(
            heatmaps, size=specs.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)
        minimum = heatmaps.amin(dim=(-2, -1), keepdim=True)
        maximum = heatmaps.amax(dim=(-2, -1), keepdim=True)
        heatmaps = (heatmaps - minimum) / (maximum - minimum).clamp_min(1e-8)
        return heatmaps.detach().cpu(), probabilities.detach().cpu()
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
        model.train(was_training)
