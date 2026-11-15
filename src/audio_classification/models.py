import torch
from torch import nn


class AudioCNN(nn.Module):
    """Classify log-Mel segments with a four-block convolutional network.

    Inputs have shape [segments, 1, mel_bins, time_frames]. The embedding
    layer also exposes features for track-level similarity search.
    """

    def __init__(self, num_classes: int = 8, embedding_dim: int = 512):
        super().__init__()
        layers = []
        in_channels = 1
        for out_channels in (32, 64, 128, 256):
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                ]
            )
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Linear(256, embedding_dim)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize each segment and return [segments, embedding_dim] features."""
        mean = x.mean(dim=(-2, -1), keepdim=True)
        # Keep silent or constant segments finite during normalization.
        std = x.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-6)
        x = self.features((x - mean) / std)
        return self.embedding(self.pool(x).flatten(1)).relu()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return unnormalized class scores for each input segment."""
        return self.classifier(self.dropout(self.extract_embeddings(x)))
