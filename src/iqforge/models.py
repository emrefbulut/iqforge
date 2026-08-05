"""The baseline classifier.

The point is not a record accuracy: it is to show that the dataset is actually
trainable. The model is deliberately small (under 50k parameters) so that high
accuracy comes from the data rather than from capacity.
"""

from __future__ import annotations

from iqforge import _require_torch

_require_torch("the baseline model")

import torch  # noqa: E402
from torch import nn  # noqa: E402

#: Parameter budget the model must not exceed.
MAX_PARAMETERS = 50_000


class BaselineCNN(nn.Module):
    """A small 1-D convolutional classifier.

    Architecture:
        Conv1d(2->16, k=7)  - BN - ReLU - MaxPool2
        Conv1d(16->32, k=5) - BN - ReLU - MaxPool2
        Conv1d(32->64, k=5) - BN - ReLU - AdaptiveAvgPool1d(1)
        Dropout(0.3) - Linear(64->n)

    Global average pooling is used at the end rather than a flatten. A flatten
    ties the parameter count to the window length (about 16k*n parameters on its
    own for a 1024-sample window) and makes the model sensitive to ABSOLUTE
    position inside the window. Global pooling is independent of window length
    and produces a shift-invariant summary — which is what we want, since the
    burst position varies from recording to recording.

    Attributes:
        features: The convolutional stack.
        head: Dropout plus the linear classifier.
    """

    def __init__(self, in_channels: int = 2, num_classes: int = 2, dropout: float = 0.3) -> None:
        """Build the model.

        Args:
            in_channels: Number of input channels (2 for iq2ch and magphase).
            num_classes: Number of classes.
            dropout: Dropout rate before the classifier.
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map the input to class scores.

        Args:
            x: A `(batch, channels, window)` float32 tensor.

        Returns:
            A `(batch, num_classes)` tensor of logits.
        """
        return self.head(self.features(x).squeeze(-1))


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Return the model's parameter count.

    Args:
        model: The model to count.
        trainable_only: Count only parameters that require gradients.

    Returns:
        The number of parameters.
    """
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)
