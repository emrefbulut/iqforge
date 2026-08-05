"""The baseline training loop.

This module needs `torch`; `cli.py` imports it only when the `train` command
actually runs.

The scope is deliberately narrow: no checkpointing, no learning-rate schedule,
no hyperparameter search. The goal is to show the dataset is trainable.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from iqforge.dataset import IQForgeDataset
from iqforge.io import IQForgeError
from iqforge.models import MAX_PARAMETERS, BaselineCNN, count_parameters


@dataclass
class EpochResult:
    """The result of a single epoch."""

    epoch: int
    train_loss: float
    train_accuracy: float
    val_accuracy: float | None


@dataclass
class TrainingResult:
    """Everything one training run produced.

    Attributes:
        parameters: The model's trainable parameter count.
        epochs: Per-epoch results.
        test_accuracy: Accuracy on the test split, or None if it is empty.
        test_per_class: Per-class accuracy on the test split.
        classes: Label names, in integer order.
    """

    parameters: int
    epochs: list[EpochResult] = field(default_factory=list)
    test_accuracy: float | None = None
    test_per_class: dict[str, float] = field(default_factory=dict)
    classes: list[str] = field(default_factory=list)

    @property
    def final_train_accuracy(self) -> float:
        """Training accuracy of the last epoch."""
        return self.epochs[-1].train_accuracy if self.epochs else 0.0


def seed_everything(seed: int) -> None:
    """Fix every RNG that affects weight initialisation and batch order.

    Separate from the split seed (`build --seed`): this one only affects
    training, never the contents of the dataset.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute accuracy over a loader."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            predictions = model(x.to(device)).argmax(dim=1).cpu()
            correct += int((predictions == y).sum())
            total += int(y.numel())
    return correct / total if total else 0.0


def _per_class_accuracy(
    model: nn.Module, loader: DataLoader, device: torch.device, classes: list[str]
) -> dict[str, float]:
    """Compute per-class accuracy."""
    model.eval()
    correct = np.zeros(len(classes), dtype=np.int64)
    total = np.zeros(len(classes), dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            predictions = model(x.to(device)).argmax(dim=1).cpu()
            for label in range(len(classes)):
                mask = y == label
                total[label] += int(mask.sum())
                correct[label] += int((predictions[mask] == label).sum())
    return {
        name: (float(correct[i] / total[i]) if total[i] else float("nan"))
        for i, name in enumerate(classes)
    }


def train_baseline(
    dataset_dir: str | Path,
    *,
    epochs: int = 10,
    batch_size: int = 64,
    seed: int = 0,
    learning_rate: float = 1e-3,
    on_epoch: Callable[[EpochResult], None] | None = None,
) -> TrainingResult:
    """Train the baseline CNN and measure test accuracy.

    Args:
        dataset_dir: Output of `iqforge build`.
        epochs: Number of epochs.
        batch_size: Batch size.
        seed: TRAINING seed (weight init + batch order). Separate from the split
            seed.
        learning_rate: Adam learning rate.
        on_epoch: Callback invoked at the end of every epoch.

    Returns:
        The results of the run.

    Raises:
        IQForgeError: If the dataset is in a form torch cannot use, or the model
            exceeds the parameter budget.
    """
    seed_everything(seed)
    device = torch.device("cpu")

    train_set = IQForgeDataset(dataset_dir, split="train")
    if train_set[0][0].is_complex():
        raise IQForgeError(
            "The baseline model does not accept complex input. Build the dataset "
            "with --repr iq2ch (the default) or --repr magphase."
        )

    def _loader(split: str, shuffle: bool) -> DataLoader | None:
        try:
            data = IQForgeDataset(dataset_dir, split=split)
        except IQForgeError:
            return None
        generator = torch.Generator().manual_seed(seed) if shuffle else None
        return DataLoader(data, batch_size=batch_size, shuffle=shuffle, generator=generator)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = _loader("val", False)
    test_loader = _loader("test", False)

    classes = train_set.classes
    model = BaselineCNN(in_channels=train_set[0][0].shape[0], num_classes=len(classes)).to(device)
    parameters = count_parameters(model)
    if parameters > MAX_PARAMETERS:
        raise IQForgeError(
            f"The baseline model has {parameters} parameters, budget is {MAX_PARAMETERS}. "
            "Reduce the channel counts in models.py."
        )

    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    result = TrainingResult(parameters=parameters, classes=classes)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimiser.step()

            running_loss += float(loss.detach()) * y.numel()
            correct += int((logits.argmax(dim=1) == y).sum())
            total += int(y.numel())

        epoch_result = EpochResult(
            epoch=epoch,
            train_loss=running_loss / total,
            train_accuracy=correct / total,
            val_accuracy=_accuracy(model, val_loader, device) if val_loader else None,
        )
        result.epochs.append(epoch_result)
        if on_epoch is not None:
            on_epoch(epoch_result)

    if test_loader is not None:
        result.test_accuracy = _accuracy(model, test_loader, device)
        result.test_per_class = _per_class_accuracy(model, test_loader, device, classes)
    return result
