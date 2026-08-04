"""Baseline eğitim döngüsü.

Bu modül `torch` gerektirir; `cli.py` yalnızca `train` komutu çağrıldığında
import eder.

Kapsam bilerek dar: checkpoint yönetimi, learning-rate zamanlaması,
hyperparameter arama yok. Amaç veri setinin eğitilebilir olduğunu göstermek.
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
    """Tek bir epoch'un sonucu."""

    epoch: int
    train_loss: float
    train_accuracy: float
    val_accuracy: float | None


@dataclass
class TrainingResult:
    """Bir eğitim koşusunun tüm sonuçları.

    Attributes:
        parameters: Modelin eğitilebilir parametre sayısı.
        epochs: Epoch bazında sonuçlar.
        test_accuracy: Test split'i doğruluğu; split boşsa None.
        test_per_class: Test split'inde sınıf bazında doğruluk.
        classes: Etiket adları, tamsayı sırasına göre.
    """

    parameters: int
    epochs: list[EpochResult] = field(default_factory=list)
    test_accuracy: float | None = None
    test_per_class: dict[str, float] = field(default_factory=dict)
    classes: list[str] = field(default_factory=list)

    @property
    def final_train_accuracy(self) -> float:
        """Son epoch'un eğitim doğruluğu."""
        return self.epochs[-1].train_accuracy if self.epochs else 0.0


def seed_everything(seed: int) -> None:
    """Ağırlık ilklendirme ve batch sırası için tüm RNG'leri sabitler.

    Bölme tohumundan (`build --seed`) ayrıdır: bu tohum yalnızca eğitimi
    etkiler, veri setinin içeriğini değil.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Bir yükleyici üzerinde doğruluk hesaplar."""
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
    """Sınıf bazında doğruluk hesaplar."""
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
    """Baseline CNN'i eğitir ve test doğruluğunu ölçer.

    Args:
        dataset_dir: `iqforge build` çıktısı.
        epochs: Epoch sayısı.
        batch_size: Batch boyutu.
        seed: EĞİTİM tohumu (ağırlık init + batch sırası). Bölme tohumundan ayrı.
        learning_rate: Adam öğrenme oranı.
        on_epoch: Her epoch sonunda çağrılan geri çağırım.

    Returns:
        Koşunun sonuçları.

    Raises:
        IQForgeError: Veri seti torch ile kullanılamaz biçimdeyse veya model
            parametre bütçesini aşıyorsa.
    """
    seed_everything(seed)
    device = torch.device("cpu")

    train_set = IQForgeDataset(dataset_dir, split="train")
    if train_set[0][0].is_complex():
        raise IQForgeError(
            "Baseline model kompleks girdiyle çalışmaz. Veri setini "
            "--repr iq2ch (varsayılan) veya --repr magphase ile kurun."
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
            f"Baseline model {parameters} parametre içeriyor, bütçe {MAX_PARAMETERS}. "
            "models.py içindeki kanal sayılarını küçültün."
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
