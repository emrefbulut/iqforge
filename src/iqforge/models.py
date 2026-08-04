"""Baseline sınıflandırıcı.

Amaç doğruluk rekoru değil: veri setinin gerçekten eğitilebilir olduğunu
kanıtlamak. Model bilerek küçük tutulur (50k parametrenin altında), böylece
yüksek doğruluk kapasiteden değil veriden gelir.
"""

from __future__ import annotations

import torch
from torch import nn

#: Modelin aşmaması gereken parametre bütçesi.
MAX_PARAMETERS = 50_000


class BaselineCNN(nn.Module):
    """Küçük 1B evrişimli sınıflandırıcı.

    Yapı:
        Conv1d(2->16, k=7)  - BN - ReLU - MaxPool2
        Conv1d(16->32, k=5) - BN - ReLU - MaxPool2
        Conv1d(32->64, k=5) - BN - ReLU - AdaptiveAvgPool1d(1)
        Dropout(0.3) - Linear(64->n)

    Sonda flatten yerine global ortalama havuzlama kullanılır. Flatten,
    parametre sayısını pencere uzunluğuna bağlar (1024 örnekte tek başına
    ~16k*n parametre) ve modeli pencere içindeki MUTLAK konuma duyarlı hale
    getirir. Global havuzlama ise pencere uzunluğundan bağımsızdır ve
    öteleme-değişmez bir özet üretir — burst konumu kayıttan kayda değiştiği
    için istenen davranış budur.

    Attributes:
        features: Evrişim yığını.
        head: Dropout + doğrusal sınıflandırıcı.
    """

    def __init__(self, in_channels: int = 2, num_classes: int = 2, dropout: float = 0.3) -> None:
        """Modeli kurar.

        Args:
            in_channels: Girdi kanal sayısı (iq2ch ve magphase için 2).
            num_classes: Sınıf sayısı.
            dropout: Sınıflandırıcı öncesi dropout oranı.
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
        """Girdiyi sınıf skorlarına çevirir.

        Args:
            x: `(batch, channels, window)` float32 tensör.

        Returns:
            `(batch, num_classes)` logit tensörü.
        """
        return self.head(self.features(x).squeeze(-1))


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Modelin parametre sayısını verir.

    Args:
        model: Sayılacak model.
        trainable_only: Yalnızca gradyan alan parametreler sayılsın mı.

    Returns:
        Parametre sayısı.
    """
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)
