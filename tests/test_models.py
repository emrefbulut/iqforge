"""iqforge.models testleri. torch kurulu değilse atlanır."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="baseline model için torch gerekli")

from iqforge.models import MAX_PARAMETERS, BaselineCNN, count_parameters  # noqa: E402


def test_model_fits_the_parameter_budget() -> None:
    """Baseline bilerek küçük: yüksek doğruluk kapasiteden değil veriden gelmeli."""
    parameters = count_parameters(BaselineCNN())

    assert parameters == 13_490
    assert parameters < MAX_PARAMETERS


def test_output_shape_matches_class_count() -> None:
    """Çıktı `(batch, num_classes)` olmalı."""
    model = BaselineCNN(num_classes=3)

    assert model(torch.zeros(5, 2, 1024)).shape == (5, 3)


@pytest.mark.parametrize("window", [256, 1024, 4096])
def test_global_pooling_makes_the_model_window_agnostic(window: int) -> None:
    """Global ortalama havuzlama sayesinde pencere uzunluğu değişse de model çalışmalı.

    Flatten kullanılsaydı doğrusal katman pencere uzunluğuna bağlanır ve farklı
    uzunlukta girdi hata verirdi.
    """
    model = BaselineCNN()

    assert model(torch.zeros(2, 2, window)).shape == (2, 2)


def test_parameter_count_does_not_depend_on_window_length() -> None:
    """Parametre sayısı pencere uzunluğundan bağımsız olmalı (flatten yok)."""
    model = BaselineCNN()
    before = count_parameters(model)
    model(torch.zeros(1, 2, 8192))

    assert count_parameters(model) == before


def test_gradients_reach_the_first_layer() -> None:
    """Geri yayılım ilk evrişime kadar ulaşmalı."""
    model = BaselineCNN()
    logits = model(torch.randn(4, 2, 512))
    logits.sum().backward()

    first = model.features[0].weight
    assert first.grad is not None
    assert torch.any(first.grad != 0)


def test_count_parameters_can_include_frozen_weights() -> None:
    """Dondurulmuş parametreler yalnızca istenirse sayılmalı."""
    model = BaselineCNN()
    for param in model.features.parameters():
        param.requires_grad = False

    assert count_parameters(model, trainable_only=True) == 130  # yalnızca Linear(64->2)
    assert count_parameters(model, trainable_only=False) == 13_490
