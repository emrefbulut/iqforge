"""Tests for iqforge.models. Skipped when torch is not installed."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="the baseline model needs torch")

from iqforge.models import MAX_PARAMETERS, BaselineCNN, count_parameters  # noqa: E402


def test_model_fits_the_parameter_budget() -> None:
    """The baseline is deliberately small: accuracy must come from the data."""
    parameters = count_parameters(BaselineCNN())

    assert parameters == 13_490
    assert parameters < MAX_PARAMETERS


def test_output_shape_matches_class_count() -> None:
    """The output is `(batch, num_classes)`."""
    model = BaselineCNN(num_classes=3)

    assert model(torch.zeros(5, 2, 1024)).shape == (5, 3)


@pytest.mark.parametrize("window", [256, 1024, 4096])
def test_global_pooling_makes_the_model_window_agnostic(window: int) -> None:
    """Global average pooling lets the model run at any window length.

    With a flatten, the linear layer would be tied to the window length and a
    different input size would fail.
    """
    model = BaselineCNN()

    assert model(torch.zeros(2, 2, window)).shape == (2, 2)


def test_parameter_count_does_not_depend_on_window_length() -> None:
    """The parameter count is independent of window length (no flatten)."""
    model = BaselineCNN()
    before = count_parameters(model)
    model(torch.zeros(1, 2, 8192))

    assert count_parameters(model) == before


def test_gradients_reach_the_first_layer() -> None:
    """Backpropagation must reach the first convolution."""
    model = BaselineCNN()
    logits = model(torch.randn(4, 2, 512))
    logits.sum().backward()

    first = model.features[0].weight
    assert first.grad is not None
    assert torch.any(first.grad != 0)


def test_count_parameters_can_include_frozen_weights() -> None:
    """Frozen parameters are counted only when asked for."""
    model = BaselineCNN()
    for param in model.features.parameters():
        param.requires_grad = False

    assert count_parameters(model, trainable_only=True) == 130  # only Linear(64->2)
    assert count_parameters(model, trainable_only=False) == 13_490


def test_device_defaults_to_cpu_and_is_recorded(tmp_path):
    """The CPU default is a reproducibility promise, not an accident."""
    from iqforge.training import DEFAULT_DEVICE, describe_environment, resolve_device

    assert DEFAULT_DEVICE == "cpu"
    device = resolve_device()
    assert device.type == "cpu"
    env = describe_environment(device)
    assert env["device"] == "cpu"
    assert env["torch"]
    assert "cuda" in env
    assert env["numpy"]
    assert env["scipy"]
    assert env["sigmf"]
    assert env["numpy"] != "absent"
    assert env["scipy"] != "absent"
    assert env["sigmf"] != "absent"


def test_an_unavailable_device_errors_rather_than_falling_back():
    """A run that silently used another device is worse than one that stops."""
    import torch

    from iqforge.io import IQForgeError
    from iqforge.training import resolve_device

    if torch.cuda.is_available():
        pytest.skip("CUDA is available here, so the refusal cannot be exercised")
    with pytest.raises(IQForgeError, match="no CUDA device"):
        resolve_device("cuda")


def test_an_unknown_device_name_is_rejected():
    from iqforge.io import IQForgeError
    from iqforge.training import resolve_device

    with pytest.raises(IQForgeError, match="must be auto, cpu or cuda"):
        resolve_device("gpu")


def test_a_sweep_refuses_to_extend_a_checkpoint_from_another_device(tmp_path):
    """Rows measured on different environments are not comparable, and no table shows it."""
    import json
    import sys

    sys.path.insert(0, "scripts")
    from leakage_experiment import check_environment, current_environment

    same = tmp_path / "same.json"
    same.write_text(json.dumps([{"environment": current_environment()}]), encoding="utf-8")
    check_environment(same)  # must not raise

    other = tmp_path / "other.json"
    other.write_text(
        json.dumps([{"environment": {"device": "cuda", "torch": "9.9", "cuda": "12.4"}}]),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="not comparable"):
        check_environment(other)

    check_environment(tmp_path / "absent.json")  # nothing to compare against
