"""Tests for iqforge.models. Skipped when torch is not installed."""

from __future__ import annotations

import numpy as np
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


def _pretend_cuda_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """A GPU is present, without requiring one on this machine."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args, **kwargs: "mocked-gpu")


def test_default_stays_cpu_even_when_cuda_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """A GPU being present is not an opt-in. The default must stay CPU."""
    from iqforge.training import DEFAULT_DEVICE, resolve_device

    _pretend_cuda_is_available(monkeypatch)
    assert DEFAULT_DEVICE == "cpu"
    assert resolve_device().type == "cpu"
    assert resolve_device(DEFAULT_DEVICE).type == "cpu"


def test_opt_in_cuda_is_requested_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--device cuda` is the opt-in; it must actually select CUDA."""
    from iqforge.training import resolve_device

    _pretend_cuda_is_available(monkeypatch)
    assert resolve_device("cuda").type == "cuda"


def test_environment_stamps_the_resolved_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """cpu vs cuda must be distinguishable later from the results file alone."""
    from iqforge.training import describe_environment, resolve_device

    _pretend_cuda_is_available(monkeypatch)
    cpu = describe_environment(resolve_device("cpu"))
    cuda = describe_environment(resolve_device("cuda"))
    assert cpu["device"] == "cpu"
    assert cuda["device"] == "cuda"
    assert cuda["device_name"] == "mocked-gpu"


def test_cpu_flag_must_not_silently_use_cuda(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If training started using CUDA whenever a GPU exists, this must go red.

    The failure mode is not an explicit `--device cuda`. It is the default, or
    an explicit `--device cpu`, landing on a GPU because one was present.
    """
    from iqforge.storage import ShardWriter, write_manifest
    from iqforge.training import train_baseline

    _pretend_cuda_is_available(monkeypatch)

    window = 32
    splits: dict[str, dict] = {}
    for split, rows in (("train", 8), ("val", 0), ("test", 4)):
        writer = ShardWriter(tmp_path, split)
        for i in range(rows):
            writer.add(np.full((1, 2, window), float(i), dtype=np.float32), [i % 2])
        writer.flush()
        splits[split] = {
            "shards": writer.shards,
            "labels": writer.labels,
            "count": writer.count,
            "records": [],
        }
    write_manifest(
        tmp_path,
        version="0.1.0",
        config={"window": window, "stride": 8, "repr": "iq2ch", "normalize": True, "seed": 42},
        label_map={"a": 0, "b": 1},
        source_files=[],
        splits=splits,
    )

    destinations: list[str] = []
    original_to = torch.nn.Module.to

    def tracking_to(self, *args, **kwargs):
        target = args[0] if args else kwargs.get("device")
        if isinstance(target, (str, torch.device)):
            destinations.append(torch.device(target).type)
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.nn.Module, "to", tracking_to)

    result = train_baseline(tmp_path, epochs=1, batch_size=4, device_choice="cpu")
    assert result.environment["device"] == "cpu"
    assert destinations
    assert all(kind == "cpu" for kind in destinations)
    assert "cuda" not in destinations


def test_an_unavailable_device_errors_rather_than_falling_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run that silently used another device is worse than one that stops."""
    from iqforge.io import IQForgeError
    from iqforge.training import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
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

    from iqforge.measurement import check_environment, current_environment

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
