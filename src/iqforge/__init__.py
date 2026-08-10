"""iqforge - turn SigMF recordings into ML-ready datasets."""

from typing import TYPE_CHECKING, Any

from iqforge.io import Annotation, IQForgeError, Recording, load

__version__ = "0.2.0"

#: Message shown when an optional-torch entry point is reached without torch.
#: Kept here so the CLI and the library say the same thing; `{what}` names the
#: thing that needs it. This module never imports torch, so importing the
#: constant is always safe.
TORCH_REQUIRED = (
    "torch is required for {what}. Install it with `uv sync --extra torch` "
    "or `pip install 'iqforge[torch]'`."
)

__all__ = [
    "Annotation",
    "IQForgeDataset",
    "IQForgeError",
    "Recording",
    "TORCH_REQUIRED",
    "load",
    "__version__",
]

if TYPE_CHECKING:
    from iqforge.dataset import IQForgeDataset


def __getattr__(name: str) -> Any:
    """Load `IQForgeDataset` lazily.

    `torch` is an optional dependency: `info`, `inspect`, `build` and `stats`
    must work without it. Importing at module level would make `import iqforge`
    require torch.
    """
    if name == "IQForgeDataset":
        from iqforge.dataset import IQForgeDataset

        return IQForgeDataset
    raise AttributeError(f"module 'iqforge' has no attribute '{name}'")


def _require_torch(what: str) -> None:
    """Raise a readable error when torch is missing.

    Called from the modules that need torch, so `import iqforge.dataset`
    without torch explains what to install instead of reporting a bare
    `ModuleNotFoundError: No module named 'torch'`.
    """
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(TORCH_REQUIRED.format(what=what)) from exc
