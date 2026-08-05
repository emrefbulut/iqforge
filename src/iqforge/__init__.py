"""iqforge - turn SigMF recordings into ML-ready datasets."""

from typing import TYPE_CHECKING, Any

from iqforge.io import Annotation, IQForgeError, Recording, load

__version__ = "0.1.0"

__all__ = ["Annotation", "IQForgeDataset", "IQForgeError", "Recording", "load", "__version__"]

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
