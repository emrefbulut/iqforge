"""iqforge — SigMF kayıtlarından ML'e hazır veri setleri."""

from typing import TYPE_CHECKING, Any

from iqforge.io import Annotation, IQForgeError, Recording, load

__version__ = "0.1.0"

__all__ = ["Annotation", "IQForgeDataset", "IQForgeError", "Recording", "load", "__version__"]

if TYPE_CHECKING:
    from iqforge.dataset import IQForgeDataset


def __getattr__(name: str) -> Any:
    """`IQForgeDataset`'i tembel yükler.

    `torch` opsiyonel bir bağımlılıktır: `info`, `inspect`, `build` ve `stats`
    torch kurulu olmadan çalışmalı. Modül seviyesinde import edilseydi
    `import iqforge` torch'u zorunlu kılardı.
    """
    if name == "IQForgeDataset":
        from iqforge.dataset import IQForgeDataset

        return IQForgeDataset
    raise AttributeError(f"module 'iqforge' has no attribute '{name}'")
