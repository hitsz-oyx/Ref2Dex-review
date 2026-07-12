"""Static hand-object correspondence task.

The task kernel defines correspondence data, target, and runner semantics.

Experiment implementations live in:
- sampling/
- supervision/
- objectives/
- models/

Experiment recipes are resolved through composition.py.
"""

from .config import Config
from .runner import CorrespondencePTV3Runner

__all__ = [
    "Config",
    "CorrespondencePTV3Runner",
]
