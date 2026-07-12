"""Static hand-object correspondence task.

The task kernel defines correspondence data, target, and runner semantics.

Experiment implementations live in:
- sampling/
- supervision/
- objectives.py
- models/
"""

from .config_loader import load_correspondence_config
from .runner import CorrespondencePTV3Runner

__all__ = [
    "CorrespondencePTV3Runner",
    "load_correspondence_config",
]
