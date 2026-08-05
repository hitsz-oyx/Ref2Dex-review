"""Pytest configuration: monkey-patch numpy for smplx/chumpy compatibility.

``smplx`` pulls in ``chumpy`` at import time, and chumpy uses ``np.bool``,
``np.int``, ``np.float`` etc. that have been removed in numpy >= 1.20. The
graspenv ships numpy 1.24, so we re-add the aliases that chumpy needs.
This file is loaded before any test module.
"""

from __future__ import annotations

import numpy as _np

# numpy >= 1.20 removed these aliases; chumpy still expects them.
for _name, _value in (
    ("bool", _np.bool_),
    ("int", _np.int_),
    ("float", _np.float_),
    ("complex", _np.complex_),
    ("object", _np.object_),
    ("unicode", _np.str_),
    ("str", _np.str_),
):
    if not hasattr(_np, _name):
        setattr(_np, _name, _value)

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _ensure_numpy_aliases() -> None:
    """Re-assert the aliases at the start of every test, in case a previous
    test triggered an import of numpy that already stripped them.
    """
    for _name, _value in (
        ("bool", _np.bool_),
        ("int", _np.int_),
        ("float", _np.float_),
        ("complex", _np.complex_),
        ("object", _np.object_),
        ("unicode", _np.str_),
        ("str", _np.str_),
    ):
        if not hasattr(_np, _name):
            setattr(_np, _name, _value)
