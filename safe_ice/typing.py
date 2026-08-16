# safe_ice/typing.py
"""Shared type aliases for the Safe-ICE public API.

Keeping these in one place means the limit-state signature is written once and
stays consistent across the core algorithm, the benchmark problems and the
analysis helpers.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

__all__ = ["LimitStateFunction", "NDArrayF", "RNGLike", "SeedLike"]

#: The float64 ndarray used throughout the package.
NDArrayF = npt.NDArray[np.float64]

#: A limit-state function g(u): failure occurs where g(u) <= 0.
#:
#: Implementations receive an ``(n, d)`` batch of points and should return the
#: matching ``(n,)`` array. Scalar-only functions are also accepted; callers
#: fall back to evaluating row by row when a batch call does not work.
LimitStateFunction = Callable[[NDArrayF], "float | NDArrayF"]

#: Random generators accepted by the sampling helpers.
RNGLike = np.random.Generator | np.random.RandomState

#: Anything accepted by a ``random_state`` argument.
SeedLike = int | np.random.Generator | None
