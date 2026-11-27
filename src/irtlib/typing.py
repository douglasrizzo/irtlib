"""Type aliases for NumPy arrays, used in irtlib's public signatures.

Code that builds on irtlib can annotate its arrays the way irtlib does.
"""

from typing import Any

import numpy
import numpy.typing as npt

FloatArray = npt.NDArray[numpy.floating[Any]]
"""A NumPy array of floating-point values of any precision."""

IntArray = npt.NDArray[numpy.integer[Any]]
"""A NumPy array of integers of any precision, such as item indices or parameter counts."""

BoolArray = npt.NDArray[numpy.bool_]
"""A NumPy array of Booleans, such as a mask of examinees."""
