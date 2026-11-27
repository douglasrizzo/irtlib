r"""Drawing responses from the model, for simulation and parameter-recovery studies.

:py:func:`draw_responses` draws each response on its own: examinee :math:`j` answers item :math:`i` correctly when a
number drawn uniformly from :math:`[0, 1)` falls below :math:`P_i(\theta_j)`, the probability of a correct response
that :py:func:`irtlib.irt.icc` gives. Examinee :math:`j`'s row takes the generator's next ``n_items`` numbers after
those of the examinees before it.
"""

import numpy
import numpy.typing as npt

from . import irt
from .irt import _N_COLUMNS
from .typing import FloatArray


def draw_responses(
  theta: npt.ArrayLike, items: FloatArray, rng: int | numpy.random.Generator | None = None
) -> FloatArray:
  """Draw the responses of examinees of the given abilities to the given items, a response matrix.

  Each response is correct with the probability :py:func:`irtlib.irt.icc` gives, independently of the others. The
  numbers are drawn row by row, so drawing one examinee at a time from one generator gives the same rows.

  Parameters
  ----------
  theta : array_like
      One ability per examinee, a one-dimensional array of finite numbers.
  items : numpy.ndarray
      A matrix of items, one row ``[a, b, c, d]`` per item, shape ``(n_items, 4)``. Only its shape and the finiteness
      of its parameters are checked; :py:class:`irtlib.ItemBank` checks their ranges too.
  rng : int, numpy.random.Generator or None, optional
      Where the responses are drawn from. None, the default, draws fresh randomness, so every call gives other
      responses. An int seeds a new generator, so the same seed gives the same responses. A generator is drawn from as
      is, and the call advances it.

  Returns
  -------
  numpy.ndarray
      A float matrix of shape ``(len(theta), n_items)``, one row per examinee and one column per item, holding 1.0 for
      a correct response and 0.0 for an incorrect one.

  Raises
  ------
  TypeError
      If ``theta`` or ``items`` holds values that are not real numbers.
  ValueError
      If ``theta`` is not one-dimensional or holds a value that is not finite, or ``items`` is not a matrix of four
      columns of finite numbers.

  Examples
  --------
  >>> import numpy
  >>> items = numpy.array([[1.0, 0.0, 0.0, 1.0], [1.5, 1.0, 0.2, 1.0], [0.8, -1.0, 0.1, 0.95]])
  >>> draw_responses(numpy.array([-1.0, 0.0, 2.5]), items, rng=3)
  array([[1., 1., 0.],
         [0., 1., 1.],
         [1., 1., 1.]])
  """
  thetas = _abilities(theta)
  matrix = _item_matrix(items)
  draws = numpy.random.default_rng(rng).random((thetas.size, matrix.shape[0]))
  return (draws < irt.icc(thetas, matrix)).astype(float)


def _abilities(theta: npt.ArrayLike) -> FloatArray:
  """Return the abilities as a new float array, after checking that they are one finite real number per examinee."""
  thetas = numpy.asarray(theta)
  # Booleans would be read as abilities 0 and 1, and None as NaN.
  if thetas.dtype.kind not in "iuf":
    msg = f"theta holds {thetas.dtype} values; abilities must be real numbers"
    raise TypeError(msg)
  if thetas.ndim != 1:
    msg = f"theta must hold one ability per examinee, a one-dimensional array, got shape {thetas.shape}"
    raise ValueError(msg)
  thetas = thetas.astype(float)
  # The probability of a correct response at NaN is NaN, and every response drawn against it would be incorrect.
  if not numpy.isfinite(thetas).all():
    position = int(numpy.flatnonzero(~numpy.isfinite(thetas))[0])
    msg = f"theta[{position}] is {thetas[position].item()!r}; abilities must be finite numbers"
    raise ValueError(msg)
  return thetas


def _item_matrix(items: FloatArray) -> FloatArray:
  """Return the items as a float array, after checking that they are a matrix of four columns of finite real numbers."""
  matrix = numpy.asarray(items)
  if matrix.dtype.kind not in "iuf":
    msg = f"items holds {matrix.dtype} values; item parameters must be real numbers"
    raise TypeError(msg)
  matrix = matrix.astype(float)
  if matrix.shape[1:] != (_N_COLUMNS,):
    msg = f"items must be a matrix with one row [a, b, c, d] per item, shape (n_items, 4), got shape {matrix.shape}"
    raise ValueError(msg)
  if not numpy.isfinite(matrix).all():
    item = int(numpy.flatnonzero(~numpy.isfinite(matrix).all(axis=1))[0])
    msg = f"items[{item}] is {matrix[item].tolist()}; item parameters must be finite numbers"
    raise ValueError(msg)
  return matrix
