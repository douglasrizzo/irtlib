"""The check that every function taking a response matrix makes of it."""

import numpy
import numpy.typing as npt

from .typing import FloatArray

_MATRIX_NDIM = 2


def response_matrix(responses: npt.ArrayLike) -> FloatArray:
  """Return a response matrix as a new float array, after checking its shape, its type and its values.

  The matrix has one row per examinee and one column per item, and holds 1 for a correct response, 0 for an incorrect
  one and, among floats, NaN for a missing one. Booleans and integers are read as 1 and 0. Any number of rows and
  columns is accepted, as are an examinee with no answers and an item nobody answered.

  Parameters
  ----------
  responses : array_like
      The response matrix, of Booleans, integers or floats.

  Returns
  -------
  numpy.ndarray
      A float copy of the responses, of the same shape.

  Raises
  ------
  TypeError
      If the responses are neither Booleans, integers nor floats.
  ValueError
      If the responses are not two-dimensional, or a response is not 0, 1 or, among floats, NaN; the message names
      the first such response by its position.
  """
  matrix = numpy.asarray(responses)
  # Strings and objects are no responses: None would mark a missing answer that NaN marks.
  if matrix.dtype.kind not in "biuf":
    msg = (
      f"responses holds {matrix.dtype} values; responses must be Booleans, the integers 0 and 1, or the floats 0.0 "
      "and 1.0 with NaN for a missing response"
    )
    raise TypeError(msg)
  if matrix.ndim != _MATRIX_NDIM:
    msg = f"responses must be a matrix, one row per examinee and one column per item, got shape {matrix.shape}"
    raise ValueError(msg)
  valid = (matrix == 0) | (matrix == 1) | numpy.isnan(matrix)
  if not valid.all():
    row, column = (int(index) for index in numpy.argwhere(~valid)[0])
    rule = (
      "an integer response must be 0 or 1"
      if matrix.dtype.kind in "iu"
      else "a response must be 0 or 1, or NaN where it is missing"
    )
    msg = f"responses[{row}, {column}] is {matrix[row, column].item()!r}; {rule}"
    raise ValueError(msg)
  return matrix.astype(float)
