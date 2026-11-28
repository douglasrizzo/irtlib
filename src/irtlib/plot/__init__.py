"""Plots of item response quantities, drawn with matplotlib.

The module needs matplotlib, which the plot extra installs: ``pip install 'irtlib[plot]'``. Each function takes its
data first, as irtlib's own objects and arrays, and its options as keyword-only arguments, draws on the ``ax`` it is
given or on the axes of a new figure, and returns the axes it drew on. Each draws on one axes with one y scale: views
of two quantities, such as an item's characteristic and information curves, are two calls on axes the caller lays
out. A plot of one series takes a ``label``, so that several drawn on one axes compare. Plots follow the caller's
matplotlib style, set with ``plt.style.use`` or ``plt.style.context``. :doc:`/adr/0011-plotting` gives the reasons.

Items are labelled by their names in an :py:class:`irtlib.ItemBank`; an item matrix names them by position, as a bank
built from it does.
"""

# The guard runs before the private modules import matplotlib, so it must sit here.
try:  # noqa: RUF067
  import matplotlib as _matplotlib  # noqa: F401 - the import that tells a user without the extra what to install
except ImportError as error:
  _missing = "irtlib.plot needs matplotlib; install it with: pip install 'irtlib[plot]'"
  raise ImportError(_missing) from error

from ._items import Curve, item_curve

__all__ = ["Curve", "item_curve"]
