"""Sphinx configuration for the irtlib documentation."""

import datetime
import enum
import importlib
import pathlib
import pkgutil
import re
import sys
from importlib.metadata import version as get_version

from matplotlib import rc_params_from_file

# Sphinx runs this file from the sphinx/ directory.
sys.path.insert(0, str(pathlib.Path("../src").resolve()))

import irtlib

# -- General configuration ------------------------------------------------

extensions = [
  "sphinx.ext.autodoc",
  "sphinx.ext.githubpages",
  "sphinx.ext.intersphinx",
  "sphinx.ext.mathjax",
  "sphinx.ext.viewcode",
  "numpydoc",
  "myst_parser",
  "sphinxcontrib.bibtex",
  "sphinx_design",
  "matplotlib.sphinxext.plot_directive",
]

# The plot directive embeds the first format in the page, so figures are vector SVG, sharp on any screen. Every
# figure takes the documentation's style, _static/figures.mplstyle. The tight bounding box keeps labels that sit
# outside the axes inside the image. The links to other formats and to the figure's source are left out.
plot_formats = [("svg", 100)]
plot_rcparams = {
  **rc_params_from_file("_static/figures.mplstyle", use_default_template=False),
  "savefig.bbox": "tight",
}
plot_html_show_formats = False
plot_html_show_source_link = False

myst_enable_extensions = [
  "colon_fence",
  "deflist",
  "dollarmath",
  "fieldlist",
]

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "alpha"
bibtex_reference_style = "author_year"
# A page that lists its own sources shares entries with other pages and with the global Bibliography
# page. The alpha style gives an entry the same label on every page.
suppress_warnings = ["bibtex.duplicate_citation"]

# Each inventory comes from its project's site, or from the copy stored in _inventories/ when the
# site cannot be reached, so that the strict build does not fail on a network timeout.
# CONTRIBUTING.md ("Building the documentation") gives the command that refreshes the copies.
# A site that stalls rather than refusing the connection falls back to the copy after this many seconds.
intersphinx_timeout = 10
intersphinx_mapping = {
  name: (url, (None, f"_inventories/{name}.inv"))
  for name, url in {
    "python": "https://docs.python.org/3/",
    "numpy": "https://numpy.org/doc/stable/",
    "scipy": "https://docs.scipy.org/doc/scipy/",
    "matplotlib": "https://matplotlib.org/stable/",
  }.items()
}

source_suffix = {
  ".md": "markdown",
  ".rst": "restructuredtext",
}

master_doc = "index"

project = "irtlib"
copyright = f"2026-{datetime.datetime.now(tz=datetime.UTC).year}, Douglas De Rizzo Meneghetti"  # noqa: A001
author = "Douglas De Rizzo Meneghetti"

# |version| and |release| come from the installed package's metadata.
version = get_version("irtlib")
release = version

language = "en"

# AGENTS.md and CLAUDE.md are instructions for agents and contributors, and the templates are copied,
# not read: none of them is a page.
exclude_patterns = ["_build", "AGENTS.md", "CLAUDE.md", "**/_template.md"]

pygments_style = "sphinx"

numfig = True
numfig_secnum_depth = 0

# -- Options for HTML output ----------------------------------------------

html_theme = "pydata_sphinx_theme"

html_theme_options = {
  "github_url": "https://github.com/douglasrizzo/irtlib",
  "show_toc_level": 2,
  "navigation_depth": 2,
  "collapse_navigation": True,
  "show_nav_level": 2,
  # The navbar shows every top-level entry of the table of contents without a "More" menu.
  "header_links_before_dropdown": 7,
  "navbar_end": ["navbar-icon-links", "theme-switcher"],
  "logo": {
    "image_light": "logo/horizontal-light.svg",
    "image_dark": "logo/horizontal-dark.svg",
  },
}

# The favicon is one file shown on the browser's own tab bar, whatever the page's color mode, so it
# uses the tile version of the mark, which carries its own background.
html_favicon = "logo/favicon.svg"

html_static_path = ["_static"]
# irtlib's colors on the theme's, for the light and the dark mode.
html_css_files = ["irtlib.css"]

# -- Options for the linkcheck builder ------------------------------------

# Hosts that answer automated clients with 403 Forbidden, although their pages open in a browser.
# A DOI link to one of them is still checked: doi.org must redirect to it, which a DOI that does not
# exist does not, and linkcheck reports the redirect as ignored instead of following it. A direct
# link to one of these hosts is not checked.
_REFUSE_AUTOMATED_CLIENTS = [
  r"https?://journals\.sagepub\.com/",
  r"https?://(bpspsychub\.)?onlinelibrary\.wiley\.com/",
  r"https?://www\.tandfonline\.com/",
  r"https?://www\.jstor\.org/",
  r"https?://www\.researchgate\.net/",
  r"https?://ieeexplore\.ieee\.org/",
]
linkcheck_ignore = _REFUSE_AUTOMATED_CLIENTS
# GitHub draws its anchors, such as #readme, with JavaScript, which linkcheck does not run.
linkcheck_anchors_ignore_for_url = [r"https://github\.com/"]

numpydoc_show_class_members = False

# Every cross-reference must resolve (sphinx-build -n). NumPy's array aliases, which irtlib.typing's
# aliases stand for, evaluate to classes under private module paths that NumPy's inventory does not list.
nitpicky = True
nitpick_ignore = [
  ("py:class", "numpy._typing._array_like.NDArray"),
  ("py:class", "numpy._typing._array_like.ArrayLike"),
]


def _own(defined_in: str, module: str) -> bool:
  """Return whether an object defined in ``defined_in`` is ``module``'s own: defined there or in a private submodule.

  A package such as irtlib.plot defines its functions in private submodules and documents them at its own path.

  Returns
  -------
  bool
      Whether the object belongs to ``module``'s public API.
  """
  return defined_in == module or defined_in.startswith(f"{module}._")


def _unvalidated() -> set[str]:
  """Name the objects whose docstrings numpydoc's validation skips, as regular expressions.

  Autodoc reads the docstrings of the names a module binds to objects defined elsewhere, such as its
  imports, before it leaves them off the page, and numpydoc reads the call signature of an Enum's
  metaclass as the enum's parameters.

  Returns
  -------
  set of str
      One pattern per public name a module of irtlib binds to an object defined in another module,
      and one per enum, wherever it is documented.
  """
  patterns = set()
  for info in pkgutil.iter_modules(irtlib.__path__, "irtlib."):
    module = importlib.import_module(info.name)
    for name, obj in vars(module).items():
      if isinstance(obj, type) and issubclass(obj, enum.Enum):
        patterns.add(rf"^irtlib(\.\w+)?\.{name}$")
      elif not name.startswith("_") and not _own(getattr(obj, "__module__", info.name), info.name):
        patterns.add(rf"^{re.escape(info.name)}\.{name}$")
  return patterns


# The build fails for a public callable whose docstring leaves out a parameter (numpydoc's PR01).
# Ruff's DOC rules check the sections a docstring has against the code, not that they exist. A class
# documents its constructor's parameters; autodoc renders the class docstring alone, so __init__ is
# never checked.
numpydoc_validation_checks = {"PR01"}
numpydoc_validation_exclude = _unvalidated()
