# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Astropy documentation build configuration file.
#
# This file is execfile()d with the current dir set to its containing dir.
#
# Note that not all possible configuration values are present in this file.
#
# All configuration values have a default. Some values are defined in
# the global Astropy configuration which is loaded here before anything else.
# See astropy.sphinx.conf for which values are set there.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# sys.path.insert(0, os.path.abspath('..'))
# IMPORTANT: the above commented section was generated by sphinx-quickstart,
# but is *NOT* appropriate for astropy or Astropy affiliated packages. It is
# left commented out with this explanation to make it clear why this should not
# be done. If the sys.path entry above is added, when the astropy.sphinx.conf
# import occurs, it will import the *source* version of astropy instead of the
# version installed (if invoked as "make html" or directly with sphinx), or the
# version in the build directory (if "python setup.py build_sphinx" is used).
# Thus, any C-extensions that are needed to build the documentation will *not*
# be accessible, and the documentation will not build correctly.

# STDLIB
import datetime
import pathlib
import sys
from importlib import import_module

# THIRD PARTY
import tomlkit
from sphinx_astropy.conf.v1 import *  # type: ignore # noqa: F401, F403
from sphinx_astropy.conf.v1 import (
    exclude_patterns,
    extensions,
    numpydoc_xref_aliases,
    numpydoc_xref_astropy_aliases,
    numpydoc_xref_ignore,
    rst_epilog,
)

# Get configuration information from pyproject.toml
path = pathlib.Path(__file__).parent.parent / "pyproject.toml"
with path.open() as f:
    toml = tomlkit.load(f)
setup_cfg = dict(toml["project"])  # type: ignore

# -- General configuration ----------------------------------------------------

# By default, highlight as Python 3.
highlight_language = "python3"

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.2'

# To perform a Sphinx version check that needs to be more specific than
# major.minor, call `check_sphinx_version("x.y.z")` here.
# check_sphinx_version("1.2.1")

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns.append("_templates")

# This is added to the end of RST files - a good place to put substitutions to
# be used globally.
rst_epilog += """

..
  RST REPLACEMENT

.. TRACKSTREAM

.. |SOM| replace:: :class:`trackstream.track.som.SelfOrganizingMap1DBase`

.. ASTROPY

.. |Quantity| replace:: :class:`~astropy.units.Quantity`
.. |Angle| replace:: :class:`~astropy.coordinates.Angle`
.. |Latitude| replace:: :class:`~astropy.coordinates.Latitude`
.. |Longitude| replace:: :class:`~astropy.coordinates.Longitude`

.. |Representation| replace:: :class:`~astropy.coordinates.BaseRepresentation`
.. |CartesianRep| replace:: :class:`~astropy.coordinates.CartesianRepresentation`
.. |Frame| replace:: `~astropy.coordinates.BaseCoordinateFrame`
.. |ICRS| replace:: `~astropy.coordinates.ICRS`
.. |SkyCoord| replace:: :class:`~astropy.coordinates.SkyCoord`

.. |Table| replace:: :class:`~astropy.table.Table`
.. |QTable| replace:: :class:`~astropy.table.QTable`


.. MATPLOTLIB

.. |Pyplot| replace:: :mod:`~matplotlib.pyplot`
.. |Axes| replace:: :class:`~matplotlib.pyplot.Axes`
.. |Figure| replace:: :class:`~matplotlib.figure.Figure`
.. |Ellipse| replace:: :class:`matplotlib.patches.Ellipse`

.. NUMPY

.. |NDArray| replace:: :class:`~numpy.ndarray`

"""

# Whether to create cross-references for the parameter types in the
# Parameters, Other Parameters, Returns and Yields sections of the docstring.
numpydoc_xref_param_type = True

# Words not to cross-reference. Most likely, these are common words used in
# parameter type descriptions that may be confused for classes of the same
# name. The base set comes from sphinx-astropy. We add more here.
numpydoc_xref_ignore.update(
    {
        "mixin",
        "Any",  # aka something that would be annotated with `typing.Any`
    },
)

# Mappings to fully qualified paths (or correct ReST references) for the
# aliases/shortcuts used when specifying the types of parameters.
# Numpy provides some defaults
# https://github.com/numpy/numpydoc/blob/b352cd7635f2ea7748722f410a31f937d92545cc/numpydoc/xref.py#L62-L94
# and a base set comes from sphinx-astropy.
# so here we mostly need to define Astropy-specific x-refs
numpydoc_xref_aliases.update(
    {
        # python & adjacent
        "Any": "`~typing.Any`",
        "number": ":term:`number`",
        # for astropy
        "Unit": ":class:`~astropy.units.Unit`",
        "Quantity": ":class:`~astropy.units.Quantity`",
        "Representation": ":class:`~astropy.coordinates.BaseRepresentation`",
        "Differential": ":class:`~astropy.coordinates.BaseDifferential`",
        "CoordinateFrame": ":class:`~astropy.coordinates.BaseCoordinateFrame`",
    },
)
# Add from sphinx-astropy 1) glossary aliases 2) physical types.
numpydoc_xref_aliases.update(numpydoc_xref_astropy_aliases)


# extensions
extensions += [
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinxcontrib.bibtex",
]

# Show / hide TODO blocks
todo_include_todos = True


# Plot Configuration Options

plot_rcparams = {}
plot_rcparams["figure.figsize"] = (6, 6)
plot_rcparams["savefig.facecolor"] = "none"  # type: ignore
plot_rcparams["savefig.bbox"] = "tight"  # type: ignore
plot_rcparams["axes.labelsize"] = "large"  # type: ignore
plot_rcparams["figure.subplot.hspace"] = 0.5  # type: ignore

plot_apply_rcparams = True
plot_include_source = False
plot_html_show_source_link = True


# Bibtex
bibtex_bibfiles = "refs.bib"


# -- Project information ------------------------------------------------------

# This does not *have* to match the package name, but typically does
project = str(setup_cfg["name"])
author = ", ".join(d["name"] for d in setup_cfg["authors"])  # type: ignore
copyright = "{0}, {1}".format(datetime.datetime.now().year, author)

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.

import_module(project)
package = sys.modules[project]

# The short X.Y version.
version = package.__version__.split("-", 1)[0]
# The full version, including alpha/beta/rc tags.
release = package.__version__

# -- Options for the module index ---------------------------------------------

# Prefixes that are ignored for sorting the Python module index
# Currently only works on the html output
modindex_common_prefix = [f"{project}."]

# -- Options for HTML output --------------------------------------------------

# A NOTE ON HTML THEMES
# The global astropy configuration uses a custom theme, 'bootstrap-astropy',
# which is installed along with astropy. A different theme can be used or
# the options for this theme can be modified by overriding some of the
# variables set in the global configuration. The variables set in the
# global configuration are listed below, commented out.


# Add any paths that contain custom themes here, relative to this directory.
# To use a different custom theme, add the directory containing the theme.
# html_theme_path = []

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes. To override the custom theme, set this to the
# name of a builtin theme or the name of a custom theme in html_theme_path.
# html_theme = None


html_theme_options = {
    "logotext1": "Track-Stream",  # white,  semi-bold
    "logotext2": "",  # orange, light
    "logotext3": ":docs",  # white,  light
}


# Custom sidebar templates, maps document names to template names.
# html_sidebars = {}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = ''

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = ''

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
# html_last_updated_fmt = ''

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = "{0} v{1}".format(project, release)

# Output file base name for HTML help builder.
htmlhelp_basename = project + "doc"

# custom css
# html_static_path = ["_static"]
# html_style = "trackstream.css"


# -- Options for LaTeX output -------------------------------------------------

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual])
latex_documents = [
    ("index", project + ".tex", project + " Documentation", author, "manual"),
]


# -- Options for manual page output -------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ("index", project.lower(), project + " Documentation", [author], 1),
]


# -- Resolving issue number to links in changelog -----------------------------

github_issues_url = setup_cfg["urls"]["repository"] + "/issues/"  # type: ignore

# -- Turn on nitpicky mode for sphinx (to warn about references not found) ----
#
# nitpicky = True
# nitpick_ignore = []
#
# Some warnings are impossible to suppress, and you can list specific
# references that should be ignored in a nitpick-exceptions file which should
# be inside the docs/ directory. The format of the file should be:
#
# <type> <class>
#
# for example:
#
# py:class astropy.io.votable.tree.Element
# py:class astropy.io.votable.tree.SimpleElement
# py:class astropy.io.votable.tree.SimpleElementWithContent
#
# Uncomment the following lines to enable the exceptions:
#
# for line in open('nitpick-exceptions'):
#     if line.strip() == "" or line.startswith("#"):
#         continue
#     dtype, target = line.split(None, 1)
#     target = target.strip()
#     nitpick_ignore.append((dtype, six.u(target)))
