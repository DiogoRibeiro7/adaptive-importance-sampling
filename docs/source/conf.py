"""Sphinx configuration file for Safe-ICE documentation."""

import importlib.metadata
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath('../..'))

# Project information
project = 'Safe-ICE'
copyright = f'{datetime.now().year}, Diogo Ribeiro'
author = 'Diogo Ribeiro'

# Read the version from the installed package so it is stated once, in
# pyproject.toml, rather than duplicated here.
try:
    release = importlib.metadata.version('safe-ice')
except importlib.metadata.PackageNotFoundError:
    release = '0.0.0.dev0'
version = release

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx_rtd_theme',
    'sphinx.ext.todo',
]

# Add any paths that contain templates here
templates_path = ['_templates']

# List of patterns to exclude
exclude_patterns = []

# HTML theme
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
    'prev_next_buttons_location': 'both',
}

# Static files
# No _static assets are shipped; leaving this empty avoids a
# build warning about a missing directory.
html_static_path = []

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'undoc-members': True,
    'show-inheritance': True,
}

# Napoleon settings for NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
# Render 'Attributes' sections as :ivar: fields inside the class body rather
# than as separate attribute objects. Otherwise a dataclass that documents
# its fields in the docstring registers each one twice, once from autodoc and
# once from napoleon, and Sphinx warns about duplicate descriptions.
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

# LaTeX output configuration
latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '11pt',
}

# Autosummary
autosummary_generate = True

# Todo extension
todo_include_todos = True