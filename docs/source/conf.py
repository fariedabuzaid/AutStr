# Configuration file for the Sphinx documentation builder.
#
# The API reference is generated from the docstrings in `autstr/`; the docs
# workflow refreshes the stubs on every build with
#     sphinx-apidoc -f -o docs/source/api autstr
#
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from importlib.metadata import PackageNotFoundError, version as _version

project = 'AutStr'
copyright = '2022-2026, Faried Abu Zaid'
author = 'Faried Abu Zaid'

try:                                    # the installed package is the truth
    release = _version('autstr')
except PackageNotFoundError:            # building from an uninstalled tree
    release = '0.0.0'
version = '.'.join(release.split('.')[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx_autodoc_typehints',
]

templates_path = ['_templates']
exclude_patterns = []

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'member-order': 'bysource',
}
autodoc_typehints = 'description'
autoclass_content = 'both'

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = 'furo'
html_static_path = []
html_title = f'AutStr {release}'
