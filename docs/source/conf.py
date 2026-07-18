# Configuration file for the Sphinx documentation builder.
#
# The API reference is generated from the docstrings in `autstr/`; the docs
# workflow refreshes the stubs on every build with
#     sphinx-apidoc -f -o docs/source/api autstr
#
# The showcase notebooks live output-free in `notebooks/` at the repository
# root; they are copied into the source tree below and EXECUTED during the
# build (myst-nb), so the published pages carry fresh outputs. Executing them
# needs the package importable and the graphviz `dot` binary on the PATH.
#
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import shutil
from importlib.metadata import PackageNotFoundError, version as _version
from pathlib import Path

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
    'myst_nb',
]

# -- Notebooks ----------------------------------------------------------------
# Copy the repository's notebooks into the source tree (docnames must be
# ASCII, so `büchi` becomes `buechi`) and execute them all during the build.

_here = Path(__file__).parent
_nb_dst = _here / 'notebooks'
_nb_dst.mkdir(exist_ok=True)
for _nb in sorted((_here.parent.parent / 'notebooks').glob('*.ipynb')):
    shutil.copyfile(_nb, _nb_dst / _nb.name.replace('ü', 'ue'))

nb_execution_mode = 'force'
nb_execution_timeout = 900
nb_execution_raise_on_error = True
nb_merge_streams = True

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
