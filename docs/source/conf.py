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
# Copy the repository's notebooks into the source tree and execute them all
# during the build.

_here = Path(__file__).parent
_nb_dst = _here / 'notebooks'
# Rebuild the copy directory from scratch so a renamed/removed notebook can't
# leave a stale execution behind in an incremental build.
if _nb_dst.exists():
    shutil.rmtree(_nb_dst)
_nb_dst.mkdir(exist_ok=True)
for _nb in sorted((_here.parent.parent / 'notebooks').glob('*.ipynb')):
    shutil.copyfile(_nb, _nb_dst / _nb.name)

# Copy the README media images into the source tree so overview.md can embed
# them (the PNGs only; the history GIF is git-lfs and unused in the docs).
_media_dst = _here / '_media'
if _media_dst.exists():
    shutil.rmtree(_media_dst)
_media_dst.mkdir(exist_ok=True)
for _img in sorted((_here.parent / 'media').glob('*.png')):
    shutil.copyfile(_img, _media_dst / _img.name)

nb_execution_mode = 'force'
nb_execution_timeout = 900
nb_execution_raise_on_error = True
nb_merge_streams = True

# MyST: render $...$ / $$...$$ and amsmath environments in the markdown cells
# (without dollarmath the notebooks' inline LaTeX is shown verbatim).
myst_enable_extensions = ['dollarmath', 'amsmath']

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

html_theme = 'sphinx_book_theme'
html_static_path = []
html_title = f'AutStr {release}'
html_theme_options = {
    'repository_url': 'https://github.com/fariedabuzaid/AutStr',
    'repository_branch': 'main',
    'path_to_docs': 'docs/source',
    'use_repository_button': True,
    'use_issues_button': True,
    'use_download_button': True,
    'home_page_in_toc': True,
    'show_toc_level': 2,
    'show_navbar_depth': 1,
}
