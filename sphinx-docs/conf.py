# Configuration file for the Sphinx documentation builder.

import os
import sys
from datetime import date

# Make the package importable for autodoc when building from a source checkout.
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'Prosemble'
author = 'Nana Abeka Otoo'
copyright = f'2022–{date.today().year}, {author}'

try:
    from prosemble import __version__ as release
except Exception:  # pragma: no cover - the package should always import
    release = 'unknown'
version = '.'.join(release.split('.')[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_copybutton',
    'sphinx_design',
]

templates_path = ['_templates']
exclude_patterns = ['_build', '_scripts', 'Thumbs.db', '.DS_Store']
root_doc = 'index'

# -- Docstrings ----------------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
# Attributes as fields, so that NamedTuple fields are not documented twice.
napoleon_use_ivar = True
# Escape trailing underscores (centroids_) in field names.
strip_signature_backslash = True
# A "Methods" section is a summary; the methods are documented by autodoc.
napoleon_custom_sections = [('Methods', 'params_style')]

# Docstrings list their references as numbered footnotes without citing
# them in the text; that is intended.
suppress_warnings = ['ref.footnote']

autosummary_generate = True
autosummary_imported_members = False
autodoc_member_order = 'groupwise'
autodoc_typehints = 'none'
autodoc_class_signature = 'mixed'
autodoc_default_options = {
    'show-inheritance': True,
}

# -- Cross-references ----------------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'jax': ('https://docs.jax.dev/en/latest/', None),
}

# -- HTML output ---------------------------------------------------------------
html_theme = 'pydata_sphinx_theme'
html_title = 'Prosemble'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_favicon = '_static/favicon.svg'
html_show_sourcelink = False
html_last_updated_fmt = '%d %b %Y'

html_theme_options = {
    'logo': {
        'text': 'Prosemble',
        'image_light': '_static/logo-light.svg',
        'image_dark': '_static/logo-dark.svg',
    },
    'icon_links': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/naotoo1/prosemble',
            'icon': 'fa-brands fa-github',
        },
        {
            'name': 'PyPI',
            'url': 'https://pypi.org/project/prosemble/',
            'icon': 'fa-brands fa-python',
        },
    ],
    'navbar_align': 'left',
    'navbar_end': ['theme-switcher', 'navbar-icon-links'],
    'header_links_before_dropdown': 7,
    'secondary_sidebar_items': ['page-toc', 'edit-this-page'],
    'show_toc_level': 2,
    'navigation_with_keys': False,
    'use_edit_page_button': True,
    'footer_start': ['copyright'],
    'footer_end': ['theme-version'],
    'pygments_light_style': 'friendly',
    'pygments_dark_style': 'github-dark',
}

html_context = {
    'github_user': 'naotoo1',
    'github_repo': 'prosemble',
    'github_version': 'main',
    'doc_path': 'sphinx-docs',
    'default_mode': 'auto',
}

# The landing page and section overviews use the full width.
html_sidebars = {
    'index': [],
    'getting_started/index': [],
    'tutorials/index': [],
}

# -- Copy button ---------------------------------------------------------------
copybutton_prompt_text = r'>>> |\.\.\. |\$ '
copybutton_prompt_is_regexp = True
