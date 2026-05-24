# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add project root to path for autodoc
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'Prosemble'
copyright = '2024-2026, Nana Abeka Otoo'
author = 'Nana Abeka Otoo'
release = '2.3.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_copybutton',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Napoleon settings (Google-style docstrings) ----------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- Autodoc settings -------------------------------------------------------
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autosummary_generate = True

# -- Intersphinx mapping ----------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'
html_static_path = ['_static']
html_title = 'Prosemble'
html_theme_options = {
    'light_css_variables': {
        'color-brand-primary': '#1a1a2e',
        'color-brand-content': '#16213e',
    },
    'dark_css_variables': {
        'color-brand-primary': '#7c83ff',
        'color-brand-content': '#7c83ff',
    },
}

# -- Copy button settings ---------------------------------------------------
copybutton_prompt_text = r'>>> |\.\.\. |\$ '
copybutton_prompt_is_regexp = True
