# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
from unittest.mock import MagicMock


os.system('pip install ..')
sys.path.insert(0, os.path.abspath('../bbstrader'))

# Mock the MetaTrader5 module
class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

# List the modules you want to mock
MOCK_MODULES = ['MetaTrader5']
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

project = 'BBSTrader'
copyright = '2024, Bertin Balouki SIMYELI'
author = 'Bertin Balouki SIMYELI'
release = '0.1.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
]


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
