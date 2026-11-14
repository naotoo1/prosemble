"""
Prosemble package
"""
from importlib.metadata import PackageNotFoundError, version as _version

from . import models
from . import core
from . import datasets


__author__ = """Nana Abeka Otoo"""
__email__ = 'nanaabekaotoo@gmail.com'
try:
    __version__ = _version('prosemble')
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = '0+unknown'
