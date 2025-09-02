"""
============================================
XML helper functions (:mod:`sarkit.xmlhelp`)
============================================

.. WARNING:: This module is documented for reference only and stable functionality is not guaranteed.
   Prefer to use the related objects from the standard-specific modules instead.

Common functions for working with XML metadata.

.. autosummary::
   :toctree: generated/

   ElementWrapper
   XmlHelper
   XsdHelper
   XsdTypeDef
   ChildDef

"""

from ._core import (
    ChildDef,
    ElementWrapper,
    XmlHelper,
    XsdHelper,
    XsdTypeDef,
    dumps_xsdtypes,
    loads_xsdtypes,
    split_elempath,
)

__all__ = [
    "ChildDef",
    "ElementWrapper",
    "XmlHelper",
    "XsdHelper",
    "XsdTypeDef",
    "dumps_xsdtypes",
    "loads_xsdtypes",
    "split_elempath",
]
