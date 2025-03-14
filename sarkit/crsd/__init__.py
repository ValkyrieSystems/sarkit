"""
==================================================
Compensated Radar Signal Data (:mod:`sarkit.crsd`)
==================================================

Python reference implementations of the suite of NGA.STND.0080 standardization
documents that define the Compensated Radar Signal Data (CRSD) format.

Supported Versions
==================

* `CRSD 1.0 DRAFT 2025-02-25`_

.. WARNING:: v1.0 DRAFT support is temporary and will be replaced upon NTB approval

Data Structure & File Format
============================

.. autosummary::
   :toctree: generated/

   FileHeaderPart
   Metadata
   Reader
   Writer
   read_file_header
   get_pvp_dtype
   get_ppp_dtype
   binary_format_string_to_dtype
   dtype_to_binary_format_string
   mask_support_array

XML Metadata
============

.. autosummary::
   :toctree: generated/

   XmlHelper
   TxtType
   EnuType
   BoolType
   XdtType
   IntType
   DblType
   HexType
   LineSampType
   XyType
   XyzType
   LatLonType
   LatLonHaeType
   PolyType
   Poly2dType
   XyzPolyType
   PxpType
   AddedPxpType
   MtxType
   EdfType
   ImageAreaCornerPointsType
   ParameterType

Reference Geometry Computations
===============================

.. autosummary::
   :toctree: generated/

   compute_ref_point_parameters
   compute_apc_to_pt_geometry_parameters
   compute_apc_to_pt_geometry_parameters_xmlnames
   arp_to_rpt_geometry_xmlnames

Polarization Parameter Computations
===================================

.. autosummary::
   :toctree: generated/

   compute_h_v_los_unit_vectors
   compute_h_v_pol_parameters

Constants
=========

.. list-table::

   * - ``VERSION_INFO``
     - `dict` of {xml namespace: version-specific information}
   * - ``DEFINED_HEADER_KEYS``
     - :external:py:obj:`set` of KVP keys defined in the standard
   * - ``SECTION_TERMINATOR``
     - Two-byte sequence that marks the end of the file header
   * - ``TRANSCODERS``
     - `dict` of {name: transcoder}

References
==========

CRSD 1.0 DRAFT 2025-02-25
-------------------------
TBD

"""

from ._computations import (
    arp_to_rpt_geometry_xmlnames,
    compute_apc_to_pt_geometry_parameters,
    compute_apc_to_pt_geometry_parameters_xmlnames,
    compute_h_v_los_unit_vectors,
    compute_h_v_pol_parameters,
    compute_ref_point_parameters,
)
from ._io import (
    DEFINED_HEADER_KEYS,
    SECTION_TERMINATOR,
    VERSION_INFO,
    FileHeaderPart,
    Metadata,
    Reader,
    Writer,
    binary_format_string_to_dtype,
    dtype_to_binary_format_string,
    get_ppp_dtype,
    get_pvp_dtype,
    mask_support_array,
    read_file_header,
)
from ._xml import (
    TRANSCODERS,
    AddedPxpType,
    BoolType,
    DblType,
    EdfType,
    EnuType,
    HexType,
    ImageAreaCornerPointsType,
    IntType,
    LatLonHaeType,
    LatLonType,
    LineSampType,
    MtxType,
    ParameterType,
    Poly2dType,
    PolyType,
    PxpType,
    TxtType,
    XdtType,
    XmlHelper,
    XyType,
    XyzPolyType,
    XyzType,
)

__all__ = [
    "DEFINED_HEADER_KEYS",
    "SECTION_TERMINATOR",
    "TRANSCODERS",
    "VERSION_INFO",
    "AddedPxpType",
    "BoolType",
    "DblType",
    "EdfType",
    "EnuType",
    "FileHeaderPart",
    "HexType",
    "ImageAreaCornerPointsType",
    "IntType",
    "LatLonHaeType",
    "LatLonType",
    "LineSampType",
    "Metadata",
    "MtxType",
    "ParameterType",
    "Poly2dType",
    "PolyType",
    "PxpType",
    "Reader",
    "TxtType",
    "Writer",
    "XdtType",
    "XmlHelper",
    "XyType",
    "XyzPolyType",
    "XyzType",
    "arp_to_rpt_geometry_xmlnames",
    "binary_format_string_to_dtype",
    "compute_apc_to_pt_geometry_parameters",
    "compute_apc_to_pt_geometry_parameters_xmlnames",
    "compute_h_v_los_unit_vectors",
    "compute_h_v_pol_parameters",
    "compute_ref_point_parameters",
    "dtype_to_binary_format_string",
    "get_ppp_dtype",
    "get_pvp_dtype",
    "mask_support_array",
    "read_file_header",
]


import os  # noqa: I001
import sys  # noqa: I001

print(
    "\033[93m" if sys.stdout.isatty() and not os.environ.get("NO_COLOR") else "",
    "WARNING: SARkit's CRSD modules are provisional and implement the 2025-02-25 draft\n",
    "The modules will be updated and this message will be removed when the standard is published",
    "\x1b[0m" if sys.stdout.isatty() and not os.environ.get("NO_COLOR") else "",
    file=sys.stderr,
)
