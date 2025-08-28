"""
==================================================
Compensated Radar Signal Data (:mod:`sarkit.crsd`)
==================================================

Python reference implementations of the suite of NGA.STND.0080 standardization
documents that define the Compensated Radar Signal Data (CRSD) format.

Supported Versions
==================

* `CRSD 1.0`_

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
   ElementWrapper
   XsdHelper
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

Receive Channel Parameters
==========================

.. autosummary::
   :toctree: generated/

   compute_dwelltimes_using_poly
   compute_dwelltimes_using_dta

Reference Geometry Parameters
===============================

.. autosummary::
   :toctree: generated/

   compute_ref_point_parameters
   compute_apc_to_pt_geometry_parameters
   compute_arp_to_rpt_geometry
   compute_reference_geometry

Antenna Parameters
==================

.. autosummary::
   :toctree: generated/

   interpolate_support_array
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

CLI Utilities
=============

.. _crsdinfo-cli:

.. autoprogram:: sarkit.crsd._crsdinfo:_parser()
   :prog: crsdinfo

References
==========

CRSD 1.0
--------
.. [NGA.STND.0080-1_1.0_CRSD] National Center for Geospatial Intelligence Standards,
   "Compensated Radar Signal Data (CRSD), Vol. 1, Design & Implementation Description Document,
   Version 1.0", 2025.
   https://nsgreg.nga.mil/doc/view?i=5672

.. [NGA.STND.0080-2_1.0_CRSD_schema_2025_02_25.xsd] National Center for Geospatial Intelligence Standards,
   "Compensated Radar Signal Data (CRSD) XML Schema, Version 1.0", 2025.
   https://nsgreg.nga.mil/doc/view?i=5673
"""

from ._computations import (
    compute_apc_to_pt_geometry_parameters,
    compute_arp_to_rpt_geometry,
    compute_dwelltimes_using_dta,
    compute_dwelltimes_using_poly,
    compute_h_v_los_unit_vectors,
    compute_h_v_pol_parameters,
    compute_ref_point_parameters,
    compute_reference_geometry,
    interpolate_support_array,
)
from ._constants import (
    DEFINED_HEADER_KEYS,
    SECTION_TERMINATOR,
    VERSION_INFO,
)
from ._io import (
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
    AddedPxpType,
    BoolType,
    DblType,
    EdfType,
    ElementWrapper,
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
    XsdHelper,
    XyType,
    XyzPolyType,
    XyzType,
)

__all__ = [
    "DEFINED_HEADER_KEYS",
    "SECTION_TERMINATOR",
    "VERSION_INFO",
    "AddedPxpType",
    "BoolType",
    "DblType",
    "EdfType",
    "ElementWrapper",
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
    "XsdHelper",
    "XyType",
    "XyzPolyType",
    "XyzType",
    "binary_format_string_to_dtype",
    "compute_apc_to_pt_geometry_parameters",
    "compute_arp_to_rpt_geometry",
    "compute_dwelltimes_using_dta",
    "compute_dwelltimes_using_poly",
    "compute_h_v_los_unit_vectors",
    "compute_h_v_pol_parameters",
    "compute_ref_point_parameters",
    "compute_reference_geometry",
    "dtype_to_binary_format_string",
    "get_ppp_dtype",
    "get_pvp_dtype",
    "interpolate_support_array",
    "mask_support_array",
    "read_file_header",
]
