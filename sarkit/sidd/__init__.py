"""
====================================================
Sensor Independent Derived Data (:mod:`sarkit.sidd`)
====================================================

Python reference implementations of the suite of NGA.STND.0025 standardization
documents that define the Sensor Independent Derived Data (SIDD) format.

Supported Versions
==================

* `SIDD 2.0`_
* `SIDD 3.0`_

Data Structure & File Format
============================

.. autosummary::
   :toctree: generated/

   NitfSecurityFields
   NitfFileHeaderPart
   NitfImSubheaderPart
   NitfDeSubheaderPart
   NitfReader
   NitfMetadata
   NitfProductImageMetadata
   NitfLegendMetadata
   NitfDedMetadata
   NitfProductSupportXmlMetadata
   NitfSicdXmlMetadata
   NitfWriter
   SegmentationImhdr
   segmentation_algorithm

XML Metadata
============

.. autosummary::
   :toctree: generated/

   XmlHelper
   BoolType
   DblType
   EnuType
   IntType
   TxtType
   XdtType
   ParameterType
   FilterCoefficientType
   IntListType
   SfaPointType

Transcoders with children in the ``urn:SICommon:1.0`` namespace.

.. autosummary::
   :toctree: generated/

   XyzType
   AngleMagnitudeType
   LatLonType
   PolyCoef1dType
   PolyCoef2dType
   RangeAzimuthType
   RowColDblType
   RowColIntType
   XyzPolyType
   ImageCornersType

Calculations
============

Calculations defined by the SIDD standard.  For more information see the `sarkit.sidd.calculations` namespace.

.. autosummary::
   :toctree: generated/

    ecef_to_pixel
    pixel_to_ecef
    compute_angles
    get_coordinate_system_type

Constants
=========

.. list-table::

   * - ``VERSION_INFO``
     - `dict` of {xml namespace: version-specific information}
   * - ``PIXEL_TYPES``
     - `dict` of {PixelType: pixel-type-specific information}
   * - ``LI_MAX``
     - maximum NITF image segment length in bytes (:math:`10^{10}-2`)
   * - ``ILOC_MAX``
     - maximum number of rows contained in a NITF image segment (99,999)
   * - ``TRANSCODERS``
     - `dict` of {name: transcoder}

References
==========

SIDD 2.0
--------
.. [NGA.STND.0025-1_2.0] National Center for Geospatial Intelligence Standards,
   "Sensor Independent Derived Data (SIDD), Vol. 1, Design & Implementation Description Document,
   Version 2.0", 2019.
   https://nsgreg.nga.mil/doc/view?i=4906

.. [NGA.STND.0025-2_2.0] National Center for Geospatial Intelligence Standards,
   "Sensor Independent Derived Data (SIDD), Vol. 2, NITF File Format Description Document,
   Version 2.0", 2019.
   https://nsgreg.nga.mil/doc/view?i=4907

SIDD 3.0
--------
.. [NGA.STND.0025-1_3.0] National Center for Geospatial Intelligence Standards,
   "Sensor Independent Derived Data (SIDD), Vol. 1, Design & Implementation Description Document,
   Version 3.0", 2021.
   https://nsgreg.nga.mil/doc/view?i=5440

.. [NGA.STND.0025-2_3.0] National Center for Geospatial Intelligence Standards,
   "Sensor Independent Derived Data (SIDD), Vol. 2, NITF File Format Description Document,
   Version 3.0", 2021.
   https://nsgreg.nga.mil/doc/view?i=5441

"""

from ._io import (
    ILOC_MAX,
    LI_MAX,
    PIXEL_TYPES,
    VERSION_INFO,
    NitfDedMetadata,
    NitfDeSubheaderPart,
    NitfFileHeaderPart,
    NitfImSubheaderPart,
    NitfLegendMetadata,
    NitfMetadata,
    NitfProductImageMetadata,
    NitfProductSupportXmlMetadata,
    NitfReader,
    NitfSecurityFields,
    NitfSicdXmlMetadata,
    NitfWriter,
    SegmentationImhdr,
    segmentation_algorithm,
)
from ._xml import (
    TRANSCODERS,
    AngleMagnitudeType,
    BoolType,
    DblType,
    EnuType,
    FilterCoefficientType,
    ImageCornersType,
    IntListType,
    IntType,
    LatLonType,
    ParameterType,
    PolyCoef1dType,
    PolyCoef2dType,
    RangeAzimuthType,
    RowColDblType,
    RowColIntType,
    SfaPointType,
    TxtType,
    XdtType,
    XmlHelper,
    XyzPolyType,
    XyzType,
)
from .calculations import (
    compute_angles,
    ecef_to_pixel,
    get_coordinate_system_type,
    pixel_to_ecef,
)

__all__ = [
    "ILOC_MAX",
    "LI_MAX",
    "PIXEL_TYPES",
    "TRANSCODERS",
    "VERSION_INFO",
    "AngleMagnitudeType",
    "BoolType",
    "DblType",
    "EnuType",
    "FilterCoefficientType",
    "ImageCornersType",
    "IntListType",
    "IntType",
    "LatLonType",
    "NitfDeSubheaderPart",
    "NitfDedMetadata",
    "NitfFileHeaderPart",
    "NitfImSubheaderPart",
    "NitfLegendMetadata",
    "NitfMetadata",
    "NitfProductImageMetadata",
    "NitfProductSupportXmlMetadata",
    "NitfReader",
    "NitfSecurityFields",
    "NitfSicdXmlMetadata",
    "NitfWriter",
    "ParameterType",
    "PolyCoef1dType",
    "PolyCoef2dType",
    "RangeAzimuthType",
    "RowColDblType",
    "RowColIntType",
    "SegmentationImhdr",
    "SfaPointType",
    "TxtType",
    "XdtType",
    "XmlHelper",
    "XyzPolyType",
    "XyzType",
    "compute_angles",
    "ecef_to_pixel",
    "get_coordinate_system_type",
    "pixel_to_ecef",
    "segmentation_algorithm",
]
