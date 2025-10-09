"""
====================================================
Sensor Independent Derived Data (:mod:`sarkit.sidd`)
====================================================

Python reference implementations of the suite of NGA.STND.0025 standardization
documents that define the Sensor Independent Derived Data (SIDD) format.

Supported Versions
==================

.. note:: As of 2025-09-26, there is only a single link for SIDD schema files on the NSG standards registry.
   The page has a name (SIDD XML Schema version 1.1.0) that does not match the contents (SIDD v3.0 schemas) and schemas
   for the ``urn:us:gov:ic:ism:13`` namespace are omitted.

   To get around these limitations, SARkit pulled SIDD schemas from other sources.
   Consult READMEs in the `SIDD schema source directory <https://github.com/ValkyrieSystems/sarkit/tree/main/sarkit/sidd/schemas>`_
   for information on their provenance.

* `SIDD 1.0`_
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
   jbp_from_nitf_metadata
   product_image_segment_mapping
   segmentation_algorithm

XML Metadata
============

.. autosummary::
   :toctree: generated/

   XmlHelper
   ElementWrapper
   XsdHelper
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
   LUTInfoType
   LookupTableType
   Lookup3TableType

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

CLI Utilities
=============

.. _siddinfo-cli:

.. autoprogram:: sarkit.sidd._siddinfo:_parser()
   :prog: siddinfo

References
==========

SIDD 1.0
--------
.. [NGA.STND.0025-1_1.0] National Center for Geospatial Intelligence Standards,
   "Sensor Independent Derived Data (SIDD), Vol. 1, Design & Implementation Description Document,
   Version 1.0", 2011.
   https://nsgreg.nga.mil/doc/view?i=2203

.. [NGA.STND.0025-2_1.0] National Center for Geospatial Intelligence Standards,
   "Sensor Independent Derived Data (SIDD), Vol. 2, NITF File Format Description Document,
   Version 1.0", 2011.
   https://nsgreg.nga.mil/doc/view?i=2204


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

from ._constants import (
    ILOC_MAX,
    LI_MAX,
    PIXEL_TYPES,
    VERSION_INFO,
)
from ._io import (
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
    jbp_from_nitf_metadata,
    product_image_segment_mapping,
    segmentation_algorithm,
)
from ._xml import (
    AngleMagnitudeType,
    BoolType,
    DblType,
    ElementWrapper,
    EnuType,
    FilterCoefficientType,
    ImageCornersType,
    IntListType,
    IntType,
    LatLonType,
    Lookup3TableType,
    LookupTableType,
    LUTInfoType,
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
    XsdHelper,
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
    "VERSION_INFO",
    "AngleMagnitudeType",
    "BoolType",
    "DblType",
    "ElementWrapper",
    "EnuType",
    "FilterCoefficientType",
    "ImageCornersType",
    "IntListType",
    "IntType",
    "LUTInfoType",
    "LatLonType",
    "Lookup3TableType",
    "LookupTableType",
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
    "XsdHelper",
    "XyzPolyType",
    "XyzType",
    "compute_angles",
    "ecef_to_pixel",
    "get_coordinate_system_type",
    "jbp_from_nitf_metadata",
    "pixel_to_ecef",
    "product_image_segment_mapping",
    "segmentation_algorithm",
]
