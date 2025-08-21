"""
====================================================
Sensor Independent Complex Data (:mod:`sarkit.sicd`)
====================================================

Python reference implementations of the suite of NGA.STND.0024 standardization
documents that define the Sensor Independent Complex Data (SICD) format.

Supported Versions
==================

* `SICD 1.1.0`_
* `SICD 1.2.1`_
* `SICD 1.3.0`_
* `SICD 1.4.0`_

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
   NitfWriter
   SizingImhdr
   image_segment_sizing_calculations
   jbp_from_nitf_metadata

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
   IntType
   DblType
   XdtType
   RowColType
   CmplxType
   XyzType
   LatLonHaeType
   LatLonType
   PolyType
   Poly2dType
   XyzPolyType
   MtxType
   ParameterType
   ImageCornersType
   compute_scp_coa

Projections
===========
For most of the functions from SICD Volume 3, see the `sarkit.sicd.projection` namespace.

In the `sarkit.sicd` namespace, there are convenience functions that operate on parsed SICD XML trees:

.. autosummary::
   :toctree: generated/

    image_to_ground_plane
    image_to_constant_hae_surface
    image_to_dem_surface
    scene_to_image

Constants
=========

.. list-table::

   * - ``VERSION_INFO``
     - `dict` of {xml namespace: version-specific information}
   * - ``PIXEL_TYPES``
     - `dict` of {PixelType: pixel-type-specific information}
   * - ``IS_SIZE_MAX``
     - maximum NITF image segment length in bytes (:math:`10^{10}-2`)
   * - ``ILOC_MAX``
     - maximum number of rows contained in a segmented SICD NITF image segment (99,999)

CLI Utilities
=============

.. _sicdinfo-cli:

.. autoprogram:: sarkit.sicd._sicdinfo:_parser()
   :prog: sicdinfo

References
==========

SICD 1.1.0
----------
.. [NGA.STND.0024-1_1.1] National Center for Geospatial Intelligence Standards,
   "Sensor Independent Complex Data (SICD), Vol. 1, Design & Implementation Description Document,
   Version 1.1", 2014.
   https://nsgreg.nga.mil/doc/view?i=4192

.. [NGA.STND.0024-2_1.1] National Center for Geospatial Intelligence Standards,
   "Sensor Independent Complex Data (SICD), Vol. 2, File Format Description Document,
   Version 1.1", 2014.
   https://nsgreg.nga.mil/doc/view?i=4194

.. [NGA.STND.0024-3_1.1] National Center for Geospatial Intelligence Standards,
   "Sensor Independent Complex Data (SICD), Vol. 3, Image Projections Description Document,
   Version 1.1", 2016.
   https://nsgreg.nga.mil/doc/view?i=4249

.. [SICD_schema_V1.1.0_2014_09_30.xsd] National Center for Geospatial Intelligence Standards,
   "Sensor Independent Complex Data (SICD) XML Schema, Version 1.1.0", 2014.
   https://nsgreg.nga.mil/doc/view?i=4251

SICD 1.2.1
----------
.. [NGA.STND.0024-1_1.2.1] National Center for Geospatial Intelligence Standards,
   "Sensor Independent Complex Data (SICD), Vol. 1, Design & Implementation Description Document,
   Version 1.2.1", 2018.
   https://nsgreg.nga.mil/doc/view?i=4900

.. [NGA.STND.0024-2_1.2.1] National Center for Geospatial Intelligence Standards,
   "Sensor Independent Complex Data (SICD), Vol. 2, File Format Description Document,
   Version 1.2.1", 2018.
   https://nsgreg.nga.mil/doc/view?i=4901

.. [NGA.STND.0024-3_1.2.1] National Center for Geospatial Intelligence Standards,
   "Sensor Independent Complex Data (SICD), Vol. 3, Image Projections Description Document,
   Version 1.2.1", 2018.
   https://nsgreg.nga.mil/doc/view?i=4902

.. [SICD_schema_V1.2.1_2018_12_13.xsd] National Center for Geospatial Intelligence Standards,
   "Sensor Independent Complex Data (SICD) XML Schema, Version 1.2.1", 2018.
   https://nsgreg.nga.mil/doc/view?i=5230

SICD 1.3.0
----------
.. [NGA.STND.0024-1_1.3.0] National Center for Geospatial Intelligence Standards,
   "Sensor Independent Complex Data (SICD), Vol. 1, Design & Implementation Description Document,
   Version 1.3.0", 2021.
   https://nsgreg.nga.mil/doc/view?i=5381

.. [NGA.STND.0024-2_1.3.0] National Center for Geospatial Intelligence Standards,
   "Sensor Independent Complex Data (SICD), Vol. 2, File Format Description Document,
   Version 1.3.0", 2021.
   https://nsgreg.nga.mil/doc/view?i=5382

.. [NGA.STND.0024-3_1.3.0] National Center for Geospatial Intelligence Standards,
   "Sensor Independent Complex Data (SICD), Vol. 3, Image Projections Description Document,
   Version 1.3.0", 2021.
   https://nsgreg.nga.mil/doc/view?i=5383

.. [SICD_schema_V1.3.0_2021_11_30.xsd] National Center for Geospatial Intelligence Standards,
   "Sensor Independent Complex Data (SICD) XML Schema, Version 1.3.0", 2021.
   https://nsgreg.nga.mil/doc/view?i=5418

SICD 1.4.0
----------
.. [NGA.STND.0024-1_1.4.0] National Center for Geospatial Intelligence Standards,
   "Sensor Independent Complex Data (SICD), Vol. 1, Design & Implementation Description Document,
   Version 1.4.0", 2023.
   https://nsgreg.nga.mil/doc/view?i=5529

.. [NGA.STND.0024-2_1.4.0] National Center for Geospatial Intelligence Standards,
   "Sensor Independent Complex Data (SICD), Vol. 2, File Format Description Document,
   Version 1.4.0", 2023.
   https://nsgreg.nga.mil/doc/view?i=5531

.. [NGA.STND.0024-3_1.4.0] National Center for Geospatial Intelligence Standards,
   "Sensor Independent Complex Data (SICD), Vol. 3, Image Projections Description Document,
   Version 1.4.0", 2023.
   https://nsgreg.nga.mil/doc/view?i=5532

.. [SICD_schema_V1.4.0.xsd] National Center for Geospatial Intelligence Standards,
   "Sensor Independent Complex Data (SICD) XML Schema, Version 1.4.0", 2024.
   https://nsgreg.nga.mil/doc/view?i=5538
"""

from ._constants import (
    ILOC_MAX,
    IS_SIZE_MAX,
    PIXEL_TYPES,
    VERSION_INFO,
)
from ._io import (
    NitfDeSubheaderPart,
    NitfFileHeaderPart,
    NitfImSubheaderPart,
    NitfMetadata,
    NitfReader,
    NitfSecurityFields,
    NitfWriter,
    SizingImhdr,
    image_segment_sizing_calculations,
    jbp_from_nitf_metadata,
)
from ._xml import (
    BoolType,
    CmplxType,
    DblType,
    ElementWrapper,
    EnuType,
    ImageCornersType,
    IntType,
    LatLonHaeType,
    LatLonType,
    MtxType,
    ParameterType,
    Poly2dType,
    PolyType,
    RowColType,
    TxtType,
    XdtType,
    XmlHelper,
    XsdHelper,
    XyzPolyType,
    XyzType,
    compute_scp_coa,
)
from .projection._derived import (
    image_to_constant_hae_surface,
    image_to_dem_surface,
    image_to_ground_plane,
    scene_to_image,
)

__all__ = [
    "ILOC_MAX",
    "IS_SIZE_MAX",
    "PIXEL_TYPES",
    "VERSION_INFO",
    "BoolType",
    "CmplxType",
    "DblType",
    "ElementWrapper",
    "EnuType",
    "ImageCornersType",
    "IntType",
    "LatLonHaeType",
    "LatLonType",
    "MtxType",
    "NitfDeSubheaderPart",
    "NitfFileHeaderPart",
    "NitfImSubheaderPart",
    "NitfMetadata",
    "NitfReader",
    "NitfSecurityFields",
    "NitfWriter",
    "ParameterType",
    "Poly2dType",
    "PolyType",
    "RowColType",
    "SizingImhdr",
    "TxtType",
    "XdtType",
    "XmlHelper",
    "XsdHelper",
    "XyzPolyType",
    "XyzType",
    "compute_scp_coa",
    "image_segment_sizing_calculations",
    "image_to_constant_hae_surface",
    "image_to_dem_surface",
    "image_to_ground_plane",
    "jbp_from_nitf_metadata",
    "scene_to_image",
]
