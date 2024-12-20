"""
======================================
Sensor Independent Complex Data (SICD)
======================================

Python reference implementations of the suite of NGA.STND.0024 standardization
documents that define the Sensor Independent Complex Data (SICD) format.

Supported Versions
==================

* `SICD 1.1.0`_
* `SICD 1.2.1`_
* `SICD 1.3.0`_
* `SICD 1.4.0`_

Functions
=========

Reading and Writing
-------------------

.. autosummary::
   :toctree: generated/

   SicdNitfReader
   SicdNitfWriter

I/O Helpers
-----------

.. autosummary::
   :toctree: generated/

   SicdNitfPlan
   SicdNitfSecurityFields
   SicdNitfHeaderFields
   SicdNitfImageSegmentFields
   SicdNitfDESegmentFields

Processing
----------

.. autosummary::
   :toctree: generated/

    image_to_ground_plane
    image_to_constant_hae_surface
    scene_to_image

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

from .io import (
    SicdNitfDESegmentFields,
    SicdNitfHeaderFields,
    SicdNitfImageSegmentFields,
    SicdNitfPlan,
    SicdNitfReader,
    SicdNitfSecurityFields,
    SicdNitfWriter,
)
from .projection.derived import (
    image_to_constant_hae_surface,
    image_to_ground_plane,
    scene_to_image,
)

# IO
__all__ = [
    "SicdNitfDESegmentFields",
    "SicdNitfHeaderFields",
    "SicdNitfImageSegmentFields",
    "SicdNitfPlan",
    "SicdNitfReader",
    "SicdNitfSecurityFields",
    "SicdNitfWriter",
]

# Projections
__all__ += [
    "image_to_constant_hae_surface",
    "image_to_ground_plane",
    "scene_to_image",
]
