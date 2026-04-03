"""
===================================================
SIDD Calculations (:mod:`sarkit.sidd.calculations`)
===================================================

Sub-package for objects and methods that implement the calculations
described in SIDD Volume 1

Section 2  SIDD Image Pixel Array
---------------------------------

.. autosummary::
   :toctree: generated/

    CoordinateSystem
    get_coordinate_system_type

Section 3  Coordinate Transformations
-------------------------------------

.. autosummary::
   :toctree: generated/

    pixel_to_ecef
    ecef_to_pixel
    pgd_pixel_to_ecef
    ecef_to_pgd_pixel
    ggd_pixel_to_geodetic
    geodetic_to_ggd_pixel
    cgd_pixel_to_ecef
    ecef_to_cgd_pixel
    pfgd_pixel_to_latlon
    latlon_to_pfgd_pixel

Section 7  ExploitationFeatures Calculations
--------------------------------------------

.. autosummary::
   :toctree: generated/

    Angles
    compute_angles

"""

from .coordinate_transformations import (
    cgd_pixel_to_ecef,
    ecef_to_cgd_pixel,
    ecef_to_pgd_pixel,
    ecef_to_pixel,
    geodetic_to_ggd_pixel,
    ggd_pixel_to_geodetic,
    latlon_to_pfgd_pixel,
    pfgd_pixel_to_latlon,
    pgd_pixel_to_ecef,
    pixel_to_ecef,
)
from .exploitation_features import Angles, compute_angles
from .image_pixel_array import (
    CoordinateSystem,
    get_coordinate_system_type,
)

__all__ = []

# Coordinate Transformations
__all__ += [
    "cgd_pixel_to_ecef",
    "ecef_to_cgd_pixel",
    "ecef_to_pgd_pixel",
    "ecef_to_pixel",
    "geodetic_to_ggd_pixel",
    "ggd_pixel_to_geodetic",
    "latlon_to_pfgd_pixel",
    "pfgd_pixel_to_latlon",
    "pgd_pixel_to_ecef",
    "pixel_to_ecef",
]

# ExploitationFeatures
__all__ += ["Angles", "compute_angles"]

# Image Pixel Array
__all__ += [
    "CoordinateSystem",
    "get_coordinate_system_type",
]
