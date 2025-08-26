"""
######################################
Processing  (:mod:`sarkit.processing`)
######################################

Interacting with SAR data in proven, but nonstandard ways.

.. warning:: Functions in this module require the ``processing`` :ref:`extra <installation>`.

SICD
====

Deskew
------

.. autosummary::
   :toctree: generated/

   sicd_get_deskew_phase_poly
   sicd_apply_phase_poly
   sicd_deskew

Pixel Types
-----------

.. autosummary::
   :toctree: generated/

   sicd_as_amp8i_phs8i
   sicd_as_re16i_im16i
   sicd_as_re32f_im32f

Subimage (chipping)
-------------------

.. autosummary::
   :toctree: generated/

   sicd_subimage

"""

from ._sicd.deskew import (
    sicd_apply_phase_poly,
    sicd_deskew,
    sicd_get_deskew_phase_poly,
)
from ._sicd.pixel_type import (
    sicd_as_amp8i_phs8i,
    sicd_as_re16i_im16i,
    sicd_as_re32f_im32f,
)
from ._sicd.subimage import sicd_subimage

__all__ = [
    "sicd_apply_phase_poly",
    "sicd_as_amp8i_phs8i",
    "sicd_as_re16i_im16i",
    "sicd_as_re32f_im32f",
    "sicd_deskew",
    "sicd_get_deskew_phase_poly",
    "sicd_subimage",
]
