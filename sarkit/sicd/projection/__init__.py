"""
===============================================
SICD Projection (:mod:`sarkit.sicd.projection`)
===============================================

Objects and methods that implement the exploitation processing
described in SICD Volume 3 Image Projections Description Document (IPDD).

Data Classes
============

To simplify interfaces, some collections of metadata parameters are encapsulated in
dataclasses with attributes named as similar as feasible to the IPDD.

.. autosummary::
   :toctree: generated/

   MetadataParams
   ErrorStatParams
   ComponentErrorStatMono
   ComponentErrorStatBi
   AdjustableParameterOffsets
   ApoErrorParams
   CoaPosVelsMono
   CoaPosVelsBi
   ProjectionSetsMono
   ProjectionSetsBi
   ScenePointRRdotParams
   ScenePointGpXyParams
   ProjGeomParamsMono
   ProjGeomParamsBi
   SlantPlaneSensitivityMatrices
   ImageLocationSensitivityMatrices
   PVTSensitivityMatricesMono
   PVTSensitivityMatricesBi
   SensitivityMatricesMono
   SensitivityMatricesBi

Type Aliases
------------

.. py:type:: CoaPosVelsLike
   :canonical: CoaPosVelsMono | CoaPosVelsBi

   Represent either a monostatic or bistatic ensemble of COA positions and velocities

.. py:type:: ProjectionSetsLike
   :canonical: ProjectionSetsMono | ProjectionSetsBi

   Represent either a monostatic or bistatic ensemble of COA projection sets

.. py:type:: SensitivityMatricesLike
   :canonical: SensitivityMatricesMono | SensitivityMatricesBi

   Represent either a set of monostatic or bistatic ensemble sensitivity matrices

Image Plane Parameters
======================

.. autosummary::
   :toctree: generated/

   image_grid_to_image_plane_point
   image_plane_point_to_image_grid

Image Grid to COA Positions & Velocities
========================================

.. autosummary::
   :toctree: generated/

   compute_coa_time
   compute_coa_pos_vel

SCP Pixel Projection
====================

.. autosummary::
   :toctree: generated/

   compute_scp_coa_r_rdot
   compute_scp_coa_slant_plane_normal
   compute_ric_basis_vectors

Image Grid to R/Rdot Contour
============================

.. autosummary::
   :toctree: generated/

   compute_coa_r_rdot
   compute_projection_sets

Precise R/Rdot to Ground Plane Projection
=========================================

.. autosummary::
   :toctree: generated/

   r_rdot_to_ground_plane_mono
   r_rdot_to_ground_plane_bi
   compute_pt_r_rdot_parameters
   compute_gp_xy_parameters

Scene To Image Grid Projection
==============================

.. autosummary::
   :toctree: generated/

   scene_to_image

Adjustable Parameters
=====================

.. autosummary::
   :toctree: generated/

   apply_apos

Precise R/Rdot to Constant HAE Surface Projection
=================================================

.. autosummary::
   :toctree: generated/

   r_rdot_to_constant_hae_surface

Precise R/Rdot to DEM Surface Projection
========================================

.. autosummary::
   :toctree: generated/

   r_rdot_to_dem_surface


Projection Sensitivity Parameters
=================================
.. autosummary::
   :toctree: generated/

   compute_proj_geom_params_mono
   compute_proj_geom_params_bi
   compute_slant_plane_sensitivity_matrices
   compute_image_location_sensitivity_matrices
   compute_pvt_sensitivity_matrices_mono
   compute_pvt_sensitivity_matrices_bi
   compute_sensitivity_matrices

Projection Error Propagation
============================
.. autosummary::
   :toctree: generated/

   compute_ecef_pv_transformation
   compute_composite_error_no_apo_mono
   compute_composite_error_apo_mono
   compute_composite_error_no_apo_bi
   compute_composite_error_apo_bi
   compute_i2s_error
   compute_s2i_error
"""

from ._calc import (
    apply_apos,
    compute_coa_pos_vel,
    compute_coa_r_rdot,
    compute_coa_time,
    compute_gp_xy_parameters,
    compute_projection_sets,
    compute_pt_r_rdot_parameters,
    compute_scp_coa_r_rdot,
    compute_scp_coa_slant_plane_normal,
    image_grid_to_image_plane_point,
    image_plane_point_to_image_grid,
    r_rdot_to_constant_hae_surface,
    r_rdot_to_dem_surface,
    r_rdot_to_ground_plane_bi,
    r_rdot_to_ground_plane_mono,
    scene_to_image,
)
from ._errorprop import (
    compute_composite_error_apo_bi,
    compute_composite_error_apo_mono,
    compute_composite_error_no_apo_bi,
    compute_composite_error_no_apo_mono,
    compute_ecef_pv_transformation,
    compute_i2s_error,
    compute_ric_basis_vectors,
    compute_s2i_error,
)
from ._params import (
    AdjustableParameterOffsets,
    ApoErrorParams,
    CoaPosVelsBi,
    CoaPosVelsLike,
    CoaPosVelsMono,
    ComponentErrorStatBi,
    ComponentErrorStatMono,
    ErrorStatParams,
    MetadataParams,
    ProjectionSetsBi,
    ProjectionSetsLike,
    ProjectionSetsMono,
    ScenePointGpXyParams,
    ScenePointRRdotParams,
)
from ._sensitivity import (
    ImageLocationSensitivityMatrices,
    ProjGeomParamsBi,
    ProjGeomParamsMono,
    PVTSensitivityMatricesBi,
    PVTSensitivityMatricesMono,
    SensitivityMatricesBi,
    SensitivityMatricesLike,
    SensitivityMatricesMono,
    SlantPlaneSensitivityMatrices,
    compute_image_location_sensitivity_matrices,
    compute_proj_geom_params_bi,
    compute_proj_geom_params_mono,
    compute_pvt_sensitivity_matrices_bi,
    compute_pvt_sensitivity_matrices_mono,
    compute_sensitivity_matrices,
    compute_slant_plane_sensitivity_matrices,
)

__all__ = [
    "AdjustableParameterOffsets",
    "ApoErrorParams",
    "CoaPosVelsBi",
    "CoaPosVelsLike",
    "CoaPosVelsMono",
    "ComponentErrorStatBi",
    "ComponentErrorStatMono",
    "ErrorStatParams",
    "ImageLocationSensitivityMatrices",
    "MetadataParams",
    "PVTSensitivityMatricesBi",
    "PVTSensitivityMatricesMono",
    "ProjGeomParamsBi",
    "ProjGeomParamsMono",
    "ProjectionSetsBi",
    "ProjectionSetsLike",
    "ProjectionSetsMono",
    "ScenePointGpXyParams",
    "ScenePointRRdotParams",
    "SensitivityMatricesBi",
    "SensitivityMatricesLike",
    "SensitivityMatricesMono",
    "SlantPlaneSensitivityMatrices",
    "apply_apos",
    "compute_coa_pos_vel",
    "compute_coa_r_rdot",
    "compute_coa_time",
    "compute_composite_error_apo_bi",
    "compute_composite_error_apo_mono",
    "compute_composite_error_no_apo_bi",
    "compute_composite_error_no_apo_mono",
    "compute_ecef_pv_transformation",
    "compute_gp_xy_parameters",
    "compute_i2s_error",
    "compute_image_location_sensitivity_matrices",
    "compute_proj_geom_params_bi",
    "compute_proj_geom_params_mono",
    "compute_projection_sets",
    "compute_pt_r_rdot_parameters",
    "compute_pvt_sensitivity_matrices_bi",
    "compute_pvt_sensitivity_matrices_mono",
    "compute_ric_basis_vectors",
    "compute_s2i_error",
    "compute_scp_coa_r_rdot",
    "compute_scp_coa_slant_plane_normal",
    "compute_sensitivity_matrices",
    "compute_slant_plane_sensitivity_matrices",
    "image_grid_to_image_plane_point",
    "image_plane_point_to_image_grid",
    "r_rdot_to_constant_hae_surface",
    "r_rdot_to_dem_surface",
    "r_rdot_to_ground_plane_bi",
    "r_rdot_to_ground_plane_mono",
    "scene_to_image",
]
