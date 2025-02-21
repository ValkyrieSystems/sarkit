"""Functions that can select monostatic or bistatic methods."""

import lxml.etree
import numpy as np
import numpy.typing as npt

from . import _calc as calc
from . import _params as params


def image_to_ground_plane(
    sicd_xmltree: lxml.etree.ElementTree,
    image_grid_locations: npt.ArrayLike,
    gref: npt.ArrayLike,
    ugpn: npt.ArrayLike,
    *,
    method: str | None = None,
    bistat_delta_gp_gpp: float = 0.010,
    bistat_maxiter: int = 10,
) -> tuple[npt.NDArray, npt.NDArray, bool]:
    """Project image coordinates to an arbitrary plane.

    Parameters
    ----------
    sicd_xmltree : lxml.etree.ElementTree
        SICD XML metadata.
    image_grid_locations : (..., 2) array_like
        N-D array of image coordinates with xrow/ycol in meters in the last dimension.
    gref : (3,) array_like
        Ground plane reference point with WGS 84 cartesian X, Y, Z components in meters.
    ugpn : (3,) array_like
        Unit normal vector to ground plane with WGS 84 cartesian X, Y, Z components in
        meters.
    method : str, optional
        "monostatic" or "bistatic". If omitted, selects based on ``sicd_xmltree`` metadata.
    bistat_delta_gp_gpp : float, optional
        (Bistatic only) Displacement threshold for ground plane points in meters.
    bistat_maxiter : int, optional
        (Bistatic only) Maximum number of R/Rdot to Ground Plane iterations to perform.

    Returns
    -------
    gpp_tgt : (..., 3) ndarray
        Array of ground plane points with WGS 84 cartesian X, Y, Z components in meters
        in the last dimension.
    delta_gp : ndarray
        Magnitude of the displacement from estimated point to the precise intersection
        of the target R/Rdot contour.
    success : bool
        Whether or not all ``gpp_tgt`` points were properly determined. The
        criteria are dependent on the collect type.

    """
    proj_metadata = params.MetadataParams.from_xml(sicd_xmltree)
    projection_sets = calc.compute_projection_sets(proj_metadata, image_grid_locations)

    if params.AdjustableParameterOffsets.exists(sicd_xmltree):
        adjust_param_offsets = params.AdjustableParameterOffsets.from_xml(sicd_xmltree)
        projection_sets = calc.compute_and_apply_offsets(
            proj_metadata, projection_sets, adjust_param_offsets
        )
    method = (
        {True: "monostatic", False: "bistatic"}[proj_metadata.is_monostatic()]
        if method is None
        else method
    )
    if method == "monostatic":
        gpp_tgt = calc.r_rdot_to_ground_plane_mono(
            proj_metadata.LOOK, projection_sets, gref, ugpn
        )
        delta_gp = np.full(gpp_tgt.shape[:-1], np.nan)
        delta_gp[np.isfinite(gpp_tgt).all(axis=-1)] = 0
        success = np.isfinite(gpp_tgt).all()
        return gpp_tgt, delta_gp, success
    if method == "bistatic":
        bistat_projection_sets = projection_sets
        if proj_metadata.is_monostatic():
            bistat_projection_sets = params.ProjectionSets(
                t_COA=projection_sets.t_COA,
                Xmt_COA=projection_sets.ARP_COA,
                VXmt_COA=projection_sets.VARP_COA,
                Rcv_COA=projection_sets.ARP_COA,
                VRcv_COA=projection_sets.VARP_COA,
                R_Avg_COA=projection_sets.R_COA,
                Rdot_Avg_COA=projection_sets.Rdot_COA,
            )
        gpp_tgt, delta_gp, success = calc.r_rdot_to_ground_plane_bi(
            proj_metadata.LOOK,
            proj_metadata.SCP,
            bistat_projection_sets,
            gref,
            ugpn,
            delta_gp_gpp=bistat_delta_gp_gpp,
            maxiter=bistat_maxiter,
        )
        return gpp_tgt, delta_gp, success
    raise ValueError(f"Unrecognized {method=}")


def scene_to_image(
    sicd_xmltree: lxml.etree.ElementTree,
    scene_points: npt.ArrayLike,
    *,
    delta_gp_s2i=0.001,
    maxiter=10,
    bistat_delta_gp_gpp: float = 0.010,
    bistat_maxiter: int = 10,
) -> tuple[npt.NDArray, npt.NDArray, bool]:
    """Map geolocated points in the three-dimensional scene to image grid locations.

    Parameters
    ----------
    sicd_xmltree : lxml.etree.ElementTree
        SICD XML metadata.
    scene_points : (..., 3) array_like
        Array of scene points with ECEF (WGS 84 cartesian) X, Y, Z components in meters in the
        last dimension.
    delta_gp_s2i : float, optional
        Ground plane displacement threshold for final ground plane point in meters.
    maxiter : int, optional
        Maximum number of iterations to perform.
    bistat_delta_gp_gpp : float, optional
        (Bistatic only) Ground plane displacement threshold for intermediate ground
        plane points in meters.
    bistat_maxiter : int, optional
        (Bistatic only) Maximum number of intermediate bistatic R/Rdot to Ground Plane
        iterations to perform per scene-to-image iteration.

    Returns
    -------
    image_grid_locations : (..., 2) ndarray
        Array of image coordinates with xrow/ycol in meters in the last dimension.
        Coordinates are NaN where there is no projection solution.
    delta_gp : ndarray
        Ground-plane to scene displacement magnitude. Values are NaN where there is no
        projection solution.
    success : bool
        Whether or not all displacement magnitudes, ``delta_gp`` are less than or equal
        to the threshold, ``delta_gp_s2i``.
        For bistatic projections, ``success`` also requires convergence of all
        intermediate ground plane points.
    """
    proj_metadata = params.MetadataParams.from_xml(sicd_xmltree)

    adjust_param_offsets = None
    if params.AdjustableParameterOffsets.exists(sicd_xmltree):
        adjust_param_offsets = params.AdjustableParameterOffsets.from_xml(sicd_xmltree)

    return calc.scene_to_image(
        proj_metadata,
        scene_points,
        adjust_param_offsets=adjust_param_offsets,
        delta_gp_s2i=delta_gp_s2i,
        maxiter=maxiter,
        bistat_delta_gp_gpp=bistat_delta_gp_gpp,
        bistat_maxiter=bistat_maxiter,
    )


def image_to_constant_hae_surface(
    sicd_xmltree: lxml.etree.ElementTree,
    image_grid_locations: npt.ArrayLike,
    hae0: npt.ArrayLike,
    *,
    delta_hae_max: float = 1.0,
    nlim: int = 3,
    bistat_delta_gp_gpp: float = 0.010,
    bistat_maxiter: int = 10,
) -> tuple[npt.NDArray, npt.NDArray, bool]:
    """Project image coordinates to a surface of constant HAE.

    Parameters
    ----------
    sicd_xmltree : lxml.etree.ElementTree
        SICD XML metadata.
    image_grid_locations : (..., 2) array_like
        Image coordinates with xrow/ycol in meters in the last dimension.
    hae0 : array_like
        Surface height above the WGS-84 reference ellipsoid for projection points in meters.
    delta_hae_max : float, optional
        Height threshold for convergence of iterative projection sequence in meters.
    nlim : int, optional
        Maximum number of iterations to perform.
    bistat_delta_gp_gpp : float, optional
        (Bistatic only) Ground plane displacement threshold for intermediate ground
        plane points in meters.
    bistat_maxiter : int, optional
        (Bistatic only) Maximum number of intermediate bistatic R/Rdot to Ground Plane
        iterations to perform per scene-to-image iteration.

    Returns
    -------
    spp_tgt : (..., 3) ndarray
        Array of points on the HAE0 surface with ECEF (WGS 84 cartesian) X, Y, Z components in meters
        in the last dimension.
    delta_hae : ndarray
        Height difference at point GPP relative to HAE0.
    success : bool
        Whether or not all height differences, ``delta_hae`` are less than or equal
        to the threshold, ``delta_hae_max``.
    """
    proj_metadata = params.MetadataParams.from_xml(sicd_xmltree)
    projection_sets = calc.compute_projection_sets(proj_metadata, image_grid_locations)

    if params.AdjustableParameterOffsets.exists(sicd_xmltree):
        adjust_param_offsets = params.AdjustableParameterOffsets.from_xml(sicd_xmltree)
        projection_sets = calc.compute_and_apply_offsets(
            proj_metadata, projection_sets, adjust_param_offsets
        )

    return calc.r_rdot_to_constant_hae_surface(
        proj_metadata.LOOK,
        proj_metadata.SCP,
        proj_metadata.Collect_Type,
        projection_sets,
        hae0,
        delta_hae_max=delta_hae_max,
        nlim=nlim,
        bistat_delta_gp_gpp=bistat_delta_gp_gpp,
        bistat_maxiter=bistat_maxiter,
    )
