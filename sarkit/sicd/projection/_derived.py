"""Functions that can select monostatic or bistatic methods."""

from collections.abc import Callable

import lxml.etree
import numpy as np
import numpy.typing as npt

from . import _calc as calc
from . import _params as params


def _get_projsets(sicd_xmltree, image_grid_locations):
    """Extract metadata params and compute projection set(s), with APOs applied if available"""
    proj_metadata = params.MetadataParams.from_xml(sicd_xmltree)
    projection_sets = calc.compute_projection_sets(proj_metadata, image_grid_locations)

    if params.AdjustableParameterOffsets.exists(sicd_xmltree):
        adjust_param_offsets = params.AdjustableParameterOffsets.from_xml(sicd_xmltree)
        projection_sets = calc.apply_apos(
            proj_metadata, projection_sets, adjust_param_offsets
        )
    return proj_metadata, projection_sets


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
    proj_metadata, projection_sets = _get_projsets(sicd_xmltree, image_grid_locations)
    method = (
        {True: "monostatic", False: "bistatic"}[proj_metadata.is_monostatic()]
        if method is None
        else method
    )
    if method == "monostatic":
        assert isinstance(projection_sets, params.ProjectionSetsMono)
        gpp_tgt = calc.r_rdot_to_ground_plane_mono(
            proj_metadata.LOOK, projection_sets, gref, ugpn
        )
        delta_gp = np.full(gpp_tgt.shape[:-1], np.nan)
        delta_gp[np.isfinite(gpp_tgt).all(axis=-1)] = 0
        success = np.isfinite(gpp_tgt).all()
        return gpp_tgt, delta_gp, success
    if method == "bistatic":
        if isinstance(projection_sets, params.ProjectionSetsMono):
            projection_sets = params.ProjectionSetsBi(
                t_COA=projection_sets.t_COA,
                tr_COA=projection_sets.t_COA,
                tx_COA=projection_sets.t_COA,
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
            projection_sets,
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

    Refer to :py:func:`sarkit.sicd.projection.scene_to_image` for full documentation of the outputs and other
    parameters.

    Parameters
    ----------
    sicd_xmltree : lxml.etree.ElementTree
        SICD XML metadata.

    Returns
    -------
    image_grid_locations : (..., 2) ndarray
    delta_gp : ndarray
    success : bool

    Other Parameters
    ----------------
    *args, **kwargs
        For other arguments, refer to :py:func:`sarkit.sicd.projection.scene_to_image`

    See Also
    --------
    :py:func:`sarkit.sicd.projection.scene_to_image`
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

    Refer to `r_rdot_to_constant_hae_surface` for full documentation of the outputs and other parameters.

    Parameters
    ----------
    sicd_xmltree : lxml.etree.ElementTree
        SICD XML metadata.
    image_grid_locations : (..., 2) array_like
        Image coordinates with xrow/ycol in meters in the last dimension.

    Returns
    -------
    spp_tgt : (..., 3) ndarray
    delta_hae : ndarray
    success : bool

    Other Parameters
    ----------------
    *args, **kwargs
        For other arguments, refer to :py:func:`~sarkit.sicd.projection.r_rdot_to_constant_hae_surface`

    See Also
    --------
    :py:func:`~sarkit.sicd.projection.r_rdot_to_constant_hae_surface`
    """
    proj_metadata, projection_sets = _get_projsets(sicd_xmltree, image_grid_locations)
    return calc.r_rdot_to_constant_hae_surface(
        proj_metadata.LOOK,
        proj_metadata.SCP,
        projection_sets,
        hae0,
        delta_hae_max=delta_hae_max,
        nlim=nlim,
        bistat_delta_gp_gpp=bistat_delta_gp_gpp,
        bistat_maxiter=bistat_maxiter,
    )


def image_to_dem_surface(
    sicd_xmltree: lxml.etree.ElementTree,
    image_grid_location: npt.ArrayLike,
    ecef2dem_func: Callable[[np.ndarray], np.ndarray],
    hae_min: float,
    hae_max: float,
    delta_dist_dem: float,
    *,
    delta_dist_rrc: float = 10.0,
    delta_hd_lim: float = 0.001,
    **kwargs,
) -> npt.NDArray:
    """Project image coordinate to a surface described by a Digital Elevation Model.

    Refer to `r_rdot_to_dem_surface` for full documentation of the outputs and other parameters.

    Parameters
    ----------
    sicd_xmltree : lxml.etree.ElementTree
        SICD XML metadata
    image_grid_location : (2,) array_like
        Image coordinate with xrow/ycol in meters

    Returns
    -------
    s : ndarray

    Other Parameters
    ----------------
    *args, **kwargs
        For other arguments, refer to :py:func:`~sarkit.sicd.projection.r_rdot_to_dem_surface`

    See Also
    --------
    :py:func:`~sarkit.sicd.projection.r_rdot_to_dem_surface`
    """
    proj_metadata, projection_sets = _get_projsets(sicd_xmltree, image_grid_location)
    return calc.r_rdot_to_dem_surface(
        proj_metadata.LOOK,
        proj_metadata.SCP,
        projection_sets,
        ecef2dem_func=ecef2dem_func,
        hae_min=hae_min,
        hae_max=hae_max,
        delta_dist_dem=delta_dist_dem,
        delta_dist_rrc=delta_dist_rrc,
        delta_hd_lim=delta_hd_lim,
        **kwargs,
    )
