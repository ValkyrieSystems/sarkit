"""Calculations from SICD Volume 3 Image Projections Description Document"""

import itertools
from collections.abc import Callable

import numpy as np
import numpy.polynomial.polynomial as npp
import numpy.typing as npt

import sarkit.wgs84
from sarkit import _constants

from . import _params as params


def _xyzpolyval(x, c):
    """Similar to polyval but moves xyz to last dim."""
    assert c.ndim == 2
    assert c.shape[1] == 3
    return np.moveaxis(npp.polyval(x, c), 0, -1)


def image_grid_to_image_plane_point(
    scp: npt.ArrayLike,
    urow: npt.ArrayLike,
    ucol: npt.ArrayLike,
    image_grid_locations: npt.ArrayLike,
) -> npt.NDArray:
    """Convert image pixel grid locations to corresponding image plane positions.

    Parameters
    ----------
    scp : (..., 3) array_like
        SCP position in ECEF coordinates (m).
    urow, ucol : (..., 3) array_like
        Unit vectors in the increasing row and column directions in ECEF coordinates.
    image_grid_locations : (..., 2) array_like
        N-D array of image coordinates with xrow/ycol in meters in the last dimension.

    Returns
    -------
    (..., 3) ndarray
        Array of image plane points with ECEF (WGS 84 cartesian) X, Y, Z components in meters
        in the last dimension.

    """
    image_grid_locations = np.asarray(image_grid_locations)
    xrow = image_grid_locations[..., 0]
    ycol = image_grid_locations[..., 1]
    # Compute displacement from SCP to image plane points
    delta_ip_pts = xrow[..., np.newaxis] * urow + ycol[..., np.newaxis] * ucol

    # Compute image plane point positions
    return np.asarray(scp) + delta_ip_pts


def image_plane_point_to_image_grid(
    scp: npt.ArrayLike,
    urow: npt.ArrayLike,
    ucol: npt.ArrayLike,
    image_plane_points: npt.ArrayLike,
) -> npt.NDArray:
    """Convert image plane positions to corresponding image pixel grid locations.

    Parameters
    ----------
    scp : (..., 3) array_like
        SCP position in ECEF coordinates (m).
    urow, ucol : (..., 3) array_like
        Unit vectors in the increasing row and column directions in ECEF coordinates.
    image_plane_points : (..., 3) array_like
        Array of image plane points with ECEF (WGS 84 cartesian) X, Y, Z components in meters
        in the last dimension.

    Returns
    -------
    (..., 2) ndarray
        Array of image coordinates with xrow/ycol in meters in the last dimension.

    """
    # Compute cosine and sine of angle between uRow and uCol and 2x2 matrix.
    cos_theta_col = np.dot(urow, ucol)
    sin_theta_col = np.sqrt(1 - cos_theta_col**2)
    m_il_ippt = (sin_theta_col ** (-2)) * np.array(
        [[1.0, -cos_theta_col], [-cos_theta_col, 1.0]]
    )

    # Compute displacement vector from SCP to image plane points. Compute image grid locations.
    delta_ip_pt = np.asarray(image_plane_points) - np.asarray(scp)
    il = (
        m_il_ippt
        @ np.stack(
            [
                (delta_ip_pt * urow).sum(axis=-1),
                (delta_ip_pt * ucol).sum(axis=-1),
            ],
            axis=-1,
        )[..., np.newaxis]
    )
    return il[..., 0]  # remove residual dimension from matrix multiply


def compute_coa_time(
    ct_coa: npt.ArrayLike,
    image_grid_locations: npt.ArrayLike,
) -> npt.NDArray:
    """Compute Center of Aperture times for specified image grid locations.

    Parameters
    ----------
    ct_coa : array_like
        Center Of Aperture time polynomial coefficients.
    image_grid_locations : (..., 2) array_like
        N-D array of image coordinates with xrow/ycol in meters in the last dimension.

    Returns
    -------
    ndarray
        Array of shape ``image_grid_locations.shape[:-1]`` containing center of aperture
        times in seconds relative to collect start.

    """
    tgts = np.asarray(image_grid_locations)
    xrow = tgts[..., 0]
    ycol = tgts[..., 1]
    return npp.polyval2d(xrow, ycol, np.asarray(ct_coa))


def compute_coa_pos_vel(
    proj_metadata: params.MetadataParams,
    t_coa: npt.ArrayLike,
) -> params.CoaPosVelsLike:
    """Compute Center of Aperture positions and velocities at specified COA times.

    The parameters that specify the positions and velocities are dependent on
    ``proj_metadata.Collect_Type``:

    MONOSTATIC
        ARP_COA, VARP_COA

    BISTATIC
        GRP_COA, tx_COA, tr_COA, Xmt_COA, VXmt_COA, Rcv_COA, VRcv_COA

    Parameters
    ----------
    proj_metadata : MetadataParams
        Metadata parameters relevant to projection.
    t_coa : array_like
        Center of aperture times in seconds relative to collect start.

    Returns
    -------
    CoaPosVelsLike
        Ensemble of COA sensor positions and velocities
    """
    t_coa = np.asarray(t_coa)
    if proj_metadata.is_monostatic():
        return params.CoaPosVelsMono(
            ARP_COA=_xyzpolyval(t_coa, proj_metadata.ARP_Poly),
            VARP_COA=_xyzpolyval(t_coa, npp.polyder(proj_metadata.ARP_Poly)),
        )

    # Bistatic Image: COA APC Positions & Velocities
    # Compute GRP position at time t=tcoa
    grp_coa = _xyzpolyval(t_coa, proj_metadata.GRP_Poly)

    # Compute transmit time
    assert proj_metadata.Xmt_Poly is not None
    x0 = _xyzpolyval(t_coa, proj_metadata.Xmt_Poly)
    r_x0 = np.linalg.norm(x0 - grp_coa)
    tx_coa = t_coa - r_x0 / _constants.speed_of_light

    # Compute transmit APC position and velocity
    xmt_coa = _xyzpolyval(tx_coa, proj_metadata.Xmt_Poly)
    vxmt_coa = _xyzpolyval(tx_coa, npp.polyder(proj_metadata.Xmt_Poly))

    # Compute receive time
    r0 = _xyzpolyval(t_coa, proj_metadata.Rcv_Poly)
    r_r0 = np.linalg.norm(r0 - grp_coa)
    tr_coa = t_coa + r_r0 / _constants.speed_of_light

    # Compute receive APC position and velocity
    assert proj_metadata.Rcv_Poly is not None
    rcv_coa = _xyzpolyval(tr_coa, proj_metadata.Rcv_Poly)
    vrcv_coa = _xyzpolyval(tr_coa, npp.polyder(proj_metadata.Rcv_Poly))

    return params.CoaPosVelsBi(
        GRP_COA=grp_coa,
        tx_COA=tx_coa,
        tr_COA=tr_coa,
        Xmt_COA=xmt_coa,
        VXmt_COA=vxmt_coa,
        Rcv_COA=rcv_coa,
        VRcv_COA=vrcv_coa,
    )


def compute_scp_coa_r_rdot(proj_metadata: params.MetadataParams) -> tuple[float, float]:
    """Compute COA range and range-rate for the Scene Center Point.

    The SCP R/Rdot projection contour is dependent upon the Collect Type.

    Parameters
    ----------
    proj_metadata : MetadataParams
        Metadata parameters relevant to projection.

    Returns
    -------
    r, rdot : float
        Range and range rate relative to the COA positions and velocities.
        For a monostatic image, ``r`` and ``rdot`` are relative to the ARP.
        For a bistatic image, ``r`` and ``rdot`` are averages relative to the COA APCs.

    """
    if proj_metadata.is_monostatic():
        r_scp_coa = np.linalg.norm(proj_metadata.ARP_SCP_COA - proj_metadata.SCP)
        u_pt_scp_coa = (proj_metadata.ARP_SCP_COA - proj_metadata.SCP) / r_scp_coa
        rdot_scp_coa = np.dot(proj_metadata.VARP_SCP_COA, u_pt_scp_coa)
        return float(r_scp_coa), float(rdot_scp_coa)
    bistatic_results = _scp_r_rdot_projection_contour_bistatic(proj_metadata)
    return bistatic_results["r_avg_scp_coa"], bistatic_results["rdot_avg_scp_coa"]


def _scp_r_rdot_projection_contour_bistatic(proj_metadata):
    """SCP R/Rdot Projection Contour calculations for Collect Type = Bistatic

    Private method for re-use.
    """
    # Bistatic
    r_xmt_scp_coa = np.linalg.norm(proj_metadata.Xmt_SCP_COA - proj_metadata.SCP)
    u_xmt_scp_coa = (proj_metadata.Xmt_SCP_COA - proj_metadata.SCP) / r_xmt_scp_coa
    rdot_xmt_scp_coa = np.dot(proj_metadata.VXmt_SCP_COA, u_xmt_scp_coa)
    u_xmtdot_scp_coa = (
        proj_metadata.VXmt_SCP_COA - rdot_xmt_scp_coa * u_xmt_scp_coa
    ) / r_xmt_scp_coa

    r_rcv_scp_coa = np.linalg.norm(proj_metadata.Rcv_SCP_COA - proj_metadata.SCP)
    u_rcv_scp_coa = (proj_metadata.Rcv_SCP_COA - proj_metadata.SCP) / r_rcv_scp_coa
    rdot_rcv_scp_coa = np.dot(proj_metadata.VRcv_SCP_COA, u_rcv_scp_coa)
    u_rcvdot_scp_coa = (
        proj_metadata.VRcv_SCP_COA - rdot_rcv_scp_coa * u_rcv_scp_coa
    ) / r_rcv_scp_coa

    return {
        "r_avg_scp_coa": float((r_xmt_scp_coa + r_rcv_scp_coa) / 2.0),
        "rdot_avg_scp_coa": float((rdot_xmt_scp_coa + rdot_rcv_scp_coa) / 2.0),
        "bp_scp_coa": (u_xmt_scp_coa + u_rcv_scp_coa) / 2.0,
        "bpdot_scp_coa": (u_xmtdot_scp_coa + u_rcvdot_scp_coa) / 2.0,
    }


def compute_scp_coa_slant_plane_normal(
    proj_metadata: params.MetadataParams,
) -> npt.NDArray:
    """Compute the slant plane unit normal for the Scene Center Point at its COA.

    The method for computing the SCP COA slant plane unit normal is dependent upon the
    collect type.

    Parameters
    ----------
    proj_metadata : MetadataParams
        Metadata parameters relevant to projection.

    Returns
    -------
    (3,) ndarray
        SCP COA slant plane unit normal with ECEF (WGS 84 cartesian) X, Y, Z components in meters.

    """
    if proj_metadata.is_monostatic():
        spn_scp_coa = proj_metadata.LOOK * np.cross(
            (proj_metadata.ARP_SCP_COA - proj_metadata.SCP), proj_metadata.VARP_SCP_COA
        )
    else:
        bistatic_results = _scp_r_rdot_projection_contour_bistatic(proj_metadata)
        spn_scp_coa = proj_metadata.LOOK * np.cross(
            bistatic_results["bp_scp_coa"], bistatic_results["bpdot_scp_coa"]
        )
    return spn_scp_coa / np.linalg.norm(spn_scp_coa)


def compute_coa_r_rdot(
    proj_metadata: params.MetadataParams,
    image_grid_locations: npt.ArrayLike,
    t_coa: npt.ArrayLike,
    coa_pos_vels: params.CoaPosVelsLike,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute COA range and range-rate contours given other projection set components.

    COA R/Rdot computation is dependent upon Collect Type, Grid Type & IFA used.

    Parameters
    ----------
    proj_metadata : MetadataParams
        Metadata parameters relevant to projection.
    image_grid_locations : (..., 2) array_like
        N-D array of image coordinates with xrow/ycol in meters in the last dimension.
    t_coa : array_like
        Center of aperture times in seconds relative to collect start.
    coa_pos_vels : CoaPosVelsLike
        Ensemble of COA sensor positions and velocities

    Returns
    -------
    r, rdot : (...) ndarray
        N-D array containing the ranges and range rates relative to the COA positions
        and velocities.
        For a monostatic image, ``r`` and ``rdot`` are relative to the ARP.
        For a bistatic image, ``r`` and ``rdot`` are averages relative to the COA APCs.

    """
    if proj_metadata.Grid_Type == "RGAZIM":
        if proj_metadata.IFA == "PFA":
            return r_rdot_from_rgazim_pfa(
                proj_metadata, image_grid_locations, t_coa, coa_pos_vels
            )
        if proj_metadata.IFA == "RGAZCOMP":
            if not isinstance(coa_pos_vels, params.CoaPosVelsMono):
                raise ValueError("coa_pos_vels must be monostatic for RGAZCOMP")
            return r_rdot_from_rgazim_rgazcomp(
                proj_metadata, image_grid_locations, coa_pos_vels
            )
    if proj_metadata.Grid_Type == "RGZERO":
        return r_rdot_from_rgzero(proj_metadata, image_grid_locations, t_coa)
    if proj_metadata.Grid_Type == "XRGYCR":
        return r_rdot_from_xrgycr(proj_metadata, image_grid_locations, coa_pos_vels)
    if proj_metadata.Grid_Type == "XCTYAT":
        return r_rdot_from_xctyat(proj_metadata, image_grid_locations, coa_pos_vels)
    if proj_metadata.Grid_Type == "PLANE":
        return r_rdot_from_plane(proj_metadata, image_grid_locations, coa_pos_vels)
    raise ValueError("Insufficient metadata to perform projection")


def r_rdot_from_rgazim_pfa(
    proj_metadata: params.MetadataParams,
    image_grid_locations: npt.ArrayLike,
    t_coa: npt.ArrayLike,
    coa_pos_vels: params.CoaPosVelsLike,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Image Grid To R/Rdot: Grid_Type = RGAZIM & IFA = PFA."""

    tgts = np.asarray(image_grid_locations)
    rg_tgts = tgts[..., 0]
    az_tgts = tgts[..., 1]
    t_coa = np.asarray(t_coa)

    if proj_metadata.is_monostatic() and isinstance(
        coa_pos_vels, params.CoaPosVelsMono
    ):
        r_scp_vector = coa_pos_vels.ARP_COA - proj_metadata.SCP
        r_scp = np.linalg.norm(r_scp_vector, axis=-1)
        rdot_scp = (coa_pos_vels.VARP_COA * r_scp_vector).sum(-1) / r_scp
    elif proj_metadata.is_bistatic() and isinstance(coa_pos_vels, params.CoaPosVelsBi):
        pt_r_rdot_params = compute_pt_r_rdot_parameters(
            proj_metadata.LOOK,
            coa_pos_vels.Xmt_COA,
            coa_pos_vels.VXmt_COA,
            coa_pos_vels.Rcv_COA,
            coa_pos_vels.VRcv_COA,
            proj_metadata.SCP,
        )
        r_scp = pt_r_rdot_params.R_Avg_PT
        rdot_scp = pt_r_rdot_params.Rdot_Avg_PT
    else:
        raise RuntimeError(
            f"{type(coa_pos_vels)=} inconsistent with {proj_metadata.Collect_Type=}"
        )

    # Compute polar angle and its derivative with respect to time
    assert proj_metadata.cPA is not None
    theta = npp.polyval(t_coa, proj_metadata.cPA)
    dtheta_dt = npp.polyval(t_coa, npp.polyder(proj_metadata.cPA))

    # Compute polar aperture scale factor and its derivative with respect to polar angle
    assert proj_metadata.cKSF is not None
    ksf = npp.polyval(theta, proj_metadata.cKSF)
    dksf_dtheta = npp.polyval(theta, npp.polyder(proj_metadata.cKSF))

    # Compute spatial frequency phase slopes
    dphi_dka = rg_tgts * np.cos(theta) + az_tgts * np.sin(theta)
    dphi_dkc = -rg_tgts * np.sin(theta) + az_tgts * np.cos(theta)

    # Compute range relative to the SCP at COA
    delta_r = ksf * dphi_dka

    # Compute rdot relative to SCP at COA
    delta_rdot = (dksf_dtheta * dphi_dka + ksf * dphi_dkc) * dtheta_dt

    # Compute the range and range rate relative to the COA positions and velocities.
    r = r_scp + delta_r
    rdot = rdot_scp + delta_rdot
    return r, rdot


def r_rdot_from_rgazim_rgazcomp(
    proj_metadata: params.MetadataParams,
    image_grid_locations: npt.ArrayLike,
    coa_pos_vels: params.CoaPosVelsMono,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Image Grid To R/Rdot: Grid_Type = RGAZIM & IFA = RGAZCOMP."""

    assert proj_metadata.is_monostatic()

    tgts = np.asarray(image_grid_locations)
    rg_tgts = tgts[..., 0]
    az_tgts = tgts[..., 1]

    # Compute the range and range rate to the SCP at COA
    r_scp_vector = coa_pos_vels.ARP_COA - proj_metadata.SCP
    r_scp = np.linalg.norm(r_scp_vector, axis=-1)
    rdot_scp = (coa_pos_vels.VARP_COA * r_scp_vector).sum(-1) / r_scp

    # Compute the increment in cosine of the DCA at COA of the target and the increment in range rate
    delta_cos_dca = proj_metadata.AzSF * az_tgts
    delta_rdot = -np.linalg.norm(coa_pos_vels.VARP_COA, axis=-1) * delta_cos_dca

    # Compute the range and range rate to the target at COA
    r_tgt_coa = r_scp + rg_tgts
    rdot_tgt_coa = rdot_scp + delta_rdot
    return r_tgt_coa, rdot_tgt_coa


def r_rdot_from_rgzero(
    proj_metadata: params.MetadataParams,
    image_grid_locations: npt.ArrayLike,
    t_coa: npt.ArrayLike,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Image Grid To R/Rdot: Grid_Type = RGZERO."""

    assert proj_metadata.is_monostatic()

    tgts = np.asarray(image_grid_locations)
    rg_tgts = tgts[..., 0]
    az_tgts = tgts[..., 1]

    # Compute the range at closest approach and the time of closest approach for the image grid location
    assert proj_metadata.cT_CA is not None
    r_ca = proj_metadata.R_CA_SCP + rg_tgts
    t_ca = npp.polyval(az_tgts, proj_metadata.cT_CA)

    # Compute the ARP velocity at t_ca and compute the magnitude of the vector
    varp_ca = _xyzpolyval(t_ca, npp.polyder(proj_metadata.ARP_Poly))
    varp_ca_mag = np.linalg.norm(varp_ca, axis=-1)

    # Compute the Doppler Rate Scale Factor for image grid (rg_tgts, az_tgts)
    assert proj_metadata.cDRSF is not None
    drsf = npp.polyval2d(rg_tgts, az_tgts, proj_metadata.cDRSF)

    # Compute the time difference between the COA time and the CA time
    delta_t_coa = t_coa - t_ca

    # Compute the range and range rate relative to the ARP at COA
    r_tgt_coa = np.sqrt(r_ca**2 + drsf * varp_ca_mag**2 * delta_t_coa**2)
    rdot_tgt_coa = drsf / r_tgt_coa * varp_ca_mag**2 * delta_t_coa
    return r_tgt_coa, rdot_tgt_coa


def r_rdot_from_xrgycr(
    proj_metadata: params.MetadataParams,
    image_grid_locations: npt.ArrayLike,
    coa_pos_vels: params.CoaPosVelsLike,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Image Grid To R/Rdot: Grid_Type = XRGYCR.

    XRGYCR is a special case of a uniformly sampled image plane.
    """
    return r_rdot_from_plane(proj_metadata, image_grid_locations, coa_pos_vels)


def r_rdot_from_xctyat(
    proj_metadata: params.MetadataParams,
    image_grid_locations: npt.ArrayLike,
    coa_pos_vels: params.CoaPosVelsLike,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Image Grid To R/Rdot: Grid_Type = XCTYAT.

    XCTYAT is a special case of a uniformly sampled image plane.
    """
    return r_rdot_from_plane(proj_metadata, image_grid_locations, coa_pos_vels)


def r_rdot_from_plane(
    proj_metadata: params.MetadataParams,
    image_grid_locations: npt.ArrayLike,
    coa_pos_vels: params.CoaPosVelsLike,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Image Grid To R/Rdot: Grid_Type = PLANE."""

    tgts = np.asarray(image_grid_locations)
    x_tgt = tgts[..., 0]
    y_tgt = tgts[..., 1]

    # Compute the image plane point for image grid location (xrowTGT, ycolTGT), IP_TGT
    ip_tgt = (
        proj_metadata.SCP
        + (x_tgt[..., np.newaxis] * proj_metadata.uRow)
        + (y_tgt[..., np.newaxis] * proj_metadata.uCol)
    )

    # Compute the range and range rate to the image plane point IP_TGT relative to the COA positions and velocities
    if proj_metadata.is_monostatic() and isinstance(
        coa_pos_vels, params.CoaPosVelsMono
    ):
        r_tgt_coa = np.linalg.norm(coa_pos_vels.ARP_COA - ip_tgt, axis=-1)
        u_pt = (coa_pos_vels.ARP_COA - ip_tgt) / r_tgt_coa[..., np.newaxis]
        rdot_tgt_coa = (coa_pos_vels.VARP_COA * u_pt).sum(-1)

        return r_tgt_coa, rdot_tgt_coa
    if proj_metadata.is_bistatic() and isinstance(coa_pos_vels, params.CoaPosVelsBi):
        pt_r_rdot_params = compute_pt_r_rdot_parameters(
            proj_metadata.LOOK,
            coa_pos_vels.Xmt_COA,
            coa_pos_vels.VXmt_COA,
            coa_pos_vels.Rcv_COA,
            coa_pos_vels.VRcv_COA,
            ip_tgt,
        )

        return pt_r_rdot_params.R_Avg_PT, pt_r_rdot_params.Rdot_Avg_PT
    raise RuntimeError(
        f"{type(coa_pos_vels)=} inconsistent with {proj_metadata.Collect_Type}"
    )


def apply_apos(
    proj_metadata: params.MetadataParams,
    init_proj_set: params.ProjectionSetsLike,
    apo_input_set: params.AdjustableParameterOffsets,
) -> params.ProjectionSetsLike:
    """Compute adjusted Center of Aperture projection set.

    The APO input set is used to compute a set of offsets that are applied to each COA projection
    set. For each projection set, the computed offsets are functions of the COA time(s) in the
    projection set. The resulting projection set is referred to as the “adjusted” COA projection
    set.

    MONOSTATIC
        t_COA, ARP_COA, VARP_COA, R_COA, Rdot_COA

    BISTATIC
        t_COA, tx_COA, tr_COA, Xmt_COA, VXmt_COA, Rcv_COA, VRcv_COA, R_Avg_COA, Rdot_Avg_COA

    Parameters
    ----------
    proj_metadata : MetadataParams
        Metadata parameters relevant to projection.
    init_proj_set : ProjectionSetsLike
        Initial projection set.
    apo_input_set : AdjustableParameterOffsets
        Input APO set, used to compute the offsets to be added to the initial
        COA projection set to form the adjusted COA projection set.

    Returns
    -------
    ProjectionSetsLike
        Ensemble of adjusted Center of Aperture projection sets.

    """
    if proj_metadata.is_monostatic():
        assert apo_input_set.delta_ARP_SCP_COA is not None
        assert apo_input_set.delta_VARP is not None
        if not isinstance(init_proj_set, params.ProjectionSetsMono):
            raise TypeError(
                f"{type(init_proj_set)=} not instance of ProjectionSetsMono"
            )

        # The input APOs are used to compute the following offsets to be added to the initial COA projection set
        # to form the adjusted COA projection set.
        delta_ARP_COA = (  # noqa N806
            apo_input_set.delta_ARP_SCP_COA
            + apo_input_set.delta_VARP * (init_proj_set.t_COA - proj_metadata.t_SCP_COA)
        )
        delta_R_ARP = (  # noqa N806
            _constants.speed_of_light
            / 2
            * (apo_input_set.delta_tr_SCP_COA - apo_input_set.delta_tx_SCP_COA)
        )

        return params.ProjectionSetsMono(
            t_COA=init_proj_set.t_COA,
            ARP_COA=init_proj_set.ARP_COA + delta_ARP_COA,
            VARP_COA=init_proj_set.VARP_COA + apo_input_set.delta_VARP,
            R_COA=init_proj_set.R_COA + delta_R_ARP,
            Rdot_COA=init_proj_set.Rdot_COA,
        )

    assert apo_input_set.delta_Xmt_SCP_COA is not None
    assert apo_input_set.delta_VXmt is not None
    assert apo_input_set.f_Clk_X_SF is not None
    assert apo_input_set.delta_Rcv_SCP_COA is not None
    assert apo_input_set.delta_VRcv is not None
    assert apo_input_set.f_Clk_R_SF is not None
    if not isinstance(init_proj_set, params.ProjectionSetsBi):
        raise TypeError(f"{type(init_proj_set)=} not instance of ProjectionSetsBi")

    # For the transmit sensor, the transmit time offset and the clock frequency scale factor are
    # used to compute a transmit time offset
    T_Clk_X_SF = -apo_input_set.f_Clk_X_SF * (1.0 / (1 + apo_input_set.f_Clk_X_SF))  # noqa N806
    delta_tx_COA = apo_input_set.delta_tx_SCP_COA + T_Clk_X_SF * (  # noqa N806
        init_proj_set.tx_COA - proj_metadata.t_SCP_COA
    )

    # For the receive sensor, the receive time offset and the clock frequency scale factor are
    # used to compute a transmit time offset
    T_Clk_R_SF = -apo_input_set.f_Clk_R_SF * (1.0 / (1 + apo_input_set.f_Clk_R_SF))  # noqa N806
    delta_tr_COA = apo_input_set.delta_tr_SCP_COA + T_Clk_R_SF * (  # noqa N806
        init_proj_set.tr_COA - proj_metadata.t_SCP_COA
    )

    # The input APOs are used to compute the following offsets to be added to the initial COA
    # projection set to form the adjusted COA projection set
    delta_Xmt_COA = (  # noqa N806
        init_proj_set.VXmt_COA * delta_tx_COA
        + apo_input_set.delta_Xmt_SCP_COA
        + apo_input_set.delta_VXmt
        * (init_proj_set.tx_COA + delta_tx_COA - proj_metadata.t_SCP_COA)
    )
    delta_Rcv_COA = (  # noqa N806
        init_proj_set.VRcv_COA * delta_tr_COA
        + apo_input_set.delta_Rcv_SCP_COA
        + apo_input_set.delta_VRcv
        * (init_proj_set.tr_COA + delta_tr_COA - proj_metadata.t_SCP_COA)
    )
    delta_R_Avg_COA = (  # noqa N806
        _constants.speed_of_light / 2 * (delta_tr_COA - delta_tx_COA)
    )
    delta_Rdot_Avg_COA = (  # noqa N806
        _constants.speed_of_light / 2 * (T_Clk_R_SF - T_Clk_X_SF)
    )

    return params.ProjectionSetsBi(
        t_COA=init_proj_set.t_COA,
        tx_COA=init_proj_set.tx_COA + delta_tx_COA,
        tr_COA=init_proj_set.tr_COA + delta_tr_COA,
        Xmt_COA=init_proj_set.Xmt_COA + delta_Xmt_COA,
        VXmt_COA=init_proj_set.VXmt_COA + apo_input_set.delta_VXmt,
        Rcv_COA=init_proj_set.Rcv_COA + delta_Rcv_COA,
        VRcv_COA=init_proj_set.VRcv_COA + apo_input_set.delta_VRcv,
        R_Avg_COA=init_proj_set.R_Avg_COA + delta_R_Avg_COA,
        Rdot_Avg_COA=init_proj_set.Rdot_Avg_COA + delta_Rdot_Avg_COA,
    )


def compute_projection_sets(
    proj_metadata: params.MetadataParams,
    image_grid_locations: npt.ArrayLike,
) -> params.ProjectionSetsLike:
    """Compute Center of Aperture projection sets at specified image grid locations.

    For a selected image grid location, the COA projection set contains the parameters
    needed for computing precise image-to-scene projection. The parameters contained in
    the COA projection set are dependent upon the ``proj_metadata.Collect_Type``.

    MONOSTATIC
        t_COA, ARP_COA, VARP_COA, R_COA, Rdot_COA

    BISTATIC
        t_COA, tx_COA, tr_COA, Xmt_COA, VXmt_COA, Rcv_COA, VRcv_COA, R_Avg_COA, Rdot_Avg_COA

    Parameters
    ----------
    proj_metadata : MetadataParams
        Metadata parameters relevant to projection.
    image_grid_locations : (..., 2) array_like
        N-D array of image coordinates with xrow/ycol in meters in the last dimension.

    Returns
    -------
    ProjectionSetsLike
        Ensemble of Center of Aperture projection sets.

    """
    t_coa = compute_coa_time(proj_metadata.cT_COA, image_grid_locations)
    coa_pos_vels = compute_coa_pos_vel(proj_metadata, t_coa)
    r, rdot = compute_coa_r_rdot(
        proj_metadata, image_grid_locations, t_coa, coa_pos_vels
    )
    if proj_metadata.is_monostatic():
        assert isinstance(coa_pos_vels, params.CoaPosVelsMono)
        return params.ProjectionSetsMono(
            t_COA=t_coa,
            ARP_COA=coa_pos_vels.ARP_COA,
            VARP_COA=coa_pos_vels.VARP_COA,
            R_COA=r,
            Rdot_COA=rdot,
        )

    assert isinstance(coa_pos_vels, params.CoaPosVelsBi)
    return params.ProjectionSetsBi(
        t_COA=t_coa,
        tx_COA=coa_pos_vels.tx_COA,
        tr_COA=coa_pos_vels.tr_COA,
        Xmt_COA=coa_pos_vels.Xmt_COA,
        VXmt_COA=coa_pos_vels.VXmt_COA,
        Rcv_COA=coa_pos_vels.Rcv_COA,
        VRcv_COA=coa_pos_vels.VRcv_COA,
        R_Avg_COA=r,
        Rdot_Avg_COA=rdot,
    )


def _check_look(look):
    if look not in (-1, +1):
        raise ValueError(f"Invalid {look=}; must be +1 or -1")


def r_rdot_to_ground_plane_mono(
    look: int,
    projection_sets: params.ProjectionSetsMono,
    gref: npt.ArrayLike,
    ugpn: npt.ArrayLike,
) -> npt.NDArray:
    """Project along contours of constant range and range rate to an arbitrary plane.

    Parameters
    ----------
    look : {+1, -1}
        +1 if SideOfTrack = L, -1 if SideOfTrack = R
    projection_sets : ProjectionSetsMono
        Ensemble of Center of Aperture projection sets to project.
    gref : (3,) array_like
        Ground plane reference point with ECEF (WGS 84 cartesian) X, Y, Z components in meters.
    ugpn : (3,) array_like
        Unit normal vector to ground plane with ECEF (WGS 84 cartesian) X, Y, Z components in
        meters.

    Returns
    -------
    (..., 3) ndarray
        Array of ground plane points with ECEF (WGS 84 cartesian) X, Y, Z components in meters
        in the last dimension. NaNs are returned where no solution is found.

    """

    _check_look(look)

    # Assign unit vector in +Z direction
    gref = np.asarray(gref)
    uz = np.asarray(ugpn)

    # Compute ARP distance from the plane and ARP ground plane nadir (AGPN)
    arpz = np.asarray(((projection_sets.ARP_COA - gref) * uz).sum(axis=-1))
    arpz[np.abs(arpz) > projection_sets.R_COA] = np.nan  # No Solution
    agpn = projection_sets.ARP_COA - arpz[..., np.newaxis] * uz

    # Compute ground plane distance from ARP nadir to circle of constant range and sine/cosine graze
    g = np.sqrt(projection_sets.R_COA**2 - arpz**2)
    cos_graz = g / projection_sets.R_COA
    sin_graz = arpz / projection_sets.R_COA

    # Compute velocity components in x and y
    vz = (projection_sets.VARP_COA * uz).sum(axis=-1)
    vx = np.asarray(np.sqrt((projection_sets.VARP_COA**2).sum(axis=-1) - vz**2))
    vx[vx == 0] = np.nan  # No Solution

    # Orient +X direction in ground plane such that Vx > 0. Compute uX and uY
    ux = (projection_sets.VARP_COA - vz[..., np.newaxis] * uz) / vx[..., np.newaxis]
    uy = np.cross(uz, ux, axis=-1)

    # Compute the cosine of azimuth angle to ground plane points
    cos_az = np.asarray((-projection_sets.Rdot_COA + vz * sin_graz) / (vx * cos_graz))
    cos_az[(cos_az < -1.0) | (cos_az > 1.0)] = np.nan  # No Solution

    # Compute the sine of the azimuth angle
    sin_az = look * np.sqrt(1 - cos_az**2)

    # Compute the ground plane points
    return (
        agpn + (g * cos_az)[..., np.newaxis] * ux + (g * sin_az)[..., np.newaxis] * uy
    )


def r_rdot_to_ground_plane_bi(
    look: int,
    scp: npt.ArrayLike,
    projection_sets: params.ProjectionSetsBi,
    gref: npt.ArrayLike,
    ugpn: npt.ArrayLike,
    *,
    delta_gp_gpp: float = 0.010,
    maxiter: int = 10,
) -> tuple[npt.NDArray, npt.NDArray, bool]:
    """Project along bistatic contours of constant average range and range rate to an arbitrary plane.

    Parameters
    ----------
    look : {+1, -1}
        +1 if SideOfTrack = L, -1 if SideOfTrack = R
    scp : (3,) array_like
        SCP position in ECEF coordinates (m).
    projection_sets : ProjectionSetsBi
        Ensemble of Center of Aperture projection sets to project.
    gref : (3,) array_like
        Ground plane reference point with ECEF (WGS 84 cartesian) X, Y, Z components in meters.
    ugpn : (3,) array_like
        Unit normal vector to ground plane with ECEF (WGS 84 cartesian) X, Y, Z components in
        meters.
    delta_gp_gpp : float, optional
        Ground plane displacement threshold for final ground plane point in meters.
    maxiter : int, optional
        Maximum number of iterations to perform.

    Returns
    -------
    g : (..., 3) ndarray
        Array of ground plane points with ECEF (WGS 84 cartesian) X, Y, Z components in meters
        in the last dimension.
    delta_gp : ndarray
        Magnitude of the displacement from estimated point to the precise intersection
        of the target R/Rdot contour.
    success : bool
        Whether or not all displacement magnitudes, ``delta_gp`` are less than or equal
        to the threshold, ``delta_gp_gpp``.

    """
    _check_look(look)

    scp = np.asarray(scp)
    gref = np.asarray(gref)
    ugpn = np.asarray(ugpn)
    # Compute initial ground points
    scp_lat, scp_lon = sarkit.wgs84.cartesian_to_geodetic(scp)[:2]
    u_up_scp = np.stack(
        (
            np.cos(np.deg2rad(scp_lat)) * np.cos(np.deg2rad(scp_lon)),
            np.cos(np.deg2rad(scp_lat)) * np.sin(np.deg2rad(scp_lon)),
            np.sin(np.deg2rad(scp_lat)),
        ),
        axis=-1,
    )
    dist_gp = ((gref - scp) * ugpn).sum(axis=-1, keepdims=True) / (u_up_scp * ugpn).sum(
        axis=-1, keepdims=True
    )
    g_0 = scp + dist_gp * u_up_scp

    xmt, vxmt, rcv, vrcv, g, ugpn = np.broadcast_arrays(
        projection_sets.Xmt_COA,
        projection_sets.VXmt_COA,
        projection_sets.Rcv_COA,
        projection_sets.VRcv_COA,
        g_0,
        ugpn,
    )
    g = np.array(g)  # make writable
    delta_gp = np.full(g.shape[:-1], np.nan)
    success = False
    above_threshold = np.full(g.shape[:-1], True)
    for _ in range(maxiter):
        pt_r_rdot_params = compute_pt_r_rdot_parameters(
            look,
            xmt[above_threshold, :],
            vxmt[above_threshold, :],
            rcv[above_threshold, :],
            vrcv[above_threshold, :],
            g[above_threshold, :],
        )

        gp_xy_params = compute_gp_xy_parameters(
            g[above_threshold, :],
            ugpn[above_threshold, :],
            pt_r_rdot_params.bP_PT,
            pt_r_rdot_params.bPDot_PT,
        )

        delta_r_avg = (
            projection_sets.R_Avg_COA[above_threshold] - pt_r_rdot_params.R_Avg_PT
        )
        delta_rdot_avg = (
            projection_sets.Rdot_Avg_COA[above_threshold] - pt_r_rdot_params.Rdot_Avg_PT
        )

        delta_gxgy = (
            gp_xy_params.M_GPXY_RRdot
            @ np.stack((delta_r_avg, delta_rdot_avg), axis=-1)[..., np.newaxis]
        )
        delta_gp[above_threshold] = np.linalg.norm(delta_gxgy, axis=-2).squeeze(axis=-1)

        g[above_threshold, :] += (
            delta_gxgy[..., 0, :] * gp_xy_params.uGX
            + delta_gxgy[..., 1, :] * gp_xy_params.uGY
        )

        # Compare displacement to threshold.
        above_threshold = delta_gp > delta_gp_gpp
        success = bool((delta_gp <= delta_gp_gpp).all())
        if success:
            break
    return g, delta_gp, success


def compute_pt_r_rdot_parameters(
    look: int,
    xmt_coa: npt.ArrayLike,
    vxmt_coa: npt.ArrayLike,
    rcv_coa: npt.ArrayLike,
    vrcv_coa: npt.ArrayLike,
    scene_points: npt.ArrayLike,
) -> params.ScenePointRRdotParams:
    """Compute range and range rate parameters at specified scene point positions.

    Parameters
    ----------
    look : {+1, -1}
        +1 if SideOfTrack = L, -1 if SideOfTrack = R
    xmt_coa : (..., 3) ndarray
        Transmit APC positions with ECEF X, Y, Z components (m) in last dimension
    vxmt_coa : (..., 3) ndarray
        Transmit APC velocities with ECEF X, Y, Z components (m/s) in last dimension
    rcv_coa : (..., 3) ndarray
        Receive APC positions with ECEF X, Y, Z components (m) in last dimension
    vrcv_coa : (..., 3) ndarray
        Receive APC velocities with ECEF X, Y, Z components (m/s) in last dimension
    scene_points : (..., 3) array_like
        Array of scene points with ECEF X, Y, Z components (m) in last dimension

    Returns
    -------
    ScenePointRRdotParams
        Ensemble of range and range rate parameters for the specified scene points
    """
    _check_look(look)
    xmt_coa = np.asarray(xmt_coa)
    vxmt_coa = np.asarray(vxmt_coa)
    rcv_coa = np.asarray(rcv_coa)
    vrcv_coa = np.asarray(vrcv_coa)
    pt = np.asarray(scene_points)

    # Compute parameters for transmit APC relative to scene points
    r_xmt_pt = np.linalg.norm(xmt_coa - pt, axis=-1)
    u_xmt_pt = (xmt_coa - pt) / r_xmt_pt[..., np.newaxis]
    rdot_xmt_pt = (vxmt_coa * u_xmt_pt).sum(axis=-1)
    u_xmtdot_pt = (vxmt_coa - rdot_xmt_pt[..., np.newaxis] * u_xmt_pt) / r_xmt_pt[
        ..., np.newaxis
    ]

    # Compute parameters for receive APC relative to scene points
    r_rcv_pt = np.linalg.norm(rcv_coa - pt, axis=-1)
    u_rcv_pt = (rcv_coa - pt) / r_rcv_pt[..., np.newaxis]
    rdot_rcv_pt = (vrcv_coa * u_rcv_pt).sum(axis=-1)
    u_rcvdot_pt = (vrcv_coa - rdot_rcv_pt[..., np.newaxis] * u_rcv_pt) / r_rcv_pt[
        ..., np.newaxis
    ]

    # Compute average range and average range rate
    r_avg_pt = (r_xmt_pt + r_rcv_pt) / 2.0
    rdot_avg_pt = (rdot_xmt_pt + rdot_rcv_pt) / 2.0

    # Compute bistatic pointing vector and its derivative w.r.t. time
    bp_pt = (u_xmt_pt + u_rcv_pt) / 2.0
    bpdot_pt = (u_xmtdot_pt + u_rcvdot_pt) / 2.0

    # Compute bistatic slant plane unit normal vector
    spn_pt = look * np.cross(bp_pt, bpdot_pt)
    uspn_pt = spn_pt / np.linalg.norm(spn_pt)

    return params.ScenePointRRdotParams(
        R_Avg_PT=r_avg_pt,
        Rdot_Avg_PT=rdot_avg_pt,
        bP_PT=bp_pt,
        bPDot_PT=bpdot_pt,
        uSPN_PT=uspn_pt,
    )


def compute_gp_xy_parameters(
    scene_points: npt.ArrayLike,
    ugpn: npt.ArrayLike,
    bp_points: npt.ArrayLike,
    bpdot_points: npt.ArrayLike,
) -> params.ScenePointGpXyParams:
    """Compute the basis vectors and sensitivity matrices for a ground plane coordinate system.

    Parameters
    ----------
    scene_points : (..., 3) array_like
        Array of scene points with ECEF (WGS 84 cartesian) X, Y, Z components in meters in the
        last dimension.
    ugpn : (..., 3) array_like
        Unit normal vector to ground plane with ECEF (WGS 84 cartesian) X, Y, Z components in
        meters.
    bp_points, bpdot_points : (..., 3) array_like
        Bistatic pointing vector and its derivative with respect to time.

    Returns
    -------
    ScenePointGpXyParams
        Ensemble of scene point ground plane XY parameters for the specified scene points
    """
    pt = np.asarray(scene_points)
    ugpn = np.asarray(ugpn)
    bp_pt = np.asarray(bp_points)
    bpdot_pt = np.asarray(bpdot_points)

    gx = bp_pt - ugpn * (bp_pt * ugpn).sum(axis=-1, keepdims=True)
    ugx = gx / np.linalg.norm(gx, axis=-1, keepdims=True)

    _sgn_criteria = (ugpn * pt).sum(axis=-1, keepdims=True)
    sgn = np.full_like(_sgn_criteria, -1.0)
    sgn[_sgn_criteria > 0] = +1.0

    gy = sgn * np.cross(ugpn, ugx)
    ugy = gy / np.linalg.norm(gy, axis=-1, keepdims=True)

    m_rrdot_gpxy = np.negative(
        np.stack(
            (
                np.stack(
                    ((bp_pt * ugx).sum(axis=-1), np.zeros_like(ugx[..., 0])), axis=-1
                ),
                np.stack(
                    ((bpdot_pt * ugx).sum(axis=-1), (bpdot_pt * ugy).sum(axis=-1)),
                    axis=-1,
                ),
            ),
            axis=-1,
        )
    )

    m_gpxy_rrdot = np.linalg.inv(m_rrdot_gpxy)
    return params.ScenePointGpXyParams(
        uGX=ugx,
        uGY=ugy,
        M_RRdot_GPXY=m_rrdot_gpxy,
        M_GPXY_RRdot=m_gpxy_rrdot,
    )


def scene_to_image(
    proj_metadata: params.MetadataParams,
    scene_points: npt.ArrayLike,
    *,
    adjust_param_offsets: params.AdjustableParameterOffsets | None = None,
    delta_gp_s2i: float = 0.001,
    maxiter: int = 10,
    bistat_delta_gp_gpp: float = 0.010,
    bistat_maxiter: int = 10,
) -> tuple[npt.NDArray, npt.NDArray, bool]:
    """Map geolocated points in the three-dimensional scene to image grid locations.

    Parameters
    ----------
    proj_metadata : MetadataParams
        Metadata parameters relevant to projection.
    scene_points : (..., 3) array_like
        Array of scene points with ECEF (WGS 84 cartesian) X, Y, Z components in meters in the
        last dimension.
    adjust_param_offsets : AdjustableParameterOffsets, optional
        Used to compute the offsets to be added to the initial COA projection set to form the
        adjusted COA projection set.
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
    s = np.asarray(scene_points)
    # Compute the spherical earth ground plane unit normals (use Spherical Earth GPN)
    u_gpn = s / np.linalg.norm(s, axis=-1, keepdims=True)

    # Compute projection scale factor
    u_proj = compute_scp_coa_slant_plane_normal(proj_metadata)
    ipn = np.cross(proj_metadata.uRow, proj_metadata.uCol)
    u_ipn = ipn / np.linalg.norm(ipn)
    sf = np.dot(u_proj, u_ipn)

    # Set initial ground plane positions to scene point positions.
    g = s.copy()

    image_grid_locations = np.full(s.shape[:-1] + (2,), np.nan)
    delta_gp = np.full(s.shape[:-1], np.nan)
    success = False
    p = np.full_like(s, np.nan)
    above_threshold = np.full(s.shape[:-1], True)
    r_rdot_to_ground_success = np.full(s.shape[:-1], False)
    for _ in range(maxiter):
        # Project ground points to image plane points
        dist = sf ** (-1) * ((proj_metadata.SCP - g[above_threshold]) * u_ipn).sum(
            axis=-1
        )
        i = g[above_threshold] + dist[..., np.newaxis] * u_proj

        # For image plane points, compute the associated image grid coordinates.
        image_grid_locations[above_threshold] = image_plane_point_to_image_grid(
            proj_metadata.SCP, proj_metadata.uRow, proj_metadata.uCol, i
        )

        # Compute the COA projection sets
        projection_sets = compute_projection_sets(
            proj_metadata, image_grid_locations[above_threshold]
        )

        if adjust_param_offsets is not None:
            projection_sets = apply_apos(
                proj_metadata, projection_sets, adjust_param_offsets
            )

        # Compute precise projection to ground plane.
        if proj_metadata.is_monostatic():
            assert isinstance(projection_sets, params.ProjectionSetsMono)
            p[above_threshold] = r_rdot_to_ground_plane_mono(
                proj_metadata.LOOK,
                projection_sets,
                s[above_threshold],
                u_gpn[above_threshold],
            )
            r_rdot_to_ground_success[above_threshold] = np.isfinite(
                p[above_threshold]
            ).all(axis=-1)
        else:
            assert isinstance(projection_sets, params.ProjectionSetsBi)
            p[above_threshold], _, r_rdot_to_ground_success[above_threshold] = (
                r_rdot_to_ground_plane_bi(
                    proj_metadata.LOOK,
                    proj_metadata.SCP,
                    projection_sets,
                    s[above_threshold],
                    u_gpn[above_threshold],
                    delta_gp_gpp=bistat_delta_gp_gpp,
                    maxiter=bistat_maxiter,
                )
            )

        # Compute displacement between ground plane points and scene points.
        delta_p = s - p
        delta_gp = np.linalg.norm(delta_p, axis=-1)

        # Compare displacement to threshold.
        above_threshold = delta_gp > delta_gp_s2i
        g[above_threshold] += delta_p[above_threshold]
        success = bool(
            (delta_gp <= delta_gp_s2i).all() and r_rdot_to_ground_success.all()
        )
        if success:
            break
    return image_grid_locations, delta_gp, success


def r_rdot_to_constant_hae_surface(
    look: int,
    scp: npt.ArrayLike,
    projection_sets: params.ProjectionSetsLike,
    hae0: npt.ArrayLike,
    *,
    delta_hae_max: float = 1.0,
    nlim: int = 3,
    bistat_delta_gp_gpp: float = 0.010,
    bistat_maxiter: int = 10,
) -> tuple[npt.NDArray, npt.NDArray, bool]:
    """Project along contours of constant range and range rate to a surface of constant HAE.

    Parameters
    ----------
    look : {+1, -1}
        +1 if SideOfTrack = L, -1 if SideOfTrack = R
    scp : (3,) array_like
        SCP position in ECEF coordinates (m).
    projection_sets : ProjectionSetsLike
        Ensemble of Center of Aperture projection sets to project.
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
    hae0 = np.asarray(hae0)
    scp = np.asarray(scp)

    _check_look(look)

    def _calc_up(lat_deg, lon_deg):
        return np.stack(
            (
                np.cos(np.deg2rad(lat_deg)) * np.cos(np.deg2rad(lon_deg)),
                np.cos(np.deg2rad(lat_deg)) * np.sin(np.deg2rad(lon_deg)),
                np.sin(np.deg2rad(lat_deg)),
            ),
            axis=-1,
        )

    # Compute parameters for ground plane 1
    scp_lat, scp_lon, scp_hae = sarkit.wgs84.cartesian_to_geodetic(scp)
    u_gpn1 = _calc_up(scp_lat, scp_lon)
    gref1 = scp + (hae0 - scp_hae)[..., np.newaxis] * u_gpn1

    if isinstance(projection_sets, params.ProjectionSetsMono):
        gref, u_gpn, arp, varp = np.broadcast_arrays(
            gref1, u_gpn1, projection_sets.ARP_COA, projection_sets.VARP_COA
        )
    else:
        gref, u_gpn, xmt, vxmt, rcv, vrcv = np.broadcast_arrays(
            gref1,
            u_gpn1,
            projection_sets.Xmt_COA,
            projection_sets.VXmt_COA,
            projection_sets.Rcv_COA,
            projection_sets.VRcv_COA,
        )

    hae0 = np.broadcast_to(hae0, gref.shape[:-1])
    gref = np.array(gref)  # make writable
    u_gpn = np.array(u_gpn)  # make writable
    u_up = np.full(gref.shape, np.nan)
    gpp = np.full(gref.shape, np.nan)
    delta_hae = np.full(gref.shape[:-1], np.nan)
    success = False
    above_threshold = np.full(gref.shape[:-1], True)
    r_rdot_to_plane_success = np.full(gref.shape[:-1], False)
    for _ in range(nlim):
        # Compute precise projection to ground plane.
        if isinstance(projection_sets, params.ProjectionSetsMono):
            gpp[above_threshold, :] = r_rdot_to_ground_plane_mono(
                look,
                params.ProjectionSetsMono(
                    t_COA=projection_sets.t_COA[above_threshold],
                    ARP_COA=arp[above_threshold, :],
                    VARP_COA=varp[above_threshold, :],
                    R_COA=projection_sets.R_COA[above_threshold],
                    Rdot_COA=projection_sets.Rdot_COA[above_threshold],
                ),
                gref[above_threshold, :],
                u_gpn[above_threshold, :],
            )
            r_rdot_to_plane_success[above_threshold] = np.isfinite(
                gpp[above_threshold, :]
            ).all(axis=-1)
        else:
            gpp[above_threshold, :], _, r_rdot_to_plane_success[above_threshold] = (
                r_rdot_to_ground_plane_bi(
                    look,
                    scp,
                    params.ProjectionSetsBi(
                        t_COA=projection_sets.t_COA[above_threshold],
                        tx_COA=projection_sets.t_COA[above_threshold],  # unused
                        tr_COA=projection_sets.t_COA[above_threshold],  # unused
                        Xmt_COA=xmt[above_threshold, :],
                        VXmt_COA=vxmt[above_threshold, :],
                        Rcv_COA=rcv[above_threshold, :],
                        VRcv_COA=vrcv[above_threshold, :],
                        R_Avg_COA=projection_sets.R_Avg_COA[above_threshold],
                        Rdot_Avg_COA=projection_sets.Rdot_Avg_COA[above_threshold],
                    ),
                    gref[above_threshold, :],
                    u_gpn[above_threshold, :],
                    delta_gp_gpp=bistat_delta_gp_gpp,
                    maxiter=bistat_maxiter,
                )
            )

        # Convert from ECEF to WGS 84 geodetic
        gpp_llh = sarkit.wgs84.cartesian_to_geodetic(gpp[above_threshold, :])

        # Compute unit vector in increasing height direction and height difference at GPP.
        u_up[above_threshold, :] = _calc_up(gpp_llh[..., 0], gpp_llh[..., 1])
        delta_hae[above_threshold] = gpp_llh[..., 2] - hae0[above_threshold]

        # Check if GPP is sufficiently close to HAE0 surface.
        above_threshold = delta_hae > delta_hae_max
        success = bool(
            (delta_hae <= delta_hae_max).all() and r_rdot_to_plane_success.all()
        )
        if success:
            break
        gref[above_threshold, :] = (
            gpp[above_threshold, :]
            - delta_hae[above_threshold][..., np.newaxis] * u_up[above_threshold, :]
        )
        u_gpn[above_threshold, :] = u_up[above_threshold, :]

    # Compute slant plane normal tangent to R/Rdot contour at GPP.
    if isinstance(projection_sets, params.ProjectionSetsMono):
        spn = look * np.cross(varp, gpp - arp)
        u_spn = spn / np.linalg.norm(spn, axis=-1, keepdims=True)
    else:
        gpp_r_rdot_params = compute_pt_r_rdot_parameters(
            look,
            projection_sets.Xmt_COA,
            projection_sets.VXmt_COA,
            projection_sets.Rcv_COA,
            projection_sets.VRcv_COA,
            gpp,
        )
        u_spn = gpp_r_rdot_params.uSPN_PT

    # Compute straight-line projection from GPP along uSPN to point SLP.
    sf = (u_up * u_spn).sum(axis=-1, keepdims=True)
    slp = gpp - (delta_hae[..., np.newaxis] * u_spn) / sf

    # Convert SLP from ECEF to geodetic
    slp_llh = sarkit.wgs84.cartesian_to_geodetic(slp)

    # Assign surface point spp by adjusting HAE to be on HAE0 surface.
    spp_llh = slp_llh.copy()
    spp_llh[..., 2] = hae0

    # Convert SPP from geodetic to ECEF
    spp_tgt = sarkit.wgs84.geodetic_to_cartesian(spp_llh)

    return spp_tgt, delta_hae, success


def r_rdot_to_dem_surface(
    look: int,
    scp: npt.ArrayLike,
    projection_set: params.ProjectionSetsLike,
    ecef2dem_func: Callable[[np.ndarray], np.ndarray],
    hae_min: float,
    hae_max: float,
    delta_dist_dem: float,
    *,
    delta_dist_rrc: float = 10.0,
    delta_hd_lim: float = 0.001,
    **kwargs,
) -> npt.NDArray:
    """Project along a contour of constant range and range rate to a surface described by a Digital Elevation Model.

    Parameters
    ----------
    look : {+1, -1}
        +1 if SideOfTrack = L, -1 if SideOfTrack = R
    scp : (3,) array_like
        SCP position in ECEF coordinates (m)
    projection_set : ProjectionSetsLike
        Center of Aperture projection set for a single point
    ecef2dem_func : callable
        A function that returns an ndarray of DEM offset heights from an ndarray of positions with ECEF
        (WGS 84 cartesian) X, Y, Z components in meters in the last dimension

        .. Note:: SICD v1.4.0 volume 3 decomposes this into two steps:

           #. Convert ECF To DEM Coords
           #. Get Surface Height HD

    hae_min, hae_max : float
        WGS-84 HAE values (m) that bound DEM surface points
    delta_dist_dem : float
        Max horizontal distance between surface points (m) for which the surface is well approximated by a straight line
    delta_dist_rrc : float, optional
        Max distance between adjacent points along R/Rdot contour (m)
    delta_hd_lim : float, optional
        Height difference threshold for determining if a point on the R/Rdot contour is on DEM surface (m)

    Returns
    -------
    s : ndarray
        Set of point(s) where the R/Rdot contour intersects the DEM surface with ECEF (WGS 84 cartesian) X, Y, Z
        components in meters in the last dimension. Ordered by increasing WGS-84 HAE.

    Other Parameters
    ----------------
    **kwargs
        Keyword-only arguments for intermediate `r_rdot_to_constant_hae_surface` calls
    """
    if (
        getattr(
            projection_set, "ARP_COA", getattr(projection_set, "Xmt_COA", np.empty(()))
        ).size
        != 3
    ):
        raise ValueError("DEM projection only supported for scalar projection sets")

    if isinstance(projection_set, params.ProjectionSetsMono):
        # Compute center point, ctr, and the radius of R/Rdot contour
        vmag = np.linalg.norm(projection_set.VARP_COA)
        u_vel = projection_set.VARP_COA / vmag
        cos_dca = -projection_set.Rdot_COA / vmag
        sin_dca = np.sqrt(1 - cos_dca**2)
        ctr = projection_set.ARP_COA + projection_set.R_COA * cos_dca * u_vel
        r_rrc = projection_set.R_COA * sin_dca

        # Compute unit vectors to be used to compute points located on R/Rdot contour
        dec_arp = np.linalg.norm(projection_set.ARP_COA)
        u_up = projection_set.ARP_COA / dec_arp
        rry = np.cross(u_up, u_vel)
        u_rry = rry / np.linalg.norm(rry)
        u_rrx = np.cross(u_rry, u_vel)

        # Compute projection along R/Rdot contour to surface of constant HAE at hae_max, "a"
        a, _, success = r_rdot_to_constant_hae_surface(
            look, scp, projection_set, hae_max, **kwargs
        )
        if not success:
            return np.array([])

        # Also compute the cosine and sine of the contour angle to point "a"
        cos_ca_a = np.dot(a - ctr, u_rrx) / r_rrc
        # sin_ca_a is unused

        # Compute projection along R/Rdot contour to surface of constant HAE at hae_min, "b"
        b, _, success = r_rdot_to_constant_hae_surface(
            look, scp, projection_set, hae_min, **kwargs
        )
        if not success:
            return np.array([])

        # Also compute the cosine and sine of the contour angle to point "b"
        cos_ca_b = np.dot(b - ctr, u_rrx) / r_rrc
        sin_ca_b = look * np.sqrt(1 - cos_ca_b**2)

        # Compute contour angle step size
        delta_cos_rrc = delta_dist_rrc * np.abs(sin_ca_b) / r_rrc
        delta_cos_dem = delta_dist_dem * np.abs(sin_ca_b) / r_rrc / cos_ca_b
        delta_cos_ca = -min(delta_cos_rrc, delta_cos_dem)

        # Determine number of points along R/Rdot contour to be computed
        npts = (cos_ca_a - cos_ca_b) // delta_cos_ca + 2

        # Compute the set of points along R/Rdot contour
        n_minus_1s = np.arange(npts)
        cos_ca = cos_ca_b + n_minus_1s * delta_cos_ca
        sin_ca = look * np.sqrt(1 - cos_ca**2)
        pn = ctr + r_rrc * (
            cos_ca[..., np.newaxis] * u_rrx + sin_ca[..., np.newaxis] * u_rry
        )
    else:
        raise NotImplementedError("Bistatic is not implemented yet")

    # Compute DEM surface points
    delta_hdn = ecef2dem_func(pn)
    aobn = np.full(delta_hdn.shape, -1)
    aobn[delta_hdn > delta_hd_lim] = 1
    aobn[np.abs(delta_hdn) <= delta_hd_lim] = 0

    s = []
    for n_minus_1, ((p, _), (aob, next_aob), (delta_hd, next_delta_hd)) in enumerate(
        zip(
            itertools.pairwise(pn),
            itertools.pairwise(aobn),
            itertools.pairwise(delta_hdn),
            strict=True,
        )
    ):
        if aob == 0:
            s.append(p)
        if (aob * next_aob) == -1:
            frac = delta_hd / (delta_hd - next_delta_hd)
            cos_ca_s = cos_ca_b + (n_minus_1 + frac) * delta_cos_ca
            sin_ca_s = look * np.sqrt(1 - cos_ca_s**2)
            s.append(
                ctr
                + r_rrc
                * (
                    cos_ca_s[..., np.newaxis] * u_rrx
                    + sin_ca_s[..., np.newaxis] * u_rry
                )
            )

    return np.asarray(s)
